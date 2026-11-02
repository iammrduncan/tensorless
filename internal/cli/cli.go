// Package cli implements the tensorless commands.
package cli

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"net"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/iammrduncan/tensorless/internal/execx"
	"github.com/iammrduncan/tensorless/internal/hfcache"
)

// App holds everything the commands touch outside the process, so that tests
// can replace it.
type App struct {
	Stdout, Stderr io.Writer
	Runner         execx.Runner
	Getenv         func(string) string
	Home           func() (string, error)
	HTTP           *http.Client // for the model download
	Recipes        fs.FS        // the embedded recipes directory
	License        string       // the license text bench builds its prompt from
	NumCPU         func() int
	FileExists     func(string) bool
	FreeSpace      func(dir string) (uint64, error)
	PortFree       func(host string, port int) error
	StderrTTY      bool
	Version        string
	Commit         string

	HealthInterval time.Duration
	HealthTimeout  time.Duration
}

// New returns an App wired to the host.
func New(version, commit string, recipes fs.FS, license string) *App {
	tty := false
	if fi, err := os.Stderr.Stat(); err == nil {
		tty = fi.Mode()&os.ModeCharDevice != 0
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.ResponseHeaderTimeout = time.Minute
	return &App{
		Stdout:  os.Stdout,
		Stderr:  os.Stderr,
		Runner:  execx.OS{},
		Getenv:  os.Getenv,
		Home:    os.UserHomeDir,
		HTTP:    &http.Client{Transport: transport},
		Recipes: recipes,
		License: license,
		NumCPU:  runtime.NumCPU,
		FileExists: func(p string) bool {
			_, err := os.Stat(p)
			return err == nil
		},
		FreeSpace:      hfcache.FreeSpace,
		PortFree:       portFree,
		StderrTTY:      tty,
		Version:        version,
		Commit:         commit,
		HealthInterval: time.Second,
		HealthTimeout:  20 * time.Minute,
	}
}

func portFree(host string, port int) error {
	l, err := net.Listen("tcp", net.JoinHostPort(host, strconv.Itoa(port)))
	if err != nil {
		return err
	}
	return l.Close()
}

type usageError struct{ msg string }

func (e usageError) Error() string { return e.msg }

func usagef(format string, args ...any) error { return usageError{fmt.Sprintf(format, args...)} }

// errFailed ends a command with exit status 1 after it has reported why.
var errFailed = errors.New("failed")

// errInterrupted ends a command that Ctrl-C or SIGTERM stopped.
var errInterrupted = errors.New("interrupted")

type command struct {
	name    string
	args    string
	summary string
	model   bool // takes exactly one MODEL argument
	extra   bool // accepts "-- extra llama-server args"
	json    bool
	detach  bool
	bench   bool
	run     func(a *App, ctx context.Context, o *options) error
}

var commands = []*command{
	{name: "run", args: "MODEL [flags] [-- llama-server args]", summary: "Pull, then serve.", model: true, extra: true, detach: true, run: (*App).cmdRun},
	{name: "pull", args: "MODEL [flags]", summary: "Pull the engine image by digest and download the model into the Hugging Face cache.", model: true, run: (*App).cmdPull},
	{name: "serve", args: "MODEL [flags] [-- llama-server args]", summary: "Start the server, wait until it is healthy, and print the endpoint.", model: true, extra: true, detach: true, run: (*App).cmdServe},
	{name: "stop", args: "MODEL [flags]", summary: "Stop the model's server.", model: true, run: (*App).cmdStop},
	{name: "bench", args: "MODEL [flags]", summary: "Measure decode speed against the running server and compare it with the receipt.", model: true, json: true, bench: true, run: (*App).cmdBench},
	{name: "models", args: "[flags]", summary: "List the recipes, whether they fit this GPU, and whether the model is cached.", json: true, run: (*App).cmdModels},
	{name: "info", args: "MODEL [flags]", summary: "Show a recipe exactly as it would run. Needs no GPU.", model: true, extra: true, json: true, run: (*App).cmdInfo},
	{name: "doctor", args: "[flags]", summary: "Check the GPU, driver, container runtime, NVIDIA container support and disk space.", run: (*App).cmdDoctor},
	{name: "capabilities", args: "[flags]", summary: "Describe this launcher: version, recipes, runtimes. Needs no GPU, runtime or network.", json: true, run: (*App).cmdCapabilities},
	{name: "version", args: "", summary: "Print the version.", run: (*App).cmdVersion},
}

func (a *App) usage(w io.Writer) {
	fmt.Fprint(w, `tensorless runs a GPU-tuned model in one command:

  tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF

MODEL is a Hugging Face repo id, optionally ORG/REPO@REVISION. See: tensorless models

Usage:
`)
	for _, c := range commands {
		fmt.Fprintf(w, "  tensorless %-13s %s\n", c.name, c.summary)
	}
	fmt.Fprint(w, `
Run 'tensorless COMMAND -h' for a command's flags.
`)
}

// Run runs one command line and returns the exit status.
func (a *App) Run(ctx context.Context, args []string) int {
	err := a.run(ctx, args)
	var ue usageError
	switch {
	case err == nil, errors.Is(err, flag.ErrHelp):
		return 0
	case errors.As(err, &ue):
		fmt.Fprintf(a.Stderr, "tensorless: %s\nRun 'tensorless help' for usage.\n", ue.msg)
		return 2
	case errors.Is(err, errFailed):
		return 1
	case errors.Is(err, errInterrupted):
		fmt.Fprintln(a.Stderr, "tensorless: interrupted")
		return 130
	}
	fmt.Fprintf(a.Stderr, "tensorless: %v\n", err)
	if ctx.Err() != nil {
		return 130
	}
	return 1
}

func (a *App) run(ctx context.Context, args []string) error {
	if len(args) == 0 {
		a.usage(a.Stderr)
		return usagef("no command given")
	}
	name := args[0]
	switch name {
	case "help", "-h", "-help", "--help":
		a.usage(a.Stdout)
		return nil
	case "--version", "-version":
		name = "version"
	}
	for _, c := range commands {
		if c.name == name {
			o, err := a.parse(c, args[1:])
			if err != nil {
				return err
			}
			return c.run(a, ctx, o)
		}
	}
	return usagef("unknown command %q", name)
}

func (a *App) home() string {
	h, err := a.Home()
	if err != nil {
		return ""
	}
	return h
}

func (a *App) getenv(k string) string {
	if a.Getenv == nil {
		return ""
	}
	return a.Getenv(k)
}

func (a *App) versionString() string {
	s := "tensorless " + a.Version
	if a.Commit != "" {
		s += " (commit " + a.Commit + ")"
	}
	return s + ", " + runtime.GOOS + "/" + runtime.GOARCH
}

func (a *App) cmdVersion(_ context.Context, _ *options) error {
	fmt.Fprintln(a.Stdout, a.versionString())
	return nil
}

// joinAnd joins items as "a, b and c".
func joinAnd(items []string) string {
	switch len(items) {
	case 0:
		return ""
	case 1:
		return items[0]
	}
	return strings.Join(items[:len(items)-1], ", ") + " and " + items[len(items)-1]
}
