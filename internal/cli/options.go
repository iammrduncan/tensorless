package cli

import (
	"errors"
	"flag"
	"fmt"
	"strconv"
	"strings"

	"github.com/iammrduncan/tensorless/internal/recipe"
)

type options struct {
	recipes    string
	card       string
	gpu        int // -1: the first GPU that matches
	runtime    string
	cache      string
	allowDraft bool
	dryRun     bool
	host       string
	port       int
	context    int // 0: the recipe's
	json       bool
	detach     bool
	runs       int
	tokens     int

	model    string   // the MODEL argument without @REVISION: a repo id or a short name
	revision string   // the @REVISION of the MODEL argument, if any
	extra    []string // after "--"
}

type gpuFlag struct{ v *int }

func (g gpuFlag) String() string {
	if g.v == nil || *g.v < 0 {
		return ""
	}
	return strconv.Itoa(*g.v)
}

func (g gpuFlag) Set(s string) error {
	n, err := strconv.Atoi(s)
	if err != nil || n < 0 {
		return errors.New("must be a GPU index: 0, 1, ...")
	}
	*g.v = n
	return nil
}

// parse reads a command's flags. Flags may come before or after MODEL;
// everything after "--" is extra llama-server arguments.
func (a *App) parse(c *command, args []string) (*options, error) {
	o := &options{gpu: -1, runtime: "auto", host: "127.0.0.1", port: 8080, runs: 3, tokens: 256}
	fs := flag.NewFlagSet("tensorless "+c.name, flag.ContinueOnError)
	fs.SetOutput(a.Stderr)
	fs.Usage = func() {
		fmt.Fprintf(a.Stderr, "Usage: tensorless %s %s\n\n%s\n", c.name, c.args, c.summary)
		if c.name != "version" {
			fmt.Fprintln(a.Stderr, "\nFlags:")
			fs.PrintDefaults()
		}
	}
	if c.name != "version" {
		fs.StringVar(&o.recipes, "recipes", "", "read recipes from `DIR` (DIR/<org>/<repo>/<card>.toml) instead of the built-in ones")
		fs.StringVar(&o.card, "card", "", "use card `ID`'s recipe, e.g. p100, and skip GPU detection")
		fs.Var(gpuFlag{&o.gpu}, "gpu", "use GPU index `N` (default: the first GPU that matches the recipe)")
		fs.StringVar(&o.runtime, "runtime", "auto", "container runtime: auto, podman or docker (auto prefers podman)")
		fs.StringVar(&o.cache, "cache", "", "model cache `DIR` (default: $HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub)")
		fs.BoolVar(&o.allowDraft, "allow-draft", false, "allow a draft recipe, or a MODEL@REVISION other than the recipe's (the run is then untuned)")
		fs.BoolVar(&o.dryRun, "dry-run", false, "print every command and download URL instead of running it")
		fs.StringVar(&o.host, "host", "127.0.0.1", "host address to publish the server on")
		fs.IntVar(&o.port, "port", 8080, "host port to publish the server on")
		fs.IntVar(&o.context, "context", 0, "context length `N` (default: the recipe's; any other value makes the run untuned)")
	}
	if c.json {
		fs.BoolVar(&o.json, "json", false, "print JSON")
	}
	if c.detach {
		fs.BoolVar(&o.detach, "detach", false, "return once the server is healthy instead of following its logs")
	}
	if c.bench {
		fs.IntVar(&o.runs, "runs", 3, "number of measured runs")
		fs.IntVar(&o.tokens, "tokens", 256, "tokens to generate per run")
	}

	for i, s := range args {
		if s == "--" {
			o.extra = append([]string{}, args[i+1:]...)
			args = args[:i]
			if !c.extra {
				return nil, usagef("%s does not take extra llama-server arguments", c.name)
			}
			break
		}
	}
	var positional []string
	for {
		if err := fs.Parse(args); err != nil {
			if errors.Is(err, flag.ErrHelp) {
				return nil, err
			}
			return nil, usageError{err.Error()}
		}
		rest := fs.Args()
		if len(rest) == 0 {
			break
		}
		positional = append(positional, rest[0])
		args = rest[1:]
	}

	switch {
	case c.model && len(positional) == 0:
		return nil, usagef("%s needs a MODEL, e.g. tensorless %s iamMrDuncan/Qwen3.8-27B-PN3-GGUF (see: tensorless models)", c.name, c.name)
	case c.model && len(positional) > 1:
		return nil, usagef("%s takes one MODEL, got %s", c.name, strings.Join(positional, " "))
	case !c.model && len(positional) > 0:
		return nil, usagef("%s takes no arguments, got %s", c.name, strings.Join(positional, " "))
	}
	if c.model {
		m, rev, err := recipe.SplitModel(positional[0])
		if err != nil {
			return nil, usageError{err.Error()}
		}
		o.model, o.revision = m, rev
	}
	if o.port < 1 || o.port > 65535 {
		return nil, usagef("--port must be between 1 and 65535")
	}
	if o.context < 0 {
		return nil, usagef("--context must be positive")
	}
	if o.host == "" {
		return nil, usagef("--host must not be empty")
	}
	if c.bench && (o.runs < 1 || o.tokens < 2) {
		return nil, usagef("--runs must be at least 1 and --tokens at least 2")
	}
	for _, e := range o.extra {
		if f := recipe.ManagedFlag(e); f != "" {
			return nil, usagef("do not pass %s after --: the launcher sets it. Use --host, --port or --context instead; the model, alias and threads come from the recipe", f)
		}
	}
	return o, nil
}
