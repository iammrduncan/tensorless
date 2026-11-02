// Package execx runs external commands behind an interface, so that tests can
// replace nvidia-smi, podman and docker with a fake.
package execx

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os/exec"
	"strings"
)

// Runner runs external commands. Arguments are always an argv, never a shell
// string.
type Runner interface {
	// LookPath reports where an executable is, like exec.LookPath.
	LookPath(file string) (string, error)
	// Output runs a command and returns its standard output. When the command
	// fails, the error includes its standard error.
	Output(ctx context.Context, name string, args ...string) ([]byte, error)
	// Run runs a command with its output connected to stdout and stderr, and
	// waits for it. Cancelling ctx kills the command.
	Run(ctx context.Context, stdout, stderr io.Writer, name string, args ...string) error
}

// OS runs commands on the host.
type OS struct{}

// LookPath implements Runner.
func (OS) LookPath(file string) (string, error) { return exec.LookPath(file) }

// Output implements Runner.
func (OS) Output(ctx context.Context, name string, args ...string) ([]byte, error) {
	var stdout, stderr bytes.Buffer
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if msg := strings.TrimSpace(stderr.String()); msg != "" {
			return stdout.Bytes(), fmt.Errorf("%s: %w: %s", name, err, msg)
		}
		if msg := strings.TrimSpace(stdout.String()); msg != "" {
			return stdout.Bytes(), fmt.Errorf("%s: %w: %s", name, err, msg)
		}
		return stdout.Bytes(), fmt.Errorf("%s: %w", name, err)
	}
	return stdout.Bytes(), nil
}

// Run implements Runner.
func (OS) Run(ctx context.Context, stdout, stderr io.Writer, name string, args ...string) error {
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("%s %s: %w", name, firstArg(args), err)
	}
	return nil
}

func firstArg(args []string) string {
	if len(args) == 0 {
		return ""
	}
	return args[0]
}
