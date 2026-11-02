package execx

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"sync"
)

// Fake is a Runner for tests. Only the executables in Found exist; every call
// to one of them is recorded and answered by Func.
type Fake struct {
	// Found lists the executables that LookPath finds.
	Found map[string]bool
	// Func answers a call. It may write to stdout and stderr, and may block
	// until ctx is done. A nil Func answers every call with no output.
	Func func(ctx context.Context, name string, args []string, stdout, stderr io.Writer) error

	mu    sync.Mutex
	calls []string
}

// NewFake returns a Fake that finds the named executables.
func NewFake(found ...string) *Fake {
	f := &Fake{Found: map[string]bool{}}
	for _, name := range found {
		f.Found[name] = true
	}
	return f
}

// Calls returns every recorded call, each as "name arg arg ...".
func (f *Fake) Calls() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.calls...)
}

// LookPath implements Runner.
func (f *Fake) LookPath(file string) (string, error) {
	if f.Found[file] {
		return "/usr/bin/" + file, nil
	}
	return "", &exec.Error{Name: file, Err: exec.ErrNotFound}
}

// Output implements Runner.
func (f *Fake) Output(ctx context.Context, name string, args ...string) ([]byte, error) {
	var stdout, stderr bytes.Buffer
	err := f.call(ctx, name, args, &stdout, &stderr)
	if err != nil && stderr.Len() > 0 {
		err = fmt.Errorf("%s: %w: %s", name, err, strings.TrimSpace(stderr.String()))
	}
	return stdout.Bytes(), err
}

// Run implements Runner.
func (f *Fake) Run(ctx context.Context, stdout, stderr io.Writer, name string, args ...string) error {
	return f.call(ctx, name, args, stdout, stderr)
}

func (f *Fake) call(ctx context.Context, name string, args []string, stdout, stderr io.Writer) error {
	f.mu.Lock()
	f.calls = append(f.calls, strings.Join(append([]string{name}, args...), " "))
	f.mu.Unlock()
	if !f.Found[name] {
		return &exec.Error{Name: name, Err: exec.ErrNotFound}
	}
	if f.Func == nil {
		return nil
	}
	return f.Func(ctx, name, args, stdout, stderr)
}
