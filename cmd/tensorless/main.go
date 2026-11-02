// Command tensorless runs a GPU-tuned model in one command:
//
//	tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF
//
// It detects the GPU, picks the model's recipe for that card, pulls the
// engine image by digest and the model by revision and sha256, and serves it
// with podman or docker.
package main

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"os/signal"
	"syscall"

	"github.com/iammrduncan/tensorless"
	"github.com/iammrduncan/tensorless/internal/cli"
)

// Set at build time with -ldflags "-X main.version=... -X main.commit=...".
var (
	version = "dev"
	commit  = ""
)

func main() {
	recipes, err := fs.Sub(tensorless.Recipes, "recipes")
	if err != nil {
		fmt.Fprintln(os.Stderr, "tensorless:", err)
		os.Exit(1)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	code := cli.New(version, commit, recipes, tensorless.License).Run(ctx, os.Args[1:])
	stop()
	os.Exit(code)
}
