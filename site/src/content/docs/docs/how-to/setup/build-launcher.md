---
title: "Build the launcher from source"
description: "Build the tensorless command with Go, for a platform without a release archive or to try a change, with the version stamped in."
type: how-to
sidebar:
  order: 3
---

Build the launcher yourself when your platform has no release archive, or to try a change to the launcher or its recipes. The launcher is plain Go with one dependency, and builds into a static binary.

## What you need

- Go 1.23 or newer.
- `git`, to build from a checkout.

## Install with `go install`

```sh
go install github.com/iammrduncan/tensorless/cmd/tensorless@v0.1.0
```

Go puts the binary in `$(go env GOPATH)/bin`, or in `$GOBIN` if you set it. `@latest` installs the newest release, and `@main` the newest commit of `main`.

A binary built this way has no version stamped in:

```sh
tensorless version
```

```text
tensorless dev, linux/amd64
```

## Build from a checkout

1. Get the source:

   ```sh
   git clone https://github.com/iammrduncan/tensorless
   cd tensorless
   ```

2. Build it, with the version and commit stamped in:

   ```sh
   go build -ldflags "-X main.version=$(git describe --tags --always --match 'v*') -X main.commit=$(git rev-parse --short HEAD)" ./cmd/tensorless
   ```

   `--match 'v*'` names the launcher's own tags (`v0.1.0`). The repository also has engine tags (`engine-v0.1.0`), which `git describe` would otherwise pick.

3. Check it:

   ```sh
   ./tensorless version
   ```

   At the tag `v0.1.0` this prints `tensorless v0.1.0 (commit` and the short commit. After the tag, `git describe` adds the number of commits since it, such as `v0.1.0-3-g147bf17`.

To build the way the release does, static and with the full commit, for any platform Go supports:

```sh
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath \
  -ldflags "-s -w -X main.version=0.1.0 -X main.commit=$(git rev-parse HEAD)" \
  -o tensorless ./cmd/tensorless
```

## Recipes are built in

The launcher embeds the `recipes/` directory when it is built ([recipes_embed.go](https://github.com/iammrduncan/tensorless/blob/main/recipes_embed.go)). A recipe you add or change in your checkout is part of the binary you build next.

To try recipes without rebuilding, point the launcher at a directory with `--recipes DIR`. See [Try a recipe of your own](../advanced/write-a-recipe.md).

## Run the checks

The project's CI runs these on every change. Run them before you send a pull request:

```sh
gofmt -l cmd internal *.go
go vet ./...
go test ./...
```

`gofmt -l` prints nothing when every file is formatted.

## How to tell it worked

`./tensorless version` prints your version and commit, and `./tensorless models` lists the recipes from your checkout's `recipes/` directory.
