---
title: "Install the launcher"
description: "Install the tensorless command on Linux or macOS, with the install script or from a release archive, checked by sha256."
type: how-to
sidebar:
  order: 1
---

The launcher, `tensorless`, is one static binary with no runtime dependencies. Install it with the install script, or download a release archive and check it yourself.

```sh
curl -fsSL https://tensorless.dev/install.sh | sh
```

## Where it runs

Each release has an archive for three platforms:

| Platform | Archive | What it can do |
|---|---|---|
| Linux x86_64 | `linux-x86_64` | Everything: pull, serve and benchmark models. |
| Linux aarch64 | `linux-aarch64` | The launcher runs, but the 0.2.0 engine image is published for `linux/amd64` only, so it cannot serve a model on an Arm host. |
| macOS, Apple silicon | `macos-arm64` | Read recipes and see exactly what would run: `models`, `info`, `capabilities` and `--dry-run`. A Mac has no NVIDIA GPU, so it serves no model. |

Serving a model also needs an NVIDIA driver, podman or docker, and NVIDIA container support. [Prepare a Linux host for a P100](prepare-host.md) sets those up.

## Install with the script

You need `curl` or `wget`, and `sha256sum` or `shasum`.

1. Run the script:

   ```sh
   curl -fsSL https://tensorless.dev/install.sh | sh
   ```

   It finds your platform, downloads that archive and its `.sha256` file from the [GitHub release](https://github.com/iammrduncan/tensorless/releases), checks the archive's sha256, and copies the one `tensorless` binary to `~/.local/bin`. It changes nothing else. You see:

   ```text
   Downloading tensorless 0.2.0 for linux-x86_64
   Checked sha256 cb0c796fd83a334eb67f163fffb30016f51513f8f6ca4a30e7144875487415eb
   Installed /home/you/.local/bin/tensorless
   ```

2. If the directory is not on your `PATH`, the script says so and prints the line to add, for example:

   ```text
   /home/you/.local/bin is not on your PATH. Add it, for example:
     export PATH="/home/you/.local/bin:$PATH"
   ```

   Run that line, and add it to your shell's startup file (such as `~/.bashrc`) to keep it.

Two environment variables change what the script does:

| Variable | Default | Meaning |
|---|---|---|
| `TENSORLESS_VERSION` | `0.2.0` | The release to install. |
| `TENSORLESS_INSTALL_DIR` | `$HOME/.local/bin` | Where the binary goes. The script creates the directory if needed. |

Set them for the `sh` that runs the script:

```sh
curl -fsSL https://tensorless.dev/install.sh | TENSORLESS_INSTALL_DIR="$HOME/bin" sh
```

:::tip
To read the script before it runs, download it first: `curl -fsSL https://tensorless.dev/install.sh -o install.sh`, read `install.sh`, then run `sh install.sh`.
:::

On any other platform, the script stops with `no release archive for` your system, and suggests building from source. See [Build the launcher from source](build-launcher.md).

## Install from a release archive

1. Download the archive for your platform and its `.sha256` file. Set `P` to `linux-x86_64`, `linux-aarch64` or `macos-arm64`:

   ```sh
   V=0.2.0
   P=linux-x86_64
   curl -fLO https://github.com/iammrduncan/tensorless/releases/download/v$V/tensorless-$V-$P.tar.gz
   curl -fLO https://github.com/iammrduncan/tensorless/releases/download/v$V/tensorless-$V-$P.tar.gz.sha256
   ```

2. Check the archive, then unpack it. On Linux:

   ```sh
   sha256sum -c tensorless-$V-$P.tar.gz.sha256 && tar -xzf tensorless-$V-$P.tar.gz
   ```

   On macOS, use `shasum -a 256 -c` in place of `sha256sum -c`. The check prints the archive's name followed by `OK`.

3. The archive's folder, `tensorless-0.2.0-linux-x86_64/`, holds the `tensorless` binary with `LICENSE`, `LICENSES/`, `NOTICE`, `THIRD_PARTY_NOTICES.md` and `README.md`. Put the binary on your `PATH`:

   ```sh
   mkdir -p ~/.local/bin
   install -m 755 tensorless-$V-$P/tensorless ~/.local/bin/
   ```

## Check that it worked

```sh
tensorless version
```

```text
tensorless 0.2.0 (commit bc8d564127a625363917c84491786dd41f91d1e0), linux/amd64
```

`tensorless models` lists the launch targets this launcher knows. On a machine without an NVIDIA GPU, it cannot tell whether they fit:

```text
MODEL                                            CARD  STATUS    ENGINE         FITS     CACHED
iamMrDuncan/Qwen3.8-27B-PN3-GGUF                 p100  released  engine-v0.2.0  unknown  no
iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF  p100  released  engine-v0.2.0  unknown  no
fits is unknown: nvidia-smi not found
```

Next: [Prepare a Linux host for a P100](prepare-host.md), or go straight to [Run your first model on a P100](../../tutorials/first-model.md) if the host is ready.
