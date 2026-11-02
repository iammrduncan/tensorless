---
title: "Files and environment variables"
description: "Where tensorless keeps models, verification markers and containers, and every environment variable the launcher and the install script read."
type: reference
sidebar:
  order: 8
---

## Files and directories

| What | Where |
|---|---|
| Model cache | `--cache DIR`, else `$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`. The path is made absolute, because the container mounts it. It must not contain `:`. |
| Model file | `<cache>/models--<org>--<repo>/blobs/<sha256>`. For example `~/.cache/huggingface/hub/models--iamMrDuncan--Qwen3.8-27B-PN3-GGUF/blobs/15ce2d44…`. |
| Snapshot link | `<cache>/models--<org>--<repo>/snapshots/<revision>/<file>`, a relative symlink to the blob, so other Hugging Face tools find the file. |
| Partial download | `<blob>.incomplete`. The next `pull` or `run` resumes it. |
| Verification markers | `$XDG_STATE_HOME/tensorless/verified/<sha256>.json`, else `~/.local/state/tensorless/verified/<sha256>.json`. |
| Engine images | The container runtime's own image storage. |
| Containers | `tensorless-<repo id>`: lowercased, with every character other than `a`–`z`, `0`–`9`, `_`, `.` and `-` replaced by `-`. Started with `--rm`, so a stopped container is removed. |
| Model inside the container | `/models/<file name>`, mounted read-only. |
| Recipes | Built into the binary. `--recipes DIR` reads `DIR/<org>/<repo>/<card>.toml` instead. |
| The launcher | Wherever you put it. The install script puts it in `$HOME/.local/bin/tensorless` unless `TENSORLESS_INSTALL_DIR` says otherwise. |

The cache directory names keep the recipe's spelling of the repo id, whatever case you typed.

### Verification markers

A marker records that a model file was hashed and matched its sha256, so that later commands need not hash 12 GB again. It holds the file's sha256, size, modification time (in nanoseconds) and absolute path:

```text title="~/.local/state/tensorless/verified/<sha256>.json (shape)"
{"sha256":"…","size":12954114496,"mtime_ns":…,"path":"/home/user/.cache/huggingface/hub/models--iamMrDuncan--Qwen3.8-27B-PN3-GGUF/blobs/15ce2d44…"}
```

A marker counts only while the file's size, modification time and path are the same. Otherwise the file is hashed again. Deleting the markers is safe: the next command hashes the file once and writes a new one. When the state directory cannot be written, the launcher says so and verifies again on the next run.

## Environment variables

### Launcher

| Variable | Default | Meaning |
|---|---|---|
| `HF_HUB_CACHE` | not set | The model cache directory. `--cache` takes precedence. |
| `HF_HOME` | not set | When `HF_HUB_CACHE` is not set, the cache is `$HF_HOME/hub`. |
| `HF_ENDPOINT` | `https://huggingface.co` | The Hugging Face Hub address model files are downloaded from: `<endpoint>/<org>/<repo>/resolve/<revision>/<file>`. |
| `HF_TOKEN` | not set | Sent as `Authorization: Bearer` with downloads, for private or gated repositories. It is never printed; `--dry-run` shows only that it is sent. |
| `XDG_STATE_HOME` | `~/.local/state` | The base of the state directory, `$XDG_STATE_HOME/tensorless`, which holds the verification markers. |

The launcher sets `OMP_NUM_THREADS` and the recipe's `GGML_CUDA_*` [engine switches](engine-switches.md) inside the container. It does not pass its own environment into the container.

### Install script

The install script, `https://tensorless.dev/install.sh`, reads:

| Variable | Default | Meaning |
|---|---|---|
| `TENSORLESS_VERSION` | `0.1.0` | The launcher release to install. Letters, digits, `.` and `-` only. |
| `TENSORLESS_INSTALL_DIR` | `$HOME/.local/bin` | Where to put the `tensorless` binary. Created if missing. |

It downloads `tensorless-<version>-<platform>.tar.gz` and its `.sha256` from the GitHub release, checks the sha256 with `sha256sum` or `shasum -a 256`, and installs one file, the binary. Platforms: Linux x86_64 (`linux-x86_64`), Linux aarch64 (`linux-aarch64`) and macOS on Apple silicon (`macos-arm64`). It needs `curl` or `wget`, and `sha256sum` or `shasum`.

## Network

The launcher sends no telemetry. It uses the network for one thing: downloading model files from the Hub (`HF_ENDPOINT`). The container runtime pulls the engine image from `ghcr.io`. `tensorless bench` talks only to the server at `--host` and `--port`.
