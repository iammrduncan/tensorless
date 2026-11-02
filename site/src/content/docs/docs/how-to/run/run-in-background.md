---
title: "Run a model in the background, and stop it"
description: "Start a model's server with --detach so it keeps running after the command returns, read its logs, and stop it."
type: how-to
sidebar:
  order: 1
---

By default, `tensorless run` stays in the foreground: it shows the server's logs, and Ctrl-C stops the server. With `--detach` it returns as soon as the model has loaded and leaves the server running, so you can close the terminal.

You need the launcher installed and a host that passes `tensorless doctor`. See [Prepare a Linux host for a P100](../setup/prepare-host.md).

## 1. Start the server in the background

```sh
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF --detach
```

`run` downloads and checks whatever is missing first, as it always does. Then it starts the server's container, waits for the model to load, and returns. Loading takes a few seconds once the file is in the page cache, and longer the first time. While it waits it prints:

```text
waiting for http://127.0.0.1:8080/health while the model loads (up to 20m)
```

When the server answers, it prints the endpoint and how to stop it:

```text
iamMrDuncan/Qwen3.8-27B-PN3-GGUF is ready at http://127.0.0.1:8080 (OpenAI-compatible API under /v1)
try it:
  curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model": "iamMrDuncan/Qwen3.8-27B-PN3-GGUF", "messages": [{"role": "user", "content": "Hello"}]}'
stop it with: tensorless stop iamMrDuncan/Qwen3.8-27B-PN3-GGUF
```

If the image and the model are already pulled, `tensorless serve iamMrDuncan/Qwen3.8-27B-PN3-GGUF --detach` starts the server without pulling or downloading anything. It still checks both, and fails if either is missing.

:::note
If the server stops before it is ready, `--detach` prints the last 60 lines of its log and fails. [Fix a failed check or a failed start](../use/troubleshoot.md) says what the usual lines mean.
:::

## 2. Read the logs

The server runs in a container named `tensorless-` followed by the model's repo id in lower case, with `/` replaced by `-`:

| Model | Container |
|---|---|
| `iamMrDuncan/Qwen3.8-27B-PN3-GGUF` | `tensorless-iammrduncan-qwen3.8-27b-pn3-gguf` |
| `iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF` | `tensorless-iammrduncan-qwen3.8-27b-ud-iq3_xxs-q3floor-gguf` |

Follow its log with the container tool that started it, podman or docker:

```sh
podman logs -f tensorless-iammrduncan-qwen3.8-27b-pn3-gguf
# or:
docker logs -f tensorless-iammrduncan-qwen3.8-27b-pn3-gguf
```

Ctrl-C stops following the log. The server keeps running.

## 3. Stop the server

```sh
tensorless stop iamMrDuncan/Qwen3.8-27B-PN3-GGUF
```

```text
stopped tensorless-iammrduncan-qwen3.8-27b-pn3-gguf
```

`stop` looks for the container in both podman and docker, whichever are installed, so you don't need to remember which one started it. The container was started with `--rm`, so stopping it also removes it. The model file and the image stay where they are, and the next `run` starts again in seconds.

## Check that it worked

While the server runs, it answers on its health address:

```sh
curl -s http://127.0.0.1:8080/health
```

After `tensorless stop`, the same command fails to connect, and `tensorless stop` again says:

```text
tensorless-iammrduncan-qwen3.8-27b-pn3-gguf is not running
```

The server does not start again by itself after a reboot. Run `tensorless serve MODEL --detach` again, or start it from your own service manager.
