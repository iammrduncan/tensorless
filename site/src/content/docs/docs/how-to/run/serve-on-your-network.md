---
title: "Serve on another port or address"
description: "Move the server off port 8080, or make it reachable from other machines through an SSH tunnel, a private network or every interface."
type: how-to
sidebar:
  order: 3
---

The server listens on `127.0.0.1:8080` by default, so only programs on the same machine can reach it. This page moves it to another port, and shows three ways to reach it from other machines, safest first.

You need the launcher installed and a host that passes `tensorless doctor`. See [Prepare a Linux host for a P100](../setup/prepare-host.md).

:::caution
The recipes start `llama-server` with no API key and no login. Anyone who can reach the port can use the model. Prefer an SSH tunnel or a private network to publishing the port on a network you don't control.
:::

## Use another port

Pass `--port` to `run` or `serve`:

```sh
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF --port 9000
```

The server is then at `http://127.0.0.1:9000`, with the API under `http://127.0.0.1:9000/v1`. Inside the container it still listens on 8080; `--port` changes only the host's side.

If something else already uses the port, the launcher refuses before it starts anything, with `cannot use port 8080 on 127.0.0.1` and the reason. Choose another port with `--port`.

`tensorless bench` takes the same `--port`, so pass it there too: `tensorless bench pn3 --port 9000`.

## Reach it from another machine through an SSH tunnel

This needs no change on the GPU host. On the machine you want to use it from, forward a local port to the server:

```sh
ssh -N -L 8080:127.0.0.1:8080 gpu-host-1
```

While that command runs, `http://127.0.0.1:8080/v1` on your machine reaches the server on `gpu-host-1`. Only people who can log in to `gpu-host-1` over SSH can use it.

## Serve on a private network

To publish the server on one address of the host only, give that address to `--host`. With [Tailscale](https://tailscale.com), for example, use the host's Tailscale address:

```sh
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF --detach --host "$(tailscale ip -4)"
```

The port is published on that address only, so devices on your tailnet reach the server at that address and port 8080, and other networks don't. The launcher waits for the server and prints the endpoint at the same address.

## Serve on every interface

`--host 0.0.0.0` publishes the port on every IPv4 address of the host (`--host ::` for IPv6):

```sh
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF --detach --host 0.0.0.0
```

When it is ready, the launcher prints the local endpoint and a reminder:

```text
iamMrDuncan/Qwen3.8-27B-PN3-GGUF is ready at http://127.0.0.1:8080 (OpenAI-compatible API under /v1)
it listens on every interface of this host (--host 0.0.0.0)
```

Use this only on a network you trust, or behind a firewall that limits who can connect.

:::note
`llama-server` can check an API key, with its `--api-key` option, which you can pass after `--`: `tensorless run pn3 -- --api-key "$KEY"`. The launcher labels any extra argument as untuned, because it cannot tell which arguments change the measured configuration (see [Change the context length or server arguments](change-context-or-args.md)). `tensorless bench` sends no key, so it cannot measure a server started with one.
:::

## Check that it worked

From the other machine, ask the server which model it serves, at the address you chose:

```sh
curl -s http://127.0.0.1:8080/v1/models              # through the SSH tunnel
curl -s http://gpu-host-1:8080/v1/models             # --host 0.0.0.0, on your network
```

The answer lists the model's repo id. Then point your client at that address: [Connect an OpenAI-compatible client](../use/openai-clients.md).
