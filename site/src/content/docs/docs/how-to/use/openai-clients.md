---
title: "Connect an OpenAI-compatible client"
description: "Point curl, the Python openai library or any client that takes a base URL at the server tensorless starts."
type: how-to
sidebar:
  order: 1
---

The server tensorless starts is `llama-server`, which speaks the OpenAI API. Any client that lets you set a base URL can use it. This page connects curl and the Python `openai` library, and lists what any other client needs.

You need a model running: `tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF`, or the [first tutorial](../../tutorials/first-model.md).

## What every client needs

| Setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:8080/v1` (another address or port if you passed `--host` or `--port`) |
| Model | the repo id, `iamMrDuncan/Qwen3.8-27B-PN3-GGUF` (or `iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF`) |
| API key | none: the recipes start the server without one. A client that insists on a key can send any text. |

The model name is the recipe's alias, which is the repo id. `GET /v1/models` lists it:

```sh
curl -s http://127.0.0.1:8080/v1/models
```

## curl

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"iamMrDuncan/Qwen3.8-27B-PN3-GGUF","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64}'
```

The answer is the usual chat completion JSON; the text is in `choices[0].message.content`. The server applies the model's own chat template to `/v1/chat/completions`, because the recipes pass `--jinja`.

## Python

Install the `openai` library, then point it at the server:

```sh
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")

reply = client.chat.completions.create(
    model="iamMrDuncan/Qwen3.8-27B-PN3-GGUF",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    max_tokens=64,
)
print(reply.choices[0].message.content)
```

The library needs some `api_key`; the server does not check it.

To print the answer as it is generated, ask for a stream:

```python
stream = client.chat.completions.create(
    model="iamMrDuncan/Qwen3.8-27B-PN3-GGUF",
    messages=[{"role": "user", "content": "Write a haiku about an old GPU."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

## Sampling

Qwen3.8-27B-PN3's [model card](https://huggingface.co/iamMrDuncan/Qwen3.8-27B-PN3-GGUF) recommends temperature 1.0, top_p 0.95 and top_k 20. Send them with each request; that leaves the run as it was measured. `top_k` is not part of the OpenAI API, so the Python library sends it in `extra_body`:

```python
reply = client.chat.completions.create(
    model="iamMrDuncan/Qwen3.8-27B-PN3-GGUF",
    messages=[{"role": "user", "content": "Name three uses of a 16 GB GPU."}],
    temperature=1.0,
    top_p=0.95,
    extra_body={"top_k": 20},
)
```

## Other clients

Chat front ends, editors and agent tools that support "OpenAI-compatible" servers need the same three settings: the base URL, the model name and an empty or dummy key. Their names for these vary, often *API base*, *endpoint* or *base URL*.

- **A client on another machine** needs to reach the server first. See [Serve on another port or address](../run/serve-on-your-network.md).
- **A client in a container** on the same host cannot reach `127.0.0.1` of the host. Use the host's address, and start the server with `--host` on an address that container can reach.

:::note
The recipes run the server with one slot (`-np 1`): it works on one request at a time, and others wait their turn. Each request can use the whole 65,536-token context.
:::

## Check that it worked

The curl command above returns a JSON answer with the model's text in `choices[0].message.content`, and the server's log (in the terminal running `tensorless run`, or `podman logs`) shows the request. If the client says it cannot connect, check the server with `curl -s http://127.0.0.1:8080/health` on the GPU host, then the address and port the client uses.
