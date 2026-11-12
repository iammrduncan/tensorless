---
title: "Switch to another model"
description: "Stop the model that holds the P100, then run another launch target, by its repo id or its short name."
type: how-to
sidebar:
  order: 2
---

A Tesla P100 has 16 GB, and one launch target with its 65,536-token context takes nearly all of it. To use another model, stop the one that is running, then run the other. This page switches from Qwen3.8-27B-PN3 to Qwen3.8-27B-UD-IQ3_XXS-Q3Floor (Q3Floor); the steps are the same in the other direction.

You need the launcher installed and a host that passes `tensorless doctor`. See [Prepare a Linux host for a P100](../setup/prepare-host.md).

## 1. See which models your launcher knows

```sh
tensorless models
```

```text
MODEL                                            CARD  STATUS    ENGINE         FITS  CACHED
iamMrDuncan/Qwen3.8-27B-PN3-GGUF                 p100  released  engine-v0.2.0  yes   yes
iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF  p100  released  engine-v0.2.0  yes   no
```

- **FITS** says whether a GPU on this host matches the recipe (`yes`, `no`, or `unknown` without a GPU).
- **CACHED** says whether the model file is already downloaded: `yes`, `partial` (an interrupted download) or `no`.

The launch targets, with their sizes and measured speeds, are in [Launch targets](../../reference/launch-targets.md).

## 2. Stop the running model

```sh
tensorless stop iamMrDuncan/Qwen3.8-27B-PN3-GGUF
```

If you started it in the foreground, Ctrl-C in its terminal does the same.

:::caution
Don't start a second model while the first holds the card. On the same port, the launcher refuses with `cannot use port 8080`. On another port, the second server runs out of GPU memory while it loads. On a host with several GPUs, you can run one model per card instead: see [Pick a GPU on a host with several](pick-a-gpu.md).
:::

## 3. Run the other model

```sh
tensorless run iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF
```

The first time, this downloads the 11.76 GB model file and checks its sha256. The engine image is the same one both recipes use, so it is not pulled again.

Each recipe also defines a short name, which works anywhere a repo id does:

| Short name | Repo id |
|---|---|
| `pn3` | `iamMrDuncan/Qwen3.8-27B-PN3-GGUF` |
| `q3floor` | `iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF` |

```sh
tensorless stop pn3
tensorless run q3floor --detach
```

The repo id is not case-sensitive, as on the Hugging Face Hub.

## Check that it worked

Ask the server which model it serves:

```sh
curl -s http://127.0.0.1:8080/v1/models
```

The answer lists `iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF`. Clients that name the model in their requests need the new repo id: see [Connect an OpenAI-compatible client](../use/openai-clients.md).
