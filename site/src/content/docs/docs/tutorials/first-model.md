---
title: "Run your first model on a P100"
description: "Install tensorless on a Linux host with a Tesla P100, check the machine, and ask Qwen3.8-27B-PN3 a question through its OpenAI-compatible API."
type: tutorial
sidebar:
  order: 1
---

In this tutorial you install the tensorless launcher, check that your machine can run a model, start Qwen3.8-27B-PN3 and ask it a question. Qwen3.8-27B-PN3 is a 27-billion-parameter model, quantized to about 3 bits per weight for the Tesla P100.

At the end, the model answers on `http://127.0.0.1:8080`. It runs on the engine image and settings that the release measured at about 30 tokens per second on this card.

**How long it takes:** the downloads take most of the time. The model is 12.95 GB and the engine image is about 1.6 GB, so expect about 20 minutes at 100 Mbit/s or a few minutes at 1 Gbit/s. The rest takes about ten minutes.

## Before you start

You need:

- a Linux x86_64 machine with one NVIDIA Tesla P100 16 GB, which nothing else is using;
- an x86-64 CPU with AVX2, FMA and F16C;
- the NVIDIA driver, version 525.60.13 or newer, from the R580 branch or older;
- podman and the NVIDIA Container Toolkit, which provides the `nvidia-ctk` command;
- about 18 GB of free disk space: 12.95 GB for the model in your home directory, and a few GB for the engine image;
- `curl`.

If the driver or the toolkit is missing, follow [Prepare a Linux host for a P100](../how-to/setup/prepare-host.md) first, then come back here.

## 1. Install the launcher

The launcher, `tensorless`, is one program with no dependencies. This script downloads it from the GitHub release, checks its sha256 and puts it in `~/.local/bin`:

```sh
curl -fsSL https://tensorless.dev/install.sh | sh
```

You see:

```text
Downloading tensorless 0.2.0 for linux-x86_64
Checked sha256 cb0c796fd83a334eb67f163fffb30016f51513f8f6ca4a30e7144875487415eb
Installed /home/you/.local/bin/tensorless

Next: check this machine, then run a model.
  tensorless doctor
  tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF
```

If the script also says that `~/.local/bin` is not on your `PATH`, run the `export` line it prints. Then check the version:

```sh
tensorless version
```

```text
tensorless 0.2.0 (commit bc8d564127a625363917c84491786dd41f91d1e0), linux/amd64
```

## 2. Give podman the GPU

The engine runs in a container. Podman learns how to hand a container the GPU from a CDI spec: a file that lists the driver's devices and libraries. `nvidia-ctk` writes it:

```sh
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

The file `/etc/cdi/nvidia.yaml` now exists. It names the driver's files by version, so generate it again whenever you upgrade the driver.

## 3. Check the machine

```sh
tensorless doctor
```

`doctor` checks the GPU, the driver, the container runtime, the CDI spec and the free disk space. On a ready machine every line starts with `PASS`. The parts in angle brackets are your machine's own values:

```text
PASS  nvidia-smi lists 1 GPU
PASS  GPU 0: Tesla P100-PCIE-16GB, compute 6.0, <memory> MiB, driver <driver>; fits iamMrDuncan/Qwen3.8-27B-PN3-GGUF on p100, iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF on p100
PASS  driver <driver> is at least 525.60.13 (needed by iamMrDuncan/Qwen3.8-27B-PN3-GGUF on p100)
PASS  container runtime: podman (/usr/bin/podman)
PASS  NVIDIA CDI spec for podman: /etc/cdi/nvidia.yaml
PASS  model cache /home/you/.cache/huggingface/hub: <free> GB free, the model needs 12.95 GB

all checks passed.
```

A `FAIL` line has a `fix:` line under it with the command that fixes it. Run that command, then run `tensorless doctor` again. [Fix a failed check or start](../how-to/use/troubleshoot.md) explains each check.

## 4. Start the model

```sh
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF
```

The name is the model's Hugging Face repository. The launcher finds its recipe for your card: the exact engine image, model file and settings that were measured on a P100. Then it does four things, and tells you as it goes:

1. It pulls the engine image by its digest:

   ```text
   pulling the engine image ghcr.io/iammrduncan/tensorless-engine@sha256:b95f77cd01cf0f785ee8cc853fc09454bdf4851f7214e8253c6529759c1eb7a7
   ```

   Podman's own download progress follows.

2. It downloads the model into the Hugging Face cache, and checks the file's sha256 as it goes:

   ```text
   model Qwen3.8-27B-PN3-P0k19.gguf from iamMrDuncan/Qwen3.8-27B-PN3-GGUF at c701b6690ec3
   ```

   A progress line follows. It starts with `downloading Qwen3.8-27B-PN3-P0k19.gguf` and shows the percentage, how much of the 12.95 GB has arrived, and the speed in MB/s. If the download stops, run the same command again: it resumes where it stopped.

3. It starts the engine in a container:

   ```text
   starting tensorless-iammrduncan-qwen3.8-27b-pn3-gguf (podman, GPU all)
   ```

   The engine's own log lines follow while the model loads onto the GPU. The first start reads the whole 12.95 GB file from disk.

4. When the server answers, the launcher prints:

   ```text
   iamMrDuncan/Qwen3.8-27B-PN3-GGUF is ready at http://127.0.0.1:8080 (OpenAI-compatible API under /v1)
   try it:
     curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
       -d '{"model": "iamMrDuncan/Qwen3.8-27B-PN3-GGUF", "messages": [{"role": "user", "content": "Hello"}]}'
   Ctrl-C stops the server.
   ```

Leave this terminal open. The server runs for as long as the command does.

## 5. Ask it something

Open a second terminal, and send the model a chat request:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"iamMrDuncan/Qwen3.8-27B-PN3-GGUF","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64}'
```

The reply is one JSON object in the OpenAI chat format. The model's answer is in `choices[0].message.content`. In the first terminal, the engine's log shows the request.

The API under `/v1` is OpenAI-compatible, so chat apps and SDKs that speak to OpenAI can use this server too. The model name they send is the repository id.

## 6. Stop the server

Press Ctrl-C in the first terminal. The launcher prints:

```text
stopping tensorless-iammrduncan-qwen3.8-27b-pn3-gguf
```

The container is removed. The model stays in the Hugging Face cache and the image stays in podman's storage, so the next `tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF` downloads nothing and starts in seconds.

## What you did

You ran a 27-billion-parameter model on one P100, on the engine it was tuned with, from one command. The launcher pinned everything: it pulled the image by digest, downloaded the model at a fixed revision and checked its sha256.

Next:

- [Measure your card against the receipt](measure-your-card.md): see how fast your P100 runs this model.
- [Connect an OpenAI-compatible client](../how-to/use/openai-clients.md): use the model from a chat app or your own code.
- [Run a model in the background, and stop it](../how-to/run/run-in-background.md).
- [How tensorless works](../explanation/how-tensorless-works.md): recipes, images and receipts.
