---
title: "Measure your card against the receipt"
description: "Start Qwen3.8-27B-PN3 in the background, measure its decode speed with tensorless bench, and compare it with the speed the release measured."
type: tutorial
sidebar:
  order: 2
---

In this tutorial you measure how fast your P100 generates text with Qwen3.8-27B-PN3, the same way the release was measured. Then you compare your number with the recipe's receipt: the speed recorded in the recipe when it was released.

It takes about five minutes.

## Before you start

You need to have finished [Run your first model on a P100](first-model.md). The engine image and the model are then already on this machine. Nothing else should be using the GPU: another process on the card slows the model down.

## 1. Start the server in the background

```sh
tensorless serve iamMrDuncan/Qwen3.8-27B-PN3-GGUF --detach
```

`serve` starts the model from the image and model file that are already on the machine, and `--detach` returns as soon as the server is ready. You see:

```text
starting tensorless-iammrduncan-qwen3.8-27b-pn3-gguf (podman, GPU all)
waiting for http://127.0.0.1:8080/health while the model loads (up to 20m)

iamMrDuncan/Qwen3.8-27B-PN3-GGUF is ready at http://127.0.0.1:8080 (OpenAI-compatible API under /v1)
try it:
  curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model": "iamMrDuncan/Qwen3.8-27B-PN3-GGUF", "messages": [{"role": "user", "content": "Hello"}]}'
stop it with: tensorless stop iamMrDuncan/Qwen3.8-27B-PN3-GGUF
```

If loading takes longer than 30 seconds, a `still loading` line shows every 30 seconds until the server is ready.

## 2. Run the benchmark

```sh
tensorless bench iamMrDuncan/Qwen3.8-27B-PN3-GGUF
```

`bench` sends one short warm-up request. Then it sends three requests that each generate 256 tokens, and it times every token as it streams back. It takes about a minute. While it runs, it shows one line per run:

```text
benchmarking iamMrDuncan/Qwen3.8-27B-PN3-GGUF at http://127.0.0.1:8080: warm-up, then 3 runs of 256 tokens
run 1: <decode> tok/s decode, TTFT <ms> ms
run 2: <decode> tok/s decode, TTFT <ms> ms
run 3: <decode> tok/s decode, TTFT <ms> ms
```

Then it prints the result. Your own numbers replace the parts in angle brackets:

```text
iamMrDuncan/Qwen3.8-27B-PN3-GGUF on p100, 3 runs of 256 tokens (median):
  decode         <decode> tok/s
  TTFT           <ms> ms
  prompt tokens  429
  receipt        30.06 tok/s decode (short); this run is <difference>%
The receipt was measured with a different prompt, so expect a few percent of difference.
```

## 3. Read the result

- **decode** is how many tokens per second the model generates once it has started answering: the median of the three runs. This is the number you feel when you chat.
- **TTFT**, time to first token, is how long the request took before the first token arrived. Most of it is reading the 429-token prompt, so it grows with the length of the prompt.
- **receipt** compares your median with the recipe's receipt, `decode_tps.short = 30.06`. The research hill climb's judge measured that number when it tuned the engine.

When the release was made, the published image measured **30.03 tok/s** on a P100 with this same command: 0.1% below the receipt. If your result is within a few percent of 30.06, your card runs the model as it was measured.

If yours is much lower, look for these:

- **Another process on the GPU.** `nvidia-smi` lists the processes that use the card.
- **A hot or power-limited card.** `nvidia-smi` shows the card's temperature and power draw. A card that runs too hot slows its clocks.
- **An untuned run.** If the server was started with another `--context`, extra server arguments or another model revision, the launcher printed `UNTUNED` when it started, and the receipt does not apply. [Tuned and untuned runs](../explanation/tuned-and-untuned.md) explains why.

## 4. Stop the server

```sh
tensorless stop iamMrDuncan/Qwen3.8-27B-PN3-GGUF
```

```text
stopped tensorless-iammrduncan-qwen3.8-27b-pn3-gguf
```

## What you did

You measured your card the way the release gate measures a candidate engine, and you compared the result with the recipe's receipt. Every number tensorless publishes is a measurement like this one, written down with its scope.

Next:

- [Share your results](../how-to/use/share-results.md): numbers from other P100s help everyone.
- [Benchmark a running server](../how-to/use/benchmark.md): more runs, longer runs and JSON output.
- [How a release is made, and why you can trust its numbers](../explanation/releases-and-receipts.md).
- [The receipts](../reference/receipts.md) of engine-v0.1.0, for both launch targets.
