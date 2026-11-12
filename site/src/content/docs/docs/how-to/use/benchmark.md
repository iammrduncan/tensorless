---
title: "Benchmark a running server"
description: "Measure a server's decode speed with tensorless bench, the way the receipts were measured, and compare it with the recipe's receipt."
type: how-to
sidebar:
  order: 2
---

`tensorless bench` measures how fast a running server generates text, and compares the result with the number the recipe's receipt promises. Use it to check that your card performs as measured, or to compare settings.

You need a host that passes `tensorless doctor` ([Prepare a Linux host for a P100](../setup/prepare-host.md)) and nothing else using the GPU. For a guided first run, follow [Measure your card against the receipt](../../tutorials/measure-your-card.md).

## 1. Start the server in the background

```sh
tensorless serve iamMrDuncan/Qwen3.8-27B-PN3-GGUF --detach
```

`bench` does not start a server: it measures the one that answers on `--host` and `--port`, by default `127.0.0.1:8080`. Without one, it stops with:

```text
tensorless: no server answers http://127.0.0.1:8080/health. Start one with: tensorless serve iamMrDuncan/Qwen3.8-27B-PN3-GGUF --detach
```

## 2. Run the benchmark

```sh
tensorless bench iamMrDuncan/Qwen3.8-27B-PN3-GGUF
```

It sends one short warm-up request, then three measured ones, and prints each run as it ends. Then it prints the medians:

```text
benchmarking iamMrDuncan/Qwen3.8-27B-PN3-GGUF at http://127.0.0.1:8080: warm-up, then 3 runs of 256 tokens
run 1: <decode> tok/s decode, TTFT <ttft> ms
run 2: ...
run 3: ...
iamMrDuncan/Qwen3.8-27B-PN3-GGUF on p100, 3 runs of 256 tokens (median):
  decode         <decode> tok/s
  TTFT           <ttft> ms
  prompt tokens  <prompt>
  receipt        30.43 tok/s decode (short); this run is <difference>%
The receipt was measured with a different prompt, so expect a few percent of difference.
```

On the P100 that measured the release, the published image gave a median of 30.48 tok/s for this model, with a 429-token prompt: +0.2% against the receipt. Qwen3.8-27B-UD-IQ3_XXS-Q3Floor gave 26.69 tok/s against its receipt of 26.53. The [receipts](../../reference/receipts.md) have the details.

## What it measures

`bench` mirrors the hill climb's judge, which timed every change to the engine:

1. **Warm-up:** one request for 8 tokens, not counted.
2. **Measured runs:** `--runs` requests (3 by default) to `/completion`, each generating `--tokens` tokens (256 by default). They are greedy (temperature 0, top_k 1, seed 424242), ignore the end-of-text token so every run generates the full count, and don't reuse the prompt cache.
3. **Timing:** every streamed token is time-stamped as it arrives. **Decode** is (tokens − 1) divided by the time from the first token to the last. **TTFT** (time to first token) is the time from sending the request to the first token, so it includes reading the prompt.

The prompt is built in: the first 2,000 characters of the Apache-2.0 license, followed by "Continue the text above." The result is the median of the runs.

## Options

| Flag | Default | Use it to |
|---|---|---|
| `--runs N` | 3 | take more runs, for a steadier median |
| `--tokens N` | 256 | generate more tokens per run (at least 2) |
| `--json` | off | print the result as JSON |
| `--host`, `--port` | `127.0.0.1`, `8080` | reach a server on another address or port |

```sh
tensorless bench pn3 --runs 5 --tokens 256 --json
```

The JSON has the `model`, the `card` and the `endpoint`; each run's `decode_tps`, `ttft_ms`, `prompt_tokens` and `tokens`; the `median_decode_tps`, `median_ttft_ms` and `prompt_tokens`; and, when the recipe has a receipt, `receipt_decode_tps` and `vs_receipt_percent`.

## Benchmark a server on another machine

`bench` can run on your laptop against the GPU host. Decode speed is timed between tokens, so the network does not change it; TTFT includes the network's delay. Forward the port over SSH:

```sh
ssh -N -L 8080:127.0.0.1:8080 gpu-host-1
```

Then, in another terminal on your laptop:

```sh
tensorless bench iamMrDuncan/Qwen3.8-27B-PN3-GGUF
```

`bench` needs no GPU on the machine it runs on: it only reads the recipe to know which receipt to compare with. On a machine that has another NVIDIA GPU, add `--card p100`, or the launcher looks for a P100 there and finds none.

## Read the result

- **Within a few percent of the receipt:** your card and host perform as measured. The receipt used a different prompt (about 600 tokens), so a small difference is expected.
- **Much slower:** check that nothing else uses the GPU (`nvidia-smi`), that the server was started by tensorless with no `--context` or extra arguments, and that the card is not held back by heat or a low power limit (`nvidia-smi -q -d PERFORMANCE` lists the reasons a card is slowed).
- **A warning that the server serves another model:** `bench` asks the server which model it runs, and warns when it is not the one you named. The comparison with the receipt then does not apply.

If you started the server with another `--context`, pass the same `--context` to `bench`. It then notes that the run is untuned, and that the receipt's number is only a rough guide.

## Check that it worked

The `receipt` line compares your median with the receipt, or the output says the recipe has no decode receipt to compare with. Stop the server when you are done:

```sh
tensorless stop iamMrDuncan/Qwen3.8-27B-PN3-GGUF
```

To share what you measured, see [Share your results](share-results.md).
