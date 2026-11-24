# Receipt: unsloth/North-Mini-Code-1.0-GGUF on one Tesla P100, engine-v0.4.0

What has been measured for `recipes/unsloth/North-Mini-Code-1.0-GGUF/p100.toml`, by whom, and on what. Numbers go here only with their scope.

**Status: released.** The recipe moved to engine-v0.4.0 after the published image was measured with this model on a P100 ([below](#the-published-image-on-a-p100-2026-11-24)). Its `[run]` section changed in one line: `GGML_CUDA_SYNC_SCHED = "auto"`, which keeps engine-v0.3.0's host sync schedule ([below](#the-host-sync-schedule)).

## What changed in the engine

engine-v0.4.0 adds Thinking Machines' Inkling-2-Mini, read in place from LookAtMyQuant's pack (its [receipt draft](Inkling-2-Mini-LookAtMyQuant-12.8GB-p100.md)): the `inkling2_loop` model, the pack's decoders, flash attention with an indexed relative-position bias, and a prepare step. That work runs only for Inkling-2-Mini's architecture (`inkling2_loop`) or with the indexed bias, which this model does not use.

One change reaches every model: **the host sync schedule.** How the host thread waits for the GPU in a CUDA synchronize is now set by `GGML_CUDA_SYNC_SCHED`, and its default in this engine is `blocking` (the thread sleeps until the GPU signals). engine-v0.3.0 and earlier always used the CUDA driver's default, `auto`, which spins when there are fewer GPUs than CPU cores. The recipe sets `auto`.

The research numbers in the recipe are still engine-v0.3.0's: the decode hill climb's judge, on a source build of its hill-climb commit (decode 79.77 tok/s after 4,096 tokens and 66.06 after 32,768, prefill 446.4 tok/s). [The engine-v0.3.0 receipt](../engine-v0.3.0/North-Mini-Code-1.0-GGUF-p100.md) has them, with what that climb held.

## The published image on a P100 (2026-11-24)

The image `ghcr.io/iammrduncan/tensorless-engine@sha256:4a0153230be9e2b5957a5aaf58cbb0ba1f79bfeb94eacd4617189f16ca35e997` was built once by CI from commit `b4db2870`, measured as `0.4.0-rc1-sm60`, and released as `0.4.0-sm60`, the same digest.
- It ran as a cluster workload on one Tesla P100-PCIE-16GB, pulled by that digest, with the recipe's arguments and 21 engine switches, `-t 8 -tb 8`, `OMP_NUM_THREADS=8` and 8 CPUs. 0 restarts.
- **Run 1** had no `GGML_CUDA_SYNC_SCHED`, so it ran at this engine's default, `blocking`: the full gate, as engine-v0.3.0's. **Run 1b** set `GGML_CUDA_SYNC_SCHED=auto`, as the released recipe does: decode after a 3,911-token prompt and `tensorless bench` only.
- The model file was a copy of the file the recipe pins, sha256 `029b51f9…`. The client and the prompts were engine-v0.3.0's gate's.

| Measure | Run 1b: `auto`, the recipe | Run 1: `blocking` | engine-v0.3.0's gate (2026-11-23) |
|---|---:|---:|---:|
| `tensorless bench` decode, median of 3 runs (256 tokens after a 411-token prompt) | **94.82 tok/s** (95.15 / 93.05 / 94.82) | 93.57 (93.57 / 93.53 / 93.79) | 93.70 |
| Time to first token, 411-token prompt, median | 1.382 s | 1.407 s | 1.415 s |
| Decode, 256 greedy tokens after a 3,911-token code prompt, median of 3 | **79.31 tok/s** (78.90 / 80.51 / 79.31) | 79.51 (78.25 / 79.51 / 79.54) | 79.94 |
| Decode, 256 greedy tokens after a 31,832-token code prompt, median of 3 | | **66.23 tok/s** (66.23 / 66.41 / 66.08) | 66.63 |
| Prefill of the 3,911- / 31,832-token prompts, llama-server's own timer | 445.7 / | 444.2 / 333.6 tok/s | 446 to 447 / 337.2 |
| Prefill, a cold prompt of 3,122 tokens (code) / 3,827 tokens (prose), median of 3 | | **438.5 / 438.3 tok/s** (llama-server: 456.4 / 452.3) | 441.6 / 439.3 (458.7 / 455.0) |
| A chat of three turns with the prompt cache on, sent twice | | every request returned, no abort; the two passes' answers differ | the same |
| GPU memory, idle / from the first prompt on | 13,877 / 14,409 MiB | 13,877 / 14,409 MiB, then 15,273 from the chat's second pass ([below](#gpu-memory-in-run-1)) | 13,877 / 14,409 |
| Least memory free | 1,862 MiB | 998 MiB | 1,862 |
| Load to listening | 3.5 s, page cache warm | 58.1 s, page cache cold | 3.79 s, warm |

The model takes the same buffers on the GPU as on engine-v0.3.0: 11,228.03 MiB of weights, KV caches of 497.25 MiB (13 full-attention layers) and 191.25 MiB (36 sliding-window layers), and 342.54 MiB of compute buffers.

Against the judge, run 1b read 79.31 tok/s after about 4,000 tokens (79.77: −0.6%), and run 1 read 66.23 after about 32,000 (66.06: +0.3%). `tensorless bench` reads 18.9% above the recipe's `decode_tps.short` (79.77), because its prompt is 411 tokens, not 4,096: compare `bench` on your own card with 94.82, not with the receipt.

**How it was timed** is engine-v0.3.0's gate's way ([its receipt](../engine-v0.3.0/North-Mini-Code-1.0-GGUF-p100.md#the-published-image-on-a-p100-2026-11-23)): `/completion` through a port-forward, temperature 0, `top_k` 1, `ignore_eos`, `cache_prompt` false, and the same prompts. In the chat's first pass, the third turn ended at its 1,024-token limit while the model was still reasoning, as it did on engine-v0.3.0.

### The host sync schedule

For this model, `auto` and `blocking` read the same within the runs' spread: 79.31 against 79.51 tok/s after 3,911 tokens, and a bench of 94.82 against 93.57, where run 1b's own three bench runs spread from 93.05 to 95.15. For Qwen3.8-27B-PN3, `blocking` read about 1% slower ([its receipt](Qwen3.8-27B-PN3-GGUF-p100.md#the-host-sync-schedule)). So the three released recipes set `auto`, which is how engine-v0.3.0 ran them; the numbers this receipt quotes for the recipe are run 1b's.

### GPU memory in run 1

In run 1, the GPU's used memory stepped from 14,409 to 15,273 MiB at the first turn of the chat's second pass, and held there to the end of the run. At that turn the server saved the slot's 1,779-token state to its prompt cache, took a cached prompt that was 0.992 similar to the request, and evaluated 6 new tokens.
- engine-v0.3.0's gate ran the same chat and never went above 14,409 MiB.
- It was not reproduced: run 1b had no chat.
- The cause is not known. Apart from the host sync schedule, which run 1 used at its new default, the engine's changes since engine-v0.3.0 outside Inkling-2-Mini's own code take effect only for Inkling-2-Mini's architecture (`inkling2_loop`) or with the indexed bias.
- The server did not fail: 998 MiB stayed free.

### The prompt cache

As on engine-v0.3.0, the chat's two passes gave different answers; the [engine-v0.3.0 receipt](../engine-v0.3.0/North-Mini-Code-1.0-GGUF-p100.md#the-prompt-cache) explains it. Reusing the prompt cache can change this model's greedy output on this engine. If you need the same answer for the same request, send it with `cache_prompt` false.

## Not measured yet

- **With `auto`, the gate's other measures:** decode after 32K, prefill and the chat ran with `blocking` only.
- **The memory step in run 1,** a second time, and its cause.
- **Whether this model's greedy text is engine-v0.3.0's.** The gate did not compare the text of the two engines.
- Everything the [engine-v0.3.0 receipt](../engine-v0.3.0/North-Mini-Code-1.0-GGUF-p100.md#not-measured-yet) lists: the model's quality checks, contexts above 36,864 tokens, speculative decoding, and more than one request at a time.
