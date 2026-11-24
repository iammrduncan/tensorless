# Receipt: iamMrDuncan/Qwen3.8-27B-PN3-GGUF on one Tesla P100, engine-v0.4.0

What has been measured for `recipes/iamMrDuncan/Qwen3.8-27B-PN3-GGUF/p100.toml`, by whom, and on what. Numbers go here only with their scope.

**Status: released.** The recipe moved to engine-v0.4.0 after the published image was measured with this model on a P100 ([below](#the-published-image-on-a-p100-2026-11-24)). Its `[run]` section changed in one line: `GGML_CUDA_SYNC_SCHED = "auto"`, which keeps engine-v0.3.0's host sync schedule ([below](#the-host-sync-schedule)).

## What changed in the engine

engine-v0.4.0 adds Thinking Machines' Inkling-2-Mini, read in place from LookAtMyQuant's pack (its [receipt draft](Inkling-2-Mini-LookAtMyQuant-12.8GB-p100.md)). That work takes effect only for Inkling-2-Mini's architecture (`inkling2_loop`) or with the indexed bias of its flash attention, which this model does not use.

One change reaches every model: **the host sync schedule.** How the host thread waits for the GPU in a CUDA synchronize is now set by `GGML_CUDA_SYNC_SCHED`, and its default in this engine is `blocking` (the thread sleeps until the GPU signals). engine-v0.3.0 and earlier always used the CUDA driver's default, `auto`, which spins when there are fewer GPUs than CPU cores. The recipe sets `auto`.

The research numbers in the recipe are still engine-v0.2.0's: the prefill hill climb's judge, on a source build of its research commit (decode 30.43 tok/s, prefill 200.1 tok/s). [The engine-v0.2.0 receipt](../engine-v0.2.0/Qwen3.8-27B-PN3-GGUF-p100.md) has them, with what that climb held.

## The published image on a P100 (2026-11-24)

The image `ghcr.io/iammrduncan/tensorless-engine@sha256:4a0153230be9e2b5957a5aaf58cbb0ba1f79bfeb94eacd4617189f16ca35e997` was built once by CI from commit `b4db2870`, measured as `0.4.0-rc1-sm60`, and released as `0.4.0-sm60`, the same digest.
- It ran as a cluster workload on one Tesla P100-PCIE-16GB, pulled by that digest, with the recipe's arguments and 21 engine switches, `-t 8 -tb 8`, `OMP_NUM_THREADS=8` and 8 CPUs. 0 restarts.
- **Run 2** had no `GGML_CUDA_SYNC_SCHED`, so it ran at this engine's default, `blocking`. **Run 2b** set `GGML_CUDA_SYNC_SCHED=auto`, as the released recipe does. Each ran `tensorless bench` twice.
- The model file was a copy of the file the recipe pins, sha256 `15ce2d44…`.
- During run 2, the host's two other P100s were busy at 100%. The gate's notes do not say whether they still were during run 2b.

| Measure | Run 2b: `auto`, the recipe | Run 2: `blocking` | engine-v0.3.0's gate (2026-11-23) |
|---|---:|---:|---:|
| `tensorless bench` decode, median of 3 runs (256 tokens after a 429-token prompt) | **30.46 tok/s** (30.46 / 30.51 / 30.44) | 30.20 (30.20 / 30.18 / 30.22) | 30.45 (30.32 / 30.45 / 30.46) |
| The same, a second time | 30.46 (30.52 / 30.46 / 30.44) | 30.13 (30.16 / 30.13 / 30.12) | |
| Against the research receipt (30.43) | +0.1%, +0.1% | −0.8%, −1.0% | +0.1% |
| Time to first token, 429-token prompt, median | 2.81 s | 2.80 s | 2.81 s |
| GPU memory, idle / after the first prompt | 14,139 / 15,381 MiB (890 MiB free) | 14,139 / 15,381 MiB (890 MiB free) | 14,139 / 15,381 |
| Load to listening | 8.4 s, page cache warm | 53.7 s, page cache cold | 60.3 s, cold |

### The host sync schedule

With the engine's new default, `blocking`, this model decoded about 1% slower than on engine-v0.3.0, in both bench runs; with `auto` it read engine-v0.3.0's speed again, in both. So the recipe sets `GGML_CUDA_SYNC_SCHED = "auto"`, which is how engine-v0.3.0 ran it. The other two released recipes set it too ([North-Mini-Code-1.0's receipt](North-Mini-Code-1.0-GGUF-p100.md#the-host-sync-schedule) has that model's two runs).

The comparison has one caveat: the host's other two cards were at 100% during the `blocking` run, and the gate's notes do not say whether they were during the `auto` run. Waiting for the GPU by spinning (`auto`) or by sleeping (`blocking`) can depend on how busy the host's CPUs are.

## Not measured yet

- **Prefill, decode after longer prompts and the chat on this engine.** This gate ran `tensorless bench` only for this model; engine-v0.3.0's [receipt](../engine-v0.3.0/Qwen3.8-27B-PN3-GGUF-p100.md) has the last measurements of them, on engine-v0.3.0.
- **Whether this model's greedy text is engine-v0.3.0's.** The gate did not compare the text of the two engines.
- Everything the [engine-v0.2.0 receipt](../engine-v0.2.0/Qwen3.8-27B-PN3-GGUF-p100.md#not-measured-yet) lists: the model's quality checks on this engine, prefill and decode at 32K and 65K, and speculative decoding.
