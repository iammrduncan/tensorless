# Receipt: iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF on one Tesla P100, engine-v0.4.0

What has been measured for `recipes/iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF/p100.toml`, by whom, and on what.

**Status: released.** The recipe moved to engine-v0.4.0 after the published image was measured with this model on a P100. Its `[run]` section changed in one line: `GGML_CUDA_SYNC_SCHED = "auto"`, which keeps engine-v0.3.0's host sync schedule. engine-v0.4.0's new work is Inkling-2-Mini's, on paths this model does not take, so for this model the numbers below are the release gate's; the [Qwen3.8-27B-PN3 receipt](Qwen3.8-27B-PN3-GGUF-p100.md#what-changed-in-the-engine) of this release says more.

## The published image on a P100 (2026-11-24)

The image `ghcr.io/iammrduncan/tensorless-engine@sha256:4a0153230be9e2b5957a5aaf58cbb0ba1f79bfeb94eacd4617189f16ca35e997` was measured as `0.4.0-rc1-sm60` and released as `0.4.0-sm60`, the same digest.
- It ran on one Tesla P100-PCIE-16GB, with the recipe's arguments and 21 engine switches, `-t 8 -tb 8` and `OMP_NUM_THREADS=8`. 0 restarts.
- **Run 3** had no `GGML_CUDA_SYNC_SCHED`, so it ran at this engine's default, `blocking`. **Run 3b** set `GGML_CUDA_SYNC_SCHED=auto`, as the released recipe does. Each ran `tensorless bench` twice.
- The model file was a copy of the file the recipe pins, sha256 `4b4d6a39…`.
- During run 3, the host's two other P100s were busy at 100%. The gate's notes do not say whether they still were during run 3b.

| Measure | Run 3b: `auto`, the recipe | Run 3: `blocking` | engine-v0.3.0's gate (2026-11-23) |
|---|---:|---:|---:|
| `tensorless bench` decode, median of 3 runs (256 tokens after a 429-token prompt) | **26.63 tok/s** (26.62 / 26.63 / 26.63) | 26.56 (26.57 / 26.55 / 26.56) | 26.53 (26.53 / 26.61 / 26.39) |
| The same, a second time | 26.61 (26.61 / 26.57 / 26.61) | 26.61 (26.60 / 26.61 / 26.62) | |
| Against the research receipt (26.53) | +0.4%, +0.3% | +0.1%, +0.3% | −0.02% |
| Time to first token, 429-token prompt, median | 4.04 s | 4.06 s | 4.13 s |
| GPU memory, idle / generating | 13,289 / 13,869 MiB (2,402 MiB free) | 13,289 / 13,869 MiB (2,402 MiB free) | 13,289 / 13,869 |
| Load to listening | 8.9 s, page cache warm | 49.4 s, page cache cold | 57.4 s, cold |

For this model, `auto` and `blocking` read the same within the runs' spread. The recipe sets `auto` as the other released recipes do, which is how engine-v0.3.0 ran it; the [Qwen3.8-27B-PN3 receipt](Qwen3.8-27B-PN3-GGUF-p100.md#the-host-sync-schedule) says why.

## What the research measured

The recipe's numbers are unchanged: 26.53 tok/s at short context on engine-v0.2.0's research commit, and 24.52 / 22.92 tok/s at 32K / 65K on the decode climb's starting commit, four engine releases back. [The engine-v0.2.0 receipt](../engine-v0.2.0/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF-p100.md) has them in full.

## Not measured yet

- Prefill, decode after longer prompts and the chat on this engine: this gate ran `tensorless bench` only for this model. The [engine-v0.3.0 receipt](../engine-v0.3.0/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF-p100.md) has the last measurements of them.
- Whether this model's greedy text is engine-v0.3.0's. It stayed bit-identical through both Qwen hill climbs; this gate did not compare the text of the two engines.
- 32K / 65K context on this engine.
- Speculative decoding (MTP) on this engine.
