# Receipt: iamMrDuncan/Qwen3.8-27B-PN3-GGUF on one Tesla P100, engine-v0.2.0

What has been measured for `recipes/iamMrDuncan/Qwen3.8-27B-PN3-GGUF/p100.toml`, by whom, and on what. Numbers go here only with their scope.

**Status: released.** The research build of this engine was measured by the hill climb's judge, and the published image was measured on a P100 before the recipe was marked released ([below](#the-published-image-on-a-p100-2026-11-12)).

## What was measured

The engine source of v0.2.0 is the champion of an automated prefill hill climb that started from engine-v0.1.0: research commit `df673571`, champion 27 of the run, exported unchanged except for six comment lines, plus one fix made here afterwards (see `engine/PROVENANCE.md` and [the check below](#before-the-release-image-a-check-that-found-a-fault-2026-11-12)). The judge's numbers are for the champion without that fix. The recipe's `[run]` section did not change: the climb's judge ran the same `llama-server` flags and the same 21 engine switches as engine-v0.1.0's, and a new path had to be on by default to be measured at all.

The judge built each source with the same CUDA configuration as the image and timed it on one P100, from outside the engine (the driver stamps the request and each streamed token on its own clock). It measured the starting point when the climb began, and the champion with the champion before it, both on one card:

| Measure | engine-v0.2.0 (2026-11-12) | engine-v0.1.0 (2026-11-02) | Scope |
|---|---:|---:|---|
| Prefill, code prompt | **200.1 tok/s** | 121.4 | one cold prompt of about 4,100 tokens with `-b 512 -ub 512`: its tokens / the time from the request to the first token; the median of 3 confirmation blocks |
| Prefill, prose prompt | **198.9 tok/s** | 119.6 | the same, prose prompt |
| Decode, code / prose | 30.43 / 30.31 tok/s | 29.95 / 29.96 | 256 greedy tokens after a cold prompt of about 600 tokens |
| First token of the decode prompt, code / prose | 4.35 / 4.81 s | 7.12 / 7.83 | the same request |
| Qwen3.8-27B-UD-IQ3_XXS-Q3Floor decode, same card, same cells | 26.53 / 26.58 tok/s | 26.26 / 26.27 | the comparison model, decoded after Qwen3.8-27B-PN3 in every block |

Prefill is 1.65× engine-v0.1.0 on the code prompt and 1.66× on the prose prompt. The decode had a guard, not a goal: at most 1% below the starting point. It read 1.4% above it.

## What every step had to keep

The climb ran 369 attempts and promoted 26 of them, one on top of the other. Each promotion was confirmed on fresh blocks, and each had to pass these checks in every block:

- **Numerics against engine-v0.1.0's outputs.** A teacher-forced pass prefills about 2,350 tokens of a prose text in ubatches of 512, then forces the next 128 tokens one at a time, so the first forced position comes from the batched prefill and every forced token attends to its KV cache. Mean KL to engine-v0.1.0 within 2.593e-3, the same top token at 95% or more of positions, and decode-path perplexity within 0.25%.
- **The PN format.** The PN golden blocks and one real tensor of each PN class decode bit-exactly on CPU and GPU. Batched `MUL_MAT` of PN weights at the largest shapes stays within 5e-3 of the sum of |w·x|.
- **Qwen3.8-27B-UD-IQ3_XXS-Q3Floor.** Its greedy text stays bit-identical, and its decode speed stays within 3% of the starting point.
- **Memory and placement.** Every weight on the GPU, every layer offloaded, and at least 512 MiB of GPU memory free with the whole 65,536-token context allocated.
- **A hidden prompt.** A promotion could not be more than 1% slower on another cold prompt of about 2,400 tokens.

## Where the speed came from

Prefill on the code prompt after each group of promotions (each promotion's own confirmation, so the rows were measured on different days):

| Promotions | Change | Prefill after | Output |
|---|---|---:|---|
| start | engine-v0.1.0: every batched PN `MUL_MAT` dequantizes to fp32 and runs cuBLAS SGEMM | 121.4 tok/s | |
| 2–3 | PN dequantization writes four weights per thread and reads each packed word once | 135.3 | bit-identical |
| 4–7 | an exact-order fp32 GEMM for the 5,120- and 12,288-row shapes, and cuBLASLt for the other large shapes; a batch of 2 to 4 tokens runs as single-token products | 144.7 | full ubatches bit-identical; the 2-to-4-token route inside the numerics bars |
| 8–11 | the next weight dequantizes on a side stream while the current GEMM runs; the partial last ubatch takes the new GEMM too | 152.5 | bit-identical |
| 12–19 | a half2 (fp16) GEMM with fp32 accumulation across windows for the weights of blocks 32 to 63, except `attn_k` and `attn_v`; fp16 dequantization for it | 185.4 | inside the numerics bars |
| 20–23 | a fused SwiGLU epilogue, fp16 sums across windows for the smaller products, and concat, RMS norm, gated delta net and attention-tile changes on the batched path | 191.9 | inside the numerics bars |
| 24 | two consecutive prompt batches with no outputs run as one ubatch of up to 1,024 tokens; `ssm_beta` and `ssm_alpha` run as one GEMV at decode | 195.1 | inside the numerics bars |
| 25 | the half2 GEMM starts at block 31 | 196.2 | inside the numerics bars |
| 26–27 | a precision tier: shorter half2 chains for blocks 27 to 30, in the ubatches that are neither a prompt's first 512 tokens nor its final ubatch | 200.1 | inside the numerics bars |

## What to know about the half2 path

- **Prefill arithmetic is no longer engine-v0.1.0's.** From block 27 up, a prompt's batched products run on fp16 operands, so the KV cache a prompt leaves is close to engine-v0.1.0's, not equal to it. Single-token decode (1 to 4 tokens) keeps the PN GEMV kernels. `GGML_PN_H2_OFF=1` with `GGML_PN_TIER_LO=-1` gives fp32 arithmetic for every batched product again.
- **The precision tier depends on where a ubatch sits in the prompt.** It skips a prompt's first 512 tokens and the final ubatch, the one that produces the output. A tool that asks for logits on every batch, as `llama-perplexity` does, never runs the tier, so it does not measure what the server's prefill computes. The teacher-forced pass above does.
- **The tier was accepted by the owner, not by the automated audit.** The audit sent promotions 26 and 27 to the owner for two reasons: the tier's rule reads a batch's output flag, so the arithmetic depends on how a tool labels its batches, and each stacked precision change had been compared with the champion before it, not with the starting point. A check on six other texts followed (1,394 to 4,124 prompt tokens, 512 forced tokens each, one run), against the same build with the half2 path switched off. It read: mean KL 5.0e-5 to 9.1e-4, the same top token at 98.8% or more of positions, and decode-path perplexity between 0.9989× and 1.0024×. On the first 128 forced tokens the perplexity ratio was above 1.0025 for two of the six texts (1.0029 and 1.0040). The build without promotion 27 also read above it on two of the six, and that window moved by 0.3 to 0.8 points between the two builds.
- **The tier's ubatch position is process-wide state.** The recipe runs one slot (`-np 1`) and one context, which is what was measured. Interleaved contexts were not.

## The published image on a P100 (2026-11-12)

The image itself, `ghcr.io/iammrduncan/tensorless-engine@sha256:b95f77cd01cf0f785ee8cc853fc09454bdf4851f7214e8253c6529759c1eb7a7` (0.2.0-rc1, released as `0.2.0-sm60` without a rebuild), was run on one Tesla P100-PCIE-16GB with exactly the recipe's switches and flags:
- the model file was the lab's verified copy, sha256 `15ce2d44…`;
- the image ran as a cluster workload on the lab's P100 node, pulled by that digest;
- the card reported compute capability 6.0 and 16,269 MiB.

| Measure | Value |
|---|---:|
| `tensorless bench` decode, median of 3 runs (256 tokens after a 429-token prompt) | **30.48 tok/s** (30.44 / 30.48 / 30.49) |
| Against the research receipt (30.43) | +0.2% |
| llama-server's own decode timer, 3 runs of 256 tokens after an 18-token prompt | 30.49 / 30.50 / 30.48 tok/s |
| Time to first token, 429-token prompt | 2.60 s (median); engine-v0.1.0's image: 4.46 s |
| Prefill, a cold prompt of 3,744 tokens (code) / 4,657 tokens (prose), median of 3 | **198.9 / 201.5 tok/s** |
| Decode, 256 greedy tokens after a 555- / 700-token prompt | 30.52 / 30.75 tok/s |
| A chat of three turns with the prompt cache on, sent twice | answers, the same both times |
| GPU memory, idle / after the first prompt | 14,139 / 15,383 MiB |
| Load to listening, page cache partly warm | 26.0 s |

`bench` and the prefill timing ran on a Mac through Wi-Fi, ssh and a port-forward to the server. The network adds latency to the time to first token, but not to the decode rate, which is timed between streamed tokens. The prefill prompts are the first 13,500 characters of a code text and of a prose text, plus an instruction: the judge's method, with other prompts.

**Memory.** The engine takes its prefill buffers at the first prompt and keeps them, so it sits at 15,383 MiB from then on, about 890 MiB below the card's 16,269 MiB. engine-v0.1.0's image used 13,997 MiB idle and 14,577 MiB generating.

**Greedy text is not engine-v0.1.0's.** On both decode prompts, the 256 greedy tokens differ between engine-v0.1.0 and this engine. That is the fp16 prefill arithmetic: the numerics bars hold the distributions close, not the text identical.

## Before the release image: a check that found a fault (2026-11-12)

Before any release image existed, the lab built `containers/engine.sm60.Containerfile` from this repository's local commits on the P100 host and ran each image as a cluster workload on one Tesla P100-PCIE-16GB, with exactly the recipe's 21 switches and flags. A script on a Mac timed it through ssh and a port-forward, on its own clock: the judge's method, with other prompts (the first 13,500 characters of a code text and of a prose text, plus an instruction). The released engine-v0.1.0 image ran on the same card in between, as the baseline.

| Measure | engine-v0.2.0 source, with the fix | The champion's source, without the fix | engine-v0.1.0, released image |
|---|---:|---:|---:|
| Prefill, code prompt (3,744 tokens), median of 3 | **199.0 tok/s** | 198.5 | 118.9 |
| Prefill, prose prompt (4,657 tokens), median of 3 | **201.9 tok/s** | 201.7 | 121.0 |
| Decode, 256 greedy tokens after a 555- / 700-token prompt | 30.47 / 30.50 tok/s | 30.49 / 30.51 | 30.31 / 30.35 |
| `tensorless bench` decode, median of 3 (256 tokens after a 429-token prompt) | **30.50 tok/s** | 30.54 | 30.37 |
| `tensorless bench` time to first token, 429-token prompt | 2.60 s | 2.59 s | 4.47 s |
| A chat of three turns with the prompt cache on, sent twice | answers, the same both times | **the server aborts on turn 2** | answers, the same both times |
| GPU memory, idle after load / the maximum during the run | 14,139 / 15,383 MiB | 14,139 / 15,383 MiB | 13,997 / 14,577 MiB |

Prefill is 1.67× the released engine on both prompts, on the same card.

**The abort, and the fix.** The champion's source answered the first turn of a chat and aborted on the second with `failed to remove sequence 0 with p0=442, p1=-1`, after a 447-token first turn and after a 2,586-token one, and with Qwen3.8-27B-UD-IQ3_XXS-Q3Floor as well. The released engine-v0.1.0 image answered the same chats. The cause was in the change that holds a prompt batch until the next one arrives: while a batch was held, the engine reported its first position as the sequence's minimum, where this model's recurrent memory reports its last. `llama-server` records its context checkpoints with that value, so each came out one token short, and the next request that reused the prompt cache asked for a rollback the recurrent state cannot do. The hill climb's judge sent every prompt with `cache_prompt` false, so no judged run reached it. The fix is 12 lines in `engine/src` and touches no kernel. With it, the chats answer with the same cache reuse as engine-v0.1.0 (a 2,586-token first turn, then 60 and 25 tokens processed), the first turn's answer is the same text as engine-v0.1.0's, and the generated text and the speed are the same as without the fix.

**Qwen3.8-27B-UD-IQ3_XXS-Q3Floor on the champion's source** (not a release measurement: the image build for the fix was using the host's CPUs during these timings): prefill 144.2 / 148.4 tok/s, decode 26.61 / 26.60 tok/s, `tensorless bench` 26.67 tok/s, GPU memory 13,289 MiB idle and 13,869 MiB at its maximum.

## Not measured yet

- **The fix under the judge.** The judge measured the champion without the fix. The published image has the fix and reads the same speed, on other prompts and without the judge's numerics guard. The fix changes no arithmetic, only a position the engine reports to the server.
- **Qwen3.8-27B-PN3's quality checks on this engine.** Perplexity, needle recall and the task screen were last run on the decode climb's starting point (the engine-v0.1.0 receipt). Two climbs have since each held numerics within the same bars of their own starting points, and this one changed the prefill arithmetic, so the release gate should run them again on this engine.
- **Other ways into the held batch.** The checks above covered a chat that reuses the prompt cache. Saving and restoring a slot, a context shift on a full context, and a request cancelled in the middle of a prompt were not tried.
- **Prefill at 32K and 65K, and prompts over 4,100 tokens in general.** Attention costs a larger share of a long prompt, so expect less than 200 tok/s there.
- **Decode at 32K and 65K context.** As for engine-v0.1.0.
- **Speculative decoding** for Qwen3.8-27B-PN3.
