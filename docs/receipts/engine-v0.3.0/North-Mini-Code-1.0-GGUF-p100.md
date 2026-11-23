# Receipt: unsloth/North-Mini-Code-1.0-GGUF on one Tesla P100, engine-v0.3.0

What has been measured for `recipes/unsloth/North-Mini-Code-1.0-GGUF/p100.toml`, by whom, and on what. Numbers go here only with their scope.

**Status: released.** The research build of this engine was measured by the hill climb's judge, and the published image was measured on a P100 before the recipe was marked released ([below](#the-published-image-on-a-p100-2026-11-23)).

## The model and its settings

CohereLabs' North-Mini-Code-1.0 is a 30B mixture-of-experts model for code: 49 layers, 128 experts of which 8 run for each token, and a 4,096-token sliding window on 36 of the 49 layers. The recipe runs Unsloth's UD-IQ3_XXS file of it, unmodified:
- repository `unsloth/North-Mini-Code-1.0-GGUF`, revision `e306bb4b`, file `North-Mini-Code-1.0-UD-IQ3_XXS.gguf`;
- 11,708,375,136 bytes, sha256 `029b51f9…`;
- the experts in IQ2_S and IQ3_S, attention and the output head in Q6_K;
- Apache-2.0, from CohereLabs/North-Mini-Code-1.0 and unsloth/North-Mini-Code-1.0-GGUF.

The recipe's `[run]` is what the hill climb's judge ran: a 36,864-token context with a q8_0 KV cache, one slot (`-np 1`), every layer on the GPU, flash attention, batches of 512, and the same 21 engine switches as the Qwen3.8-27B recipes. The climb's decode paths are on by default for this model and need no switch.

## How the engine source was made

engine-v0.3.0 is the champion of an automated decode hill climb on this model, run on one P100: hill-climb commit `7d8be264`, champion 23 of the run (see `engine/PROVENANCE.md`).
- **The starting point** was engine-v0.2.0 plus one fix: the sliding-window KV cache keeps room for two ubatches, so a prompt longer than the window no longer streams nothing (`src/llama-kv-cache-iswa.cpp`).
- **The tree in `engine/`** was rebuilt from the run's export: the 17-patch chain from the starting point to the champion, checked blob for blob against every patch, then scanned for lab-only and private terms (none found). 18 files differ from engine-v0.2.0, none of them new.
- **Changed after the import,** and not measured by the judge: a guard for the router's thread blocks in the Q6_K GEMV, a branch that is never taken (the output is unchanged); the fix for a load with no GPU ([below](#with-no-gpu)); and `-DLLAMA_USE_PREBUILT_UI=OFF` in the Containerfile, so the build downloads nothing.

## What the judge measured

The judge built each source with the image's CUDA configuration and timed it on one P100, from outside the engine: it stamps the request and each streamed token on its own clock. Decode is 256 greedy tokens after a cold code prompt, as tokens over the time after the first token.

| Measure | engine-v0.3.0's research commit | The starting point | Scope |
|---|---:|---:|---|
| Decode after a 4,096-token prompt | **79.77 tok/s** | 49.28 | champion 23's confirmation on 5 fresh blocks |
| Decode after a 32,768-token prompt | **66.06 tok/s** | 35.83 | the same |
| The climb's decode metric, both prompts | 72.62 tok/s | 42.03 | +74.25%: the product of the confirmed gains |
| Prefill, 4,096-token prompt | 446.4 tok/s | | the prompt's tokens over the time from the request to the first token |
| Prefill, 32,768-token prompt | 334.7 tok/s | | the same |

The climb judged 127 submissions and confirmed 22 improvements. Sampled decode rose by 61.7%. Prefill was not a goal, and the climb left it unchanged.

## What the climb held

Every promotion had to keep the model's output close to the starting point's own:
- **Teacher-forced numerics:** mean KL divergence at most 0.003 nat, and the same top token at 98% or more of positions.
- **Greedy text:** 0 of the judge's greedy replies changed.

**Above about 8,192 tokens of context, the output is not the starting point's.** Below that, it is bit-identical. Above it, kernels chosen by context length add in another order, and the output differs from the starting point's inside the limits above.

## Where the speed came from

All on the CUDA path, and on by default for this model (`engine/PROVENANCE.md` has the files):
- **Attention** for one decoded token with a q8_0 KV cache at head size 128: a 16-lane KQ mapping, one K load shared by the two query heads of a group (four heads for long contexts), whole-row KQ, a transposed reduction, and the KV split count chosen by context length.
- **The Q6_K attention projections:** rows per warp from the device fill, a fused activation prep shared by Q, K and V, Q, K and V merged at load into one GEMV, and the router GEMV as extra thread blocks of that launch.
- **The experts:** IQ3_S rows owned by a warp for the down projection, a leaner IQ2_S dot product for gate and up, one q8_1 quantization shared by gate and up, and SwiGLU writing q8_1 for the down projection.
- **Fusions:** the mixture-of-experts tail (router weights, sum and residual add) as one kernel, and RoPE fused with the q8_0 KV cache write.
- **Graph order** in `cohere2moe.cpp`, and contiguous one-token Q, K and V views, so that the fusions above can match.

## The published image on a P100 (2026-11-23)

The image `ghcr.io/iammrduncan/tensorless-engine@sha256:3423499a1676c557341bc7e959b65845b59cd791cc7b30e36f862c4c5528bd5e` was built once by CI from commit `6ca26460`, measured as `0.3.0-rc1-sm60`, and released as `0.3.0-sm60`, the same digest.
- It ran as a cluster workload on the lab's P100 node, pulled by that digest, with exactly the recipe's arguments and switches and `OMP_NUM_THREADS=8`, as the launcher runs it.
- The model file was the lab's copy of the file the recipe pins, sha256 `029b51f9…`.
- The card was a Tesla P100-PCIE-16GB. It reported compute capability 6.0 and 16,269 MiB.

| Measure | Value |
|---|---:|
| `tensorless bench` decode, median of 3 runs (256 tokens after a 411-token prompt) | **93.70 tok/s** (93.70 / 93.52 / 94.15) |
| Time to first token, 411-token prompt | 1.415 s (median) |
| Decode, 256 greedy tokens after a 3,911-token code prompt, median of 3 | **79.94 tok/s** (llama-server's own timer: 79.8) |
| Decode, 256 greedy tokens after a 31,832-token code prompt, median of 3 | **66.63 tok/s** (llama-server: 66.6) |
| Prefill of the 31,832-token prompt, llama-server's own timer | 337.2 tok/s (94.4 s) |
| Prefill, a cold prompt of 3,122 tokens (code) / 3,827 tokens (prose), median of 3 | **441.6 / 439.3 tok/s** (llama-server: 458.7 / 455.0) |
| A chat of three turns with the prompt cache on, sent twice | answers, no abort; the answers differ between the two passes ([below](#the-prompt-cache)) |
| GPU memory, idle / from the first prompt on | 13,877 / 14,409 MiB (2,394 / 1,862 MiB free) |
| GPU memory after the 31,832-token prompt | 14,409 MiB |
| Load to listening, page cache warm | 3.79 s |

Against the judge: decode 79.94 against 79.77 after about 4,000 tokens (+0.2%), and 66.63 against 66.06 after about 32,000 (+0.9%).

`bench` reads 17.5% above the recipe's `decode_tps.short` (79.77), because its prompt is 411 tokens, not 4,096: attention reads less of the cache for each token. Compare `bench` on your own card with the 93.70 above, not with the receipt.

**How it was timed.** A client reached the server through a port-forward and sent `/completion` with temperature 0, `top_k` 1, `ignore_eos` and `cache_prompt` false. Prefill is the prompt's tokens over the time from the request to the first streamed event, so it includes the network; llama-server's own timer does not. The prompts are the start of two texts in this repository, plus an instruction ("Explain what the code above does." or "Summarize the text above."):
- code: `engine/src/llama-model.cpp` at `6ca26460`, its first 13,500 characters for prefill, 16,571 and 125,579 for the two decode prompts;
- prose: `engine/docs/llamampere-v0.3/ARTICLE.md`, its first 13,500 characters.

**Memory.** The weights take 11,228 MiB on the GPU, the KV cache 497 MiB for the 13 full-attention layers (36,864 cells) and 191 MiB for the 36 sliding-window layers (5,120 cells), both q8_0, and the compute buffers 343 MiB. The engine takes 532 MiB more at the first prompt and keeps it, so it sits at 14,409 MiB from then on, 32K prompts included.

### The prompt cache

The three-turn chat answered on both passes, and every turn reused the start of the prompt from the cache (122 to 513 tokens). But the two passes did not give the same answers: the first turn's reply, reasoning and answer, was 670 tokens on one pass and 729 on the other. The same request sent twice with `cache_prompt` false gave the same text both times.

The hill climb's judge saw the same thing. Its check compares a second turn that reuses the cache with the same turn sent cold, and it found a difference for the champion and for the candidates alike. So reusing the prompt cache can change this model's greedy output on this engine. That is a known limit, not something this release introduced. If you need the same answer for the same request, send it with `cache_prompt` false.

**The model reasons before it answers.** `/v1/chat/completions` returns its reasoning in `reasoning_content` and the answer in `content`. With `max_tokens` at 200, every turn spent its tokens on reasoning and returned an empty answer; with 1,024, the third turn still ended at the limit before it answered. Give it room.

### With no GPU

A load without a GPU is not a recipe, but it no longer aborts. The image ran in a container with no GPU device: CUDA did not start ("no usable GPU found"), the model loaded on the CPU (11,156 MiB), and the server was listening after 25.8 s. A greedy completion of 32 tokens after `def fibonacci(n):` was correct code, at 9.74 tok/s on 8 threads. Before the fix, a CUDA build of champion 23's tree aborted on the same load with `GGML_ASSERT(tensor->view_src == NULL)`: the load-time merge of Q, K and V made them views of one tensor, and the CPU loader cannot map a view. The merge now happens only for a layer on a GPU.

## Not measured yet

- **The model's quality checks.** The release gate ran no perplexity, code-task or recall check for this model. The climb held its output against the starting point's, not against a quality bar.
- **The two changes after the import, under the judge.** The judge measured champion 23 without them. The published image has them and reads the judge's speed, on other prompts. Neither changes the arithmetic on the P100.
- **Contexts above 36,864 tokens.** The recipe allocates 36,864; the model was trained for more.
- **Speculative decoding,** and more than one request at a time.
