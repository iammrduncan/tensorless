# Receipt: iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF on one Tesla P100, engine-v0.2.0

What has been measured for `recipes/iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF/p100.toml`, by whom, and on what.

## The published image on a P100 (2026-11-12)

The image `ghcr.io/iammrduncan/tensorless-engine@sha256:b95f77cd01cf0f785ee8cc853fc09454bdf4851f7214e8253c6529759c1eb7a7` was measured as `0.2.0-rc1` and released as `0.2.0-sm60`, the same digest.
- It ran on one Tesla P100-PCIE-16GB, with exactly the recipe's switches and flags.
- The model file was the lab's verified copy, sha256 `4b4d6a39…`, the same bytes as the Hugging Face upload at revision `11f76ec7`.

| Measure | Value |
|---|---:|
| `tensorless bench` decode, median of 3 runs (256 tokens after a 429-token prompt) | **26.69 tok/s** (26.69 / 26.77 / 26.66) |
| Against the research receipt (26.53) | +0.6% |
| llama-server's own decode timer, 3 runs of 256 tokens after an 18-token prompt | 26.70 / 26.71 / 26.71 tok/s |
| Time to first token, 429-token prompt | 3.86 s (median) |
| Prefill, a cold prompt of 3,744 tokens (code) / 4,657 tokens (prose), median of 3 | 144.7 / 148.4 tok/s |
| Decode, 256 greedy tokens after a 555- / 700-token prompt | 26.61 / 26.78 tok/s |
| A chat of three turns with the prompt cache on, sent twice | answers, the same both times |
| GPU memory, idle / generating | 13,289 / 13,869 MiB |
| Load to listening, page cache partly warm | 29.8 s |

`bench` and the prefill timing ran on a Mac through Wi-Fi, ssh and a port-forward to the server. The network adds latency to the time to first token, but not to the decode rate, which is timed between streamed tokens. The prefill prompts are the first 13,500 characters of a code text and of a prose text, plus an instruction.

On engine-v0.1.0's image the same `bench` read 26.18 tok/s, with 13,157 MiB idle and 13,737 MiB generating (the engine-v0.1.0 receipt). This model's prefill was not measured on engine-v0.1.0's image, so this receipt has no before-and-after for it.

## What the research measured

The prefill hill climb's judge decoded this model in every block, as the comparison model for Qwen3.8-27B-PN3: 26.53 / 26.58 tok/s (code / prose) on this engine's research commit, against 26.26 / 26.27 on engine-v0.1.0 when the climb began. Its greedy text stayed bit-identical through the whole climb, as it had through the decode climb before it.

At 32K / 65K context, on the decode climb's starting commit, it decoded at 24.52 / 22.92 tok/s (code) and 24.54 / 22.97 (prose), with 1,024 greedy tokens after a prompt filling the context. That commit is two engine releases back.

## A fault found before the release

An image built from the hill climb's champion, before one fix, aborted on the second turn of any chat that reused the prompt cache, with this model as with Qwen3.8-27B-PN3. The published image has the fix. The Qwen3.8-27B-PN3 receipt of this release has the account.

## Not measured yet

- 32K / 65K context on this engine.
- This model's prefill on engine-v0.1.0, for comparison.
- Speculative decoding (MTP) on this engine: the earlier kernel campaign ran it at about 28 tok/s on its own engine version.
