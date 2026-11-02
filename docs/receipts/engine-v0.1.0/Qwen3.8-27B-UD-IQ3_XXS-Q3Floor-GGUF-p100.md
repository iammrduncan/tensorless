# Receipt: iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF on one Tesla P100, engine-v0.1.0

What has been measured for `recipes/iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF/p100.toml`, by whom, and on what.

## The published image on a P100 (2026-11-02)

The image `ghcr.io/iammrduncan/tensorless-engine@sha256:35409ff97c2f518e468790c28bd19206ae777a945fcffe5775585e28fd9939a7` was measured as `0.1.0-rc1` and released as `0.1.0-sm60`, the same digest.
- It ran on one Tesla P100-PCIE-16GB, with exactly the recipe's switches and flags.
- The model file was checked against sha256 `4b4d6a39…`, the same bytes as the Hugging Face upload at revision `11f76ec7`.

| Measure | Value |
|---|---:|
| `tensorless bench` decode, median of 3 runs (256 tokens after a 429-token prompt) | **26.18 tok/s** (26.16 / 26.18 / 26.19) |
| Against the research receipt (26.26) | −0.3% |
| llama-server's own decode timer, 3 runs of 256 tokens after a 45-token prompt | 26.16 / 26.19 / 26.21 tok/s |
| Time to first token, 429-token prompt | 3.85 s (median) |
| GPU memory, idle / generating | 13,157 / 13,737 MiB |
| Load to listening, page cache warm | 3.5 s |

`bench` ran on a Mac through Wi-Fi, ssh and a port-forward to the server. The network adds latency to the time to first token, but not to the decode rate, which is timed between streamed tokens.

## What the research measured

The research hill climb's judge decoded this model in every block, as the comparison model for Qwen3.8-27B-PN3: 26.26 / 26.27 tok/s (code / prose) on this engine commit. At 32K / 65K context, on the climb's starting commit, it decoded at 24.52 / 22.92 tok/s (code) and 24.54 / 22.97 (prose), with 1,024 greedy tokens after a prompt filling the context. Its greedy text stayed bit-identical through the whole climb.

## Not measured yet

- 32K / 65K context on this engine commit.
- Speculative decoding (MTP) on this engine: the earlier kernel campaign ran it at about 28 tok/s on its own engine version.
