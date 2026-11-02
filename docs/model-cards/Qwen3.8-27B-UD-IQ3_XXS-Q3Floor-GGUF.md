---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: gguf
tags:
  - gguf
  - llama.cpp
  - unsloth
  - tensorless
  - p100
---

# Qwen3.8-27B-UD-IQ3_XXS-Q3Floor (GGUF)

**Join us on [Discord](https://discord.gg/3Qs2uejUf9)**: questions, results from your own P100, and ideas.

[Unsloth](https://huggingface.co/unsloth)'s `UD-IQ3_XXS` quantization of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with a **3-bit floor**: every weight tensor is at 3 bits or more. Unsloth's file stores 91 tensors below 3 bits. This one raises them to IQ3_XXS and changes nothing else.

It uses only standard llama.cpp quantization types, so it loads in stock llama.cpp. It is also a tuned launch target of [tensorless](https://github.com/iammrduncan/tensorless) on the NVIDIA Tesla P100:

```sh
tensorless run iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF
```

## What changed from Unsloth's file

The base is `Qwen3.8-27B-UD-IQ3_XXS.gguf` from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) at revision `4ca720788d1e01f1bff70c033e0d0028fd02e502`.

| Part | In Unsloth's UD-IQ3_XXS | Here |
|---|---|---|
| 91 GEMV tensors (about 27% of the decode weights) | IQ2_S 35, IQ2_XXS 24, IQ2_XS 12, Q2_K 9, IQ1_S 8, IQ1_M 3 | **IQ3_XXS**. Re-quantized with stock `llama-quantize` from Unsloth's BF16 GGUF and Unsloth's importance matrix (`imatrix_unsloth.gguf`), the matrix their file was made with |
| `token_embd` | Q2_K | **Q3_K**, copied byte for byte from Unsloth's `Qwen3.8-27B-UD-IQ4_XS.gguf` |
| Every other tensor, and all metadata | | Unchanged, byte for byte |

The GGUF metadata is still Unsloth's, including `general.name`: the file was renamed, not rewritten.

## File

| File | Size | sha256 |
|---|---:|---|
| `Qwen3.8-27B-UD-IQ3_XXS-Q3Floor.gguf` | 11,758,422,944 bytes | `4b4d6a39dca07c915472b3fdf70267385f6e6f3f38befd5223c9a1ac272cecd0` |

## Why

Some GPUs have no fast paths for the lowest-bit formats: the Tesla P100 (sm_60) has no DP4A, for example. There, a GEMV's cost follows instructions per weight, not bits per weight. In our P100 tests, raising the sub-3-bit tensors made decode **faster**: about 4.5% on a 32K/65K benchmark, compared with the original file on the same engine. It also made it better: on a short CPU check (1,024 tokens, not a full evaluation), perplexity was 8.70 against 9.25.

On one Tesla P100 with the tensorless engine (engine-v0.1.0's source, greedy decode, q4_0 KV cache):

| Measure | tok/s |
|---|---:|
| Decode, 256 tokens after a ~600-token prompt | 26.3 |
| Decode, 1,024 tokens after a 32K / 65K prompt | 24.5 / 22.9 |

## Credits

- **The Qwen team (Alibaba)**: Qwen3.8-27B.
- **Unsloth**: the UD-IQ3_XXS quantization this file is built on, the UD-IQ4_XS `token_embd`, the BF16 GGUF conversion and the importance matrix.
- **llama.cpp / ggml contributors**: the GGUF format, the quantization types and `llama-quantize`.
- **tensorless**: the 3-bit-floor recipe, the verification, and the P100 engine and testing.

## License

Apache-2.0, inherited from Qwen/Qwen3.8-27B and unsloth/Qwen3.8-27B-GGUF.
