# Credits

tensorless stands on a great deal of other people's work: an inference engine several forks deep, open models and quantizations, P100 owners who published their patches and numbers, and the papers behind the format, the kernels and the way we measured them. This page thanks them.

License obligations for the code tensorless ships are in [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSES/](LICENSES/). Model licenses are in each model's card. If we missed you, or got something wrong, open an issue or tell us on [Discord](https://discord.gg/3Qs2uejUf9).

## The code tensorless ships

- **[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)**, Georgi Gerganov and contributors. The foundation of the engine, plus its quantization formats (K-quants and i-quants), `llama-quantize`, the importance matrix, `llama-server` and the test tools. MIT.
- **[llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)**, TheTom: TurboQuant KV-cache types and native MTP, the base of llamAmpere. MIT.
- **[llamAmpere](https://github.com/JakeATX/llamAmpere)**, JakeATX: the fork our engine starts from (`2cb16936`), with its EXL3 GGUF path and Qwen3.8 work. MIT.
- **[exllamav3](https://github.com/turboderp-org/exllamav3)**, turboderp: the EXL3 format, whose code the engine uses for its EXL3 weight types. MIT.
- **Vendored libraries:** cpp-httplib, nlohmann/json, miniaudio, stb_image, subprocess.h, xxHash, SHA-1, SHA-256 and rotate-bits. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- **[BurntSushi/toml](https://github.com/BurntSushi/toml)**, used by the launcher. MIT.

## Models and quantizations

- **The Qwen team (Alibaba): [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).** The base model of every launch target. Apache-2.0.
- **[Unsloth](https://huggingface.co/unsloth): [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).** Qwen3.8-27B-UD-IQ3_XXS-Q3Floor is mostly their work:
  - about 73% of its bytes are their UD-IQ3_XXS, unchanged;
  - its `token_embd` is from their UD-IQ4_XS;
  - its re-quantized tensors come from their BF16 GGUF, quantized with their importance matrix.
  Their quantization ladder was also our main comparison, and their Dynamic 3.0 notes gave us the trajectory-divergence check we used to judge quantization quality.
- **[turboderp/Qwen3.8-27B-exl3](https://huggingface.co/turboderp/Qwen3.8-27B-exl3):** the EXL3 quantizations we measured on the P100.
- **bartowski and AtomicChat:** Qwen3.8-27B GGUF ladders we compared against.
- **Drafters we measured for speculative decoding on the P100:** [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) (with [DFlash](https://github.com/z-lab/dflash)), incoai's DFlash2 builds, and the RadixArk and Anbeeld DSpark drafters.

## P100 and Pascal know-how from the community

- **[shinbunbun/llama-cpp-p100-patches](https://github.com/shinbunbun/llama-cpp-p100-patches):** 29 patches for llama.cpp on the P100, each measured on our card. Two shaped our engine directly:
  - the half2 Q4_1 GEMV with LOP3 unpacking (patch 12);
  - the MTP draft-vocabulary subset (patch 15), which led to our draft-vocab maps.
- **[qr0n/p100-llama-cpp](https://github.com/qr0n/p100-llama-cpp):** a P100 fork whose fp16-subnormal Q4_K GEMV (patch 0007) is behind our output-head kernel.
- **[Joe11221/p100-llama-cpp](https://github.com/Joe11221/p100-llama-cpp):** P100 issue-rate measurements (packed half2, fp32, emulated int8).
- **[Kmic-68/llama.cpp](https://github.com/Kmic-68/llama.cpp):** P100 optimization patches and a detailed optimization log, including the blocked fp16 cuBLAS algorithm for prefill, GEMM-based flash attention and head-256 GQA attention.
- **[hma02/cublasHgemm-P100](https://github.com/hma02/cublasHgemm-P100):** HGEMM and SGEMM rates on the P100.
- **P100 owners who posted numbers:** on r/LocalLLM, a [thread of P100 owners' token rates](https://www.reddit.com/r/LocalLLM/comments/1vwvng2/tesla_p100_folks_whats_your_tps/) and a [P40 vs P100 vs V100 comparison on Qwen3.8](https://www.reddit.com/r/LocalLLM/comments/1w41178/the_lost_world_p40_vs_p100_vs_v100_in_qwen_38/). Also the llama.cpp issue on [SM60 accumulation](https://github.com/ggml-org/llama.cpp/issues/25593).

## Engines and projects we learned from

- **[TensorFold](https://github.com/ashhart/TensorFold)**, ashhart:
  - Its exact speculative decoding, in-place regrouping and chain kernels shaped our kernel and speculation work.
  - Its recipe harness was the evaluation baseline for the PN quantization.
  - Its `pull` / `serve` / `info` / `capabilities` command line shaped the tensorless launcher.
- **[Lucebox](https://github.com/Luce-Org/lucebox-hub)**, Luce-Org: DFlash with DDTree on the RTX 3090, and a hybrid DeltaNet megakernel.
- **[0xSero/qwen38-b70](https://github.com/0xSero/qwen38-b70)** (Qwen3.8-27B on Intel Arc Pro B70) and **[0xSero/trellis-serve](https://github.com/0xSero/trellis-serve)** (EXL3 trellis kernels on the RTX 3090).
- **[MiaAI-Lab's one-click Qwen3.8-27B installer](https://github.com/MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install):** an EXL3 configuration for 16 GB cards.
- **[exllamav2](https://github.com/turboderp-org/exllamav2)**, for reconstructing to fp16 for wide batches. **[SGLang](https://github.com/sgl-project/sglang)** and **[vLLM](https://github.com/vllm-project/vllm)**, for their Qwen3.8 and hybrid-model implementations; vLLM also ran our structured-output evaluation.
- **llama.cpp pull requests and discussions we built on:**
  - the fused gated-delta-net kernel ([#19504](https://github.com/ggml-org/llama.cpp/pull/19504));
  - Hadamard rotation for the quantized KV cache ([#21038](https://github.com/ggml-org/llama.cpp/pull/21038));
  - deterministic mode ([#16016](https://github.com/ggml-org/llama.cpp/pull/16016));
  - DFlash2 support ([#27342](https://github.com/ggml-org/llama.cpp/pull/27342));
  - the CUDA graphs work ([#6766](https://github.com/ggml-org/llama.cpp/pull/6766));
  - the KV-cache quantization KLD tables ([discussion #23470](https://github.com/ggml-org/llama.cpp/discussions/23470)).

## Papers

### Quantization: the PN format and its quantizer

- **GPTQ / OPTQ** (Frantar et al.), [arXiv 2210.17323](https://arxiv.org/abs/2210.17323): the quantizer behind PN3.
- **QuIP** (Chee et al.), [2307.13304](https://arxiv.org/abs/2307.13304), and **QuIP#** (Tseng et al.), [2402.04396](https://arxiv.org/abs/2402.04396): incoherence processing. No QuIP# code is used.
- **QuaRot** (Ashkboos et al.), [2404.00456](https://arxiv.org/abs/2404.00456), and **SpinQuant** (Liu et al.), [2405.16406](https://arxiv.org/abs/2405.16406): the rotation line our Hadamard-rotated basis follows.
- **QTIP** (Tseng, Sun, Hou, De Sa), [2406.11235](https://arxiv.org/abs/2406.11235): trellis-coded quantization, the basis of EXL3.
- **Other methods studied while designing the format:**
  - HIGGS (Malinovskii et al.), [2411.17525](https://arxiv.org/abs/2411.17525);
  - GPTAQ, [2504.02692](https://arxiv.org/abs/2504.02692);
  - Qronos, [2505.11695](https://arxiv.org/abs/2505.11695);
  - GAR, [2505.14638](https://arxiv.org/abs/2505.14638);
  - FlatQuant, [2410.09426](https://arxiv.org/abs/2410.09426);
  - AutoRound / SignRound, [2309.05516](https://arxiv.org/abs/2309.05516);
  - GSQ, [2604.18556](https://arxiv.org/abs/2604.18556);
  - RCO, [2605.00649](https://arxiv.org/abs/2605.00649);
  - ReQuant, [2608.07019](https://arxiv.org/abs/2608.07019).
- **Outliers and sensitive weights:**
  - Massive Activations (Sun et al.), [2402.17762](https://arxiv.org/abs/2402.17762);
  - Super Weights (Yu, Wang, Shan, Reed, Wan), [2411.07191](https://arxiv.org/abs/2411.07191);
  - Bondarenko, Nagel and Blankevoort, [2306.12929](https://arxiv.org/abs/2306.12929);
  - LLM.int8() (Dettmers et al.), [2208.07339](https://arxiv.org/abs/2208.07339);
  - Kozyrev and Maiboroda, [2609.04098](https://arxiv.org/abs/2609.04098), on which Qwen3.8-27B components tolerate low precision.

### Measuring quality and exactness

- **Answer flips** (Dutta et al.), [2407.09141](https://arxiv.org/abs/2407.09141), and **MarginGate**, [2605.30218](https://arxiv.org/abs/2605.30218): why KL divergence and top-token agreement are the right gates.
- **The Curious Case of Neural Text Degeneration** (Holtzman et al.), [1904.09751](https://arxiv.org/abs/1904.09751).
- **Run-to-run variance:** Yuan et al., [2506.09501](https://arxiv.org/abs/2506.09501), and [2510.26788](https://arxiv.org/abs/2510.26788), on numerical nondeterminism in inference.
- **Thinking Machines, ["Defeating Nondeterminism in LLM Inference"](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/):** batch invariance. **RepOps / Verde**, [2502.19405](https://arxiv.org/abs/2502.19405): reproducible bitwise inference.

### Speculative decoding

- **Speculative sampling:** Leviathan et al., [2211.17192](https://arxiv.org/abs/2211.17192); Chen et al., [2302.01318](https://arxiv.org/abs/2302.01318).
- **Drafters and verifiers:**
  - DFlash, [2602.06036](https://arxiv.org/abs/2602.06036);
  - DSpark, [2607.05147](https://arxiv.org/abs/2607.05147);
  - DFlare, [2606.02091](https://arxiv.org/abs/2606.02091);
  - xPress, [2608.02438](https://arxiv.org/abs/2608.02438);
  - DBloom, [2608.30427](https://arxiv.org/abs/2608.30427);
  - Judge Decoding, [2501.19309](https://arxiv.org/abs/2501.19309);
  - lossy verification, [2607.26627](https://arxiv.org/abs/2607.26627);
  - speculative decoding meets quantization, [2505.22179](https://arxiv.org/abs/2505.22179).
- **The RTX 3090 Qwen3.8-27B [speculative decoding study](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090)** by thc1006.

### Kernels and systems

- **Flash-Decoding** (Dao, Haziza, Massa, Sizov): the split-KV idea behind our P100 decode-attention kernel. **FlashAttention** (Dao et al.), [2205.14135](https://arxiv.org/abs/2205.14135).
- **DeepSeek-V3**, [2412.19437](https://arxiv.org/abs/2412.19437): promoting partial sums to fp32 at fixed intervals.
- **GPU synchronization microbenchmarks** (Zhang et al.), [2004.05371](https://arxiv.org/abs/2004.05371): the cost of a grid barrier on the P100.
- **Ideas we evaluated:**
  - TEAL, [2408.14690](https://arxiv.org/abs/2408.14690), activation sparsity;
  - Strassen on GPU, [1808.07984](https://arxiv.org/abs/1808.07984);
  - DistServe, [2401.09670](https://arxiv.org/abs/2401.09670), and Mooncake, [2407.00079](https://arxiv.org/abs/2407.00079), prefill/decode disaggregation.

## Data

The PN quantization was calibrated and evaluated on these datasets; tensorless does not redistribute any of them.

- **Calibration:**
  - [mlabonne/open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend);
  - [open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k);
  - [AI-MO/NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5);
  - [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2);
  - [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu);
  - [allenai/c4](https://huggingface.co/datasets/allenai/c4);
  - [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k).
- **Evaluation:**
  - WikiText-2 (Merity et al.; [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext));
  - JSONSchemaBench (Geng et al., [2501.10868](https://arxiv.org/abs/2501.10868));
  - BFCL v3 ([gorilla-llm](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard));
  - StructEval ([2505.20139](https://arxiv.org/abs/2505.20139));
  - IFEval ([2311.07911](https://arxiv.org/abs/2311.07911));
  - RULER ([2404.06654](https://arxiv.org/abs/2404.06654)).

## Hardware documentation

**NVIDIA:**
- the Pascal Tuning Guide;
- the Tesla P100 whitepaper and datasheet;
- the CUDA C Programming Guide, Math API and cuBLAS documentation;
- the CUDA 13 release notes, which set our CUDA 12.x floor for Pascal;
- the Container Toolkit's CDI guide.

## Tools, harnesses and the agents that did the work

- **[Thermite](https://github.com/iammrduncan/thermite):** the automated research harness that ran the PN campaigns: a judge the agents can't see, held-out checks, A/A noise calibration, an auditor, and the decode hill climb (131 judged attempts, 28 confirmed wins).
- **[pi-autoresearch](https://github.com/davebcn87/pi-autoresearch)** (davebcn87) and the **[Pi](https://pi.dev/)** coding agent: the experiment loop of the earlier P100 kernel campaigns. **[pod-pi](https://github.com/Hackers-in-the-Loop/pod-pi):** their rootless podman workspace.
- **NVIDIA's nvprof, compute-sanitizer, cuobjdump and nvdisasm**, which still support sm_60.
- **PyTorch, transformers, vLLM, llm-compressor, compressed-tensors and flash-linear-attention**, for the quantization pipeline. **LiteLLM**, for model routing. **Hugging Face**, for hosting the models.
- **The models that worked as researchers:**
  - GLM-5.3 Flash, DeepSeek V4.1 Flash, Ornith-1.5-122B and Claude Haiku 5.5 as workers;
  - GPT 6.1 Sol and Claude Opus 5.5 as teachers and auditors;
  - Claude Sonnet 5.5 as planner;
  - Claude Code and Codex for supervision, research and this repository.
