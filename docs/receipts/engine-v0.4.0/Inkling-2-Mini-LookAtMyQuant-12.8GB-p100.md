# Receipt: shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-12.8GB on one Tesla P100, engine-v0.4.0

What has been measured for `recipes/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-12.8GB/p100.toml`, by whom, and on what. Numbers go here only with their scope.

**Status: draft. engine-v0.4.0's published image has not been measured with this model yet.** The numbers below are the research build's: the goal campaign's judge measured a source build of this engine's champion commit on one P100. tensorless v0.4.0 was released with this recipe as a draft: its release gate measured the published image, `0.4.0-sm60` (`sha256:4a0153230be9e2b5957a5aaf58cbb0ba1f79bfeb94eacd4617189f16ca35e997`, built by CI from `b4db2870`), with the three released recipes only. The recipe pins that image, so `--allow-draft` runs it, untried on a P100. It becomes released in a later patch release, once that image has been gated on a P100 with this model.

## The model and its settings

Thinking Machines' Inkling-2-Mini is a mixture-of-experts model (architecture `inkling2_loop`): 24 layers, 128 experts of which 8 run for each token, and a core of middle layers that runs three times. Its model card gives 26.4B parameters, 3.4B of them active for a token, about 5.4B counting the loop's extra passes. The recipe runs LookAtMyQuant's 12.8 GB pack of it, unmodified:
- repository `shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-12.8GB`, revision `758a177f`;
- 63 files, 12,760,459,456 bytes in all, each pinned in the recipe by name, sha256 and size;
- the format `inkling2-packed-v1`: the weight matrices in LookAtMyQuant's trellis codes and NVFP4, one `.npz` file per layer, plus the embedding, the output head, lookup tables and steering tensors;
- Apache-2.0, from thinkingmachines/Inkling-2-Mini and the pack's repository.

**The pack is not one GGUF file, and it is not converted.** The engine reads the pack's files in place. Before each serve, a prepare step (`engine/tools/inkling-prepare/prepare.py`, python3 and numpy only, a few seconds) writes a GGUF of about 12 MB that holds the hyperparameters, the tokenizer and a table of the pack's arrays by file and offset, and the pack's folder. `llama-server` is given that GGUF and memory-maps the pack's files from the folder it names. The launcher runs the step in the engine image, with the pack's snapshot folder at the same path for the step and for the server (see [recipes/README.md](../../../recipes/README.md#a-pack-instead-of-one-file)). The judge ran the model the same way: the 63 files linked under their own names into one folder, the prepare step, then `-m` set to the GGUF.

The recipe's `[run]` is what the judge ran: a 32,768-token context with an f16 KV cache, one slot (`-np 1`), every layer on the GPU, no memory map of the GGUF (`-lm none`), flash attention as the engine chooses (`-fa auto`), batches of 512, 8 threads, and the same 21 engine switches as the North-Mini-Code-1.0 recipe.

## How the engine source was made

engine-v0.4.0 is the champion of a Thermite goal campaign on this model, run on one P100: champion 7 of the run, commit `3e343003`, engine tree `28aeb14d`, which started from engine-v0.3.0 (see `engine/PROVENANCE.md`).
- **What it adds:** the `inkling2_loop` model, which reads the pack in place; the pack's trellis, Lloyd-Max and NVFP4 decoders for the CPU and for `sm_60`; flash attention with an indexed relative-position bias, after the contract of upstream llama.cpp PR 32740; the prepare step; and `tools/inkling-dump`. 41 files change from engine-v0.3.0.
- **The tree in `engine/`** was rebuilt from the run's export: the 4-patch chain from engine-v0.3.0 to champion 7, checked blob for blob against every patch. One file of the champion is left out, a scratch script; nothing else differs from the champion's tree.

## What the judge measured

The research build of champion 7, on one P100. A client timed the server from outside the engine.

| Measure | Value | Scope |
|---|---:|---|
| Decode after a 512-token prompt | **28.52 tok/s** | the median of 3 greedy replies of 128 tokens, cold prompt |
| Decode after a 32,000-token prompt | **25.45 tok/s** | the median of 2 greedy replies of 128 tokens, cold prompt |
| Prefill, 4,096-token prompt | **239.89 tok/s** | a cold prompt |
| Peak card memory | 14,151 MiB | at a 32,768-token context |

The recipe's `decode_tps.short` is the first row. Unlike the other recipes' `short`, it is after a 512-token prompt, not 4,096. `tensorless bench` compares its median with it; its prompt is not the judge's, so expect a difference of a few percent.

## What the campaign held

The goal campaign had seven goals, and champion 7 met all of them on the judge's P100: the pack read losslessly; a 32K context on the card; prefill and greedy decode that match the pack's own float32 reference; and the two decode speeds and the prefill speed above. The held-out checks also passed.

Against the pack's float32 reference (LookAtMyQuant's own reference implementation, with the embedding in float32):

| Measure | Value | Prompts |
|---|---:|---|
| Mean KL divergence | 1.09e-4 | LookAtMyQuant's four reference prompts |
| Same top token | 99.67% of positions | the same |
| Greedy chat replies that match the reference's | 4 of 4 | four chat prompts |

A greedy reply is 64 tokens, decoded one at a time by the judge's own logits tool, not through `/v1/chat/completions`. It matches when its tokens equal the reference's to the end, or when the first difference is where the reference's top two tokens are within 0.05 nats.

## The prepare step in the image

Checked locally, without a GPU, on the image's runtime stage as the Containerfile builds it (a stub in place of the CUDA build stage), Python 3.12.3 and numpy 1.26.4 from Ubuntu 24.04:
- `tensorless serve` verified the 63 files of a cache that holds the pack at the recipe's revision, then ran the prepare step in the image exactly as it does for this recipe: no GPU, no network, the repository's cache folder read-only at `/models/repo` and an empty output folder at `/models/prepared`.
- The step listed 1,548 arrays and 160 packed matrices, a vocabulary of 200,058 tokens and 424,413 merges, and wrote `/models/prepared/inkling2.gguf`, 12,088,985 bytes, whose pack folder is `/models/repo/snapshots/758a177fdaba30e53cce1266504b5428290aa0c9`.
- With the server's mounts, every one of the 28 files the GGUF names opened at that folder, through the snapshot's links into `blobs/`, and each of the 1,548 arrays lay inside its file.

The server did not start: that host has no GPU. That check used a local build of the runtime stage, not the published image.

## Not measured yet

- **The published image.** Speed, memory, a cold load and chat replies on `0.4.0-sm60`, on a P100, through the launcher. Its host sync schedule is this engine's default, `blocking`, as the research build was measured; the three released recipes set `auto` ([why](Qwen3.8-27B-PN3-GGUF-p100.md#the-host-sync-schedule)).
- **`tensorless bench`** on this model.
- **The model's quality benchmarks** on this engine (the pack's model card reports GSM8K, HumanEval and others for its own runtime). The campaign held the output against the pack's float32 reference, not against a benchmark.
- **`/v1/chat/completions`.** The chat template has a thinking channel and sets a thinking effort of 0.9 unless it is given `reasoning_effort`. How the server shows the thinking, and whether `--jinja` renders the template as the reference does, was not checked on this engine.
- **The prompt cache** across turns, and more than one request at a time.
- **Contexts above 32,768 tokens,** and speculative decoding.
- **A load with no GPU.** The engine has CPU decoders for the pack, but no CPU load of this model was measured.
