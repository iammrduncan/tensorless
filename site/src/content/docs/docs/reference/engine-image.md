---
title: "Engine image"
description: "The tensorless engine container image: its tags and digest, base images, contents, entrypoint, build configuration and host requirements."
type: reference
sidebar:
  order: 4
---

The engine image holds `llama-server` built from the [engine source](https://github.com/iammrduncan/tensorless/tree/main/engine) for one GPU architecture. There is one image per engine release and architecture. Recipes pin it by digest, never by tag.

## Published images

Registry: `ghcr.io/iammrduncan/tensorless-engine`. The images are public.

| Tag | Digest | Engine release | Architecture | Meaning |
|---|---|---|---|---|
| `0.2.0-sm60` | `sha256:b95f77cd01cf0f785ee8cc853fc09454bdf4851f7214e8253c6529759c1eb7a7` | `engine-v0.2.0` | `sm60` (Tesla P100) | The current release. Both recipes pin it. |
| `0.2.0-rc1-sm60` | `sha256:b95f77cd01cf0f785ee8cc853fc09454bdf4851f7214e8253c6529759c1eb7a7` | `engine-v0.2.0` | `sm60` | The release candidate the release gate measured. The release tag points at the same digest. |
| `0.1.0-sm60` | `sha256:35409ff97c2f518e468790c28bd19206ae777a945fcffe5775585e28fd9939a7` | `engine-v0.1.0` | `sm60` | The first release. It stays in the registry for the recipes of tensorless v0.1.0. |
| `0.1.0-rc1-sm60` | `sha256:35409ff97c2f518e468790c28bd19206ae777a945fcffe5775585e28fd9939a7` | `engine-v0.1.0` | `sm60` | Its release candidate, the same digest. |

```text
ghcr.io/iammrduncan/tensorless-engine@sha256:b95f77cd01cf0f785ee8cc853fc09454bdf4851f7214e8253c6529759c1eb7a7
```

## Tags

| Pattern | Example | Made by |
|---|---|---|
| `X.Y.Z-rcN-<arch>` | `0.2.0-rc1-sm60` | A release-candidate build ([`engine-image.yml`](https://github.com/iammrduncan/tensorless/blob/main/.github/workflows/engine-image.yml)). |
| `X.Y.Z-<arch>` | `0.2.0-sm60` | Promotion of a measured candidate's digest, without a rebuild ([`engine-promote.yml`](https://github.com/iammrduncan/tensorless/blob/main/.github/workflows/engine-promote.yml)). The workflow checks that the new tag points at the measured digest. |

The source of an engine release is tagged `engine-vX.Y.Z` in the repository. The launcher is versioned separately, `vX.Y.Z`.

## Host requirements

| Requirement | Value |
|---|---|
| GPU | Compute capability 6.0 (`sm60` images): the NVIDIA Tesla P100. |
| NVIDIA driver | R525 or newer, on the R580 branch or older. R580 is the last branch with Pascal support. The recipes' `min_driver` is 525.60.13. |
| CUDA on the host | Not needed: the CUDA 12.6 runtime is in the image. CUDA 13 dropped Pascal, so `sm60` images stay on CUDA 12.x. |
| Container runtime | podman with an NVIDIA CDI spec, or docker with the NVIDIA container toolkit. |
| CPU | x86-64 with AVX2, FMA and F16C. The image is built for `linux/amd64`. |

## Contents

| Path | Contents |
|---|---|
| `/opt/tensorless/bin/llama-server` | The server. |
| `/opt/tensorless/bin/*.so` | Its shared libraries, built with it. |
| `/opt/tensorless/LICENSE` | Apache-2.0. |
| `/opt/tensorless/NOTICE`, `/opt/tensorless/THIRD_PARTY_NOTICES.md` | Attribution, including the MIT-licensed upstreams. |
| `/opt/tensorless/LICENSES/` | The upstream license texts, the engine's `LICENSE` (as `engine-LICENSE.txt`) and the licenses of code vendored into the engine (`LICENSES/engine/`). |
| `/opt/tensorless/PROVENANCE.md` | Where the engine source came from: the upstream base, the research commit, the tree digest, what the export removed and rewrote, and known issues. |
| `/opt/tensorless/VERSION` | The version label the image was built with. For `0.2.0-sm60` it is the candidate's label, `0.2.0-rc1`, because the release is the candidate's digest. |
| `/opt/tensorless/SHA256SUMS` | The sha256 of every other file under `/opt/tensorless`. |

The image holds no model weights and no build tree.

| Setting | Value |
|---|---|
| Entrypoint | `/opt/tensorless/bin/llama-server` |
| Exposed port | `8080` |
| `PATH` | `/opt/tensorless/bin` first |
| `LD_LIBRARY_PATH` | `/opt/tensorless/bin` |
| Labels | `org.opencontainers.image.title` "tensorless engine (sm_60)", `.description`, `.source` https://github.com/iammrduncan/tensorless, `.licenses` "Apache-2.0 AND MIT", `.version` |

Arguments after the image name go to `llama-server`. The launcher passes the model path, `--host 0.0.0.0 --port 8080`, the context, the alias, the thread counts and the recipe's arguments; see [Command line](cli.md#serve).

## Build

The image is built from [`containers/engine.sm60.Containerfile`](https://github.com/iammrduncan/tensorless/blob/main/containers/engine.sm60.Containerfile), with the repository root as the build context. It has two stages.

| Stage | Base image | What happens |
|---|---|---|
| Build | `docker.io/nvidia/cuda:12.6.3-devel-ubuntu24.04` | Installs cmake, ninja and git; configures and builds `llama-server` and `test-pn-format`; runs `test-pn-format`, so a PN decoding error fails the build; copies the server and its libraries out. |
| Runtime | `docker.io/nvidia/cuda:12.6.3-runtime-ubuntu24.04` | Installs `libgomp1`; copies the server, libraries and license files into `/opt/tensorless`; writes `VERSION` and `SHA256SUMS`. |

| Build argument | Default | Meaning |
|---|---|---|
| `CUDA_VERSION` | `12.6.3` | The CUDA base image version. |
| `UBUNTU` | `ubuntu24.04` | The base image's Ubuntu. |
| `VERSION` | `dev` | Written to `/opt/tensorless/VERSION` and the `org.opencontainers.image.version` label. |

### CMake configuration

| Option | Value | Why |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | |
| `GGML_CUDA` | `ON` | The CUDA backend. |
| `CMAKE_CUDA_ARCHITECTURES` | `60-real` | `sm_60` code only, as measured. |
| `GGML_CUDA_FA` | `ON` | Flash attention. |
| `GGML_CUDA_FA_ALL_QUANTS` | `OFF` | Only the KV-cache type pairs below. |
| `GGML_CUDA_FA_QUANTS` | `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;f16-q8_0;q8_0-f16;bf16-q8_0;q8_0-bf16` | The measured KV-quant set. |
| `GGML_CUDA_GRAPHS` | `OFF` | No CUDA graphs, as measured. |
| `GGML_CUDA_NO_VMM` | `ON` | No virtual memory management, as measured. |
| `GGML_CUDA_COMPRESSION_MODE` | `size` | |
| `GGML_NATIVE` | `OFF` | Portable CPU code, independent of the builder's CPU. |
| `GGML_AVX`, `GGML_AVX2`, `GGML_FMA`, `GGML_F16C` | `ON` | The x86-64 baseline the CPU side is built for. |
| `GGML_CCACHE` | `OFF` | |
| `LLAMA_BUILD_TESTS`, `LLAMA_BUILD_SERVER`, `LLAMA_BUILD_TOOLS` | `ON` | |
| `LLAMA_BUILD_EXAMPLES`, `LLAMA_OPENSSL`, `LLAMA_BUILD_UI`, `LLAMA_BUILD_MTMD` | `OFF` | No examples, no TLS, no web UI, no multimodal tools. |
| `CMAKE_BUILD_RPATH_USE_ORIGIN` | `ON` | The server finds its libraries beside it. |
| `CMAKE_PLATFORM_NO_VERSIONED_SONAME` | `ON` | |

The research build of the same source used `GGML_NATIVE=ON`; the release build does not. The CUDA side keeps the configuration the hill climbs measured.

To build one yourself, see [Build the engine image yourself](../how-to/advanced/build-engine-image.md).
