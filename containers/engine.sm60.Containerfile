# tensorless engine image for sm_60 (NVIDIA Tesla P100).
#
#   docker buildx build -f containers/engine.sm60.Containerfile --build-arg VERSION=0.1.0 -t tensorless-engine:0.1.0-sm60 .
#
# Build context: the repository root. The CUDA side uses the configuration the hill climbs measured
# (sm_60 only, flash attention with the measured KV-quant set, no CUDA graphs, no VMM). The CPU side
# is built for a portable x86-64 baseline (AVX2/FMA/F16C) instead of the builder's own CPU.
# CUDA stays on 12.x: CUDA 13 dropped Pascal. The host needs only the NVIDIA driver (R525 or newer,
# on the R580 branch or older, which is Pascal's last).

ARG CUDA_VERSION=12.6.3
ARG UBUNTU=ubuntu24.04

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends cmake ninja-build git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY engine/ /src/engine/
RUN cmake -S engine -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60-real \
      -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=OFF \
      "-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;f16-q8_0;q8_0-f16;bf16-q8_0;q8_0-bf16" \
      -DGGML_CUDA_GRAPHS=OFF -DGGML_CUDA_NO_VMM=ON -DGGML_CUDA_COMPRESSION_MODE=size \
      -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
      -DGGML_CCACHE=OFF \
      -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_MTMD=OFF \
      -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON -DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON \
 && cmake --build build --parallel "$(nproc)" --target llama-server test-pn-format \
 && ./build/bin/test-pn-format \
 && mkdir -p /out/bin \
 && cp build/bin/llama-server /out/bin/ \
 && cp -a build/bin/*.so /out/bin/ \
 && objdump -p /out/bin/llama-server | grep -E 'NEEDED|RUNPATH'

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU}
ARG VERSION=dev
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/bin/ /opt/tensorless/bin/
COPY LICENSE NOTICE THIRD_PARTY_NOTICES.md /opt/tensorless/
COPY LICENSES/ /opt/tensorless/LICENSES/
COPY engine/LICENSE /opt/tensorless/LICENSES/engine-LICENSE.txt
COPY engine/licenses/ /opt/tensorless/LICENSES/engine/
COPY engine/PROVENANCE.md /opt/tensorless/
RUN echo "${VERSION}" > /opt/tensorless/VERSION \
 && cd /opt/tensorless && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
ENV PATH=/opt/tensorless/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/tensorless/bin
LABEL org.opencontainers.image.title="tensorless engine (sm_60)" \
      org.opencontainers.image.description="llama.cpp-based engine tuned for the NVIDIA Tesla P100" \
      org.opencontainers.image.source="https://github.com/iammrduncan/tensorless" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      org.opencontainers.image.version="${VERSION}"
EXPOSE 8080
ENTRYPOINT ["/opt/tensorless/bin/llama-server"]
