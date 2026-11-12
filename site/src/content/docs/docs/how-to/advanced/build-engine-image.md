---
title: "Build the engine image yourself"
description: "Build the sm_60 engine image from the repository's Containerfile with podman or docker, check it, and run it."
type: how-to
sidebar:
  order: 2
---

The engine image is built from the source in [`engine/`](https://github.com/iammrduncan/tensorless/tree/main/engine) by one Containerfile per GPU architecture. Build it yourself to read exactly what goes into it, to try a change to the engine, or to run without the published registry.

## Before you start

You need:

- an x86-64 Linux machine with podman or docker. Building needs no GPU;
- `git`;
- several GB of free disk space for the CUDA build image and the build.

The build compiles the CUDA kernels on every core of the machine, so it takes a while.

## 1. Get the source at a release

```sh
git clone --branch engine-v0.2.0 --depth 1 https://github.com/iammrduncan/tensorless
cd tensorless
```

`engine-v0.2.0` is the tag of the engine source in the `0.2.0-sm60` image. To build your own change, check out your branch instead.

## 2. Build

From the repository root, which is the build context:

```sh
podman build -f containers/engine.sm60.Containerfile --build-arg VERSION=0.2.0-local \
  -t tensorless-engine:0.2.0-local-sm60 .
# or: docker build -f containers/engine.sm60.Containerfile --build-arg VERSION=0.2.0-local -t tensorless-engine:0.2.0-local-sm60 .
```

`VERSION` is written into the image as `/opt/tensorless/VERSION`. Pick a label that says the image is yours.

The build stage configures the engine with the measured CUDA settings (`sm_60` only, flash attention with the measured KV-cache types, no CUDA graphs, no VMM) and a portable x86-64 CPU baseline. The full list is in [Engine image](../../reference/engine-image.md#cmake-configuration). It then builds `llama-server` and `test-pn-format`, and runs `test-pn-format`. If the PN format test fails, the build fails.

## 3. Check the image

The image lists the sha256 of every file it ships under `/opt/tensorless`. Check them, and read the version:

```sh
podman run --rm --entrypoint sh tensorless-engine:0.2.0-local-sm60 \
  -c 'cd /opt/tensorless && sha256sum -c --quiet SHA256SUMS && cat VERSION'
```

It prints `0.2.0-local` and nothing else. A mismatched file prints `FAILED`.

## 4. Run it

Your image's digest is not the release's. Builds are not byte for byte reproducible: the base images and packages they install change over time, and so do the layers' timestamps. The recipes pin the published digest, so the launcher does not use your build.

Run it as [Run the engine image without the launcher](without-launcher.md) shows, with `tensorless-engine:0.2.0-local-sm60` in place of the published image. To use it through the launcher instead, push it to a registry you control and pin its digest in a recipe of your own: see [Try a recipe of your own](write-a-recipe.md).

:::note
A run on your own build is not what the receipts measured, even from the same source. To compare, measure it with `tensorless bench`: see [Benchmark a running server](../use/benchmark.md).
:::

## How release images are built

Release images are built by the [`engine-image`](https://github.com/iammrduncan/tensorless/blob/main/.github/workflows/engine-image.yml) workflow, from the same Containerfile, as a release candidate (`X.Y.Z-rcN-sm60`). After the release gate has measured that candidate on a P100, the [`engine-promote`](https://github.com/iammrduncan/tensorless/blob/main/.github/workflows/engine-promote.yml) workflow gives its digest the release tag without rebuilding it. Why is in [How a release is made](../../explanation/releases-and-receipts.md).
