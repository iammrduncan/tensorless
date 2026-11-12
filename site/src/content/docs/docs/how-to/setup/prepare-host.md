---
title: "Prepare a Linux host for a P100"
description: "Install a driver that supports Pascal, give podman or docker the GPU, make room on the disk and the card, and get every tensorless doctor check to pass."
type: how-to
sidebar:
  order: 2
---

tensorless runs its engine in a container. So the host needs an NVIDIA driver that supports the P100, a container runtime (podman or docker), and NVIDIA container support, so that the runtime can hand the GPU to the container. The host does not need CUDA: the CUDA runtime is inside the engine image.

When every check of `tensorless doctor` passes, the host is ready.

## What you need

- A Linux x86_64 machine with an NVIDIA Tesla P100 **16 GB** installed. The 12 GB P100 does not fit the recipes.
- An x86-64 CPU with AVX2, FMA and F16C. The engine image's CPU code is built for that baseline, and some older Xeons that hosted P100s lack AVX2.
- `sudo` on that machine.
- The launcher: see [Install the launcher](install-launcher.md).

## 1. Install the NVIDIA driver

The P100 recipes need a driver from **525.60.13** up to the **R580** branch:

- **525.60.13** is the oldest driver that runs programs built with CUDA 12. The engine image is built with CUDA 12.6, because CUDA 13 dropped Pascal cards.
- **R580** is the last driver branch that supports Pascal. Newer branches do not drive the P100.

Install the driver from your distribution's packages, or from [NVIDIA's driver downloads](https://www.nvidia.com/en-us/drivers/), then reboot.

Check it with a query like the one the launcher uses:

```sh
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv
```

The P100 shows compute capability `6.0` and about 16 GB of memory, with the driver version in the last column.

## 2. Install a container runtime

Install podman or docker. If both are installed, tensorless uses podman, unless you pass `--runtime docker`.

```sh
sudo apt install podman   # Debian, Ubuntu
sudo dnf install podman   # Fedora, RHEL
```

If you use docker, your user must be able to run it without `sudo`, usually by being in the `docker` group.

## 3. Install the NVIDIA Container Toolkit

The toolkit connects the container runtime to the driver, and provides the `nvidia-ctk` command. Install it with [NVIDIA's installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for your distribution.

## 4. Give the runtime the GPU

**With podman,** generate a CDI spec, the file that tells podman which device files and driver libraries to give a container:

```sh
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

The spec names the driver's files by version. **Generate it again after every driver upgrade,** or containers fail to start.

**With docker,** register the NVIDIA runtime and restart docker:

```sh
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

## 5. Make room on the disk

The model goes into the Hugging Face cache, `~/.cache/huggingface/hub` unless you choose another place:

| Model | File size |
|---|---:|
| Qwen3.8-27B-PN3 | 12.95 GB |
| Qwen3.8-27B-UD-IQ3_XXS-Q3Floor | 11.76 GB |

The engine image goes into the container runtime's own storage. It is about 1.6 GB to download, and takes a few GB once unpacked.

To keep models on a larger disk, see [Keep models on another disk](../run/model-cache.md).

## 6. Free the GPU

A model and its 65,536-token context take about 15 GB of the card's 16 GB: the release measured 14,139 MiB with the server idle and 15,383 MiB once it had read its first prompt. Anything else on the card can make the server fail to start, or run slower.

`nvidia-smi` lists the processes that use the card, at the bottom of its output. Stop them before you start a model.

## 7. Check the host

```sh
tensorless doctor
```

`doctor` checks, in order:

1. that `nvidia-smi` is present and lists the GPUs;
2. that a GPU matches a recipe: compute capability 6.0 and at least 16,000 MiB;
3. that the driver is at least the recipe's `min_driver`, 525.60.13;
4. that podman or docker is installed;
5. for podman, that a CDI spec exists at `/etc/cdi/nvidia.yaml` or `/var/run/cdi/nvidia.yaml`; for docker, that it has the nvidia runtime, a CDI spec or the toolkit's hook;
6. that the model cache has room for the largest model that is not downloaded yet.

Each check prints `PASS`, `WARN` or `FAIL`. A `FAIL` comes with the command that fixes it:

```text
FAIL  no NVIDIA CDI spec for podman in /etc/cdi/nvidia.yaml or /var/run/cdi/nvidia.yaml
      fix: install nvidia-container-toolkit, then: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

A `WARN` does not fail the check. For example, a second GPU that fits no recipe gets a `WARN`, and tensorless uses the card that fits.

## How to tell it worked

`tensorless doctor` ends with:

```text
all checks passed.
```

and exits with status 0. If any check fails, it ends with the number of failed checks and exits with status 1. [Fix a failed check or start](../use/troubleshoot.md) covers each failure.

Next: [Run your first model on a P100](../../tutorials/first-model.md).
