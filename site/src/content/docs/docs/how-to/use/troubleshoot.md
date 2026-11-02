---
title: "Fix a failed check or a failed start"
description: "Read tensorless doctor's checks and the launcher's error messages, and do what each one asks."
type: how-to
sidebar:
  order: 4
---

When something is missing on the host, or a server does not start, the launcher says what failed and, almost always, what to run next. This page lists those checks and messages, and what to do about each.

## 1. Run the doctor

```sh
tensorless doctor
```

It checks the host in order and prints `PASS`, `WARN` or `FAIL` for each item, with a `fix:` line under each failure. On a machine without an NVIDIA driver and without the container toolkit, for example:

```text
FAIL  nvidia-smi not found: no NVIDIA driver is installed
      fix: install the NVIDIA driver, 525.60.13 or newer, and reboot
PASS  container runtime: podman (/usr/bin/podman)
FAIL  no NVIDIA CDI spec for podman in /etc/cdi/nvidia.yaml or /var/run/cdi/nvidia.yaml
      fix: install nvidia-container-toolkit, then: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
PASS  model cache /home/you/.cache/huggingface/hub: 231.19 GB free, the model needs 12.95 GB

2 checks failed.
```

It exits with status 1 when a check fails, and ends with `all checks passed.` when none does. Do the fixes in order, then run it again.

## 2. Fix what the doctor found

| Check | If it fails |
|---|---|
| `nvidia-smi` is present and lists the GPUs | Install the NVIDIA driver, 525.60.13 or newer, and reboot. If `nvidia-smi` is installed but fails, the driver is not loaded: reboot, then run `nvidia-smi`. For Pascal cards such as the P100, R580 is the last driver branch. |
| A GPU fits a recipe | The card's compute capability or memory matches no recipe. Every recipe today needs a 16 GB Tesla P100 (compute capability 6.0); a 12 GB P100 does not fit. `WARN ... fits no recipe` names a GPU that is skipped. |
| The driver is new enough | Install the driver the line names, or newer, and reboot. |
| podman or docker is installed | `sudo apt install podman` (Debian, Ubuntu) or `sudo dnf install podman` (Fedora, RHEL), or install docker. |
| podman: a CDI spec at `/etc/cdi/nvidia.yaml` or `/var/run/cdi/nvidia.yaml` | Install `nvidia-container-toolkit`, then `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`. Generate it again after every driver upgrade. |
| docker: the nvidia runtime, a CDI spec or the toolkit's hook | Install `nvidia-container-toolkit`, then `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`. If `docker info failed`, start docker and make sure your user may use it (the `docker` group). |
| Free space in the model cache | Free some space, or move the cache to a larger disk: [Keep models on another disk](../run/model-cache.md). |

The full setup is in [Prepare a Linux host for a P100](../setup/prepare-host.md).

## 3. Fix a failed command

Find the start of the message, then do what the right-hand column says. Each message starts with `tensorless:`.

| Message | What to do |
|---|---|
| `cannot detect the GPU: nvidia-smi not found` | Install the driver (step 2). To only look at a recipe on a machine without the card, pass `--card p100`. |
| `no recipe for ... fits this host's GPUs` | No GPU here matches the model's recipe. The message lists your GPUs and what the recipe needs. |
| `the NVIDIA driver ... is older than ...` | Install the driver version it names, or newer, then run `tensorless doctor`. |
| `there is no GPU N` | `--gpu` named an index `nvidia-smi` does not list. Use one of the indexes it prints. |
| `no recipe for "..."` | The launcher does not know that model. It lists the models it knows; so does `tensorless models`. A model added after your launcher's release needs a newer launcher. |
| `... is a draft: it has not passed the release gate` | The recipe has not been measured for release yet. Pass `--allow-draft` to try it anyway. |
| `... is not published yet` | The recipe's model upload or engine image does not exist yet. Wait for a release. |
| `revision ... is not the one the recipe ... pins` | You gave `MODEL@REVISION`. Pass `--allow-draft` to use it anyway; the run is then untuned. |
| `neither podman nor docker is installed` | Install podman or docker and the NVIDIA container toolkit (step 2). |
| `could not pull the engine image` | Check that this host can reach `ghcr.io`, then run the command again. |
| `HTTP 401` or `HTTP 403` for a Hugging Face address | The repository is private or gated. Accept its terms on the Hub, and set `HF_TOKEN`. |
| `HTTP 404` for a Hugging Face address | The recipe's repository, revision or file does not exist on the Hub. Please report it. |
| `not enough disk space in ...` | Free some space, or pass `--cache DIR` on a larger disk. |
| `the download of ... stopped at ...` or `download interrupted` | Run the same command again: it resumes where it stopped. |
| `sha256 mismatch` | The download was deleted. Run the command again. If it happens again, the upload or the recipe is wrong; please report it. |
| `the engine image is not pulled` or `the model is not downloaded` | `serve` only starts what `pull` fetched. Run `tensorless pull MODEL`, or use `tensorless run MODEL`, which pulls first. |
| `... is already running in podman` | The model's server is already up. Use it, or stop it first with `tensorless stop MODEL`. |
| `cannot use port 8080 on 127.0.0.1` | Something else listens on that port, often another model's server. Pass `--port`, or stop the other server. |
| `could not start the container` | The container tool refused the command, usually because it cannot reach the GPU. Run `tensorless doctor`. |
| `the server exited before it was ready` | Read the server's log lines above the message (with `--detach`, the last 60 are printed). See below. |
| `the server was not ready after 20m, so it was stopped` | The model did not finish loading in 20 minutes. Check the log lines above it, and that the disk holding the cache is not very slow. |
| `do not pass -c after --` (or another flag) | The launcher sets that flag itself. Use `--context`, `--port` or `--host` instead. |
| `no server answers http://.../health` | From `bench`: start the server first with `tensorless serve MODEL --detach`, or pass the `--host` and `--port` it runs on. |

### When the server exits before it is ready

Read the log lines the launcher printed:

- **`no kernel image is available`**: the engine image was built for another GPU architecture than this card's. The P100 image is built for sm_60 only.
- **An out-of-memory error**: another process is using the GPU. `nvidia-smi` lists the processes on each card; stop them, or another model's server, and try again. A `--context` larger than the recipe's can also run out of memory.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success. |
| 1 | The command failed, or `doctor` found a failure. |
| 2 | Bad usage: an unknown command or flag, a missing MODEL. |
| 130 | Interrupted, for example with Ctrl-C. |

[The command line reference](../../reference/cli.md) describes every command and flag.

## Check that it worked

`tensorless doctor` ends with `all checks passed.`, and `tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF` prints `is ready at http://127.0.0.1:8080`. If a message here does not help, ask on [Discord](https://discord.gg/3Qs2uejUf9) or [open an issue](https://github.com/iammrduncan/tensorless/issues) with the full output: see [Share your results](share-results.md).
