# tensorless: design

Status: agreed direction, 2026-11-02. The first release is engine-v0.1.0 with launcher v0.1.0.

## The goal

One command runs a model that has been tuned for a GPU, on the engine it was tuned with:

```sh
tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF
```

The launcher detects the card, picks the recipe for that model on that card, pulls the engine image and the model weights it names, checks both, and starts an OpenAI-compatible server. The first target is the Tesla P100 (sm_60, 16 GB) running Qwen3.8-27B-PN3, a P100-native 3-bit quantization of Qwen3.8-27B.

## Decisions

| Question | Decision |
|---|---|
| One image per engine, or one configurable image? | **One engine lineage. One image per engine release and GPU architecture.** Model-specific tuning lives in a recipe, not in an image. |
| Is the engine source public? | **Yes.** It is in this repository, under `engine/`. |
| License | **Apache-2.0** for tensorless. The engine keeps the MIT notices of the projects it is built on (see [Licensing](#licensing)). |
| Launcher | A single static binary with no Python dependency, shaped after TensorFold's `pull` / `serve` / `info` / `capabilities` commands (see [The launcher](#the-launcher)). |

### Why not one image per model

- **One engine build already serves several models.** The P100 engine runs Qwen3.8-27B-PN3 and Qwen3.8-27B-UD-IQ3_XXS-Q3Floor (Unsloth's UD-IQ3_XXS with a 3-bit floor) from the same binary. Every tuning step was checked against Qwen3.8-27B-UD-IQ3_XXS-Q3Floor, which had to produce bit-identical text, so a change for one model could not silently break the other. The PN kernels only engage on PN tensor types.
- **What differs per model is runtime configuration:** the engine's `GGML_CUDA_*` switches and the server's flags (context length, KV-cache type, batch sizes, offload). That is data, so it goes in a recipe.
- **A fix should be one rebuild, not one rebuild per model.**

### Why one image per architecture

- **The tuned kernels are built for one architecture.** The P100 build is `sm_60` only, and its acceptance checks confirm that the library holds sm_60 code and nothing else. A fat binary for several cards is not what was measured.
- **Each card family gets its own build of the same release.**
- **P100 images pin CUDA 12.x.** CUDA 13 dropped Pascal, and the R580 driver branch is Pascal's last. The host needs only the driver; the CUDA runtime is in the image.

### Why recipes pin images by digest

A recipe moves to a new engine release only after that release has passed that model's checks on that card. Older releases stay pullable, so a model whose recipe has not moved yet keeps working on the release it was measured with. One lineage moves forward; the registry holds as many versions as the recipes need.

A second engine lineage would be justified only by a different runtime, or by a change that cannot coexist with another model's guarantees. The default answer to such a conflict is to leave the other model's recipe on its last good release, not to fork.

## Terms

| Term | Meaning |
|---|---|
| **Engine release** | A tagged engine version, `engine-vX.Y.Z` (semver; the first is `engine-v0.1.0`). It is the exact source in `engine/` at that tag, plus its build configuration. The launcher is versioned separately, `vX.Y.Z`. |
| **Image** | One build of a release for one GPU architecture, published as `ghcr.io/iammrduncan/tensorless-engine:<X.Y.Z>-<arch>` (e.g. `0.1.0-sm60`). Recipes refer to it by digest, never by tag. |
| **Recipe** | One model on one card: the image digest, the model's Hugging Face repository, revision, file and sha256, the engine switches and server flags, the hardware requirements, and the measured numbers. `recipes/<org>/<repo>/<card>.toml`, named after the model's Hugging Face repository. |
| **Receipt** | The measurements the release gate took for a recipe: decode and prefill speed at stated contexts, quality checks, the image digest and model hash they were taken on. `docs/receipts/<release>/`. |
| **Champion** | A change that won a hill-climb campaign: it was faster than the campaign's best so far, under that campaign's guards. |

## Repository layout

```text
LICENSE                 Apache-2.0
NOTICE                  attribution, including the MIT-licensed upstreams
LICENSES/               upstream license texts (llama.cpp, llamAmpere, ...)
THIRD_PARTY_NOTICES.md
engine/                 the engine source (a llama.cpp fork), exported per release; PROVENANCE.md says from where
containers/             one Containerfile per architecture: engine.sm60.Containerfile, ...
recipes/                <model>/<card>.toml, plus the schema; embedded into the launcher at build time
go.mod, cmd/, internal/ the tensorless CLI (Go); recipes_embed.go embeds recipes/
docs/                   design, the PN format, the launcher, receipts
.github/workflows/      engine-image.yml (images on engine-v* tags), launcher.yml (tests; archives on v* tags)
```

## The launcher

TensorFold's native CLI is the model: a binary that needs no Python, verifies what it downloads, refuses what it cannot run, and can describe itself without a GPU or a model. tensorless differs in one way: the engine runs in a container, so the launcher orchestrates podman or docker instead of serving in-process.

| Command | What it does |
|---|---|
| `tensorless run ORG/REPO` | `pull`, then `serve`. The one-command path. |
| `tensorless pull ORG/REPO` | Resolves the recipe for the detected card. Pulls the image by digest. Downloads the model at its pinned revision into the Hugging Face cache, resumably, and checks it by sha256. |
| `tensorless serve ORG/REPO` | Starts the engine container with the recipe's switches and flags. The model is mounted read-only. The server binds to `127.0.0.1` by default. Waits for `/health` and prints the endpoint. |
| `tensorless models` | Lists the recipes, and marks which ones fit the detected card and are already pulled. |
| `tensorless info ORG/REPO` | Shows the recipe: engine release, image digest, model hash, memory needed, context, and the receipt numbers. |
| `tensorless doctor` | Checks the GPU and its compute capability, the driver branch, podman or docker, and NVIDIA container support (CDI or the container toolkit). Names each missing piece with the command that fixes it. |
| `tensorless capabilities --json` | Version, the supported cards and architectures, and the known recipes. Needs no GPU, driver or model. |

Rules the launcher keeps:
- **Refuse rather than guess.** It refuses a model with no recipe for the detected card, and an image or model whose hash does not match. It refuses a driver outside the image's supported range, with the reason. Overrides exist, such as `--context` or another port, but a run with changed tuning settings is labelled untuned.
- **Pin everything.** Image by digest, model by revision and sha256. `tensorless info` shows what would run before anything is downloaded.
- **Stay small.** The launcher holds no inference code. Everything numerical happens in the image that the receipts measured.

Distribution: per-platform release archives with `.sha256` companions, as TensorFold does, and an install script that checks them. Linux x86_64 and aarch64 come first.

## Recipes

The schema is in [recipes/README.md](../recipes/README.md). An excerpt of the first recipe:

```toml
# recipes/iamMrDuncan/Qwen3.8-27B-PN3-GGUF/p100.toml
schema = 1
status = "draft"

[model]
hf_repo  = "iamMrDuncan/Qwen3.8-27B-PN3-GGUF"   # what users type
revision = "<40-hex commit>"
file     = "<the GGUF file>"
sha256   = "<64-hex>"

[card]
compute_capability = "6.0"
min_memory_mib     = 16000

[engine]
release = "engine-v0.1.0"
image   = "ghcr.io/iammrduncan/tensorless-engine@sha256:<digest>"

[run]
context = 65536
args    = ["-np", "1", "-ngl", "99", "-ctk", "q4_0", "-ctv", "q4_0", "-b", "512", "-ub", "512", "-fa", "on", "..."]

[run.env]
GGML_CUDA_FATTN_SM60 = "1"   # ... the 21 switches the hill climb's judge ran with

[receipt]
decode_tps = { short = 30.06 }
```

A recipe's `[run]` section is the exact configuration the gate measured. Recipes are reviewed like code: a pull request that changes `[run]` or `[engine]` needs a new receipt.

## Models built on someone else's file

Some tuned models are a small change to another publisher's file. The UD-IQ3_XXS Q3Floor model is one: Unsloth's `Qwen3.8-27B-UD-IQ3_XXS.gguf` with its 91 sub-3-bit tensors re-quantized to IQ3_XXS, and `token_embd` from their UD-IQ4_XS. Its other 73% of bytes are Unsloth's, unchanged. Such models are hosted whole, in our namespace:

- **The name says what the file is.** It keeps the original publisher's own label, so their users recognize it ("UD" is Unsloth Dynamic), and adds what changed ("Q3Floor").
- **The bytes are exactly the file that was measured.** Only the repository and file names change. The GGUF's internal metadata stays the original publisher's, because editing it would change the sha256 the receipts were measured on.
- **The model card credits the original publisher,** names their repository and revision, lists every change, and carries their license.

## The engine image

- **Two-stage Containerfile per architecture.** Build on `nvidia/cuda:12.6.x-devel`, run on the matching `-runtime` image.
- **The image holds** `llama-server` and its libraries, `LICENSE`, `NOTICE`, `LICENSES/`, `THIRD_PARTY_NOTICES.md`, `VERSION` and `SHA256SUMS`. It holds no model weights and no build tree.
- **Portable CPU code.** The release build uses `GGML_NATIVE=OFF`, so the CPU side does not depend on the builder's CPU. The research build used `GGML_NATIVE=ON`; that must not ship.
- **The CUDA side keeps the measured configuration:** `CMAKE_CUDA_ARCHITECTURES=60-real`, flash attention on with the measured KV-quant set, CUDA graphs off, and no VMM.
- **CI builds each image once, from the tag.** The release gate measures that exact digest. Publishing a digest that the gate did not measure is not allowed.

## Releases and merging champions

Hill-climb campaigns run in the research repository with an automated harness. Each campaign has its own metric and guards. Each campaign's best change is a git commit there. The release process turns those into an engine release here:

1. **Every campaign starts from the latest engine release**, not from another campaign's champion, so two campaigns never diverge by more than their own work.
2. **A finished campaign's champion is merged into `engine-next`**, by an explicit merge with an ancestry check. A campaign's champion only has to beat the campaign's previous best, so it may not contain every earlier winning change. Conflicting or missing changes become ports: re-applied changes that are measured again.
3. **The release gate measures the candidate image**, built by CI from `engine-next`, for every recipe on every card:
   - build checks;
   - each model's numerics against the previous release, within that model's recorded bars;
   - speed no worse than any merged champion, within measured noise;
   - decode at the long contexts (32K and 65K on the P100) and prefill;
   - the model's quality checks, once per release.
4. **If the gate passes:** tag `engine-vX.Y.Z`, publish the measured digest, write the receipts, and move the recipes in one pull request.
5. **If a change helps one model and fails another's check,** the failing model's recipe stays on the previous release, and the conflict is recorded in the release notes.

The engine source arrives here as one snapshot commit per release, exported from the research repository by its export tool. The export:
- takes `engine/` at a commit with `git archive`, so nothing uncommitted can leak in;
- drops the research-only files: autoresearch state, the research lane's switch file, a status note and a commit map;
- rewrites, one line for one line, the few comments that point into the private repository. Line numbers and the compiled code do not change;
- refuses to finish if a scan finds a lab-only term (hostnames, network names, private repository names, credentials) in any file, or a home path or personal name in any file that differs from the public upstream base;
- writes `engine/PROVENANCE.md`: the upstream base, the research commit, the tree digest, what was removed and rewritten, and known issues.

The image build is CI's: `.github/workflows/engine-image.yml` builds `containers/engine.<arch>.Containerfile` on an `engine-v*` tag (or by hand, for release candidates) and prints the digest to pin.

## Licensing

- tensorless is Apache-2.0.
- The engine descends from MIT-licensed projects: llama.cpp (the ggml authors), TheTom/llama-cpp-turboquant, and JakeATX/llamAmpere. It vendors code from turboderp's exllamav3 (MIT) and nlohmann/json (MIT), whose texts are already in `engine/licenses/`. Their copyright and permission notices stay in `LICENSES/` and in the engine files that carry them. Changes made for tensorless are Apache-2.0.
- `NOTICE` and `THIRD_PARTY_NOTICES.md` list every upstream, including the libraries vendored into the engine.
- Model weights are not part of tensorless and keep their own licenses. Each recipe names its model's license.

## The PN format

Qwen3.8-27B-PN3 uses two weight types, `pn3t` and `pn4t`, that only this engine implements. Their GGUF type ids (124 and 125) are fork-specific, so stock llama.cpp cannot load the file and could misread it.
- The model's GGUF should carry a metadata key that names the format and its version.
- The model card should say plainly that it needs the tensorless engine.
- The format is specified in `engine/docs/` alongside its reference decoder and test vectors.

## Open before the first release

- **Receipts at the long contexts for the current champion.** Only the starting point of the decode climb has been measured at 32K and 65K; the champion has numbers only at short context.
- **The P100 recipe's switch set.** The research lane's switch file accumulated history and lab paths. The Qwen3.8-27B-PN3 recipe starts from the switches the measurements actually used, then is pruned to those that change its output or speed.
- **A known CPU-path defect.** The CPU backend's PN dot product decodes the wrong codes. GPU paths are correct, and every recipe runs fully on the GPU. Fix it before any CPU or partial-offload recipe exists.
- **The image's driver range,** confirmed on a real P100 host.
- **The launcher's container support,** confirmed on rootless podman with CDI and on docker with the NVIDIA toolkit.
