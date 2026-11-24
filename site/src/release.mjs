// What the site says about the current release, read from the repository itself so the pages
// never drift from it: the recipes (recipes/<org>/<repo>/<card>.toml) and the launcher version.
import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { parse } from "smol-toml";

// The repository root. The site is built from site/, and bundling moves this module, so the root
// comes from the working directory rather than from this file's own location.
const root = process.env.TENSORLESS_ROOT ?? path.resolve(process.cwd(), "..");
const recipesDir = path.join(root, "recipes");

function walk(dir) {
  return readdirSync(dir).flatMap((name) => {
    const p = path.join(dir, name);
    if (statSync(p).isDirectory()) return walk(p);
    return p.endsWith(".toml") ? [p] : [];
  });
}

/** Every recipe in the repository, released ones first. */
export const recipes = walk(recipesDir)
  .map((file) => ({ file: path.relative(root, file), ...parse(readFileSync(file, "utf8")) }))
  .sort((a, b) => (a.status === b.status ? 0 : a.status === "released" ? -1 : 1));

/** Versions and the measurements of the published image, as the receipts state them. */
export const release = {
  launcher: "0.4.0",
  engine: "engine-v0.4.0",
  image: "ghcr.io/iammrduncan/tensorless-engine:0.4.0-sm60",
  digest: "sha256:4a0153230be9e2b5957a5aaf58cbb0ba1f79bfeb94eacd4617189f16ca35e997",
  // tensorless bench on the published image, one Tesla P100-PCIE-16GB, 2026-11-24, with
  // GGML_CUDA_SYNC_SCHED=auto as the recipes set it (docs/receipts/engine-v0.4.0/).
  // North-Mini-Code-1.0's prompt was 411 tokens, the others' 429.
  measured: {
    "iamMrDuncan/Qwen3.8-27B-PN3-GGUF": 30.46,
    "iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF": 26.63,
    "unsloth/North-Mini-Code-1.0-GGUF": 94.82,
  },
  // The decode hill climb that produced engine-v0.1.0's PN kernels, the prefill hill climb that
  // produced engine-v0.2.0's, and the North-Mini-Code decode hill climb that produced engine-v0.3.0
  // (engine/PROVENANCE.md; its from / to are the climb's decode metric over both prompt lengths).
  climb: { attempts: 131, champions: 28, from: 7.74, to: 30.06 },
  prefillClimb: { attempts: 369, champions: 26, from: 121.4, to: 200.1 },
  nmcClimb: { attempts: 127, champions: 22, from: 42.03, to: 72.62 },
};

for (const r of recipes) {
  if (r.status === "released" && r.engine.image.split("@")[1] !== release.digest) {
    throw new Error(`${r.file} pins ${r.engine.image}, not the release digest ${release.digest}`);
  }
}
