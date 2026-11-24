#!/usr/bin/env python3
# Reference-logit dumper for the llama.cpp Inkling-2-Mini port (copy of work/ref/reference_logits.py).
# Wraps LookAtMyQuant (github.com/ShinMegamiBoson/lookatmyquant, MIT) runtime_cuda/inkling/reference.py with the
# embedding table kept in float32. Not run by CI; needs torch(CPU) + transformers@main + the LAMQ repo on PYTHONPATH.
"""Float32 CPU reference logits for LookAtMyQuant Inkling-2-Mini (for comparison with the llama.cpp port).

Runs LookAtMyQuant's `runtime_cuda/inkling/reference.py::load_reference(pack, "cpu", dtype=torch.float32)`
(transformers' Inkling2LoopForCausalLM with the packed matrices decoded to dense float32), with ONE change that
matches the judge's reference: the token embedding table is kept in float32 (upstream PackedEmbedding casts it
to bfloat16). One no-cache forward per prompt.

Needs on PYTHONPATH: torch (CPU), transformers from git main (src/), its runtime deps, and the lookatmyquant
repo root (so `runtime_cuda.inkling.reference` imports). Needs ~20 GB RAM.

Prompts: a JSON file, either {"name": [ids, ...], ...} or LAMQ's reference_prompts.json format
[{"name": ..., "ids": [...]}, ...]. Default: reference_prompts.json from the lookatmyquant checkout.

Output .npz, per prompt <name>:
  <name>.ids       int32  [T]       input token ids
  <name>.top_ids   int32  [T, K]    top-K token ids per position (descending logit)
  <name>.top_vals  float32[T, K]    their logits (after the model's final softcap, i.e. what the model returns)
  <name>.lse       float32[T]       logsumexp over the full vocab (log-softmax = logit - lse)
  <name>.logits    float32[T, V]    full logits (only with --full)
and __meta__ (JSON string).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def load_prompts(path: Path) -> dict[str, list[int]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return {str(k): [int(t) for t in v] for k, v in data.items()}
    return {str(p["name"]): [int(t) for t in p["ids"]] for p in data}


def patch_embedding_f32(reference) -> None:
    """Replace reference.PackedEmbedding by a copy whose dense table stays float32 (no .to(torch.bfloat16))."""

    class PackedEmbeddingF32(reference.PackedEmbedding):
        def __init__(self, packed_dir, luts, embed_norm, device):
            nn.Module.__init__(self)
            self.b = reference._Buckets(packed_dir, "embed", luts, device)
            self.embed_norm = embed_norm
            rows = [reference.rht_out(Q.float(), s) for Q, s in zip(self.b.Qs, self.b.signs)]
            table = torch.cat(rows, 0)[self.b.inv_order]  # token-indexed, float32
            self.register_buffer("weight", table, persistent=False)
            self.b.Qs = []

    reference.PackedEmbedding = PackedEmbeddingF32


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack", type=Path, help="LookAtMyQuant packed Inkling-2-Mini directory")
    ap.add_argument("--prompts", type=Path, default=None, help="JSON prompts (default: LAMQ reference_prompts.json)")
    ap.add_argument("--out", type=Path, required=True, help="output .npz")
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--full", action="store_true", help="also store full [T, V] float32 logits")
    ap.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    from runtime_cuda.inkling import reference

    if args.prompts is None:
        args.prompts = Path(reference.__file__).resolve().parents[1] / "reference_prompts.json"
    prompts = load_prompts(args.prompts)
    print(f"[ref] torch {torch.__version__}, threads {torch.get_num_threads()}, {len(prompts)} prompts from {args.prompts}",
          flush=True)

    patch_embedding_f32(reference)
    t0 = time.time()
    model, _tok = reference.load_reference(args.pack, "cpu", dtype=torch.float32)
    emb = model.model.embed_tokens
    assert isinstance(emb, reference.PackedEmbedding) and emb.weight.dtype == torch.float32, emb.weight.dtype
    print(f"[ref] loaded in {time.time() - t0:.0f}s; embed table {tuple(emb.weight.shape)} {emb.weight.dtype}", flush=True)

    out: dict[str, np.ndarray] = {}
    meta = {"pack": str(args.pack.resolve()), "prompts": str(args.prompts), "dtype": "float32", "embed_dtype": "float32",
            "topk": args.topk, "full": args.full, "torch": torch.__version__, "threads": torch.get_num_threads(),
            "names": list(prompts)}
    with torch.no_grad():
        for name, ids in prompts.items():
            t1 = time.time()
            x = torch.tensor([ids], dtype=torch.long)
            logits = model(input_ids=x, use_cache=False).logits[0].float()  # [T, V]
            lse = torch.logsumexp(logits, dim=-1)
            vals, top = torch.topk(logits, min(args.topk, logits.shape[-1]), dim=-1)
            out[f"{name}.ids"] = np.asarray(ids, dtype=np.int32)
            out[f"{name}.top_ids"] = top.to(torch.int32).numpy()
            out[f"{name}.top_vals"] = vals.numpy().astype(np.float32)
            out[f"{name}.lse"] = lse.numpy().astype(np.float32)
            if args.full:
                out[f"{name}.logits"] = logits.numpy().astype(np.float32)
            meta.setdefault("vocab", int(logits.shape[-1]))
            print(f"[ref] {name}: T={len(ids)} V={logits.shape[-1]} {time.time() - t1:.1f}s "
                  f"argmax[-1]={int(top[-1, 0])} finite={bool(torch.isfinite(logits).all())}", flush=True)
    out["__meta__"] = np.asarray(json.dumps(meta))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"[ref] wrote {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
