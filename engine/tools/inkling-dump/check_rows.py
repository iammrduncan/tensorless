#!/usr/bin/env python3
"""Check llama-inkling-dump against a numpy port of LookAtMyQuant's formats.py decoders.

    python3 check_rows.py <pack folder> <llama-inkling-dump> <model gguf> <work dir> [seed]

Builds a request file like the judge's (two rows of every attention linear, two rows of one expert of each format of
each side of each layer, two rows of every embedding and head bucket), runs the dump tool and compares each row
within 1e-6 of the row's largest absolute value.
"""
import json
import os
import random
import struct
import subprocess
import sys

import numpy as np

FMT = {"ht1": (1, 4, 512, 1), "ht2": (2, 4, 512, 1), "ht3": (3, 2, 1024, 1),
       "rtr1": (1, 1, 4096, 0), "rtr2": (2, 1, 4096, 0), "rtr3": (3, 1, 4096, 0), "rtr4": (4, 1, 4096, 0)}
ID2 = ["ht1", "ht2", "ht3", "rtr1", "rtr2", "rtr3", "rtr4", "nvfp4"]
E2M1 = np.array((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0), np.float32)


def e4m3():
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        sign = -1.0 if b & 0x80 else 1.0
        exp, mant = (b >> 3) & 0xF, b & 0x7
        if exp == 15 and mant == 7:
            t[b] = 0.0
        elif exp == 0:
            t[b] = sign * (mant / 8.0) * 2.0 ** -6
        else:
            t[b] = sign * (1.0 + mant / 8.0) * 2.0 ** (exp - 7)
    return t


def dec_trellis(words, scale, lut, n, fmt):
    k, g, K, h = FMT[fmt]
    w = np.concatenate([words.astype(np.uint64), np.zeros(1, np.uint64)])
    last = np.arange(g - 1, n, g, dtype=np.int64)
    bp = (n - 1 - last) * k
    wi, off = bp >> 5, (bp & 31).astype(np.uint64)
    pair = w[wi] | (w[wi + 1] << np.uint64(32))
    st = (pair >> off) & np.uint64(4095)
    if h:
        st = ((st * np.uint64(0x9E3779B1)) & np.uint64(0xFFFFFFFF)) >> np.uint64(32 - int(np.log2(K)))
    vals = lut.astype(np.float32).reshape(K, g)[st.astype(np.int64)].reshape(n)
    return vals * np.float32(scale)


def dec_lm(codes, scale, levels, n, k, r):
    if k == 8:
        idx = codes[r * n:(r + 1) * n].astype(np.int64)
    else:
        b = codes[r * n * 3 // 4:(r + 1) * n * 3 // 4].astype(np.int64).reshape(-1, 3)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        idx = np.stack([v & 63, (v >> 6) & 63, (v >> 12) & 63, (v >> 18) & 63], -1).reshape(n)
    return levels.astype(np.float32)[idx] * np.float32(scale)


def main():
    pack, tool, model, work = sys.argv[1:5]
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    rng = random.Random(seed)
    luts = np.load(os.path.join(pack, "luts.npz"))
    t4 = e4m3()
    reqs, refs = [], []
    for L in range(24):
        meta = json.load(open(os.path.join(pack, f"layer_{L:02d}_meta.json")))
        z = np.load(os.path.join(pack, f"layer_{L:02d}_packed.npz"), mmap_mode="r")
        for lin, d in meta["linears"].items():
            s, m, n, fmt = d["safe"], d["m"], d["n"], d["fmt"]
            for r in rng.sample(range(m), 2):
                sc = z[f"lin_{s}_scales"][r].astype(np.float32)
                if fmt.startswith("lm"):
                    ref = dec_lm(np.asarray(z[f"lin_{s}_codes"]), sc, z[f"lin_{s}_levels"], n, d["k"], r)
                else:
                    ref = dec_trellis(z[f"lin_{s}_codes"][r], sc, luts[fmt], n, fmt)
                reqs.append(f"layer_{L:02d}.{lin} {r}")
                refs.append(ref)
        for side, n in (("gu", 2304), ("dn", 1152)):
            mp = np.asarray(z[f"{side}_map"])
            for fid in sorted(set(mp[:, 0].tolist())):
                es = [e for e in range(128) if mp[e, 0] == fid]
                e = rng.choice(es)
                loc = int(mp[e, 1])
                fmt = ID2[fid]
                for r in rng.sample(range(2304), 2):
                    if fmt == "nvfp4":
                        c = z[f"{side}_nvfp4_codes"][loc, r].astype(np.int64)
                        vals = np.stack([E2M1[c & 15], E2M1[c >> 4]], -1).reshape(n)
                        bs = t4[z[f"{side}_nvfp4_scales"][loc, r].astype(np.int64)]
                        ref = (vals.reshape(-1, 16) * bs[:, None]).reshape(n) * np.float32(z[f"{side}_nvfp4_outer"][loc])
                    else:
                        ref = dec_trellis(z[f"{side}_{fmt}_codes"][loc, r], z[f"{side}_{fmt}_scales"][loc, r].astype(np.float32),
                                          luts[fmt], n, fmt)
                    reqs.append(f"layer_{L:02d}.experts.{side}.{e} {r}")
                    refs.append(ref.astype(np.float32))
    for kind in ("embed", "head"):
        meta = json.load(open(os.path.join(pack, f"{kind}_meta.json")))
        z = np.load(os.path.join(pack, f"{kind}_packed.npz"), mmap_mode="r")
        for bm in meta:
            kb, rows, fmt = bm["kb"], bm["rows"], bm["opt"]
            for r in rng.sample(range(rows), 2):
                sc = z[f"b{kb}_scales"][r].astype(np.float32)
                if fmt.startswith("lm"):
                    ref = dec_lm(np.asarray(z[f"b{kb}_codes"]), sc, z[f"b{kb}_levels"], 2304, bm["k"], r)
                else:
                    ref = dec_trellis(z[f"b{kb}_codes"][r], sc, luts[fmt], 2304, fmt)
                reqs.append(f"{kind}.b{kb} {r}")
                refs.append(ref)
    rq = os.path.join(work, "rows.txt")
    out = os.path.join(work, "rows.bin")
    open(rq, "w").write("\n".join(reqs) + "\n")
    env = dict(os.environ, INK_ROWS=rq, INK_OUT=out, CUDA_VISIBLE_DEVICES="")
    subprocess.run([tool, "-m", model, "-ngl", "0", "-t", "4"], env=env, check=True)
    data = open(out, "rb").read()
    p, bad, worst = 0, 0, 0.0
    for q, ref in zip(reqs, refs):
        n = struct.unpack_from("<i", data, p)[0]
        p += 4
        got = np.frombuffer(data, np.float32, n, p)
        p += 4 * n
        if n != ref.size:
            bad += 1
            print("size", q, n, ref.size)
            continue
        err = float(np.max(np.abs(got - ref)) / max(float(np.max(np.abs(ref))), 1e-30))
        worst = max(worst, err)
        if err > 1e-6:
            bad += 1
            if bad < 10:
                print("bad", q, err)
    print(f"rows {len(reqs)} bad {bad} worst_rel_err {worst:.3g}")


if __name__ == "__main__":
    main()
