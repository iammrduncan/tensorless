#!/usr/bin/env python3
"""Prepare step for LookAtMyQuant's Inkling-2-Mini pack (inkling2-packed-v1).

    python3 -E -s prepare.py <pack folder> <empty output folder>

Writes <out>/inkling2.gguf: the model's hyperparameters, the tokenizer (tokenizer.json's byte-level BPE) and a table
of the pack's arrays (every .npz member and every steering tensor) by file and absolute byte offset. The weights are
not copied: the engine memory-maps them in place from the pack folder. The last line printed is the GGUF's path.

numpy and the standard library only (the judge's image has nothing else; gguf-py needs yaml), a few seconds.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import zipfile

import numpy as np

ARCH = "inkling2_loop"

# GGUF value types
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)

NPY_DTYPES = {"<u4": "u32", "<f2": "f16", "|i1": "i8", "|u1": "u8", "<f4": "f32", "<i2": "i16", "<i4": "i32",
              "<u2": "u16", "<i8": "i64"}
ST_DTYPES = {"F32": "f32", "BF16": "bf16", "F16": "f16", "I8": "i8", "U8": "u8", "I32": "i32", "I64": "i64"}

# matrix formats as the engine numbers them: trellis 0-6 (the pack's fmt ids), nvfp4 7, lm6 8, lm8 9
FMT_IDS = {"ht1": 0, "ht2": 1, "ht3": 2, "rtr1": 3, "rtr2": 4, "rtr3": 5, "rtr4": 6, "nvfp4": 7, "lm6": 8, "lm8": 9}


class Writer:
    def __init__(self):
        self.kv: list[bytes] = []

    @staticmethod
    def _s(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def _key(self, k: str, t: int) -> bytes:
        return self._s(k) + struct.pack("<I", t)

    def str(self, k, v):
        self.kv.append(self._key(k, T_STR) + self._s(v))

    def u32(self, k, v):
        self.kv.append(self._key(k, T_U32) + struct.pack("<I", int(v)))

    def i32(self, k, v):
        self.kv.append(self._key(k, T_I32) + struct.pack("<i", int(v)))

    def f32(self, k, v):
        self.kv.append(self._key(k, T_F32) + struct.pack("<f", float(v)))

    def bool(self, k, v):
        self.kv.append(self._key(k, T_BOOL) + struct.pack("<B", 1 if v else 0))

    def arr_str(self, k, vs):
        parts = [self._key(k, T_ARR), struct.pack("<IQ", T_STR, len(vs))]
        parts += [self._s(v) for v in vs]
        self.kv.append(b"".join(parts))

    def arr_num(self, k, vs, t):
        dt = {T_U32: "<u4", T_I32: "<i4", T_U64: "<u8", T_I64: "<i8", T_F32: "<f4", T_U8: "u1", T_BOOL: "u1"}[t]
        a = np.asarray(vs, dtype=dt)
        self.kv.append(self._key(k, T_ARR) + struct.pack("<IQ", t, a.size) + a.tobytes())

    def write(self, path):
        with open(path, "wb") as f:
            f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(self.kv)))
            for b in self.kv:
                f.write(b)


def npz_members(path):
    """[(member name, absolute data offset, dtype tag, shape)] for an uncompressed .npz, read from the zip headers."""
    out = []
    with zipfile.ZipFile(path) as z, open(path, "rb") as f:
        for info in z.infolist():
            if info.compress_type != zipfile.ZIP_STORED:
                raise SystemExit(f"{path}: member {info.filename} is compressed")
            f.seek(info.header_offset)
            lh = f.read(30)
            if lh[:4] != b"PK\x03\x04":
                raise SystemExit(f"{path}: bad local header for {info.filename}")
            nlen, xlen = struct.unpack("<HH", lh[26:30])
            start = info.header_offset + 30 + nlen + xlen
            f.seek(start)
            magic = f.read(8)
            if magic[:6] != b"\x93NUMPY":
                raise SystemExit(f"{path}: {info.filename} is not .npy")
            major = magic[6]
            if major == 1:
                hlen = struct.unpack("<H", f.read(2))[0]
                hdr_end = start + 10 + hlen
            else:
                hlen = struct.unpack("<I", f.read(4))[0]
                hdr_end = start + 12 + hlen
            header = f.read(hlen).decode("latin1")
            d = eval(header, {"__builtins__": {}}, {})  # the .npy header is a Python dict literal
            if d["fortran_order"]:
                raise SystemExit(f"{path}: {info.filename} is Fortran-ordered")
            descr = d["descr"]
            if descr not in NPY_DTYPES:
                raise SystemExit(f"{path}: {info.filename} has dtype {descr}")
            shape = tuple(int(x) for x in d["shape"])
            nbytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(descr).itemsize
            if hdr_end + nbytes > start + info.file_size:
                raise SystemExit(f"{path}: {info.filename} runs past its member")
            out.append((info.filename[:-4] if info.filename.endswith(".npy") else info.filename, hdr_end,
                        NPY_DTYPES[descr], shape, nbytes))
    return out


def safetensors_members(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    base = 8 + n
    out = []
    for k, v in h.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        out.append((k, base + a, ST_DTYPES[v["dtype"]], tuple(v["shape"]), b - a))
    return out


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: prepare.py <pack folder> <output folder>")
    pack = os.path.abspath(sys.argv[1])
    outd = os.path.abspath(sys.argv[2])
    os.makedirs(outd, exist_ok=True)
    P = lambda n: os.path.join(pack, n)  # noqa: E731

    cfg = json.load(open(P("config.json")))
    tc = cfg.get("text_config", cfg)
    lc = tc["loop_config"]
    n_layer = int(tc["num_hidden_layers"])

    w = Writer()
    w.str("general.architecture", ARCH)
    w.str("general.name", "Inkling-2-Mini (LookAtMyQuant 12.8GB pack)")
    w.u32("general.alignment", 32)
    a = ARCH
    w.u32(f"{a}.context_length", int(tc.get("model_max_length", 1048576)))
    w.u32(f"{a}.embedding_length", tc["hidden_size"])
    w.u32(f"{a}.block_count", n_layer)
    w.u32(f"{a}.feed_forward_length", tc["intermediate_size"])
    w.u32(f"{a}.expert_feed_forward_length", tc["intermediate_size"])
    w.u32(f"{a}.attention.head_count", tc["num_attention_heads"])
    w.u32(f"{a}.attention.head_count_kv", tc["num_key_value_heads"])
    w.u32(f"{a}.attention.key_length", tc["head_dim"])
    w.u32(f"{a}.attention.value_length", tc["head_dim"])
    w.f32(f"{a}.attention.layer_norm_rms_epsilon", tc["rms_norm_eps"])
    w.u32(f"{a}.attention.sliding_window", tc["sliding_window_size"])
    w.u32(f"{a}.expert_count", tc["n_routed_experts"])
    w.u32(f"{a}.expert_used_count", tc["num_experts_per_tok"])
    w.u32(f"{a}.expert_shared_count", tc.get("n_shared_experts", 0))
    w.f32(f"{a}.expert_weights_scale", tc["route_scale"])
    w.f32(f"{a}.final_logit_softcapping", tc["final_logit_softcapping"])
    w.u32(f"{a}.vocab_size", tc["unpadded_vocab_size"])
    local = set(int(x) for x in tc["local_layer_ids"])
    w.arr_num(f"{a}.swa_layers", [1 if i in local else 0 for i in range(n_layer)], T_BOOL)
    w.u32(f"{a}.rel_dim", tc["d_rel"])
    w.u32(f"{a}.rel_extent", tc["rel_extent"])
    w.f32(f"{a}.qk_l2_clip", tc["qk_l2_clip_threshold"])
    w.f32(f"{a}.rel_l2_clip", tc["rel_l2_clip_threshold"])
    w.u32(f"{a}.log_scaling_n_floor", tc["log_scaling_n_floor"])
    w.f32(f"{a}.log_scaling_alpha", tc["log_scaling_alpha"])
    w.u32(f"{a}.conv_kernel", tc["sconv_kernel_size"])
    w.u32(f"{a}.attn_res_block", tc["n_layers_per_attn_res_block"])
    w.f32(f"{a}.mlp_gate_softcap", tc["mlp_gate_softcap"])
    w.f32(f"{a}.mlp_up_softcap", tc["mlp_up_softcap"])
    w.u32(f"{a}.loop.core_start", lc["core_start"])
    w.u32(f"{a}.loop.core_end", lc["core_end"])
    w.u32(f"{a}.loop.num_passes", lc["num_passes"])
    w.u32(f"{a}.loop.gate_rank", lc["gate_rank"])

    # ---- tokenizer: tokenizer.json (byte-level BPE), added tokens as control tokens
    tj = json.load(open(P("tokenizer.json"), encoding="utf-8"))
    model = tj["model"]
    vocab = model["vocab"]
    n_vocab = int(tc["unpadded_vocab_size"])
    tokens = [None] * n_vocab
    types = [1] * n_vocab  # NORMAL
    for t, i in vocab.items():
        tokens[i] = t
    for at in tj["added_tokens"]:
        tokens[at["id"]] = at["content"]
        types[at["id"]] = 3 if at.get("special", True) else 4  # CONTROL / USER_DEFINED
    for i, t in enumerate(tokens):
        if t is None:
            tokens[i] = f"[PAD{i}]"
            types[i] = 5  # UNUSED
    merges = [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]
    eos = int(cfg.get("eos_token_id", 200006))
    w.str("tokenizer.ggml.model", "gpt2")
    w.str("tokenizer.ggml.pre", "inkling2")
    w.arr_str("tokenizer.ggml.tokens", tokens)
    w.arr_num("tokenizer.ggml.token_type", types, T_I32)
    w.arr_str("tokenizer.ggml.merges", merges)
    w.u32("tokenizer.ggml.eos_token_id", eos)
    w.bool("tokenizer.ggml.add_bos_token", False)
    w.bool("tokenizer.ggml.add_eos_token", False)
    tmpl = P("chat_template.jinja")
    if os.path.exists(tmpl):
        w.str("tokenizer.chat_template", open(tmpl, encoding="utf-8").read())

    # ---- the pack's arrays, in place
    files, names, fidx, offs, nbytes, dtypes, ndims, dims = [], [], [], [], [], [], [], []

    def add_file(fname, prefix, members):
        files.append(fname)
        fi = len(files) - 1
        size = os.path.getsize(P(fname))
        for name, off, dt, shape, nb in members:
            if off + nb > size:
                raise SystemExit(f"{fname}: {name} runs past the end of the file")
            names.append(prefix + name)
            fidx.append(fi)
            offs.append(off)
            nbytes.append(nb)
            dtypes.append(dt)
            ndims.append(len(shape))
            dims.extend(shape)

    for i in range(n_layer):
        add_file(f"layer_{i:02d}_packed.npz", f"L{i:02d}.", npz_members(P(f"layer_{i:02d}_packed.npz")))
    add_file("embed_packed.npz", "embed.", npz_members(P("embed_packed.npz")))
    add_file("head_packed.npz", "head.", npz_members(P("head_packed.npz")))
    add_file("luts.npz", "lut.", npz_members(P("luts.npz")))
    st = [(k[len("model."):] if k.startswith("model.") else k, o, d, s, n) for k, o, d, s, n in
          safetensors_members(P("steering.safetensors"))]
    add_file("steering.safetensors", "st.", st)

    w.str("inkling2.pack_dir", pack)
    w.arr_str("inkling2.files", files)
    w.arr_str("inkling2.t.names", names)
    w.arr_num("inkling2.t.file", fidx, T_U32)
    w.arr_num("inkling2.t.offset", offs, T_U64)
    w.arr_num("inkling2.t.nbytes", nbytes, T_U64)
    w.arr_str("inkling2.t.dtype", dtypes)
    w.arr_num("inkling2.t.ndim", ndims, T_U32)
    w.arr_num("inkling2.t.dims", dims, T_U64)

    # ---- the packed matrices: name as the dump tool takes it, tensor prefix, format, rows, cols, out_rot
    mn, mp, mf, mm, mc, mo = [], [], [], [], [], []
    for i in range(n_layer):
        meta = json.load(open(P(f"layer_{i:02d}_meta.json")))
        for lin, d in meta["linears"].items():
            mn.append(f"layer_{i:02d}.{lin}")
            mp.append(f"L{i:02d}.lin_{d['safe']}")
            mf.append(FMT_IDS[d["fmt"]])
            mm.append(int(d["m"]))
            mc.append(int(d["n"]))
            mo.append(1 if d.get("out_rot") else 0)
    for kind in ("embed", "head"):
        meta = json.load(open(P(f"{kind}_meta.json")))
        for bm in meta:
            mn.append(f"{kind}.b{bm['kb']}")
            mp.append(f"{kind}.b{bm['kb']}")
            mf.append(FMT_IDS[bm["opt"]])
            mm.append(int(bm["rows"]))
            mc.append(int(tc["hidden_size"]))
            mo.append(0)
    w.arr_str("inkling2.m.names", mn)
    w.arr_str("inkling2.m.prefix", mp)
    w.arr_num("inkling2.m.fmt", mf, T_U32)
    w.arr_num("inkling2.m.rows", mm, T_U32)
    w.arr_num("inkling2.m.cols", mc, T_U32)
    w.arr_num("inkling2.m.out_rot", mo, T_U32)

    out = os.path.join(outd, "inkling2.gguf")
    w.write(out)
    print(f"tensors {len(names)} matrices {len(mn)} vocab {n_vocab} merges {len(merges)}", file=sys.stderr)
    print(out)


if __name__ == "__main__":
    main()
