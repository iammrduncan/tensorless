#!/usr/bin/env python3
"""Build src/data/model-map.json: the real structure of the launch targets, for the home page.

It reads, from Hugging Face, only the headers of the two released GGUF files (byte ranges, at the
recipes' pinned revisions) and a few hundred bytes of real PN blocks, and writes:
  - every tensor of every layer: shape, type and size in both models, and whether the engine
    rotates its input (the prism.hadamard metadata);
  - the 19 tensors promoted to pn4t; the bytes one decoded token reads;
  - the bench prompt, tokenized with the model's own vocabulary (byte-level BPE), which must come
    to the 429 tokens the server counted when the published image was measured;
  - the first blocks of one pn3t and one pn4t tensor, decoded.

Run from site/ with: python3 -I scripts/build-model-map.py   (standard library only)
"""
import json
import pathlib
import re
import struct
import urllib.request

SITE = pathlib.Path(__file__).resolve().parent.parent
REPO = SITE.parent
OUT = SITE / "src" / "data" / "model-map.json"

MODELS = {
    "pn3": ("iamMrDuncan/Qwen3.8-27B-PN3-GGUF", "c701b6690ec360b16d40f85df1cb76062e30cd38", "Qwen3.8-27B-PN3-P0k19.gguf"),
    "q3floor": ("iamMrDuncan/Qwen3.8-27B-UD-IQ3_XXS-Q3Floor-GGUF", "11f76ec7e1f7be8e76563ee2b9a912a8519e9801", "Qwen3.8-27B-UD-IQ3_XXS-Q3Floor.gguf"),
}
HEAD_BYTES = 48 * 1024 * 1024  # both headers are about 11 MB; the vocabulary is most of it

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
         16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
         30: "BF16", 124: "pn3t", 125: "pn4t"}
# (bytes per block, weights per block)
BLOCK = {"F32": (4, 1), "F16": (2, 1), "BF16": (2, 1), "Q4_0": (18, 32), "Q8_0": (34, 32), "Q2_K": (84, 256),
         "Q3_K": (110, 256), "Q4_K": (144, 256), "Q5_K": (176, 256), "Q6_K": (210, 256), "IQ2_XXS": (66, 256),
         "IQ2_XS": (74, 256), "IQ2_S": (82, 256), "IQ3_XXS": (98, 256), "IQ3_S": (110, 256), "IQ4_XS": (136, 256),
         "IQ4_NL": (18, 32), "pn3t": (34, 80), "pn4t": (34, 64)}


def url(repo, rev, file):
    return f"https://huggingface.co/{repo}/resolve/{rev}/{file}"


def fetch(u, start, end):
    req = urllib.request.Request(u, headers={"Range": f"bytes={start}-{end}", "User-Agent": "tensorless-site"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


class Reader:
    def __init__(self, data):
        self.d, self.o = data, 0

    def u(self, fmt):
        v = struct.unpack_from("<" + fmt, self.d, self.o)[0]
        self.o += struct.calcsize("<" + fmt)
        return v

    def s(self):
        n = self.u("Q")
        v = self.d[self.o:self.o + n].decode("utf-8", "replace")
        self.o += n
        return v

    def val(self, t):
        if t == 8:
            return self.s()
        if t == 9:
            at, n = self.u("I"), self.u("Q")
            return [self.val(at) for _ in range(n)]
        return self.u({0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[t])


def parse(data):
    r = Reader(data)
    assert data[:4] == b"GGUF"
    r.o = 4
    r.u("I")
    nt, nkv = r.u("Q"), r.u("Q")
    kv = {}
    for _ in range(nkv):
        k = r.s()
        kv[k] = r.val(r.u("I"))
    tensors = {}
    for _ in range(nt):
        name = r.s()
        dims = [r.u("Q") for _ in range(r.u("I"))]
        ty = TYPES[r.u("I")]
        off = r.u("Q")
        n = 1
        for x in dims:
            n *= x
        bb, bw = BLOCK[ty]
        tensors[name] = {"dims": dims, "type": ty, "offset": off, "bytes": n // bw * bb}
    align = kv.get("general.alignment", 32)
    return kv, tensors, (r.o + align - 1) // align * align


# --- byte-level BPE, as the qwen35 pre-tokenizer and the GGUF vocabulary define it -----------
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


# The engine's qwen35 regex (src/llama-vocab.cpp), for ASCII text: \p{L} = [A-Za-z], \p{N} = [0-9], no \p{M}.
PRE = re.compile(r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\nA-Za-z0-9]?[A-Za-z]+|[0-9]| ?[^\sA-Za-z0-9]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def tokenize(text, vocab, ranks):
    assert text.isascii(), "this tokenizer handles ASCII text only"
    b2u = bytes_to_unicode()
    ids = []
    for piece in PRE.findall(text):
        word = [b2u[b] for b in piece.encode()]
        while len(word) > 1:
            best = min(((ranks.get((word[i], word[i + 1]), 1 << 30), i) for i in range(len(word) - 1)))
            if best[0] == 1 << 30:
                break
            i = best[1]
            word = word[:i] + [word[i] + word[i + 1]] + word[i + 2:]
        ids.extend(vocab[w] for w in word)
    return ids


def decode_piece(tok):
    u2b = {v: k for k, v in bytes_to_unicode().items()}
    return bytes(u2b[c] for c in tok).decode("utf-8", "replace")


def short(name):
    return name.split(".", 2)[2].removesuffix(".weight") if name.startswith("blk.") else name.removesuffix(".weight")


# Execution order inside a layer, for drawing.
ORDER = ["attn_norm", "attn_qkv", "attn_gate", "ssm_conv1d", "ssm_alpha", "ssm_beta", "ssm_a", "ssm_dt.bias", "ssm_norm",
         "ssm_out", "attn_q", "attn_k", "attn_v", "attn_q_norm", "attn_k_norm", "attn_output", "post_attention_norm",
         "ffn_gate", "ffn_up", "ffn_down", "nextn.enorm", "nextn.hnorm", "nextn.eh_proj", "nextn.shared_head_norm"]


def main():
    heads = {}
    for key, (repo, rev, file) in MODELS.items():
        print(f"reading the header of {repo}@{rev[:12]}")
        heads[key] = parse(fetch(url(repo, rev, file), 0, HEAD_BYTES - 1))
    kv, pn, pn_start = heads["pn3"]
    _, q3, _ = heads["q3floor"]

    rotated = set(kv["prism.hadamard.weight_names"])
    n_layer = kv["qwen35.block_count"]
    layers = []
    for b in range(n_layer):
        names = [n for n in pn if n.startswith(f"blk.{b}.")]
        kind = "nextn" if any(".nextn." in n for n in names) else ("attention" if f"blk.{b}.attn_q.weight" in pn else "deltanet")
        names.sort(key=lambda n: ORDER.index(short(n)) if short(n) in ORDER else 99)
        rows = []
        for n in names:
            t, u = pn[n], q3[n]
            dims = t["dims"] + [1] * (2 - len(t["dims"]))
            rows.append([short(n), dims[0], dims[1], t["type"], t["bytes"], u["type"], u["bytes"], 1 if n in rotated else 0])
        layers.append({"kind": kind, "t": rows})
    globals_ = []
    for n in ("token_embd.weight", "output_norm.weight", "output.weight"):
        t, u = pn[n], q3[n]
        dims = t["dims"] + [1] * (2 - len(t["dims"]))
        globals_.append([short(n), dims[0], dims[1], t["type"], t["bytes"], u["type"], u["bytes"], 1 if n in rotated else 0])

    promoted = sorted((n for n, t in pn.items() if t["type"] == "pn4t" and t["dims"][0] == 5120
                       and ".attn_k." not in n and ".attn_v." not in n), key=lambda n: int(n.split(".")[1]))
    assert len(promoted) == 19, len(promoted)
    assert len(rotated) == sum(1 for t in pn.values() if t["type"] in ("pn3t", "pn4t")), "every PN tensor is rotated"

    def per_token(tensors):
        # A decoded token reads every layer except the unused next-token (MTP) layer, the output
        # head, and one row of the token embedding.
        total = sum(t["bytes"] for n, t in tensors.items() if n.startswith("blk.") and int(n.split(".")[1]) < n_layer - 1)
        emb = tensors["token_embd.weight"]
        bb, bw = BLOCK[emb["type"]]
        return total + tensors["output.weight"]["bytes"] + tensors["output_norm.weight"]["bytes"] + emb["dims"][0] // bw * bb

    # The bench prompt (internal/cli/bench.go): the first 2,000 characters of the license, then an instruction.
    license_text = (REPO / "LICENSE").read_text()
    prompt = license_text[:2000] + "\n\nContinue the text above."
    vocab_list = kv["tokenizer.ggml.tokens"]
    vocab = {t: i for i, t in enumerate(vocab_list)}
    ranks = {tuple(m.split(" ", 1)): i for i, m in enumerate(kv["tokenizer.ggml.merges"])}
    ids = tokenize(prompt, vocab, ranks)
    assert "".join(decode_piece(vocab_list[i]) for i in ids) == prompt, "the tokens must decode back to the prompt"
    assert len(ids) == 429, f"expected the 429 prompt tokens the server counted, got {len(ids)}"
    print(f"bench prompt: {len(ids)} tokens, matching the measured prompt_n")

    # Real PN blocks: the first 6 blocks of one pn3t and one pn4t tensor of layer 0.
    repo, rev, file = MODELS["pn3"]
    blocks = {}
    for ty, name in (("pn3t", "blk.0.ffn_gate.weight"), ("pn4t", "blk.0.ffn_down.weight")):
        t = pn[name]
        start = pn_start + t["offset"]
        raw = fetch(url(repo, rev, file), start, start + 34 * 6 - 1)
        out = []
        for k in range(6):
            blk = raw[34 * k: 34 * (k + 1)]
            d = struct.unpack_from("<e", blk, 0)[0]
            words = list(struct.unpack_from("<8I", blk, 2))
            if ty == "pn3t":
                assert all(w & 0x80008000 == 0 for w in words), "pn3t reserved bits must be zero"
            out.append({"d": d, "words": words})
        blocks[ty] = {"tensor": name, "shape": t["dims"], "blocks": out}

    data = {
        "source": {key: {"repo": r, "revision": v, "file": f} for key, (r, v, f) in MODELS.items()},
        "arch": kv["general.architecture"],
        "dims": {
            "embd": kv["qwen35.embedding_length"], "ffn": kv["qwen35.feed_forward_length"],
            "heads": kv["qwen35.attention.head_count"], "kv_heads": kv["qwen35.attention.head_count_kv"],
            "head_dim": kv["qwen35.attention.key_length"], "ssm_heads": kv["qwen35.ssm.time_step_rank"],
            "ssm_state": kv["qwen35.ssm.state_size"], "ssm_groups": kv["qwen35.ssm.group_count"],
            "ssm_inner": kv["qwen35.ssm.inner_size"], "conv_kernel": kv["qwen35.ssm.conv_kernel"],
            "vocab": len(vocab_list), "context_train": kv["qwen35.context_length"],
            "full_attention_interval": kv["qwen35.full_attention_interval"], "layers": n_layer,
        },
        "rotation": {
            "transform": kv["prism.hadamard.transform"], "block": kv["prism.hadamard.block_size"],
            "widths": kv["prism.hadamard.sign_widths"], "signs": kv["prism.hadamard.sign_mode"],
            "count": len(rotated), "first_signs": kv["prism.hadamard.sign_values"][:64],
        },
        "composition": kv["pnx.composition"],
        "layers": layers,
        "globals": globals_,
        "promoted": promoted,
        "per_token_bytes": {"pn3": per_token(pn), "q3floor": per_token(q3)},
        "file_bytes": {"pn3": pn_start + sum(t["bytes"] for t in pn.values())},
        "prompt": {"count": len(ids), "tokens": [[i, decode_piece(vocab_list[i])] for i in ids]},
        "blocks": blocks,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT.relative_to(SITE)}: {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
