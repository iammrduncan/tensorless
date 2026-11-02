#!/usr/bin/env python3
# EXL3 format and trellis layout: Turboderp, exllamav3 (https://github.com/turboderp-org/exllamav3), MIT License,
# Copyright (c) 2025 Turboderp; see licenses/LICENSE-exllamav3. Independent reimplementation.
"""Verify complete EXL3 GGUF storage against the original pinned checkpoint."""

import argparse
import contextlib
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

SCRIPT_DIR = Path(__file__).resolve().parent
LLAMA_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(LLAMA_DIR))
sys.path.insert(0, str(LLAMA_DIR / "gguf-py"))
import gguf
from gguf import GGUFReader


EXPECTED_BITS = {3: 34, 4: 297, 5: 66, 6: 4}
EXPECTED_SOURCE_COMMIT = "4acd9ad5af224ec9e8815a54d71a033f378a7e9d"
EXPECTED_SHARDS = {
    "model-00001-of-00002.safetensors": "2633b0cbcc4b250a2837f78fac315d2f4d6c9ce3decc51a625bbf30a2a5b214f",
    "model-00002-of-00002.safetensors": "23879b3c29049377af1c14ba78a1a998d0bc7053bd9237f4a352821730a83c12",
}
MUL1 = 0x83DCD12D


class SourceModel:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.index = json.loads((self.model_dir / "model.safetensors.index.json").read_text())
        self.stack = contextlib.ExitStack()
        self.shards = {}

    def __enter__(self):
        for shard in sorted(set(self.index["weight_map"].values())):
            self.shards[shard] = self.stack.enter_context(safe_open(self.model_dir / shard, framework="pt", device="cpu"))
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def tensor(self, name):
        shard = self.index["weight_map"][name]
        return self.shards[shard].get_tensor(name)

    def names(self):
        return self.index["weight_map"].keys()


def canonical_source_name(name):
    return name.replace("model.language_model.", "model.")


def gguf_name(name_map, source_name):
    canonical = canonical_source_name(source_name)
    mapped = name_map.get_name(canonical, try_suffixes=(".weight", ".bias"))
    if mapped is None:
        match = re.fullmatch(r"model\.layers\.(\d+)\.linear_attn\.dt_bias", canonical)
        if match:
            mapped = f"blk.{match.group(1)}.ssm_dt.bias"
    if mapped is None:
        raise ValueError(f"cannot map source tensor {source_name}")
    return mapped


def block_permutation(config, size, kind):
    nk = int(config["linear_num_key_heads"])
    nv = int(config["linear_num_value_heads"])
    hk = int(config["linear_key_head_dim"])
    hv = int(config["linear_value_head_dim"])

    def vperm(count, head_dim):
        expected = nv * head_dim
        if count != expected:
            raise ValueError((kind, count, expected))
        return np.arange(count).reshape(nk, nv // nk, head_dim).transpose(1, 0, 2).reshape(-1)

    if kind == "qkv_out":
        qk = 2 * nk * hk
        return np.concatenate((np.arange(qk), qk + vperm(size - qk, hv)))
    if kind in ("z_out", "out_proj_in"):
        return vperm(size, hv)
    if kind == "head_vector":
        return vperm(size, 1)
    raise ValueError(kind)


def exl3_kind(prefix):
    if ".linear_attn." not in prefix:
        return None
    if prefix.endswith(".in_proj_qkv"):
        return "qkv_out"
    if prefix.endswith(".in_proj_z"):
        return "z_out"
    if prefix.endswith(".out_proj"):
        return "out_proj_in"
    return None


def array_bytes(array):
    return memoryview(np.ascontiguousarray(array)).cast("B")


def digest(array):
    return hashlib.sha256(array_bytes(array)).hexdigest()


def field_scalar(reader, name):
    field = reader.fields.get(name)
    if field is None:
        return None
    part = field.parts[-1]
    if field.types and field.types[0] == gguf.GGUFValueType.STRING:
        return bytes(part).decode()
    value = np.asarray(part).reshape(-1)
    return value[0].item() if value.size else None


def tensor_array(tensor):
    return np.asarray(tensor.data)


def compare_raw(actual, expected):
    a = array_bytes(actual)
    b = array_bytes(expected)
    return len(a) == len(b) and a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--expected-revision", default=EXPECTED_SOURCE_COMMIT)
    parser.add_argument("--max-errors", type=int, default=50)
    args = parser.parse_args()

    config_all = json.loads((args.model_dir / "config.json").read_text())
    config = {**config_all, **config_all.get("text_config", {})}
    n_layers = int(config["num_hidden_layers"])
    if n_layers != 64:
        raise ValueError(f"expected 64 text layers, got {n_layers}")

    reader = GGUFReader(args.gguf)
    output = {tensor.name: tensor for tensor in reader.tensors}
    name_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35, n_layers)
    errors = []
    stats = Counter()

    def fail(message):
        stats["failures"] += 1
        if len(errors) < args.max_errors:
            errors.append(message)

    def require(condition, message):
        if not condition:
            fail(message)
        return condition

    require(field_scalar(reader, "qwen35.exl3.codebook") == "mul1", "missing or invalid qwen35.exl3.codebook")
    require(field_scalar(reader, "qwen35.exl3.quant_version") == "1.4.2", "missing or invalid qwen35.exl3.quant_version")
    require(field_scalar(reader, "qwen35.exl3.hadamard_block") == 128, "missing or invalid qwen35.exl3.hadamard_block")
    require("exl3_had128.weight" in output, "missing exl3_had128.weight")
    if "exl3_had128.weight" in output:
        idx = np.arange(128)
        popcount = np.array([bin(value).count("1") for value in range(128)], dtype=np.int64)
        expected_had = ((-1.0) ** popcount[idx[:, None] & idx[None, :]]).astype(np.float32) / np.float32(np.sqrt(128))
        require(output["exl3_had128.weight"].tensor_type.name == "F32", "exl3_had128.weight is not F32")
        require(compare_raw(tensor_array(output["exl3_had128.weight"]), expected_had), "exl3_had128.weight mismatch")

    metadata_revisions = set()
    metadata_dir = args.model_dir / ".cache" / "huggingface" / "download"
    for metadata_path in metadata_dir.glob("*.metadata"):
        lines = metadata_path.read_text().splitlines()
        if lines:
            metadata_revisions.add(lines[0])
    require(metadata_revisions == {args.expected_revision},
            f"local checkpoint metadata revisions {sorted(metadata_revisions)}, expected only {args.expected_revision}")

    with SourceModel(args.model_dir) as source:
        names = list(source.names())
        prefixes = sorted(name[:-len(".trellis")] for name in names
                          if name.endswith(".trellis") and ".visual." not in name and not name.startswith("mtp."))
        require(len(prefixes) == 401, f"expected 401 retained EXL3 projections, got {len(prefixes)}")

        for index, prefix in enumerate(prefixes, 1):
            mapped = gguf_name(name_map, prefix + ".weight")
            weight = output.get(mapped)
            if not require(weight is not None, f"{prefix}: missing {mapped}"):
                continue
            trellis = source.tensor(prefix + ".trellis")
            suh = source.tensor(prefix + ".suh")
            svh = source.tensor(prefix + ".svh")
            mul1 = source.tensor(prefix + ".mul1")
            if not require(trellis.dtype == torch.int16 and suh.dtype == torch.float16 and svh.dtype == torch.float16,
                           f"{prefix}: invalid source dtype set"):
                continue
            require(mul1.numel() == 1 and (int(mul1.reshape(-1)[0]) & 0xFFFFFFFF) == MUL1, f"{prefix}: invalid mul1")
            kt, nt, width = trellis.shape
            bits = int(width) // 16
            require(width == 16 * bits and bits in EXPECTED_BITS, f"{prefix}: invalid trellis shape {tuple(trellis.shape)}")
            kind = exl3_kind(prefix)
            expected_trellis = trellis.numpy()
            expected_suh = suh.numpy()
            expected_svh = svh.numpy()
            if kind in ("qkv_out", "z_out"):
                perm = block_permutation(config, nt * 16, kind)
                tile_perm = perm.reshape(-1, 16)[:, 0] // 16
                expected_trellis = expected_trellis[:, tile_perm, :]
                expected_svh = expected_svh[perm]
                stats[f"permutation_{kind}"] += 1
            elif kind == "out_proj_in":
                perm = block_permutation(config, kt * 16, kind)
                tile_perm = perm.reshape(-1, 16)[:, 0] // 16
                expected_trellis = expected_trellis[tile_perm, :, :]
                expected_suh = expected_suh[perm]
                stats[f"permutation_{kind}"] += 1

            require(weight.tensor_type.name == f"EXL3_{bits}", f"{mapped}: type {weight.tensor_type.name}, expected EXL3_{bits}")
            require(compare_raw(tensor_array(weight), expected_trellis), f"{mapped}: trellis mismatch")
            for suffix, expected_side in ((".suh", expected_suh), (".svh", expected_svh)):
                side_name = mapped.replace(".weight", suffix)
                side = output.get(side_name)
                if not require(side is not None, f"{prefix}: missing {side_name}"):
                    continue
                require(side.tensor_type.name == "F32", f"{side_name}: type {side.tensor_type.name}, expected F32")
                require(compare_raw(tensor_array(side), expected_side.astype(np.float32)), f"{side_name}: side-vector mismatch")
            stats[f"exl3_{bits}"] += 1
            if index % 50 == 0 or index == len(prefixes):
                print(f"EXL3 storage {index}/{len(prefixes)}")

        excluded_visual = sorted(name for name in names if ".visual." in name)
        excluded_mtp = sorted(name for name in names if name.startswith("mtp."))
        stats["excluded_visual_tensors"] = len(excluded_visual)
        stats["excluded_mtp_tensors"] = len(excluded_mtp)
        remaining = sorted(name for name in names
                           if ".visual." not in name and not name.startswith("mtp.")
                           and not name.endswith((".trellis", ".suh", ".svh", ".mul1", ".mcg")))
        require(len(remaining) == 450, f"expected 450 retained non-EXL3 tensors, got {len(remaining)}")
        for index, name in enumerate(remaining, 1):
            mapped = gguf_name(name_map, name)
            target = output.get(mapped)
            if not require(target is not None, f"{name}: missing {mapped}"):
                continue
            value = source.tensor(name)
            actual = tensor_array(target)

            if name.endswith("embed_tokens.weight"):
                require(value.dtype == torch.bfloat16 and target.tensor_type.name == "BF16", f"{mapped}: embedding precision mismatch")
                expected_bits = value.view(torch.uint16).numpy()
                actual_bits = np.ascontiguousarray(actual).view(np.uint16).reshape(expected_bits.shape)
                require(compare_raw(actual_bits, expected_bits), f"{mapped}: BF16 embedding mismatch")
                stats["bf16_embedding"] += 1
            elif name.endswith((".linear_attn.in_proj_a.weight", ".linear_attn.in_proj_b.weight")):
                require(value.dtype == torch.float16 and target.tensor_type.name == "F16", f"{mapped}: alpha/beta precision mismatch")
                perm = torch.from_numpy(block_permutation(config, value.shape[0], "head_vector")).long()
                expected = value.index_select(0, perm).numpy()
                require(compare_raw(actual, expected), f"{mapped}: F16 alpha/beta mismatch")
                stats["f16_alpha_beta"] += 1
            elif name.endswith(".linear_attn.A_log"):
                require(target.tensor_type.name == "F32", f"{mapped}: A_log output is not F32")
                perm = torch.from_numpy(block_permutation(config, value.shape[0], "head_vector")).long()
                expected = -torch.exp(value.float().index_select(0, perm))
                require(compare_raw(actual, expected.numpy()), f"{mapped}: -exp(A_log) mismatch")
                stats["f32_decay"] += 1
            elif name.endswith(".linear_attn.dt_bias"):
                require(target.tensor_type.name == "F32", f"{mapped}: dt_bias output is not F32")
                perm = torch.from_numpy(block_permutation(config, value.shape[0], "head_vector")).long()
                expected = value.float().index_select(0, perm)
                require(compare_raw(actual, expected.numpy()), f"{mapped}: dt_bias mismatch")
                stats["f32_dt_bias"] += 1
            elif name.endswith(".linear_attn.conv1d.weight"):
                require(target.tensor_type.name == "F32", f"{mapped}: convolution output is not F32")
                expected = value.float().squeeze()
                qk = 2 * int(config["linear_num_key_heads"]) * int(config["linear_key_head_dim"])
                perm = torch.from_numpy(block_permutation(config, expected.shape[0] - qk, "z_out")).long()
                expected = torch.cat((expected[:qk], expected[qk:].index_select(0, perm)), dim=0)
                require(compare_raw(actual, expected.numpy()), f"{mapped}: convolution transform mismatch")
                stats["f32_convolution"] += 1
            elif name.endswith("norm.weight"):
                require(target.tensor_type.name == "F32", f"{mapped}: norm output is not F32")
                expected = value.float()
                if not name.endswith("linear_attn.norm.weight"):
                    expected = expected + 1
                    stats["f32_norm_plus_one"] += 1
                else:
                    stats["f32_gdn_norm"] += 1
                require(compare_raw(actual, expected.numpy()), f"{mapped}: norm transform mismatch")
            else:
                fail(f"{name}: unclassified retained tensor")
            if index % 100 == 0 or index == len(remaining):
                print(f"non-EXL3 storage {index}/{len(remaining)}")

    actual_bits = {bits: stats[f"exl3_{bits}"] for bits in EXPECTED_BITS}
    require(actual_bits == EXPECTED_BITS, f"EXL3 bitrate coverage {actual_bits}, expected {EXPECTED_BITS}")
    expected_output_count = 1 + 401 * 3 + 450
    require(len(output) == expected_output_count, f"GGUF has {len(output)} tensors, expected {expected_output_count}")

    summary = {
        "status": "pass" if not errors and stats["failures"] == 0 else "fail",
        "expected_revision": args.expected_revision,
        "gguf": str(args.gguf.resolve()),
        "gguf_sha256": digest(np.memmap(args.gguf, mode="r", dtype=np.uint8)),
        "source_shards": {},
        "tensor_count": len(output),
        "stats": dict(sorted(stats.items())),
        "errors": errors,
    }
    for shard, expected_hash in EXPECTED_SHARDS.items():
        path = args.model_dir / shard
        actual_hash = digest(np.memmap(path, mode="r", dtype=np.uint8))
        summary["source_shards"][shard] = {"sha256": actual_hash, "expected_sha256": expected_hash}
        if actual_hash != expected_hash:
            summary["status"] = "fail"
            summary["errors"].append(f"{shard}: SHA-256 {actual_hash}, expected {expected_hash}")
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(text)
    print(text, end="")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
