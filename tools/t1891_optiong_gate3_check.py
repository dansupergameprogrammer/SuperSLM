#!/usr/bin/env python3
"""T-1891 gate G3 -- reference parity, the COMPARISON half.

DISPOSABLE. Branch brunel/t1891-optionG-spike only, never merged.

Reads the engine's dumped K/V store (`tools/t1891_optiong_gate3_probe.cpp`'s binary
format: num_docs, num_layers, num_kv_heads, head_dim, context_cap, then per document
[doc_len, raw workspace bytes] -- T-1892 Observation 2: `context_cap` is read from
the dump header, not hardcoded here, so this script cannot silently drift from the
probe's own geometry) and the reference's dumped K-landing trace records
(`tools/t1891_optiong_gate3_reference.py`'s JSON), and compares them element-wise:
for every (document, layer, token position < doc_len, kv_head, d), the engine's K
store byte at that address must equal the reference's `k_proj.requant` trace record's
`codes[d]` for that (layer, token_index, head).

The K/V store's layout (read here, not reinvented): per-(layer, kv_head)-major,
position-minor within a layer's K half -- `offset(kv_head, position, d) = kv_head *
context_cap * head_dim + position * head_dim + d`, K half starting at
`layer * context_cap * num_kv_heads * head_dim * 2` (forward_sites.h's own documented
layout, unchanged by Option G).

Usage: python tools/t1891_optiong_gate3_check.py <kv_bin_path> <reference_json_path>
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path


def read_kv_dump(path: Path):
    with open(path, "rb") as f:
        data = f.read()
    off = 0

    def u64():
        nonlocal off
        v = struct.unpack_from("<Q", data, off)[0]
        off += 8
        return v

    num_docs = u64()
    num_layers = u64()
    num_kv_heads = u64()
    head_dim = u64()
    context_cap = u64()
    docs = []
    for _ in range(num_docs):
        doc_len = u64()
        kv_bytes = num_layers * context_cap * num_kv_heads * head_dim * 2
        blob = data[off:off + kv_bytes]
        off += kv_bytes
        docs.append((doc_len, blob))
    return num_layers, num_kv_heads, head_dim, context_cap, docs


def engine_k(blob: bytes, num_kv_heads: int, head_dim: int, context_cap: int, layer: int,
             kv_head: int, position: int) -> list[int]:
    layer_stride = context_cap * num_kv_heads * head_dim * 2
    k_half_base = layer * layer_stride
    offset = k_half_base + kv_head * context_cap * head_dim + position * head_dim
    raw = blob[offset:offset + head_dim]
    return [b - 256 if b > 127 else b for b in raw]  # int8 sign-extend


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: t1891_optiong_gate3_check.py <kv_bin_path> <reference_json_path>",
              file=sys.stderr)
        return 2
    kv_path = Path(argv[1])
    ref_path = Path(argv[2])

    num_layers, num_kv_heads, head_dim, context_cap, docs = read_kv_dump(kv_path)
    with open(ref_path) as f:
        reference = json.load(f)["documents"]

    if len(docs) != len(reference):
        print(f"FAIL: engine dumped {len(docs)} documents, reference has "
              f"{len(reference)}", file=sys.stderr)
        return 1

    total_cells = 0
    mismatches = []
    for doc_idx, ((doc_len, blob), ref_doc) in enumerate(zip(docs, reference)):
        # Index the reference's K-landing records by (layer, token_index, head).
        by_key = {}
        for rec in ref_doc["k_records"]:
            site = rec["site"]  # "layer{L}.k_proj.requant"
            layer = int(site.split(".")[0].removeprefix("layer"))
            by_key[(layer, rec["token_index"], rec["head"])] = rec["codes"]

        for layer in range(num_layers):
            for position in range(min(doc_len, len(ref_doc["tokens"]))):
                for kv_head in range(num_kv_heads):
                    expected = by_key.get((layer, position, kv_head))
                    if expected is None:
                        mismatches.append(
                            f"doc={doc_idx} layer={layer} pos={position} head={kv_head}: "
                            f"no reference record")
                        continue
                    actual = engine_k(blob, num_kv_heads, head_dim, context_cap, layer, kv_head,
                                       position)
                    total_cells += 1
                    if actual != expected:
                        mismatches.append(
                            f"doc={doc_idx} ({ref_doc['doc']!r}) layer={layer} pos={position} "
                            f"head={kv_head}: engine={actual} reference={expected}")

    print(f"T-1891 gate G3: {total_cells} (document, layer, position, kv_head) K-row cells "
          f"compared across {len(docs)} documents, {num_layers} layers, {num_kv_heads} kv heads")
    if mismatches:
        print(f"FAIL: {len(mismatches)} mismatches", file=sys.stderr)
        for m in mismatches[:30]:
            print(f"  {m}", file=sys.stderr)
        if len(mismatches) > 30:
            print(f"  ... and {len(mismatches) - 30} more", file=sys.stderr)
        return 1

    print(f"PASS: engine and reference K store agree bit-for-bit on all {total_cells} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
