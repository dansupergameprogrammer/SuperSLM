#!/usr/bin/env python3
"""T-1911 -- merge a multi-shard safetensors checkpoint into one single-file checkpoint.

**This is a disposable offline tool, not shared conversion code.** `pipeline.py`'s
`_checkpoint_tensor_file` correctly rejects a checkpoint with more than one `.safetensors`
file -- a shard it did not read is a silently absent projection, and that rejection is not
being changed here (proper multi-shard support is a separate, designed change: T-1912). This
script produces a single-file checkpoint the existing loader already accepts unmodified, so a
measurement blocked on a sharded checkpoint (T-1909) can proceed while the real support is
built.

**Format, read directly, no framework.** The merge reads each shard's safetensors header (an
8-byte little-endian header length, then that many bytes of JSON) exactly as
`pipeline.py`'s own `_SafeTensors` reader does, and copies each tensor's raw on-disk bytes
verbatim from its source shard into the merged file at a recomputed offset. Nothing decodes,
widens, or reinterprets a tensor's bytes at any point -- there is no numpy/torch dependency
and nothing that could round or upcast a bf16 value. The merged file's per-tensor `dtype` and
`shape` fields are copied unchanged from the source shard's own header.

Usage:
    python t1911_merge_checkpoint_shards.py --checkpoint <sharded snapshot dir> --out <new dir>

The output directory is created fresh (must not already exist) and receives:
  - `model.safetensors`      -- the merged single-file checkpoint (this script's output)
  - every non-shard, non-index file from the source checkpoint directory, copied verbatim
    (config.json, generation_config.json, tokenizer.json, tokenizer_config.json, vocab.json,
    merges.txt, ...) -- whatever the loader needs alongside the weights.

`model.safetensors.index.json` and the original shard files are NOT copied into the output
directory (the merge supersedes them) and are never modified or deleted at the source.
"""
import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

HEADER_LEN_BYTES = 8
COPY_CHUNK = 64 * 1024 * 1024  # 64 MiB


def read_safetensors_header(path: Path):
    """(tensor_name -> {dtype, shape, data_offsets}), absolute byte offset where data begins."""
    with open(path, "rb") as f:
        raw_len = f.read(HEADER_LEN_BYTES)
        if len(raw_len) != HEADER_LEN_BYTES:
            raise SystemExit(f"{path}: file shorter than an 8-byte safetensors header")
        header_len = struct.unpack("<Q", raw_len)[0]
        header = json.loads(f.read(header_len))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    data_start = HEADER_LEN_BYTES + header_len
    return tensors, data_start


def load_index(checkpoint: Path):
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"{index_path}: not present -- {checkpoint} is not a sharded "
                          f"safetensors checkpoint (nothing for this tool to merge)")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if "weight_map" not in index:
        raise SystemExit(f"{index_path}: no 'weight_map' key -- unexpected index format")
    return index


def plan_merge(checkpoint: Path, index: dict):
    """Read every shard's header, cross-check against the index, and lay out the merged file.

    Returns (plan, header_obj, declared_total_size) where `plan` is an ordered list of
    (tensor_name, shard_name, abs_source_offset, length, dtype, shape) and `header_obj` is the
    JSON-ready header for the merged file (offsets already recomputed, `__metadata__` included).
    """
    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))

    shard_headers = {}
    shard_data_start = {}
    for shard_name in shard_names:
        shard_path = checkpoint / shard_name
        if not shard_path.exists():
            raise SystemExit(f"{shard_path}: shard named in the index is not present on disk")
        tensors, data_start = read_safetensors_header(shard_path)
        shard_headers[shard_name] = tensors
        shard_data_start[shard_name] = data_start

    # Set comparison, both directions: every name the index claims must actually be a tensor
    # in *some* shard's own header, and every tensor any shard's header actually carries must
    # be named by the index. A silent extra or a silent drop on either side is exactly the
    # failure mode this tool exists to make impossible.
    index_names = set(weight_map)
    shard_names_union = set()
    for tensors in shard_headers.values():
        shard_names_union |= set(tensors)
    symmetric_diff = index_names ^ shard_names_union
    if symmetric_diff:
        only_index = sorted(index_names - shard_names_union)
        only_shards = sorted(shard_names_union - index_names)
        raise SystemExit(
            f"Index/shard-header name mismatch, symmetric difference = {len(symmetric_diff)}. "
            f"In index but in no shard header ({len(only_index)}): {only_index[:5]}. "
            f"In a shard header but not in the index ({len(only_shards)}): {only_shards[:5]}."
        )

    # Every name must resolve through the SPECIFIC shard the index maps it to -- not merely
    # exist somewhere -- so a mis-mapped name (present, but in the wrong shard) is also caught.
    for name, shard_name in weight_map.items():
        if name not in shard_headers[shard_name]:
            raise SystemExit(
                f"{name}: index maps it to {shard_name}, but that shard's own header has no "
                f"tensor by that name"
            )

    ordered_names = sorted(weight_map)  # deterministic; lookup by name downstream, order is inert

    header_obj = {}
    plan = []
    cursor = 0
    for name in ordered_names:
        shard_name = weight_map[name]
        spec = shard_headers[shard_name][name]
        start, end = spec["data_offsets"]
        length = end - start
        abs_start = shard_data_start[shard_name] + start
        header_obj[name] = {
            "dtype": spec["dtype"],
            "shape": list(spec["shape"]),
            "data_offsets": [cursor, cursor + length],
        }
        plan.append((name, shard_name, abs_start, length, spec["dtype"], list(spec["shape"])))
        cursor += length

    declared_total = (index.get("metadata") or {}).get("total_size")
    header_obj["__metadata__"] = {"format": "pt"}
    return plan, header_obj, declared_total, cursor


def write_merged(checkpoint: Path, out_path: Path, plan, header_obj):
    header_bytes = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")
    shard_files = {}
    try:
        with open(out_path, "wb") as out:
            out.write(struct.pack("<Q", len(header_bytes)))
            out.write(header_bytes)
            for name, shard_name, abs_start, length, _dtype, _shape in plan:
                if shard_name not in shard_files:
                    shard_files[shard_name] = open(checkpoint / shard_name, "rb")
                f = shard_files[shard_name]
                f.seek(abs_start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(remaining, COPY_CHUNK))
                    if not chunk:
                        raise SystemExit(f"{name}: unexpected EOF copying from {shard_name}")
                    out.write(chunk)
                    remaining -= len(chunk)
    finally:
        for f in shard_files.values():
            f.close()


def verify_merge(checkpoint: Path, out_path: Path, weight_map: dict, plan):
    """Independent, from-disk verification -- re-reads the merged file fresh, does not reuse
    anything held in memory from the write pass."""
    merged_tensors, merged_data_start = read_safetensors_header(out_path)

    merged_names = set(merged_tensors)
    index_names = set(weight_map)
    name_symdiff = merged_names ^ index_names
    print(f"Name-set check: index names={len(index_names)}, merged-file names={len(merged_names)}, "
          f"symmetric difference={len(name_symdiff)} (expect 0)")
    if name_symdiff:
        raise SystemExit(f"Merged file's tensor names do not match the index: "
                          f"{sorted(name_symdiff)[:10]}")

    print(f"Tensor-count check: index weight_map={len(weight_map)}, merge plan={len(plan)}, "
          f"merged file header={len(merged_tensors)} (expect all equal)")
    if not (len(weight_map) == len(plan) == len(merged_tensors)):
        raise SystemExit("Tensor counts disagree across index / plan / merged-file header")

    compared = 0
    shard_files = {}
    try:
        with open(out_path, "rb") as merged_f:
            for name, shard_name, abs_start, length, dtype, shape in plan:
                mspec = merged_tensors[name]
                if mspec["dtype"] != dtype or list(mspec["shape"]) != list(shape):
                    raise SystemExit(f"{name}: dtype/shape changed across the merge -- "
                                      f"source ({dtype}, {shape}) vs merged "
                                      f"({mspec['dtype']}, {mspec['shape']})")
                mstart, mend = mspec["data_offsets"]
                if mend - mstart != length:
                    raise SystemExit(f"{name}: byte length changed across the merge -- "
                                      f"source {length}, merged {mend - mstart}")

                if shard_name not in shard_files:
                    shard_files[shard_name] = open(checkpoint / shard_name, "rb")
                sf = shard_files[shard_name]
                sf.seek(abs_start)
                source_bytes = sf.read(length)

                merged_f.seek(merged_data_start + mstart)
                merged_bytes = merged_f.read(length)

                if source_bytes != merged_bytes:
                    raise SystemExit(f"{name}: byte mismatch between source shard {shard_name} "
                                      f"and merged file -- NOT byte-for-byte identical")
                compared += 1
    finally:
        for f in shard_files.values():
            f.close()

    print(f"Byte-for-byte check: {compared} of {len(plan)} tensors compared against their "
          f"source shard, {compared} matched, 0 mismatched")
    return compared


SIDE_FILES_SKIP = {"model.safetensors.index.json"}


def copy_side_files(checkpoint: Path, out_dir: Path, shard_names):
    skip = set(SIDE_FILES_SKIP) | set(shard_names)
    copied = []
    for entry in sorted(checkpoint.iterdir()):
        if entry.name in skip or not entry.is_file():
            continue
        dest = out_dir / entry.name
        shutil.copyfile(entry, dest)
        copied.append(entry.name)
    return copied


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path,
                     help="Directory holding model.safetensors.index.json + shard files")
    ap.add_argument("--out", required=True, type=Path,
                     help="New output directory (must not already exist)")
    args = ap.parse_args()

    checkpoint = args.checkpoint.resolve()
    out_dir = args.out.resolve()

    if out_dir.exists():
        raise SystemExit(f"{out_dir}: already exists -- refusing to write into an existing "
                          f"directory (pass a fresh path)")

    index = load_index(checkpoint)
    weight_map = index["weight_map"]
    print(f"Index: {checkpoint / 'model.safetensors.index.json'}")
    print(f"  weight_map: {len(weight_map)} tensor names across "
          f"{len(set(weight_map.values()))} shards")

    plan, header_obj, declared_total, computed_total = plan_merge(checkpoint, index)
    print(f"Plan: {len(plan)} tensors, {computed_total} bytes of tensor data "
          f"({computed_total / (1024**3):.3f} GiB)")
    if declared_total is not None:
        match = "matches" if declared_total == computed_total else "DOES NOT MATCH"
        print(f"  index metadata.total_size = {declared_total} -- {match} computed total "
              f"{computed_total}")
        if declared_total != computed_total:
            raise SystemExit("Computed tensor-data size disagrees with the index's own "
                              "declared total_size -- refusing to proceed")

    out_dir.mkdir(parents=True, exist_ok=False)
    out_path = out_dir / "model.safetensors"
    print(f"Writing merged checkpoint: {out_path}")
    write_merged(checkpoint, out_path, plan, header_obj)
    on_disk = out_path.stat().st_size
    print(f"  wrote {on_disk} bytes ({on_disk / (1024**3):.3f} GiB) -- "
          f"header + {computed_total} bytes of tensor data")

    print("Verifying...")
    compared = verify_merge(checkpoint, out_path, weight_map, plan)

    shard_names = set(weight_map.values())
    copied = copy_side_files(checkpoint, out_dir, shard_names)
    print(f"Copied {len(copied)} side files alongside the merged checkpoint: {copied}")

    safetensor_glob = sorted(p.name for p in out_dir.glob("*.safetensors"))
    print(f"Output directory now has {len(safetensor_glob)} *.safetensors file(s): "
          f"{safetensor_glob} (loader needs exactly 1)")
    if len(safetensor_glob) != 1:
        raise SystemExit(f"Expected exactly one *.safetensors file in {out_dir}, "
                          f"found {len(safetensor_glob)}")

    print()
    print("=== SUMMARY ===")
    print(f"Tensors in index:           {len(weight_map)}")
    print(f"Tensors in merged header:   {len(plan)}")
    print(f"Name-set symmetric diff:    0 (verified)")
    print(f"Tensors byte-compared:      {compared}")
    print(f"Tensors byte-matched:       {compared}")
    print(f"Output:                     {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
