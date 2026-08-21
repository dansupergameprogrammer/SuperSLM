"""Add B0's explicitly selected DGC1 source scale to a copy of an existing model artifact."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import sslm_format as F


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scale-m", type=int, default=2883584,
                        help="source mantissa; default derives (q_ln2,q_b,q_c)=(493,964,487361)")
    parser.add_argument("--scale-e", type=int, default=-36)
    args = parser.parse_args()

    header = F.read_header(args.input)
    if any(s["type"] == F.SectionType.DAMPED_GREEDY_CONSTANTS for s in header["sections"]):
        raise RuntimeError(f"{args.input}: already carries a DampedGreedyConstants section")
    sections = []
    for descriptor in header["sections"]:
        begin = descriptor["offset"]
        end = begin + descriptor["byte_size"]
        sections.append(F.Section(
            descriptor["type"], bytes(header["data"][begin:end]), dtype=descriptor["dtype"],
            elem_count=descriptor["elem_count"], alignment=descriptor["alignment"]))
    sections.append(F.Section(
        F.SectionType.DAMPED_GREEDY_CONSTANTS,
        struct.pack("<qi", args.scale_m, args.scale_e), alignment=64))
    old_flags = struct.unpack_from("<I", header["data"], 16)[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = F.write_artifact(
        args.output, sections, flags=old_flags | F.DAMPED_GREEDY_CONSTANTS_FLAG)
    print(f"wrote {args.output}")
    print(f"fingerprint {fingerprint}")
    print(f"DGC1 m={args.scale_m} e={args.scale_e}; expected q=(493,964,487361)")


if __name__ == "__main__":
    main()
