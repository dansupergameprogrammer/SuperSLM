"""Independent struct/hashlib reader for the precached-prefix SPX1 wire table."""

import argparse
import hashlib
from pathlib import Path
import re
import struct
import sys


def at(data: bytes, fmt: str, offset: int):
    return struct.unpack_from("<" + fmt, data, offset)[0]


def check(args) -> None:
    blob = args.blob.read_bytes()
    ssb = args.ssb.read_bytes()
    rows = args.rows.read_bytes()
    project = args.cmake.read_text(encoding="utf-8")
    match = re.search(r"\bproject\s*\(\s*superslm\s+VERSION\s+(\d+)\.(\d+)\.(\d+)\b", project, re.I)
    assert match, "CMake project version is absent"
    assert len(blob) == 184 + args.hidden + args.length * args.layers * 2 * args.heads * args.dim
    assert blob[:4] == b"SPX1" and ssb[:4] == b"SSB6"
    assert at(blob, "I", 4) == 0, "backend"
    assert tuple(at(blob, "I", offset) for offset in (8, 12, 16)) == tuple(map(int, match.groups())), "producer version"
    assert at(blob, "I", 20) == at(blob, "I", 24) == 0, "CPU producer adapter"
    digest = blob[28:60]
    assert hashlib.sha256(blob[:28] + bytes(32) + blob[60:]).digest() == digest, "digest"
    assert blob[60:92].hex() == args.model_sha, "artifact hash"
    assert blob[60:92] == ssb[4:36], "model hash versus SSB6"
    fields = (("I", 92, 36, "precision"),
              ("Q", 96, 40, "schema hash"),
              ("I", 104, 48, "walk"),
              ("I", 108, 72, "current token"),
              ("q", 112, 60, "context length"),
              ("q", 128, 76, "m"),
              ("q", 136, 84, "e"),
              ("Q", 144, 92, "aggregate saturation"),
              ("Q", 152, 124, "kv landing"),
              ("Q", 160, 132, "k channel landing"),
              ("Q", 168, 140, "rope q"),
              ("Q", 176, 148, "rope k"))
    for fmt, pos, twin_pos, name in fields:
        assert at(blob, fmt, pos) == at(ssb, fmt, twin_pos), name
    assert at(blob, "q", 112) == args.length, "known prefix length"
    assert at(blob, "q", 120) == args.origin, "known origin"
    assert at(blob, "Q", 96) == 0, "known unbound schema hash"
    assert blob[184:184 + args.hidden] == ssb[172:172 + args.hidden], "residual"
    assert blob[184 + args.hidden:] == rows, "canonical K/V rows from page-index peek"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("blob", "ssb", "rows", "cmake"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model-sha", required=True)
    for name in ("length", "origin", "layers", "heads", "dim", "hidden"):
        parser.add_argument("--" + name, type=int, required=True)
    args = parser.parse_args()
    try:
        check(args)
    except (AssertionError, OSError, struct.error) as exc:
        print(f"check_spx1: {exc}", file=sys.stderr)
        return 1
    print("check_spx1: every fixed field, digest, residual and canonical row matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
