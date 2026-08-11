#!/usr/bin/env python3
"""T-1902 scratch tool: writes a COPY of a .sslm artifact with header `flags`
(offset 16, u32 LE) set to a caller-supplied value, and the integrity
`integrity_sha256` field (offset 32, 32 bytes) recomputed over the whole
file with that field zeroed -- the exact procedure docs/sslm_format.md's
"Load-bearing choices" #2 specifies and tools/sslm_format.py's own
write_artifact performs at build time. This does NOT touch the source file;
it reads it once and writes a new file at --out.

Rationale (T-1902 brief §3): the shipped artifact's header flags field is 0
(no converter writes the Option-G bit yet, D-SLM2463); this script is the
"scratch copy of the artifact with the header bit set" route the ticket
names as one acceptable way to force the fused path on. Every other byte
(all weights, all calibration constants, all per-head K-landing
reciprocals) is left bit-identical to the shipped artifact -- only the one
bit under test changes, plus the hash that must agree with it or the real,
unmodified production loader (SslmModel::Load -> SslmArtifact::
OpenFromMemory) rejects the file outright (BadHeader on hash mismatch) --
which would be a false pass by refusal, not a measurement.

Usage:
    python patch_flags.py <src.sslm> <out.sslm> <flags_u32>
"""
from __future__ import annotations

import hashlib
import struct
import sys


def patch(src_path: str, out_path: str, flags: int) -> None:
    with open(src_path, "rb") as f:
        data = bytearray(f.read())

    if data[0:4] != b"SSLM":
        raise SystemExit(f"not a .sslm file (bad magic): {src_path}")
    file_bytes = struct.unpack_from("<Q", data, 24)[0]
    if file_bytes != len(data):
        raise SystemExit(
            f"header file_bytes ({file_bytes}) != actual size ({len(data)}) before patch")

    struct.pack_into("<I", data, 16, flags)
    # Zero the 32-byte hash field (offset 32..64) before hashing, per the
    # format's own documented integrity procedure.
    for i in range(32, 64):
        data[i] = 0
    digest = hashlib.sha256(bytes(data)).digest()
    data[32:64] = digest

    with open(out_path, "wb") as f:
        f.write(data)

    print(f"wrote {out_path}: {len(data)} bytes, flags={flags:#x}, "
          f"sha256={digest.hex()}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} <src.sslm> <out.sslm> <flags_u32>", file=sys.stderr)
        raise SystemExit(2)
    patch(sys.argv[1], sys.argv[2], int(sys.argv[3], 0))
