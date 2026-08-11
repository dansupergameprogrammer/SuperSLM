#!/usr/bin/env python3
"""T-1902 scratch tool, vitality-demonstration only (not part of the real
measurement): writes a COPY of a .sslm artifact with ONE named entry in its
`KvLandingReciprocals` section (type 9, `(m, e, R)` int64 triples, docs/
sslm_format.md's KVC1 "Keyed numeric-constant blob") overwritten so that
head's landing composite guarantees `LandingRescale`'s own
`out_magnitude_exceeded_int64` for any nonzero input -- by driving the `e`
word (word index 1) far enough negative that `ComposedExponent` lands deep
in the negative-k (left-shift) branch, past the 128-bit carry width
(`forward_sites.cpp`'s own `shift >= 128` case: any nonzero magnitude sets
`magnitude_exceeds_int64 = true` unconditionally there).

This produces a deliberately-broken artifact used ONLY to prove the
t1902_domain_gate_probe harness's --encoder/--decode counting path (not
just the isolated LandingRescale call --selftest already checked) reports a
nonzero OptionGFusedLandingExponentOutOfDomain count end to end, before any
zero from the real, unmodified artifact is trusted (StandardsDocument.md
§4's vitality-before-trust rule). It is never used for the real leg 1/leg 2
measurement and is deleted after the demonstration.

Recomputes the header hash after the edit, same procedure as patch_flags.py.

Usage:
    python poison_kv_landing.py <src.sslm> <out.sslm> <entry_name> [new_e]
"""
from __future__ import annotations

import hashlib
import struct
import sys

SECTION_TABLE_OFFSET = 64
SECTION_DESC_SIZE = 40
KV_LANDING_RECIPROCALS_TYPE = 9
kKvLandingReciprocalMax = 1 << 32  # src/model.cpp's own load-legal ceiling


def find_section(data: bytes, wanted_type: int) -> tuple[int, int]:
    section_count = struct.unpack_from("<I", data, 12)[0]
    for i in range(section_count):
        desc_off = SECTION_TABLE_OFFSET + i * SECTION_DESC_SIZE
        stype, dtype, offset, byte_size, elem_count, alignment, reserved = struct.unpack_from(
            "<IIQQQII", data, desc_off)
        if stype == wanted_type:
            return offset, byte_size
    raise SystemExit(f"no section of type {wanted_type} found")


def poison(src_path: str, out_path: str, entry_name: str, new_e: int = -1000) -> None:
    with open(src_path, "rb") as f:
        data = bytearray(f.read())

    sec_offset, sec_size = find_section(bytes(data), KV_LANDING_RECIPROCALS_TYPE)
    magic, version, entry_count, value_words, name_blob_len, reserved = struct.unpack_from(
        "<4sIIIII", data, sec_offset)
    if magic != b"KVC1":
        raise SystemExit(f"bad KVC1 magic at section offset {sec_offset}: {magic!r}")
    if value_words != 3:
        raise SystemExit(f"expected value_words=3 (KvLandingReciprocals), got {value_words}")

    desc_table_off = sec_offset + 24
    values_off = desc_table_off + entry_count * 8
    name_blob_off = values_off + entry_count * value_words * 8

    found_index = None
    for i in range(entry_count):
        name_off, name_len = struct.unpack_from("<II", data, desc_table_off + i * 8)
        name = bytes(data[name_blob_off + name_off:name_blob_off + name_off + name_len]).decode("utf-8")
        if name == entry_name:
            found_index = i
            break
    if found_index is None:
        raise SystemExit(f"entry \"{entry_name}\" not found in KvLandingReciprocals ({entry_count} entries)")

    # (m, e, R) tuple for this entry; word index 1 is `e` (e_t), word index 2
    # is `R` (r_t) -- both LandingRescale's own runtime inputs
    # (tools/sslm_marshal.h:264-289). Pushing e_t to the load-legal floor
    # (-60) AND r_t to the load-legal ceiling (2^32) simultaneously
    # maximizes the composed magnitude for whatever real activation reaches
    # this site, rather than relying on the exponent alone against
    # data-dependent m_a/branch_code that may not be large enough to
    # overflow on its own.
    e_value_off = values_off + (found_index * value_words + 1) * 8
    r_value_off = values_off + (found_index * value_words + 2) * 8
    old_e = struct.unpack_from("<q", data, e_value_off)[0]
    old_r = struct.unpack_from("<q", data, r_value_off)[0]
    struct.pack_into("<q", data, e_value_off, new_e)
    struct.pack_into("<q", data, r_value_off, kKvLandingReciprocalMax)
    print(f"entry \"{entry_name}\" (index {found_index}): e {old_e} -> {new_e}, "
          f"R {old_r} -> {kKvLandingReciprocalMax}")

    for i in range(32, 64):
        data[i] = 0
    digest = hashlib.sha256(bytes(data)).digest()
    data[32:64] = digest

    with open(out_path, "wb") as f:
        f.write(data)
    print(f"wrote {out_path}: {len(data)} bytes, sha256={digest.hex()}")


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        print(f"usage: {sys.argv[0]} <src.sslm> <out.sslm> <entry_name> [new_e]", file=sys.stderr)
        raise SystemExit(2)
    ne = int(sys.argv[4]) if len(sys.argv) == 5 else -1000
    poison(sys.argv[1], sys.argv[2], sys.argv[3], ne)
