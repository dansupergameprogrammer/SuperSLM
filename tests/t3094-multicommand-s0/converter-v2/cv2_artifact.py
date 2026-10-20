"""Independent readers for the attachment cells: the `.sslm` container (docs/sslm_format.md: 64-byte header,
40-byte section descriptors, SHA-256 over the file with the hash field zeroed) and a multi-schema SCM1
section (t2119 design Sec13.2 as written by the T-2132 fixture mechanism: 24-byte header, 56-byte
descriptors, name blob, then per schema mask pages, CSR state offsets, accepting ids and transitions).
Written without the converter, `sslm_format` or the C++ reader."""
from __future__ import annotations

import hashlib
import struct

TOKENIZER, UNICODE_TABLES, SCHEMA_MASKS, SIGMOID_LUT, CONFIG = 20, 22, 30, 12, 0


def read_container(raw: bytes) -> dict:
    if raw[:4] != b"SSLM":
        raise ValueError("bad magic")
    version, header_bytes, count, flags, _ = struct.unpack_from("<IIIII", raw, 4)
    (file_bytes,) = struct.unpack_from("<Q", raw, 24)
    if file_bytes != len(raw):
        raise ValueError("file_bytes mismatch")
    zeroed = bytearray(raw)
    zeroed[32:64] = bytes(32)
    if hashlib.sha256(zeroed).digest() != raw[32:64]:
        raise ValueError("integrity hash mismatch")
    sections = []
    for i in range(count):
        row = header_bytes + 40 * i
        s_type, dtype = struct.unpack_from("<II", raw, row)
        off, size, elems = struct.unpack_from("<QQQ", raw, row + 8)
        align, _ = struct.unpack_from("<II", raw, row + 32)
        sections.append({"type": s_type, "dtype": dtype, "elem_count": elems, "alignment": align,
                         "data": bytes(raw[off:off + size])})
    return {"version": version, "flags": flags, "sections": sections}


def config_vocab_size(container) -> int:
    data = next(s["data"] for s in container["sections"] if s["type"] == CONFIG)
    return struct.unpack_from("<I", data, 32)[0]


def parse_scm1(data: bytes) -> dict:
    magic, version, count, vocab, name_blob, reserved = struct.unpack_from("<4sIIIII", data, 0)
    if magic != b"SCM1" or version != 1 or reserved != 0 or count == 0:
        raise ValueError("SCM1 header")
    names_at = 24 + 56 * count
    page_width = (vocab + 7) // 8
    schemas = {}
    cursor = names_at + name_blob
    for i in range(count):
        (noff, nlen, states, accepts, transitions, dres,
         masks_at, offs_at, acc_at, trans_at) = struct.unpack_from("<IIIIIIQQQQ", data, 24 + 56 * i)
        if dres != 0 or nlen == 0 or noff + nlen > name_blob:
            raise ValueError("SCM1 descriptor")
        name = data[names_at + noff:names_at + noff + nlen]
        expected = (cursor, cursor + states * page_width, cursor + states * page_width + (states + 1) * 4,
                    cursor + states * page_width + (states + 1) * 4 + accepts * 4)
        if (masks_at, offs_at, acc_at, trans_at) != expected:
            raise ValueError(f"SCM1 block offsets for {name!r}")
        cursor = trans_at + 8 * transitions
        offsets = struct.unpack_from(f"<{states + 1}I", data, offs_at)
        if offsets[0] != 0 or offsets[-1] != transitions or list(offsets) != sorted(offsets):
            raise ValueError("CSR offsets")
        accepting = struct.unpack_from(f"<{accepts}I", data, acc_at)
        if len(set(accepting)) != accepts or any(a >= states for a in accepting):
            raise ValueError("accepting ids")
        flat = struct.unpack_from(f"<{2 * transitions}I", data, trans_at)
        rows = {}
        for s in range(states):
            row = {}
            page = bytearray(page_width)
            for e in range(offsets[s], offsets[s + 1]):
                tok, tgt = flat[2 * e], flat[2 * e + 1]
                if tok >= vocab or tgt >= states or tok in row:
                    raise ValueError("CSR entry")
                row[tok] = tgt
                page[tok >> 3] |= 1 << (tok & 7)
            if bytes(page) != data[masks_at + s * page_width:masks_at + (s + 1) * page_width]:
                raise ValueError(f"mask page {s} disagrees with its transitions")
            rows[s] = row
        schemas[name] = {"states": states, "transitions": transitions, "accepting": frozenset(accepting),
                         "rows": rows}
    if cursor != len(data):
        raise ValueError("SCM1 trailing or missing bytes")
    return {"vocab": vocab, "schemas": schemas}


def fnv1a64(raw: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in raw:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


# FNV-1a 64 of these eight bytes is zero (draft-eastlake-fnv's zero-hash input; checked by a reference cell).
ZERO_HASH_NAME = bytes([0xD5, 0x6B, 0xB9, 0x53, 0x42, 0x87, 0x08, 0x36])
