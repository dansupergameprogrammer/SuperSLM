"""Python writers for the `.sslm` model sub-formats — the emit side of the runtime
C++ parsers (include/superslm/model.h, docs/sslm_format.md "Model sub-formats").

Byte-for-byte mirrors of `SslmTensorManifest::Parse` (WGT1/BIA1/ROP1/WSC1),
`SslmKeyedConstants::Parse` (KVC1), and `ParseConfig` (CFG1). A blob produced here must
parse `Ok` in the C++ ModelView and read back the exact values written. Little-endian,
explicit `struct` packing. Build-time tooling — Python ships nothing (SuperSLM_Plan.md §11).
"""

import struct
import numpy as np

# --- magics + geometry (mirror include/superslm/model.h) ---
WGT1 = b"WGT1"
BIA1 = b"BIA1"
ROP1 = b"ROP1"
WSC1 = b"WSC1"
KVC1 = b"KVC1"
CFG1 = b"CFG1"
SIL1 = b"SIL1"
# T-2046 (design §24.2 D-SLM3159): B0b's own two magics (T-2021/T-2029), written via the SAME
# write_tensor_manifest below, unmodified -- DFS1/UFS1 share WGT1/BIA1/ROP1/WSC1's own manifest
# byte layout exactly (design's own "storage-shape-identical" framing).
DFS1 = b"DFS1"
UFS1 = b"UFS1"
MANIFEST_VERSION = 1

# Sigmoid-LUT geometry (mirror include/superslm/model.h + silu_lut.h). Pinned, not per-artifact.
SIGMOID_LUT_NODES = 1024        # N, x in [-X, X]
SIGMOID_LUT_ENTRIES = SIGMOID_LUT_NODES + 1  # N+1 = 1025
SIGMOID_LUT_X = 16.0            # domain half-width X
SIGMOID_FRAC_BITS = 15          # Q15
MAX_TENSOR_RANK = 4
KV_PRECISION_INT8 = 0
KV_PRECISION_INT16 = 1

# numpy dtype -> element size, for the tensor-manifest data payload.
_ELEM_SIZE = {np.int8: 1, np.int32: 4, np.int64: 8}

# S-HARDEN-3 (F13): explicit little-endian dtype objects, keyed by the same
# caller-facing dtype identity as _ELEM_SIZE. `np.int8`/`np.int32`/`np.int64`
# are the platform's NATIVE dtype -- on every platform this project's own CI
# matrix runs (x64, ARM64), native already IS little-endian, so casting to
# the bare native dtype and calling `.tobytes()` happens to emit correct
# bytes here today. It is not a portability guarantee: a big-endian build
# host would cast to ITS native representation and emit big-endian bytes,
# silently producing an artifact the (always-little-endian) C++ reader
# misreads with no diagnostic anywhere (the finding's own words: "the writer
# additionally relies on host-native NumPy byte order while the format
# documents explicit little-endian"). Casting to the EXPLICIT little-endian
# dtype below removes the platform dependency outright rather than relying on
# every future build host happening to be little-endian.
_LE_DTYPE = {np.int8: np.dtype("<i1"), np.int32: np.dtype("<i4"), np.int64: np.dtype("<i8")}


def _align_up(v, a):
    return v if a == 0 else (v + (a - 1)) // a * a


def write_tensor_manifest(magic, element_np_dtype, tensors):
    """A WGT1/BIA1/ROP1/WSC1 blob: `tensors` is an ordered dict name -> np.ndarray, all
    of `element_np_dtype` (int8/int32/int64). Mirrors SslmTensorManifest::Parse and the
    test builder BuildManifest: 16-byte header, 48-byte descriptors, name blob, then each
    tensor's data packed contiguously from the element-aligned data region.
    """
    element_size = _ELEM_SIZE[element_np_dtype]
    le_dtype = _LE_DTYPE[element_np_dtype]
    names = list(tensors.keys())
    arrs = [np.ascontiguousarray(tensors[n], dtype=le_dtype) for n in names]
    for a in arrs:
        if a.ndim < 1 or a.ndim > MAX_TENSOR_RANK:
            raise ValueError(f"tensor rank {a.ndim} outside [1, {MAX_TENSOR_RANK}]")

    name_blob = b"".join(n.encode("utf-8") for n in names)
    name_offs, off = [], 0
    for n in names:
        name_offs.append(off)
        off += len(n.encode("utf-8"))

    header_and_desc = 16 + len(names) * 48
    data_region = _align_up(header_and_desc + len(name_blob), element_size)

    data_offs, cursor = [], data_region
    for a in arrs:
        data_offs.append(cursor)
        cursor += a.size * element_size

    buf = bytearray()
    buf += magic
    buf += struct.pack("<III", MANIFEST_VERSION, len(names), len(name_blob))
    for i, (n, a) in enumerate(zip(names, arrs)):
        shape = list(a.shape) + [0] * (MAX_TENSOR_RANK - a.ndim)
        buf += struct.pack("<III", name_offs[i], len(n.encode("utf-8")), a.ndim)
        buf += struct.pack("<IIII", *shape)
        buf += struct.pack("<QQI", data_offs[i], int(a.size), 0)
    buf += name_blob
    buf += b"\x00" * (data_region - len(buf))  # pad to the aligned data region
    for a in arrs:
        buf += a.tobytes()  # little-endian on x86/ARM; np default matches the C++ LE read
    return bytes(buf)


def write_kvc1(value_words, entries):
    """A KVC1 blob: `entries` is an ordered dict name -> tuple of `value_words` ints
    (each stored int64). Mirrors SslmKeyedConstants::Parse and BuildKvc1: 24-byte header,
    8-byte descriptors, the int64 value array (row-major), then the name blob.
    """
    if value_words not in (2, 3):
        raise ValueError("value_words must be 2 or 3")
    names = list(entries.keys())
    for n in names:
        if len(entries[n]) != value_words:
            raise ValueError(f"entry {n!r} has {len(entries[n])} values, expected {value_words}")

    name_blob = b"".join(n.encode("utf-8") for n in names)
    buf = bytearray()
    buf += KVC1
    buf += struct.pack("<IIIII", MANIFEST_VERSION, len(names), value_words, len(name_blob), 0)
    off = 0
    for n in names:
        nb = n.encode("utf-8")
        buf += struct.pack("<II", off, len(nb))
        off += len(nb)
    for n in names:
        for v in entries[n]:
            buf += struct.pack("<q", int(v))  # signed int64, two's-complement LE
    buf += name_blob
    return bytes(buf)


def write_cfg1(*, hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads,
               head_dim, intermediate_size, vocab_size, context_cap, tie_word_embeddings,
               kv_precision, kv_block_size, unicode_major, unicode_minor, unicode_patch,
               rope_theta, rms_norm_eps):
    """The fixed 84-byte CFG1 config struct. Mirrors ParseConfig / docs "Config blob"."""
    buf = bytearray()
    buf += CFG1
    buf += struct.pack(
        "<IIIIIIIIIIIIIIII",  # version + 15 u32 fields
        MANIFEST_VERSION,
        hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads,
        head_dim, intermediate_size, vocab_size, context_cap,
        1 if tie_word_embeddings else 0, int(kv_precision), kv_block_size,
        unicode_major, unicode_minor, unicode_patch, 0,  # reserved
    )
    buf += struct.pack("<dd", float(rope_theta), float(rms_norm_eps))
    assert len(buf) == 84, f"CFG1 must be 84 bytes, got {len(buf)}"
    return bytes(buf)


def build_sigmoid_lut(nodes=SIGMOID_LUT_NODES, x_half=SIGMOID_LUT_X):
    """The offline Q15 sigmoid table (double precision, round-half-to-even) — the one place
    double runs (SuperSLM_S2.4_SiLU_LUT_Design §4). Entry i, for i in [0, N], holds
    round_half_even(sigmoid(-X + i*(2X/N)) * 2^15) as int32. Round-half-to-even is the pinned
    tie rule for this offline-derived table (C14 precedent, §6.8); it is NOT C3 — C3 governs the
    runtime rounding (§5/§6). Returns an int32 numpy array of N+1 entries. int32, not int16:
    sigmoid(X)*2^15 rounds to 32768, which exceeds INT16_MAX (§8).
    """
    i = np.arange(nodes + 1, dtype=np.float64)
    x = -x_half + i * (2.0 * x_half / nodes)
    sig = 1.0 / (1.0 + np.exp(-x))              # float64 sigmoid
    entries = np.rint(sig * (1 << SIGMOID_FRAC_BITS))  # np.rint = round-half-to-even
    return entries.astype(np.int32)


def write_sil1(table=None):
    """A SIL1 sigmoid-LUT blob: a fixed 16-byte header (magic + version + entry_count +
    reserved) then N+1 little-endian int32 Q15 nodes. Mirrors ParseSigmoidLut / docs
    "Sigmoid-LUT blob — SIL1". `table` defaults to build_sigmoid_lut().
    """
    if table is None:
        table = build_sigmoid_lut()
    table = np.ascontiguousarray(table, dtype=_LE_DTYPE[np.int32])
    if table.size != SIGMOID_LUT_ENTRIES:
        raise ValueError(f"SIL1 table must have {SIGMOID_LUT_ENTRIES} entries, got {table.size}")
    buf = bytearray()
    buf += SIL1
    buf += struct.pack("<III", MANIFEST_VERSION, int(table.size), 0)  # version, entry_count, reserved
    buf += table.tobytes()  # little-endian int32 on x86/ARM; matches the C++ LE read
    assert len(buf) == 16 + SIGMOID_LUT_ENTRIES * 4, f"SIL1 size {len(buf)} != expected"
    return bytes(buf)
