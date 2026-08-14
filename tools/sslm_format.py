"""Write `.sslm` v1 artifacts from Python (build-time converter tooling).

Byte-for-byte mirror of the runtime C++ loader's format (docs/sslm_format.md,
include/superslm/artifact.h): little-endian, 64-byte header, 40-byte section
descriptors, aligned sections, SHA-256 integrity over the whole file with the hash
field zeroed. A file produced here must load `Ok` in the C++ loader and reproduce the
same fingerprint. Python ships nothing — this is offline tooling (SuperSLM_Plan.md §11).
"""

import hashlib
import struct
from dataclasses import dataclass, field

# v2 (S2.4): adds the required SigmoidLut (SIL1) section — no i-exp-sigmoid fallback once
# C10 is the LUT (D-SLM68), so v1 and v2 artifacts are mutually rejected by the version check.
ARTIFACT_FORMAT_VERSION = 2
HEADER_BYTES = 64
SECTION_DESC_BYTES = 40
MAX_SECTIONS = 4096
INTEGRITY_HASH_OFFSET = 32
INTEGRITY_HASH_BYTES = 32
MAGIC = b"SSLM"

# T-1894 (T-1822 design Sec31.2.1, round 4/D-SLM2423): the header `flags`
# field's own first reserved bit -- byte-for-byte mirror of
# `superslm::kOptionGFusedKLandingFlag` (include/superslm/artifact.h).
# Selects Option-G fused post-RoPE K landing; every other bit stays reserved
# (the C++ loader rejects any unknown bit, `artifact.cpp`'s
# `kKnownArtifactFlagsMask`).
OPTION_G_FUSED_K_LANDING_FLAG = 0x1


class SectionType:
    CONFIG = 0
    PROVENANCE = 1
    WEIGHTS = 2
    BIASES = 3
    ROPE_TABLES = 4
    SCALES = 5
    WEIGHT_SCALES = 6
    COMPOSITION_CONSTANTS = 7
    KV_LANDING_SCALES = 8
    KV_LANDING_RECIPROCALS = 9
    CALIBRATION = 10
    GOLDEN_HASHES = 11
    SIGMOID_LUT = 12    # S2: fixed-point SiLU sigmoid LUT (SIL1 fixed table); required from v2
    TOKENIZER = 20      # S1: byte-BPE vocab + merges + special tokens (self-contained blob)
    CHAT_TEMPLATE = 21  # S1: chat template + special-token metadata (JSON)
    UNICODE_TABLES = 22 # S1: pinned NFC + property-class tables (self-contained blob)
    SCHEMA_MASKS = 30
    # T-2046 (design §24.2, D-SLM3159): mirrors include/superslm/artifact.h's SslmSectionType --
    # B0b's own section types (T-2021/T-2029), never referenced from the Python writer side until
    # this fold's adapter-artifact writer needed to emit them.
    DELTA_FOLD_SCALES = 40
    U_FOLD_SCALES = 41


class Dtype:
    RAW = 0
    INT8 = 1
    INT32 = 2
    INT64 = 3
    UINT8 = 4
    FLOAT32 = 5
    FLOAT64 = 6


# The dtype every section type must carry (mirrors ExpectedDtype in the C++ loader;
# the loader rejects a mismatch with SectionDtypeMismatch).
EXPECTED_DTYPE = {
    SectionType.WEIGHTS: Dtype.INT8,
    SectionType.BIASES: Dtype.INT64,  # BIA1: C28 dynamic-bias codes reach ~10^14 at q_b=30
    SectionType.WEIGHT_SCALES: Dtype.INT32,  # WSC1 tensor manifest of (identity,mult,shift) fold ops
    SectionType.SIGMOID_LUT: Dtype.INT32,    # SIL1 fixed table of int32 Q15 nodes (int16 unsafe: 32768 > INT16_MAX)
    SectionType.ROPE_TABLES: Dtype.INT64,
    # T-2046 (design §24.2): mirrors src/artifact.cpp's ExpectedDtype -- both DFS1/UFS1 manifests
    # of (identity,mult,exponent) triples are Int32, exactly like WSC1's own fold-op manifest.
    SectionType.DELTA_FOLD_SCALES: Dtype.INT32,
    SectionType.U_FOLD_SCALES: Dtype.INT32,
}  # everything else -> RAW


def expected_dtype(section_type):
    return EXPECTED_DTYPE.get(section_type, Dtype.RAW)


@dataclass
class Section:
    type: int
    data: bytes
    dtype: int = None            # defaults to the type's required dtype
    elem_count: int = None       # defaults to len(data) // dtype_size
    alignment: int = 64

    def __post_init__(self):
        if self.dtype is None:
            self.dtype = expected_dtype(self.type)


_DTYPE_SIZE = {Dtype.RAW: 1, Dtype.INT8: 1, Dtype.INT32: 4, Dtype.INT64: 8,
               Dtype.UINT8: 1, Dtype.FLOAT32: 4, Dtype.FLOAT64: 8}


def _align_up(v, a):
    return v if a == 0 else (v + (a - 1)) // a * a


def build_artifact(sections, flags=0):
    """Serialize `sections` into `.sslm` bytes. Sections are placed in table order,
    each aligned to its own alignment, immediately after the section table.

    T-1894 (design Sec31.2.1, round 4/D-SLM2423): `flags` defaults to 0,
    preserving every existing caller's own output byte-for-byte. Pass
    `OPTION_G_FUSED_K_LANDING_FLAG` to produce an artifact whose header asks
    for Option-G's fused K-landing order (the same field
    `SslmArtifact::OptionGFusedKLandingEnabled()` reads at load time)."""
    if len(sections) > MAX_SECTIONS:
        raise ValueError(f"{len(sections)} sections > MAX_SECTIONS {MAX_SECTIONS}")

    table_end = HEADER_BYTES + len(sections) * SECTION_DESC_BYTES
    placed = []
    cursor = table_end
    for s in sections:
        dsize = _DTYPE_SIZE[s.dtype]
        if len(s.data) % dsize != 0:
            raise ValueError(f"section type {s.type}: {len(s.data)} bytes not a multiple of dtype size {dsize}")
        elem_count = s.elem_count if s.elem_count is not None else len(s.data) // dsize
        if elem_count * dsize != len(s.data):
            raise ValueError(f"section type {s.type}: elem_count {elem_count} * {dsize} != {len(s.data)}")
        off = _align_up(cursor, s.alignment)
        placed.append((s, off, elem_count))
        cursor = off + len(s.data)

    file_bytes = cursor
    buf = bytearray(file_bytes)

    buf[0:4] = MAGIC
    struct.pack_into("<I", buf, 4, ARTIFACT_FORMAT_VERSION)
    struct.pack_into("<I", buf, 8, HEADER_BYTES)
    struct.pack_into("<I", buf, 12, len(sections))
    struct.pack_into("<I", buf, 16, flags)   # flags
    struct.pack_into("<I", buf, 20, 0)   # reserved0
    struct.pack_into("<Q", buf, 24, file_bytes)
    # integrity hash [32:64] left zero for now

    for i, (s, off, elem_count) in enumerate(placed):
        row = HEADER_BYTES + i * SECTION_DESC_BYTES
        struct.pack_into("<I", buf, row + 0, s.type)
        struct.pack_into("<I", buf, row + 4, s.dtype)
        struct.pack_into("<Q", buf, row + 8, off)
        struct.pack_into("<Q", buf, row + 16, len(s.data))
        struct.pack_into("<Q", buf, row + 24, elem_count)
        struct.pack_into("<I", buf, row + 32, s.alignment)
        struct.pack_into("<I", buf, row + 36, 0)  # reserved
        buf[off:off + len(s.data)] = s.data

    # Integrity: SHA-256 over the whole file with the 32 hash bytes treated as zero.
    digest = hashlib.sha256(buf).digest()  # buf's hash field is currently zero
    buf[INTEGRITY_HASH_OFFSET:INTEGRITY_HASH_OFFSET + INTEGRITY_HASH_BYTES] = digest
    return bytes(buf), digest.hex()


def write_artifact(path, sections, flags=0):
    data, fingerprint = build_artifact(sections, flags=flags)
    with open(path, "wb") as f:
        f.write(data)
    return fingerprint


# T-2046 (design §24.2 D-SLM3161/D-SLM3158): a minimal reader, the writer's own mirror image --
# `build_artifact`'s own byte layout, read back directly, without needing the C++ loader from
# Python. Used by the adapter artifact writer to copy the bound base artifact's own CFG1 bytes
# byte-for-byte and to read its RawIntegrityHash for ADP1's own base_artifact_hash field.

def read_header(path):
    """The header + section table of a v2 `.sslm` at `path`, plus the raw file bytes."""
    with open(path, "rb") as f:
        data = f.read()
    if data[0:4] != MAGIC:
        raise ValueError(f"{path}: not an .sslm file (bad magic)")
    version, header_bytes, section_count, _flags, _reserved0 = struct.unpack_from("<IIIII", data, 4)
    file_bytes, = struct.unpack_from("<Q", data, 24)
    integrity_hash = bytes(data[INTEGRITY_HASH_OFFSET:INTEGRITY_HASH_OFFSET + INTEGRITY_HASH_BYTES])
    sections = []
    for i in range(section_count):
        row = header_bytes + i * SECTION_DESC_BYTES
        s_type, s_dtype = struct.unpack_from("<II", data, row)
        off, byte_size, elem_count = struct.unpack_from("<QQQ", data, row + 8)
        alignment, _reserved = struct.unpack_from("<II", data, row + 32)
        sections.append({"type": s_type, "dtype": s_dtype, "offset": off,
                         "byte_size": byte_size, "elem_count": elem_count, "alignment": alignment})
    return {"version": version, "file_bytes": file_bytes, "integrity_hash": integrity_hash,
           "sections": sections, "data": data}


def read_section_bytes(path, section_type):
    """The raw bytes of the first section of `section_type` in the `.sslm` at `path`, or `None`
    if absent."""
    h = read_header(path)
    for s in h["sections"]:
        if s["type"] == section_type:
            off, size = s["offset"], s["byte_size"]
            return bytes(h["data"][off:off + size])
    return None


def raw_integrity_hash(path):
    """The artifact's own stored 32-byte integrity hash -- literally the header bytes at
    `[INTEGRITY_HASH_OFFSET : +INTEGRITY_HASH_BYTES]` (`build_artifact`'s own write target),
    the identical value `SslmArtifact::RawIntegrityHash()` exposes -- not a recomputation."""
    with open(path, "rb") as f:
        f.seek(INTEGRITY_HASH_OFFSET)
        return f.read(INTEGRITY_HASH_BYTES)
