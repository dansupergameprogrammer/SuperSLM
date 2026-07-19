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

ARTIFACT_FORMAT_VERSION = 1
HEADER_BYTES = 64
SECTION_DESC_BYTES = 40
MAX_SECTIONS = 4096
INTEGRITY_HASH_OFFSET = 32
INTEGRITY_HASH_BYTES = 32
MAGIC = b"SSLM"


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
    TOKENIZER = 20      # S1: byte-BPE vocab + merges + special tokens (self-contained blob)
    CHAT_TEMPLATE = 21  # S1: chat template + special-token metadata (JSON)
    UNICODE_TABLES = 22 # S1: pinned NFC + property-class tables (self-contained blob)
    SCHEMA_MASKS = 30


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
    SectionType.BIASES: Dtype.INT32,
    SectionType.ROPE_TABLES: Dtype.INT64,
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


def build_artifact(sections):
    """Serialize `sections` into `.sslm` bytes. Sections are placed in table order,
    each aligned to its own alignment, immediately after the section table."""
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
    struct.pack_into("<I", buf, 16, 0)   # flags
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


def write_artifact(path, sections):
    data, fingerprint = build_artifact(sections)
    with open(path, "wb") as f:
        f.write(data)
    return fingerprint
