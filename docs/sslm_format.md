# The `.sslm` artifact format — version 1

A `.sslm` file is a converted, quantized SuperSLM model: a versioned header, a
section table, and a sequence of aligned sections. It is the boundary between the
offline converter (build-time Python, `SuperSLM_Plan.md` §11) and the runtime C++
loader. This document is the normative spec; `include/superslm/artifact.h` is the
machine-readable contract and must agree with it byte-for-byte.

The format is designed to be **memory-mappable** (every section is aligned so it
can be consumed in place) and **integrity-checked** (a hash over the whole file
detects any corruption or truncation). The loader is a **trust boundary**
(§17 dim 2): it treats the entire file as hostile input and validates every field
against declared bounds before any byte of a section is read.

## Load-bearing choices (called out for review)

These are the decisions a builder must not make silently. Each is fixed for v1:

1. **Little-endian, fixed.** Every multi-byte integer is stored little-endian,
   independent of host. x86 and ARM are both little-endian; fixing it makes the
   artifact bytes host-independent, which the §11 reproducibility formula requires.
   The loader reads each field with explicit little-endian byte assembly — it never
   `reinterpret_cast`s a struct over untrusted bytes (that would import host padding
   and endianness into a security boundary).
2. **Integrity hash = SHA-256 over the whole file with the hash field zeroed.**
   The 32 hash bytes in the header are treated as zero while hashing, then compared
   to the stored value. This covers the header, the section table, and every
   section — one check proves the whole artifact is intact. It is also the artifact's
   identity (the spike's `source_fingerprint`, `artifact_cache.py`).
3. **Reject over degrade (§11, D-SLM-lineage).** Any deviation — bad magic,
   unsupported version, an out-of-bounds/overlapping/misaligned section, an unknown
   section type, a size that disagrees with its dtype, a hash mismatch — is a
   **rejection with a versioned diagnostic**, never a silent partial load.
4. **Section alignment ≥ 8, a power of two, ≤ 4096.** Declared per section and
   validated; the section offset must be a multiple of it. This is what makes the
   sections mmap-consumable in place.
5. **Unknown section type is rejected in v1.** Forward-compatible section-skipping
   (optional sections a newer reader may ignore) is deferred to a later version with
   an explicit rule; v1's known set is closed and anything outside it is a rejection.

## Byte layout

All integers little-endian. Offsets are absolute from the start of the file.

### Header — 64 bytes, at offset 0

| Offset | Type       | Field              | Constraint (v1)                             |
|-------:|------------|--------------------|---------------------------------------------|
|      0 | `u8[4]`    | `magic`            | `'S','S','L','M'` (0x53 0x53 0x4C 0x4D)     |
|      4 | `u32`      | `format_version`   | `== 1`                                      |
|      8 | `u32`      | `header_bytes`     | `== 64`                                     |
|     12 | `u32`      | `section_count`    | `<= 4096`                                   |
|     16 | `u32`      | `flags`            | `== 0` (reserved)                           |
|     20 | `u32`      | `reserved0`        | `== 0`                                      |
|     24 | `u64`      | `file_bytes`       | `== actual file size`                       |
|     32 | `u8[32]`   | `integrity_sha256` | SHA-256 of the file, these 32 bytes zeroed  |

### Section table — `section_count` × 40 bytes, at offset 64

| Offset | Type  | Field        | Constraint (v1)                                              |
|-------:|-------|--------------|-------------------------------------------------------------|
|      0 | `u32` | `type`       | a known `SslmSectionType`; no duplicates                    |
|      4 | `u32` | `dtype`      | a known `SslmDtype`                                          |
|      8 | `u64` | `offset`     | `>= 64 + section_count*40`; `% alignment == 0`; in bounds   |
|     16 | `u64` | `byte_size`  | `offset + byte_size <= file_bytes` (no overflow)            |
|     24 | `u64` | `elem_count` | `byte_size == elem_count * dtype_size(dtype)`               |
|     32 | `u32` | `alignment`  | power of two, `8 <= alignment <= 4096`                      |
|     36 | `u32` | `reserved`   | `== 0`                                                       |

Sections must not overlap one another. Their order in the table is not constrained;
their byte ranges are.

### Section data

Each section's bytes live at `[offset, offset + byte_size)`, aligned as declared.

## Section types (`SslmSectionType`)

| Value | Name                    | dtype        | Slot | Notes                                              |
|------:|-------------------------|--------------|------|----------------------------------------------------|
|     0 | `Config`                | `Json`       | S0   | model config, read from `config.json` (§11); **required** |
|     1 | `Provenance`            | `Json`       | S0   | checkpoint name, license id, source hash (§11)     |
|     2 | `Weights`               | `Int8`       | S0   | quantized weight blocks                            |
|     3 | `Biases`                | `Int32`      | S0   | quantized biases                                   |
|     4 | `RopeTables`            | `Int64`      | S0   | RoPE tables (§6.4)                                 |
|     5 | `Scales`                | `Json`       | S0   | `StaticScales` (requant / rescale / nonlinear)     |
|     6 | `WeightScales`          | `Json`       | S0   | per-block weight scales                            |
|     7 | `CompositionConstants`  | `Json`       | S0   | pinned composition constants (§6.8)                |
|     8 | `KvLandingScales`       | `Json`       | S0   | per-head KV landing scales (C27)                   |
|     9 | `KvLandingReciprocals`  | `Json`       | S0   | per-head KV landing reciprocals (C27)              |
|    10 | `Calibration`           | `Json`       | S0   | calibration record                                 |
|    11 | `GoldenHashes`          | `Json`       | S0   | reference-pack hashes (§11 golden pack)            |
|    20 | `Tokenizer`             | `Raw`        | S1   | reserved — introduced at S1                        |
|    21 | `ChatTemplate`          | `Json`       | S1   | reserved — role markers / special-token ids (F-W3) |
|    30 | `SchemaMasks`           | `Raw`        | S5   | reserved — compiled DFA mask pages                 |

The S1/S5 types are **reserved, not yet emitted**: a v1 artifact that carries them
is still parsed structurally, but the loader does not yet interpret their contents.
Types outside this table are rejected.

## Dtypes (`SslmDtype`)

| Value | Name    | Element size |
|------:|---------|-------------:|
|     0 | `Raw`   |            1 |
|     1 | `Int8`  |            1 |
|     2 | `Int32` |            4 |
|     3 | `Int64` |            8 |
|     4 | `Uint8` |            1 |
|     5 | `Float32` |          4 |
|     6 | `Float64` |          8 |

`Json` sections use dtype `Raw` (UTF-8 bytes).

## Versioning

`format_version` is a single integer. A reader that does not recognize the version
rejects with a versioned diagnostic (`UnsupportedVersion`) — it never attempts a
best-effort load. `ARTIFACT_FORMAT_VERSION` is currently **1** and matches the spike
(`artifact_cache.py`). A change to any field layout, a new required section, or a
change to the integrity-hash construction bumps this number.
