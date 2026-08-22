# The `.sslm` artifact format — version 2

A `.sslm` file is a converted, quantized SuperSLM model: a versioned header, a
section table, and a sequence of aligned sections. It is the boundary between the
offline converter (build-time Python) and the runtime C++ loader. This document is
the normative spec; `include/superslm/artifact.h` is the machine-readable contract
and must agree with it byte-for-byte.

The format is designed to be **memory-mappable** (every section is aligned so it
can be consumed in place) and **integrity-checked** (a hash over the whole file
detects any corruption or truncation). The loader is a **trust boundary**: it
treats the entire file as hostile input and validates every field against
declared bounds before any byte of a section is read.

## Load-bearing choices (called out for review)

These are the decisions a builder must not make silently. Each is fixed for v1:

1. **Little-endian, fixed.** Every multi-byte integer is stored little-endian,
   independent of host. x86 and ARM are both little-endian; fixing it makes the
   artifact bytes host-independent, which reproducibility across machines requires.
   The loader reads each field with explicit little-endian byte assembly — it never
   `reinterpret_cast`s a struct over untrusted bytes (that would import host padding
   and endianness into a security boundary).
2. **Integrity hash = SHA-256 over the whole file with the hash field zeroed.**
   The 32 hash bytes in the header are treated as zero while hashing, then compared
   to the stored value. This covers the header, the section table, and every
   section — one check proves the whole artifact is intact. It also serves as the
   artifact's content-addressed identity.
3. **Reject over degrade.** Any deviation — bad magic,
   unsupported version, an out-of-bounds/overlapping/misaligned section, an unknown
   section type, a size that disagrees with its dtype, a hash mismatch — is a
   **rejection with a versioned diagnostic**, never a silent partial load.
4. **Section alignment ≥ 8, a power of two, ≤ 4096.** Declared per section and
   validated; the section offset must be a multiple of it. This is what makes the
   sections mmap-consumable in place.
5. **Unknown section type is rejected in v1.** Forward-compatible section-skipping
   (optional sections a newer reader may ignore) is deferred to a later version with
   an explicit rule; v1's known set is closed and anything outside it is a rejection.
6. **A section's dtype is fixed by its type and is validated.** Each known section
   type carries exactly one dtype (the "dtype" column of the section-types table
   below is normative, not documentation). A section whose type is known but whose
   dtype is not the one that type requires is rejected (`SectionDtypeMismatch`) — the
   reject-over-degrade rule above, applied to the descriptor, so a consumer that reads
   a section by type can trust its element width without re-checking.

## Byte layout

All integers little-endian. Offsets are absolute from the start of the file.

### Header — 64 bytes, at offset 0

| Offset | Type       | Field              | Constraint (v2)                             |
|-------:|------------|--------------------|---------------------------------------------|
|      0 | `u8[4]`    | `magic`            | `'S','S','L','M'` (0x53 0x53 0x4C 0x4D)     |
|      4 | `u32`      | `format_version`   | `== 2`                                      |
|      8 | `u32`      | `header_bytes`     | `== 64`                                     |
|     12 | `u32`      | `section_count`    | `<= 4096`                                   |
|     16 | `u32`      | `flags`            | `& ~0x1 == 0` (bit 0: Option-G fused K-landing; every other bit reserved) |
|     20 | `u32`      | `reserved0`        | `== 0`                                      |
|     24 | `u64`      | `file_bytes`       | `== actual file size`                       |
|     32 | `u8[32]`   | `integrity_sha256` | SHA-256 of the file, these 32 bytes zeroed  |

### Section table — `section_count` × 40 bytes, at offset 64

| Offset | Type  | Field        | Constraint (v2)                                              |
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
|     0 | `Config`                | `Raw`        | S0   | model config — a fixed `CFG1` binary struct (§ "Config blob"); **required** |
|     1 | `Provenance`            | `Json`       | S0   | checkpoint name, license id, source hash           |
|     2 | `Weights`               | `Int8`       | S0   | quantized weight blocks                            |
|     3 | `Biases`                | `Int64`      | S0   | dynamic-bias codes — a `BIA1` int64 tensor manifest (codes reach ~10¹⁴ at the widest observed shift) |
|     4 | `RopeTables`            | `Int64`      | S0   | RoPE tables                                        |
|     5 | `Scales`                | `Json`       | S0   | `StaticScales` (requant / rescale / nonlinear)     |
|     6 | `WeightScales`          | `Int32`      | S0   | per-channel weight-scale fold ops — a `WSC1` tensor manifest (§ "Weight-scale fold blob") |
|     7 | `CompositionConstants`  | `Raw`        | S0   | pinned composition constants — a `KVC1` keyed blob |
|     8 | `KvLandingScales`       | `Raw`        | S0   | per-head KV landing scales — a `KVC1` keyed blob    |
|     9 | `KvLandingReciprocals`  | `Raw`        | S0   | per-head KV landing reciprocals — a `KVC1` keyed blob |
|    10 | `Calibration`           | `Json`       | S0   | calibration record                                 |
|    11 | `GoldenHashes`          | `Json`       | S0   | reference-pack hashes (the golden pack)            |
|    12 | `SigmoidLut`            | `Int32`      | S2   | fixed-point SiLU sigmoid LUT — a `SIL1` fixed table (§ "Sigmoid-LUT blob"); **required from v2** |
|    20 | `Tokenizer`             | `Raw`        | S1   | byte-BPE vocab + merges + special tokens (blob)    |
|    21 | `ChatTemplate`          | `Json`       | S1   | chat template + special-token metadata             |
|    22 | `UnicodeTables`         | `Raw`        | S1   | pinned NFC + `\p{L}`/`\p{N}`/`\s` tables (blob)     |
|    30 | `SchemaMasks`           | `Raw`        | S5   | compiled DFA mask pages — an `SCM1` keyed-by-name blob (§ "SchemaMasks sub-format"); **optional** |
|    31 | `CalibrationBand`       | `Raw`        | S3.7 | token-length calibration band — a `KVC1` keyed blob; **optional** |
|    40 | `DeltaFoldScales`       | `Int32`      | S-LoRA | adapter delta fold triples — a `DFS1` tensor manifest; adapter artifacts only |
|    41 | `UFoldScales`           | `Int32`      | S-LoRA | adapter intermediate fold triples — a `UFS1` tensor manifest; adapter artifacts only |
|    42 | `DampedGreedyConstants` | `Raw`        | 1.2 | opt-in decoder scale — a 12-byte `DGC1` payload (§ "Damped-greedy constants"); **optional** |

The tokenizer types (20–22) are emitted and interpreted at S1. `SchemaMasks` (30) is
**optional at the current container version** — a v2
artifact carrying zero compiled schemas omits the section entirely, and every loader
accepts that identically to before this type existed; one carrying it must satisfy the
`SCM1` sub-format's own structural and cross-check rejections (§ "SchemaMasks
sub-format", below) before the runtime uses it. `CalibrationBand` (31) is optional at the
current container version — a new section type, not a version bump or a CFG1 extension; its
absence loads exactly as it did before this type existed. Types outside this table are
rejected.

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

## Tokenizer sub-formats (S1)

The `Tokenizer` (20) and `UnicodeTables` (22) sections carry self-contained blobs with
their own internal layout, parsed by the runtime `TokenizerView` after the loader has
verified whole-file integrity (so the parse trusts the bytes; it only fails closed on a
bad marker or a length that overruns the section). Both are little-endian. The converter
that writes them is `tools/convert_tokenizer.py` / `tools/unicode_tables.py`.

### `Tokenizer` blob — `TOK1`

| Field | Type | Notes |
|---|---|---|
| magic | `u8[4]` | `'TOK1'` |
| version | `u32` | `1` |
| vocab_count | `u32` | number of ids in the id→bytes table |
| merge_count | `u32` | number of BPE merges |
| special_count | `u32` | number of special tokens |
| reserved | `u32` | `0` |
| byte_to_id | `u32[256]` | base byte b → its single-byte token id |
| vocab_offsets | `u32[vocab_count+1]` | offsets into vocab_blob; id i → `[off[i], off[i+1])` |
| vocab_blob_len | `u32` | |
| vocab_blob | `u8[vocab_blob_len]` | id → the raw UTF-8 bytes it decodes to |
| merges | `(u32 a, u32 b, u32 merged)[merge_count]` | **rank = array index**; merging ids `(a,b)` yields `merged` |
| special_ids | `u32[special_count]` | |
| special_offsets | `u32[special_count+1]` | offsets into special_blob |
| special_blob_len | `u32` | |
| special_blob | `u8[special_blob_len]` | special-token contents, **longest-content-first** (greedy match order) |

### `UnicodeTables` blob — `UNI1`

Pinned Unicode data (version recorded in the `Config` section) for NFC and the
`\p{L}`/`\p{N}`/`\s` classes. A range is an inclusive `(lo, hi)` codepoint pair.
**Each of the three range tables (`letter`, `number`, `space`) must be sorted by `lo`,
non-overlapping, and satisfy `lo <= hi` per range** — the loader rejects a
range table violating any of these at load time, so a producer must emit ranges in
ascending, non-overlapping order.

| Field | Type | Notes |
|---|---|---|
| magic + version | `u8[4]='UNI1'`, `u32=1` | |
| letter / number / space | each: `u32 count`, `(u32 lo, u32 hi)[count]` | `\p{L}` / `\p{N}` / `\s` ranges, sorted by `lo`, non-overlapping, `lo <= hi` |
| ccc | `u32 count`, `(u32 cp, u32 class)[count]` | nonzero canonical combining classes |
| decomp | `u32 count`, `u32 cp[count]`, `u32 off[count+1]`, `u32 seq_len`, `u32 seq[seq_len]` | full canonical decomposition (Hangul is algorithmic, not tabled) |
| compose | `u32 count`, `(u32 a, u32 b, u32 c)[count]` | primary composites `(a,b)→c` |

## Model sub-formats (S2)

The array-bearing model sections carry a self-contained binary **tensor manifest** with
their concatenated tensor data, parsed by the runtime `ModelView` **after** the loader
has verified whole-file integrity. Like the tokenizer blobs (`TOK1`/`UNI1`), the parse
then trusts the bytes are intact but still **fails closed** on a bad marker, an
out-of-bounds range, or a descriptor that disagrees with its declared shape — the same
hostile-input trust boundary the loader itself is.
All fields are little-endian and read by explicit byte assembly (never a struct cast over
untrusted bytes). The converter that writes them is `tools/convert_model.py`.

The three array sections share **one manifest layout**; only the magic and the element
type differ. The section's declared dtype (the section-types table) already fixes the
element width; the manifest names and shapes the tensors packed into the section's data.

| Section (type)         | Magic    | Element  |
|------------------------|----------|----------|
| `Weights` (2)          | `'WGT1'` | `int8`   |
| `Biases` (3)           | `'BIA1'` | `int64`  |
| `RopeTables` (4)       | `'ROP1'` | `int64`  |
| `WeightScales` (6)     | `'WSC1'` | `int32`  |

### Tensor-manifest blob — `WGT1` / `BIA1` / `ROP1`

All byte offsets below are **from the start of the section** (the magic byte).

| Field | Type | Notes |
|---|---|---|
| magic | `u8[4]` | `'WGT1'` \| `'BIA1'` \| `'ROP1'`, matching the section type |
| version | `u32` | `1` |
| tensor_count | `u32` | `<= kMaxTensors` (65536) |
| name_blob_len | `u32` | byte length of the name blob |
| descriptors | `TensorDesc[tensor_count]` | fixed **48 bytes** each (below) |
| name_blob | `u8[name_blob_len]` | UTF-8 tensor names, concatenated (not NUL-terminated) |
| data | `<element>[...]` | concatenated tensor data; each tensor at its `data_off` |

`TensorDesc` — 48 bytes, fixed:

| Offset | Type | Field | Constraint |
|-------:|------|-------|------------|
| 0 | `u32` | `name_off` | `name_off + name_len <= name_blob_len` |
| 4 | `u32` | `name_len` | `> 0`; the name is unique across the manifest |
| 8 | `u32` | `rank` | `1 <= rank <= kMaxTensorRank` (4) |
| 12 | `u32[4]` | `shape` | `shape[i] > 0` for `i < rank`; `shape[i] == 0` for `i >= rank` |
| 28 | `u64` | `data_off` | multiple of the element size; `>=` end of the name blob; in bounds |
| 36 | `u64` | `elem_count` | `== product(shape[0..rank))` (64-bit, overflow-checked) |
| 44 | `u32` | `reserved` | `== 0` |

The name blob begins at `16 + tensor_count*48`; the data region begins at
`16 + tensor_count*48 + name_blob_len`, rounded up to the element size. A tensor's bytes
span `[data_off, data_off + elem_count * element_size)`.

The `ModelView` sub-parse rejects (fails closed, never a partial view) on any of: a short
section (`byte_size < 16`); a wrong magic or version; `tensor_count > kMaxTensors`; the
fixed header + descriptor table + name blob exceeding `byte_size` (computed in 64-bit, to
avoid a 32-bit overflow on the product); for **any** descriptor — a name range outside
the name blob, a duplicate name, `rank` outside `[1,4]`, a `shape` entry nonzero at or past
`rank` or zero before it, an `elem_count` disagreeing with `product(shape)`, a `data_off`
not element-aligned or below the data region, a data range exceeding `byte_size` or
overflowing, a nonzero `reserved`; or **any two tensors whose data ranges overlap** (the
converter packs them contiguously in descriptor order; the loader validates non-overlap).

`kMaxTensors` and `kMaxTensorRank` are declared in `include/superslm/model.h`.

### Keyed numeric-constant blob — `KVC1`

The pinned integer composition constants live in three keyed sections, each
mapping a string key to a fixed-width tuple of integers. They share one self-contained binary
sub-format, `KVC1`, parsed by `ModelView` as a hostile-input trust boundary (the same bar as
`WGT1`). Only the **magic is shared** (`'KVC1'`); the section type fixes how many integers each
entry carries (`value_words`), and the blob restates it so the parse can reject a mismatch:

| Section (type)                | `value_words` | Tuple |
|-------------------------------|:-------------:|-------|
| `CompositionConstants` (7)    | 2 | `(m, e)` — canonical carried scale (C26) |
| `KvLandingScales` (8)         | 2 | `(m, e)` — per-head K/V landing target (C27) |
| `KvLandingReciprocals` (9)    | 3 | `(m, e, R)` — the landing composite's offline reciprocal (C27) |
| `CalibrationBand` (31)        | 2 | `(min, max)` — inclusive token-length calibration band, entry named `token_length` |

Every value is stored as a little-endian **`int64`**, including `e` (a small exponent that fits
trivially) — one uniform layout serves both the 2-word and 3-word sections. All offsets are from
the start of the section.

| Field | Type | Notes |
|---|---|---|
| magic | `u8[4]` | `'KVC1'` |
| version | `u32` | `1` |
| entry_count | `u32` | `<= kMaxConstantEntries` (1048576) |
| value_words | `u32` | `2` or `3`, and **must equal** the section type's required count |
| name_blob_len | `u32` | byte length of the name blob |
| reserved | `u32` | `== 0` |
| descriptors | `EntryDesc[entry_count]` | fixed **8 bytes** each: `name_off u32`, `name_len u32` |
| values | `int64[entry_count * value_words]` | entry `i`'s tuple at `[i*value_words, (i+1)*value_words)`, row-major |
| name_blob | `u8[name_blob_len]` | UTF-8 keys, concatenated (not NUL-terminated) |

The `ModelView` sub-parse rejects (fails closed) on: a short section (`byte_size < 24`, the fixed
header); a wrong magic or version; `entry_count > kMaxConstantEntries`; a `value_words` not in
`{2,3}` or not the value the section type requires; the header + descriptor table + value array +
name blob exceeding `byte_size` (computed in 64-bit — `entry_count * value_words * 8` is the
overflow-prone product and is guarded); for **any** entry — a name range outside the name blob, a
zero-length name, a duplicate name, or a nonzero-reserved header. The integer values themselves are
not range-checked here — the structural parse stays value-blind by design, so it does not couple
this format to any one consumer's kernel domain. The obligation belongs to the load-time
schema-value gate (`SslmModel::Load` + `ValidateSectionValues`): each
consuming primitive documents its domain (e.g. `CompositionConstants`' `(m, e)` against
`SiluSigmoidQ15`'s no-UB floor), and the gate declares that domain as data in a per-section
descriptor table, checked once per artifact open before any typed view is exposed. This is what the
now-built `SslmModel::Load` orchestrates in place of the machinery this section previously deferred
to (an unbuilt "C29" input-domain check with no host); the parse itself guarantees only that
`entry_count` tuples of `value_words` `int64`s and their keys are safely readable. `int64`s are read
by explicit little-endian byte assembly, so the value array needs no alignment.

`kMaxConstantEntries` is declared in `include/superslm/model.h`.

### Config blob — `CFG1`

The `Config` section (type 0, **required**) is a single fixed-layout binary struct — the model's
architecture, read once at load into typed fields. Fixed rather than keyed because the field set is
closed for v1, and a fixed struct needs no name table or bounds arithmetic. The runtime integer
fields drive the forward; `rope_theta`/`rms_norm_eps` are recorded for reproducibility (the RoPE
tables are precomputed offline from θ, and the integer RMSNorm carries no eps term — neither is read
by a kernel). All little-endian; the total is exactly **84 bytes**.

| Offset | Type | Field | Constraint |
|-------:|------|-------|------------|
| 0  | `u8[4]` | magic | `'CFG1'` |
| 4  | `u32` | version | `1` |
| 8  | `u32` | hidden_size | `> 0` |
| 12 | `u32` | num_hidden_layers | `> 0` |
| 16 | `u32` | num_attention_heads | `> 0` |
| 20 | `u32` | num_key_value_heads | `> 0` |
| 24 | `u32` | head_dim | `> 0` and even |
| 28 | `u32` | intermediate_size | `> 0` |
| 32 | `u32` | vocab_size | `> 0` |
| 36 | `u32` | context_cap | `> 0` |
| 40 | `u32` | tie_word_embeddings | `0` or `1` |
| 44 | `u32` | kv_precision | `0` (int8) or `1` (int16) |
| 48 | `u32` | kv_block_size | `> 0` |
| 52 | `u32` | unicode_major | (recorded; the tokenizer's pinned Unicode version) |
| 56 | `u32` | unicode_minor | |
| 60 | `u32` | unicode_patch | |
| 64 | `u32` | reserved | `== 0` |
| 68 | `f64` | rope_theta | recorded (offline RoPE-table input; not read at runtime) |
| 76 | `f64` | rms_norm_eps | recorded (the integer RMSNorm carries no eps term) |

`ModelView` config parse rejects on: `byte_size != 84`; a wrong magic or version; any of the eight
dimension fields **or `kv_block_size`** `== 0`; `tie_word_embeddings`/`kv_precision` outside their
allowed set; a nonzero `reserved`. This is the reject-over-degrade rule (above) applied to config — a zero dimension or a
defaulted field produces "a model that loads, runs, generates fluent text, and is not the source
model," so it is rejected, never repaired.

### Weight-scale fold blob — `WSC1` (reuses the tensor manifest)

`WeightScales` (type 6, dtype **`Int32`**) does not need a new sub-format: each projection's
per-output-channel C24/C25 fold ops are `(identity, mult, shift)` int32 triples, i.e. a named
`int32` tensor of shape `[num_channels, 3]`. So `WeightScales` is a **tensor manifest** exactly like
`Weights`/`Biases`/`RopeTables`, with its own magic `'WSC1'` and element type `int32`; it is parsed
by the same `SslmTensorManifest::Parse` (already certified, S2.0a). Column 0 of each row is the
identity flag (`0`/`1`), column 1 the `mult`, column 2 the `shift`. The attention-context per-head
fold (C27) rides the same section under `{prefix}.ctx_fold_head{h}` keys. No parse code is
added — only the magic and the `int32` element type, per the tensor-manifest rules above.

### Sigmoid-LUT blob — `SIL1`

The `SigmoidLut` section (type 12, dtype **`Int32`**, **required from v2**) carries the
fixed-point SiLU sigmoid lookup table (C10). Like `CFG1` it is a **fixed-layout**
section — the table geometry (`N` nodes over `x ∈ [-X, X]`, and the runtime sub-node index
resolution) is a pinned build-time constant of the construction, **not** carried per-artifact —
so the exact-size check gates every read. `N = 1024`, so the table has `N+1 = 1025` entries; the
payload is `1025` little-endian **signed `int32`** Q15 nodes (`int16` is unsafe: `sigmoid(X)·2¹⁵`
rounds to `32768`, one past `INT16_MAX`). Entry `i` is `round_half_even(sigmoid(-X + i·(2X/N))·2¹⁵)`,
generated offline in double precision by the converter (`tools/sslm_model_writer.py::build_sigmoid_lut`).
All little-endian; total is exactly **`16 + 1025·4 = 4116` bytes**.

| Offset | Type | Field | Constraint |
|-------:|------|-------|------------|
| 0  | `u8[4]` | magic | `'SIL1'` |
| 4  | `u32` | version | `1` |
| 8  | `u32` | entry_count | `== 1025` (`N+1`) |
| 12 | `u32` | reserved | `== 0` |
| 16 | `int32[1025]` | nodes | Q15 sigmoid values, little-endian |

`ModelView` SIL1 parse (`ParseSigmoidLut`) rejects (fails closed) on: `byte_size != 4116`
(`BadSigmoidLutSize`); a wrong magic (`BadSigmoidLutMagic`) or version (`UnsupportedSigmoidLutVersion`);
`entry_count != 1025` (`BadSigmoidLutCount`); a nonzero reserved field (`BadSigmoidLutReserved`) — the
reject-over-degrade rule above. Nodes are read with the byte-assembly reader (`SigmoidLutValue`), never a
cast (the payload is not guaranteed `int32`-aligned). The runtime lookup over the table lives in
`include/superslm/silu_lut.h` (index derivation + interpolation, division-free), not in the parse.

### SchemaMasks sub-format — `SCM1`

The `SchemaMasks` section (type 30, dtype `Raw`, **optional**) carries a **table of named,
independently-compiled schemas** — the
`sslm_schema_lookup`/`sslm_schema_count`/`sslm_schema_name` ABI is lookup-by-name over a set,
not a single schema. Mirrors the `KVC1` keyed-blob shape (fixed header, fixed-size descriptor
table, name blob, then payload) for the same reason `KVC1` uses it — this table has to be
addressed by name, not position. All offsets below are **relative to the start of the
section** (the magic byte), little-endian, read by explicit byte assembly (never a struct
cast), matching every other sub-format in this file.

**Fixed header — 24 bytes, at section offset 0:**

| Offset | Type | Field | Constraint |
|-------:|------|-------|------------|
| 0  | `u8[4]` | magic | `'SCM1'` |
| 4  | `u32` | version | `1` |
| 8  | `u32` | schema_count | `> 0`; `<= 65536` |
| 12 | `u32` | vocab_size | `> 0`; must equal `Config.vocab_size` |
| 16 | `u32` | name_blob_len | byte length of the name blob |
| 20 | `u32` | reserved | `== 0` |

`vocab_size` is carried here, redundantly with `Config`, so this section's own structural
parse can compute every mask-page width without depending on section-table parse order
(unconstrained, per "Byte layout" above) — the redundancy is then closed by an explicit
equality check, never trusted.

**Schema descriptor table — `schema_count` × 56 bytes, at section offset 24:**

| Offset | Type | Field | Constraint |
|-------:|------|-------|------------|
| 0  | `u32` | name_off | `name_off + name_len <= name_blob_len` |
| 4  | `u32` | name_len | `> 0`; unique across the table (byte-exact, case-sensitive) |
| 8  | `u32` | state_count | `> 0`; `<= 1048576`; state ids are `[0, state_count)`, state `0` is start |
| 12 | `u32` | accepting_count | `<= state_count` |
| 16 | `u32` | transition_count | `<= 16777216` |
| 20 | `u32` | reserved | `== 0` |
| 24 | `u64` | mask_pages_off | section-relative, this schema's mask-page block |
| 32 | `u64` | state_offsets_off | section-relative, this schema's CSR state-offset array |
| 40 | `u64` | accepting_off | section-relative, this schema's accepting-state-id array |
| 48 | `u64` | transitions_off | section-relative, this schema's flattened transition array |

The name blob begins at `24 + schema_count*56`; every per-schema data block must start at or
past `24 + schema_count*56 + name_blob_len`.

**Per-schema data blocks (four per schema, addressed by the descriptor's four offsets):**

1. **Mask pages**, `state_count * mask_page_bytes` bytes, `mask_page_bytes = ceil(vocab_size / 8)`.
   Page `i` is a packed bitmask, one bit per vocabulary token id, LSB-first within each byte:
   bit `(token_id & 7)` of byte `i*mask_page_bytes + (token_id >> 3)` is `1` iff `token_id` is a
   valid continuation at state `i`. An accepting state's page may legitimately be all-zero.
2. **State-offset array (CSR)**, `(state_count + 1) * 4` bytes, `u32[state_count+1]`.
   `state_offsets[i]` indexes into the transition array where state `i`'s row begins; state
   `i`'s edges span `[state_offsets[i], state_offsets[i+1])`. `state_offsets[0] == 0` and
   `state_offsets[state_count] == transition_count`.
3. **Accepting-state-id array**, `accepting_count * 4` bytes, `u32[accepting_count]`, strictly
   ascending, every value `< state_count`.
4. **Transition array**, `transition_count * 8` bytes, `(u32 token_id, u32 next_state)[transition_count]`,
   flattened in state order; within each state's own row, `token_id` entries are strictly
   ascending.

**ABI binding:** `sslm_schema_count` returns `schema_count`. `sslm_schema_name(index, ...)`
returns descriptor-table row `index`'s own `[name_off, name_off+name_len)` name-blob slice —
index order is descriptor-table order, not sorted. `sslm_schema_lookup(name, ...)` scans the
name blob for a byte-exact match (linear scan — the realistic `schema_count` this format ever
carries makes this the correct "smallest sound thing," not a gap).

**Loader rejections (structural, this section's own bytes):** truncated below the 24-byte
header; bad magic/version/reserved; `schema_count` zero or over its bound (a present-but-empty
section is a defect, not a legal degrade — an artifact with no schemas omits the section
entirely); `vocab_size` zero; the manifest (header + descriptor table + name blob) overflowing
or exceeding the section; a bad/duplicate schema name; `state_count`/`accepting_count`/
`transition_count` violating their own bound or relationship; any per-schema data block
overflowing, starting before the manifest, exceeding the section, or overlapping another block
(including the manifest region itself); a malformed CSR state-offset array; a malformed or
out-of-range accepting-state array; a malformed or out-of-range/non-ascending transition table.
**Cross-check, the one rejection this format adds beyond direct precedent:** for every state
`i` and every `token_id`, bit `token_id` of that state's mask page must be set **if and only
if** `token_id` appears in that state's own CSR row — a mask page and a transition table that
disagree about the same `(state, token)` pair encode the same compiler fact in two different
wire representations, and letting them diverge would mask-permit a token the walk cannot
actually advance on (or vice versa), a "wrong but deterministic" hazard a consistency-only
check cannot see.

**Loader rejections (semantic, requires `Config` already parsed):** `SchemaMasks.vocab_size !=
Config.vocab_size`.

**Explicitly NOT a load-time check, by design:** the loader does not verify that every
reachable non-accepting state has a non-empty valid-token set — that proof belongs to the
schema compiler (`tools/sslm_convert_schema.py`), made once, at compile time; a schema that
fails it is never written into an `SCM1` section in the first place. Adding a runtime re-check
here would make the compiler's own guard redundant rather than load-bearing — a check the
runtime never exercises against a real failure is not proving anything, and this format
deliberately keeps every check it declares reachable by a real rejection path.

**Save-blob binding (the sequence-level, not artifact-level, half — `SslmModel`'s own C5
save/restore blob, `src/sslm_abi.cpp`, unrelated to this file's own container format above):** a
saved sequence's bound schema is identified by `Fnv1a64` of its own name (the project's house
64-bit name-hash convention, `tools/sslm_layer_trace.cpp`) — `0` means no schema bound. Restore
resolves this hash against the target artifact's own `SCM1` name set; no match (including no
`SchemaMasks` section at all) is a defined rejection (`SSLM_RESTORE_SCHEMA_MISMATCH`), checked
after the existing model/kv-precision validation and before any block is drawn from the pool.

### Damped-greedy constants (`DGC1`, section type 42)

An artifact converted with `convert_model.py --enable-damped-greedy` carries one Raw,
12-byte `DampedGreedyConstants` section and sets header flag `0x2`. The payload is:

| Offset | Size | Type | Meaning |
|---:|---:|---|---|
| 0 | 8 | little-endian `int64` | source scale mantissa `m` |
| 8 | 4 | little-endian `int32` | source scale exponent `e` |

The 1.2 ruled pair is `(m, e) = (2883584, -36)`, from which the runtime's
certified integer derivation produces `(q_ln2, q_b, q_c) = (493, 964,
487361)`. The runtime validates the section's dtype, exact size and element
count, and the derived softmax-width domain before making damped greedy
available. The section is optional because greedy remains the artifact and
runtime default; a caller selecting damped greedy on an artifact without both
the section and its flag receives a defined rejection.

## Versioning

`format_version` is a single integer. A reader that does not recognize the version
rejects with a versioned diagnostic (`UnsupportedVersion`) — it never attempts a
best-effort load. `ARTIFACT_FORMAT_VERSION` is currently **2**. A change to any field
layout, a new required section, or a change to the integrity-hash construction bumps this
number. **v1 → v2:** the required `SigmoidLut` (`SIL1`) section was added — once C10 is
the LUT, the forward has no i-exp-sigmoid fallback, so `SIL1` is a required section and
the bump follows the "new required section" rule. v1 and v2 artifacts are therefore mutually
incompatible by the version check: a v2 loader rejects a v1 artifact (missing the required table),
and a v1 loader rejects a v2 artifact — a rejection with a diagnostic, never a silent degrade.

**The `flags` field is a reserved-bit mechanism for optional, backward-compatible
additions — distinct from `format_version`.** A version bump is for a field-layout
change, a new required section, or an integrity-hash change; `flags` is for a
capability that is optional (an old loader keeps working, unmodified, on an artifact
that does not set the bit) and requires no new section or field. The loader accepts
`flags` values that set only known bits and rejects (`BadHeader`) any unknown bit, so
a future artifact carrying a capability an older loader does not recognize is refused
rather than silently mis-loaded. Every artifact produced before a given bit existed has
that bit clear, which every loader — before and after the bit's own introduction —
accepts identically; introducing a new flag bit changes no existing artifact's bytes,
loadability, or computed output.

**Bit allocation table — ADDED 2026-08-20 (T-2199 fold 23,
`Claude/Plans/superslm-1p2-fsd-plan-2026-08-19.md` §3 in the SuperSLM-Wizard records repo),
closing the third instance of the same collision class (T-1894/D-SLM2408 moved a canary off bit 0
once already; T-2199's own `TestD1c` re-scoped off bit 1 as a stopgap; the shipped
`tests/test_main.cpp::TestRejectsUnknownFlagsBit` then collided with bit 1 a second time). A claimed
bit is now a recorded fact here, not something a reader has to reconstruct from `grep`:**

| Bit | Value | Meaning | Format version | Decision/commit |
|---|---|---|---|---|
| 0 | `0x1` | `kOptionGFusedKLandingFlag` — Option-G fused post-RoPE K landing | 2 | `D-SLM2355` |
| 1 | `0x2` | `kDampedGreedyArtifactConstantsFlag` — damped greedy anti-repetition decoding's carried scale constants (`SslmSectionType::DampedGreedyConstants`) | 2 | `D-SLM3794` |
| 2–30 | `0x4`–`0x40000000` | Unclaimed — available for future allocation, sequentially from bit 2 | — | — |
| 31 | `0x80000000` | **PERMANENTLY RESERVED FOR TESTING — never allocated to a real capability.** Every "the loader rejects an unknown `flags` bit" fixture (this file's own canary, any red-suite fixture exercising the same claim) uses this bit and only this bit, so the claim under test can never collide with a real allocation growing sequentially from bit 0. Chosen at the opposite end of the 32-bit space from where real allocations grow, specifically so this class of collision cannot recur without exhausting every other bit first. | n/a | ruled at T-2199 fold 23 (plan §3, above) |

**Why a table now, not before.** Two collisions in one project's history (T-1894's bit-0 canary,
this ticket's bit-1 canary) is `StandardsDocument.md` §4's own "failed by search twice" shape — the
fix is a structure a future allocation cannot miss, not a third ad-hoc bit picked without checking
this file. **Allocating a new capability bit:** add a row above, claim the next unclaimed bit in the
2–30 range (never bit 31), cite the ruling decision/commit, and update any `kKnownArtifactFlagsMask`-
style constant in the same commit — this table and the code must never diverge on which bits are
claimed.
