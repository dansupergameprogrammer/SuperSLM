"""T-1691 step 1's own independent `.sslm` artifact reader (design
Claude/Vitruvius/superslm-t1683-source-attribution-design-2026-08-02.md S7
step 1): reads a `.sslm` model artifact's per-layer `LayerWeights` arrays --
the raw int8 GEMM matrices and the raw `(identity, mult, shift)` fold
triples, per docs/sslm_format.md -- DIRECTLY from the documented byte
layout. This module does not import `tools/sslm_marshal.h` (it cannot; that
is a C++ header) and does not import `tools/sslm_format.py` or
`tools/sslm_model_writer.py` -- every offset and field below is transcribed
from `docs/sslm_format.md`'s own tables, independently, the same
"no shared apparatus" argument design S7 step 1 states ("this unit does not
include or link against that header, so a defect in the marshaling code
cannot also be present here by construction"): a defect in
`tools/sslm_marshal.h`'s C++ MarshalLayer, or in the Python writer used
elsewhere in this repo to PRODUCE `.sslm` artifacts, cannot also be present
here, because this file shares no code with either.

Only the READ side is implemented -- this module never writes a `.sslm`
artifact. Integrity verification (the header's own `integrity_sha256`
field) is NOT performed here: this reader is scoped to artifacts the real
loader (`SslmModel::Load`) has already accepted (the parity/precision
shadow's own input is always an artifact the real int8 engine also loaded
successfully for the same run), not to hostile or corrupted input -- that
trust boundary belongs to the production loader, already certified
elsewhere in this codebase (docs/sslm_format.md's own "hostile-input trust
boundary... certified to the T-129 bar"). This reader DOES check every
structural bound the format documents (offsets in range, no overrun,
correct magic/version, matching tensor-manifest element counts) and raises
loudly (`ValueError`) rather than silently truncating or defaulting on any
violation -- the same reject-over-degrade posture, applied at reduced scope.

Field naming exactly mirrors `tools/sslm_marshal.h`'s own key convention
(transcribed from that file's own tensor/entry name strings, which are
DATA -- the artifact's own key namespace, not code -- so citing them here
carries no code dependency): `layer{L}.q_proj` (WGT1/WSC1), `layer{L}.k_head{h}`
/ `layer{L}.v_head{h}` (KVC1 kv_landing_reciprocals, 3 words: unused, e_t, r_t),
`layer{L}.softmax_khead{h}` / `layer{L}.attn_norm` / etc. (KVC1
composition_constants, 2 words: m, e).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

# --- container-level constants (docs/sslm_format.md "Byte layout") --------
_HEADER_MAGIC = b"SSLM"
_HEADER_BYTES = 64
_SECTION_ENTRY_BYTES = 40
_FORMAT_VERSION = 2

# docs/sslm_format.md "Section types" -- the values this reader actually uses.
_SECTION_CONFIG = 0
_SECTION_WEIGHTS = 2
_SECTION_BIASES = 3
_SECTION_ROPE_TABLES = 4
_SECTION_WEIGHT_SCALES = 6
_SECTION_COMPOSITION_CONSTANTS = 7
_SECTION_KV_LANDING_RECIPROCALS = 9
_SECTION_SIGMOID_LUT = 12

_DTYPE_ELEM_SIZE = {0: 1, 1: 1, 2: 4, 3: 8, 4: 1, 5: 4, 6: 8}  # docs "Dtypes"


class SslmArtifactFormatError(ValueError):
    """Raised on any structural violation of the documented byte layout."""


def _read_container_sections(data: bytes) -> dict[int, bytes]:
    """Parses the top-level header + section table (docs "Byte layout"),
    returning {section_type: raw_bytes}. Does not verify the SHA-256
    integrity hash (module docstring: out of this reader's scope)."""
    if len(data) < _HEADER_BYTES:
        raise SslmArtifactFormatError(f"file too short for the {_HEADER_BYTES}-byte header")
    magic = data[0:4]
    if magic != _HEADER_MAGIC:
        raise SslmArtifactFormatError(f"bad magic {magic!r}, want {_HEADER_MAGIC!r}")
    (format_version, header_bytes, section_count, flags, _reserved0) = struct.unpack(
        "<IIIII", data[4:24])
    if format_version != _FORMAT_VERSION:
        raise SslmArtifactFormatError(f"format_version {format_version}, want {_FORMAT_VERSION}")
    if header_bytes != _HEADER_BYTES:
        raise SslmArtifactFormatError(f"header_bytes {header_bytes}, want {_HEADER_BYTES}")
    if flags != 0:
        raise SslmArtifactFormatError(f"flags {flags}, want 0 (reserved)")
    (file_bytes,) = struct.unpack("<Q", data[24:32])
    if file_bytes != len(data):
        raise SslmArtifactFormatError(f"file_bytes {file_bytes} != actual file size {len(data)}")

    table_off = _HEADER_BYTES
    table_end = table_off + section_count * _SECTION_ENTRY_BYTES
    if table_end > len(data):
        raise SslmArtifactFormatError("section table overruns the file")

    sections: dict[int, bytes] = {}
    for i in range(section_count):
        entry = data[table_off + i * _SECTION_ENTRY_BYTES: table_off + (i + 1) * _SECTION_ENTRY_BYTES]
        (sec_type, dtype, offset, byte_size, elem_count, alignment, reserved) = struct.unpack(
            "<IIQQQII", entry)
        if reserved != 0:
            raise SslmArtifactFormatError(f"section {i}: reserved field nonzero")
        if offset < table_end:
            raise SslmArtifactFormatError(f"section {i}: offset {offset} below the section table end")
        if offset % alignment != 0:
            raise SslmArtifactFormatError(f"section {i}: offset {offset} not aligned to {alignment}")
        if offset + byte_size > len(data):
            raise SslmArtifactFormatError(f"section {i}: [{offset},{offset + byte_size}) overruns the file")
        elem_size = _DTYPE_ELEM_SIZE.get(dtype)
        if elem_size is not None and byte_size != elem_count * elem_size:
            raise SslmArtifactFormatError(
                f"section {i}: byte_size {byte_size} != elem_count {elem_count} * elem_size {elem_size}")
        if sec_type in sections:
            raise SslmArtifactFormatError(f"duplicate section type {sec_type}")
        sections[sec_type] = data[offset:offset + byte_size]
    return sections


def _parse_tensor_manifest(blob: bytes, expected_magic: bytes) -> dict[str, np.ndarray]:
    """WGT1 / WSC1 / ROP1 (docs "Tensor-manifest blob") -> {name: ndarray}.
    Element dtype is inferred from `expected_magic` (int8 for WGT1, int32 for
    WSC1, int64 for ROP1), matching the section-type table (docs "Model
    sub-formats (S2)")."""
    if len(blob) < 16:
        raise SslmArtifactFormatError(f"{expected_magic!r} manifest shorter than the 16-byte header")
    magic = blob[0:4]
    if magic != expected_magic:
        raise SslmArtifactFormatError(f"manifest magic {magic!r}, want {expected_magic!r}")
    (version, tensor_count, name_blob_len) = struct.unpack("<III", blob[4:16])
    if version != 1:
        raise SslmArtifactFormatError(f"{expected_magic!r} manifest version {version}, want 1")
    desc_off = 16
    desc_end = desc_off + tensor_count * 48
    name_blob_off = desc_end
    name_blob_end = name_blob_off + name_blob_len
    if name_blob_end > len(blob):
        raise SslmArtifactFormatError(f"{expected_magic!r} manifest: descriptors + name blob overrun")
    name_blob = blob[name_blob_off:name_blob_end]

    elem_size = {b"WGT1": 1, b"WSC1": 4, b"ROP1": 8}[expected_magic]
    np_dtype = {b"WGT1": np.int8, b"WSC1": np.int32, b"ROP1": np.int64}[expected_magic]

    tensors: dict[str, np.ndarray] = {}
    for i in range(tensor_count):
        d = blob[desc_off + i * 48: desc_off + (i + 1) * 48]
        (name_off, name_len, rank) = struct.unpack("<III", d[0:12])
        shape = struct.unpack("<IIII", d[12:28])
        (data_off, elem_count, reserved) = struct.unpack("<QQI", d[28:48])
        if reserved != 0:
            raise SslmArtifactFormatError(f"{expected_magic!r} tensor {i}: reserved nonzero")
        if not (1 <= rank <= 4):
            raise SslmArtifactFormatError(f"{expected_magic!r} tensor {i}: rank {rank} outside [1,4]")
        if name_off + name_len > len(name_blob):
            raise SslmArtifactFormatError(f"{expected_magic!r} tensor {i}: name range outside name blob")
        name = name_blob[name_off:name_off + name_len].decode("utf-8")
        if name in tensors:
            raise SslmArtifactFormatError(f"{expected_magic!r}: duplicate tensor name {name!r}")
        expected_elems = 1
        for k in range(rank):
            expected_elems *= shape[k]
        if elem_count != expected_elems:
            raise SslmArtifactFormatError(
                f"{expected_magic!r} tensor {name!r}: elem_count {elem_count} != product(shape) {expected_elems}")
        data_end = data_off + elem_count * elem_size
        if data_end > len(blob):
            raise SslmArtifactFormatError(f"{expected_magic!r} tensor {name!r}: data range overruns the section")
        raw = blob[data_off:data_end]
        arr = np.frombuffer(raw, dtype=f"<{np.dtype(np_dtype).kind}{elem_size}").reshape(shape[:rank])
        tensors[name] = arr
    return tensors


def _parse_kvc1(blob: bytes, expected_value_words: int) -> dict[str, tuple[int, ...]]:
    """KVC1 (docs "Keyed numeric-constant blob") -> {name: (int64, ...)}."""
    if len(blob) < 24:
        raise SslmArtifactFormatError("KVC1 blob shorter than the 24-byte header")
    magic = blob[0:4]
    if magic != b"KVC1":
        raise SslmArtifactFormatError(f"KVC1 magic {magic!r}, want b'KVC1'")
    (version, entry_count, value_words, name_blob_len, reserved) = struct.unpack("<IIIII", blob[4:24])
    if version != 1:
        raise SslmArtifactFormatError(f"KVC1 version {version}, want 1")
    if reserved != 0:
        raise SslmArtifactFormatError("KVC1: reserved nonzero")
    if value_words != expected_value_words:
        raise SslmArtifactFormatError(f"KVC1 value_words {value_words}, want {expected_value_words}")

    desc_off = 24
    desc_end = desc_off + entry_count * 8
    values_off = desc_end
    values_end = values_off + entry_count * value_words * 8
    name_blob_off = values_end
    name_blob_end = name_blob_off + name_blob_len
    if name_blob_end > len(blob):
        raise SslmArtifactFormatError("KVC1: descriptors + values + name blob overrun")
    name_blob = blob[name_blob_off:name_blob_end]

    entries: dict[str, tuple[int, ...]] = {}
    for i in range(entry_count):
        (name_off, name_len) = struct.unpack("<II", blob[desc_off + i * 8: desc_off + i * 8 + 8])
        if name_off + name_len > len(name_blob):
            raise SslmArtifactFormatError(f"KVC1 entry {i}: name range outside name blob")
        if name_len == 0:
            raise SslmArtifactFormatError(f"KVC1 entry {i}: zero-length name")
        name = name_blob[name_off:name_off + name_len].decode("utf-8")
        if name in entries:
            raise SslmArtifactFormatError(f"KVC1: duplicate entry name {name!r}")
        vraw = blob[values_off + i * value_words * 8: values_off + (i + 1) * value_words * 8]
        entries[name] = struct.unpack(f"<{value_words}q", vraw)
    return entries


def _parse_cfg1(blob: bytes) -> dict:
    if len(blob) != 84:
        raise SslmArtifactFormatError(f"CFG1 blob is {len(blob)} bytes, want exactly 84")
    magic = blob[0:4]
    if magic != b"CFG1":
        raise SslmArtifactFormatError(f"CFG1 magic {magic!r}, want b'CFG1'")
    (version,) = struct.unpack("<I", blob[4:8])
    if version != 1:
        raise SslmArtifactFormatError(f"CFG1 version {version}, want 1")
    # offset 8..68: 15 u32 fields (hidden_size .. reserved), per docs' own
    # offset table; offset 4..8 is version (already read above).
    fields = struct.unpack("<IIIIIIIIIIIIIII", blob[8:68])
    (hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim,
     intermediate_size, vocab_size, context_cap, tie_word_embeddings, kv_precision, kv_block_size,
     unicode_major, unicode_minor, unicode_patch, reserved) = fields
    if reserved != 0:
        raise SslmArtifactFormatError("CFG1: reserved nonzero")
    # rope_theta at offset 68, rms_norm_eps at offset 76 (docs "Config blob").
    (_rope_theta, _rms_norm_eps) = struct.unpack("<dd", blob[68:84])
    return {
        "hidden_size": hidden_size, "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads, "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim, "intermediate_size": intermediate_size, "vocab_size": vocab_size,
        "context_cap": context_cap, "tie_word_embeddings": bool(tie_word_embeddings),
        "kv_precision": kv_precision, "kv_block_size": kv_block_size,
    }


def _parse_sil1(blob: bytes) -> np.ndarray:
    if len(blob) != 16 + 1025 * 4:
        raise SslmArtifactFormatError(f"SIL1 blob is {len(blob)} bytes, want exactly {16 + 1025 * 4}")
    magic = blob[0:4]
    if magic != b"SIL1":
        raise SslmArtifactFormatError(f"SIL1 magic {magic!r}, want b'SIL1'")
    (version, entry_count, reserved) = struct.unpack("<III", blob[4:16])
    if version != 1 or entry_count != 1025 or reserved != 0:
        raise SslmArtifactFormatError(f"SIL1 header invalid: version={version} entry_count={entry_count}")
    return np.frombuffer(blob[16:], dtype="<i4")


@dataclass
class LayerParityWeights:
    """One layer's real, artifact-sourced arrays -- the direct Python
    analogue of `include/superslm/model.h`'s `LayerWeights`, read
    independently (module docstring)."""
    q_weight: np.ndarray  # int8 [hidden_size, hidden_size], row v = output channel v's input row.
    k_weight: np.ndarray  # int8 [kv_hidden_size, hidden_size]
    v_weight: np.ndarray
    o_weight: np.ndarray  # int8 [hidden_size, hidden_size]
    gate_weight: np.ndarray  # int8 [intermediate_size, hidden_size]
    up_weight: np.ndarray
    down_weight: np.ndarray  # int8 [hidden_size, intermediate_size]
    attn_norm_gain: np.ndarray  # int8 [hidden_size] (widened to int32 by the caller, matching WidenGainToInt32)
    mlp_norm_gain: np.ndarray

    q_fold: np.ndarray  # int32 [hidden_size, 3] -- (identity, mult, shift) per output channel.
    k_fold: np.ndarray  # int32 [kv_hidden_size, 3]
    v_fold: np.ndarray
    o_fold: np.ndarray
    gate_fold: np.ndarray
    up_fold: np.ndarray
    down_fold: np.ndarray
    ctx_fold: np.ndarray  # int32 [num_heads, 3]

    q_bias: np.ndarray | None  # int64 [hidden_size] or None
    k_bias: np.ndarray | None  # int64 [kv_hidden_size] or None
    v_bias: np.ndarray | None

    kv_landing_e_t_k: list[int]  # per kv_head
    kv_landing_r_t_k: list[int]
    kv_landing_e_t_v: list[int]
    kv_landing_r_t_v: list[int]

    iexp_softmax_khead_m: list[int]  # per kv_head
    iexp_softmax_khead_e: list[int]

    attn_norm_site_constant: tuple[int, int]  # (m, e)
    q_site_constant: tuple[int, int]
    o_site_constant: tuple[int, int]
    ctx_fold_site_constant: tuple[int, int]
    attn_residual_site_constant: tuple[int, int]
    mlp_norm_site_constant: tuple[int, int]
    gate_site_constant: tuple[int, int]
    up_site_constant: tuple[int, int]
    mlp_act_site_constant: tuple[int, int]
    down_site_constant: tuple[int, int]
    mlp_residual_site_constant: tuple[int, int]


@dataclass
class ParsedArtifact:
    config: dict
    embed: np.ndarray  # int8 [vocab_size, hidden_size]
    final_norm_gain: np.ndarray  # int8 [hidden_size]
    embed_site_constant: tuple[int, int]
    final_norm_site_constant: tuple[int, int]
    lm_head: np.ndarray | None  # int8 [vocab_size, hidden_size], only present if not tied
    rope_cos: np.ndarray  # int64 [context_cap, head_dim/2]
    rope_sin: np.ndarray
    sigmoid_lut: np.ndarray  # int32 [1025], Q15
    layers: list[LayerParityWeights] = field(default_factory=list)


def read_artifact(path: str) -> ParsedArtifact:
    with open(path, "rb") as f:
        data = f.read()
    sections = _read_container_sections(data)

    for required in (_SECTION_CONFIG, _SECTION_WEIGHTS, _SECTION_WEIGHT_SCALES, _SECTION_ROPE_TABLES,
                      _SECTION_COMPOSITION_CONSTANTS, _SECTION_KV_LANDING_RECIPROCALS,
                      _SECTION_SIGMOID_LUT):
        if required not in sections:
            raise SslmArtifactFormatError(f"missing required section type {required}")

    cfg = _parse_cfg1(sections[_SECTION_CONFIG])
    weights = _parse_tensor_manifest(sections[_SECTION_WEIGHTS], b"WGT1")
    fold = _parse_tensor_manifest(sections[_SECTION_WEIGHT_SCALES], b"WSC1")
    rope = _parse_tensor_manifest(sections[_SECTION_ROPE_TABLES], b"ROP1")
    biases = _parse_tensor_manifest(sections[_SECTION_BIASES], b"BIA1") if _SECTION_BIASES in sections else {}
    composition = _parse_kvc1(sections[_SECTION_COMPOSITION_CONSTANTS], 2)
    kv_recip = _parse_kvc1(sections[_SECTION_KV_LANDING_RECIPROCALS], 3)
    sigmoid_lut = _parse_sil1(sections[_SECTION_SIGMOID_LUT])

    def site_const(name: str) -> tuple[int, int]:
        v = composition.get(name)
        if v is None or len(v) < 2:
            raise SslmArtifactFormatError(f"missing composition_constants entry {name!r}")
        return (v[0], v[1])

    if "embed" not in weights or "final_norm.gain" not in weights:
        raise SslmArtifactFormatError("missing embed or final_norm.gain WGT1 tensor")

    num_layers = cfg["num_hidden_layers"]
    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg["num_key_value_heads"]

    layers: list[LayerParityWeights] = []
    for l in range(num_layers):
        p = f"layer{l}"

        def wgt(suffix: str) -> np.ndarray:
            key = f"{p}.{suffix}"
            if key not in weights:
                raise SslmArtifactFormatError(f"missing WGT1 tensor {key!r}")
            return weights[key]

        def wsc(suffix: str) -> np.ndarray:
            key = f"{p}.{suffix}"
            if key not in fold:
                raise SslmArtifactFormatError(f"missing WSC1 tensor {key!r}")
            return fold[key]

        def bia(suffix: str) -> np.ndarray | None:
            return biases.get(f"{p}.{suffix}")

        kv_e_t_k, kv_r_t_k, kv_e_t_v, kv_r_t_v = [], [], [], []
        for h in range(num_kv_heads):
            ke = kv_recip.get(f"{p}.k_head{h}")
            ve = kv_recip.get(f"{p}.v_head{h}")
            if ke is None or ve is None:
                raise SslmArtifactFormatError(f"missing kv_landing_reciprocals entry for {p} head {h}")
            kv_e_t_k.append(ke[1])
            kv_r_t_k.append(ke[2])
            kv_e_t_v.append(ve[1])
            kv_r_t_v.append(ve[2])

        iexp_m, iexp_e = [], []
        for h in range(num_kv_heads):
            e = composition.get(f"{p}.softmax_khead{h}")
            if e is None or len(e) < 2:
                raise SslmArtifactFormatError(f"missing composition_constants entry {p}.softmax_khead{h}")
            iexp_m.append(e[0])
            iexp_e.append(e[1])

        layers.append(LayerParityWeights(
            q_weight=wgt("q_proj"), k_weight=wgt("k_proj"), v_weight=wgt("v_proj"), o_weight=wgt("o_proj"),
            gate_weight=wgt("gate_proj"), up_weight=wgt("up_proj"), down_weight=wgt("down_proj"),
            attn_norm_gain=wgt("attn_norm.gain"), mlp_norm_gain=wgt("mlp_norm.gain"),
            q_fold=wsc("q_proj"), k_fold=wsc("k_proj"), v_fold=wsc("v_proj"), o_fold=wsc("o_proj"),
            gate_fold=wsc("gate_proj"), up_fold=wsc("up_proj"), down_fold=wsc("down_proj"),
            ctx_fold=wsc("ctx_fold"),
            q_bias=bia("q_proj"), k_bias=bia("k_proj"), v_bias=bia("v_proj"),
            kv_landing_e_t_k=kv_e_t_k, kv_landing_r_t_k=kv_r_t_k,
            kv_landing_e_t_v=kv_e_t_v, kv_landing_r_t_v=kv_r_t_v,
            iexp_softmax_khead_m=iexp_m, iexp_softmax_khead_e=iexp_e,
            attn_norm_site_constant=site_const(f"{p}.attn_norm"),
            q_site_constant=site_const(f"{p}.q_proj"),
            o_site_constant=site_const(f"{p}.o_proj"),
            ctx_fold_site_constant=site_const(f"{p}.attn_ctx"),
            attn_residual_site_constant=site_const(f"{p}.attn_residual"),
            mlp_norm_site_constant=site_const(f"{p}.mlp_norm"),
            gate_site_constant=site_const(f"{p}.gate_proj"),
            up_site_constant=site_const(f"{p}.up_proj"),
            mlp_act_site_constant=site_const(f"{p}.mlp_act"),
            down_site_constant=site_const(f"{p}.down_proj"),
            mlp_residual_site_constant=site_const(f"{p}.mlp_residual"),
        ))

    lm_head = None if cfg["tie_word_embeddings"] else weights.get("lm_head")
    if not cfg["tie_word_embeddings"] and lm_head is None:
        raise SslmArtifactFormatError("tie_word_embeddings=False but no lm_head WGT1 tensor is present")

    return ParsedArtifact(
        config=cfg,
        embed=weights["embed"],
        final_norm_gain=weights["final_norm.gain"],
        embed_site_constant=site_const("embed"),
        final_norm_site_constant=site_const("final_norm"),
        lm_head=lm_head,
        rope_cos=rope["cos"],
        rope_sin=rope["sin"],
        sigmoid_lut=sigmoid_lut,
        layers=layers,
    )
