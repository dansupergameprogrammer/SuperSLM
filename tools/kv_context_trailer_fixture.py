"""T-1691 step 4's own red-first fixture (design Claude/Vitruvius/superslm-
t1683-source-attribution-design-2026-08-02.md S7 step 4, "provenance and
shape guards"/part 3 of the red-first proof): a small, fully synthetic,
hand-constructed `.sslm` model artifact with MULTIPLE layers and MULTIPLE
prefill positions, so `sslm_layer_trace.cpp`'s new K/V-context trailer
(the `KeyRow`/`ValueRow` read for every prior position at each target
layer) has more than one position and more than one layer to read.

Reuses `kv_saturation_fixture.py`'s own hand-derived construction verbatim
(module docstring there: RmsNormSite/K-V-landing/RoPE-identity derivation) --
this fixture changes only the GEOMETRY (more layers, a bigger context_cap,
a three-token prompt) and the TOKENIZER (three distinct one-character
tokens instead of one), never the per-site arithmetic. The same "identity
K/V weight, all-zero q/o/gate/up_proj, identity down_proj" construction that
already reaches a REAL loaded artifact (D-SLM721's own constraint) is
repeated per layer, unchanged, because RmsNormSite -> K/V-landing is a fixed
point of this fold chain independent of layer index (the same normed_scale,
{2^30,-14}, recurs at every layer since q_proj is all-zero and the residual
branches reconcile a genuine zero at every exponent).

NUM_HIDDEN_LAYERS = 6 (small, but includes "row 5" -- the design's own
control-band layer -- so the trailer's target-layer selection has at least
one real hit to read; rows 21-28 do not exist at this geometry and the tool
is required to skip them, loud but non-fatal, which this fixture's own test
(tools/test_t1691_kv_context_trailer.py) asserts directly). A 3-token
prompt ("abc") gives prefill exactly 2 full-layer passes (positions 0, 1)
before the last token's own per-layer walk lands position 2 -- so the
trailer's own "every PRIOR position" (0..context_length-1) at each target
layer has two real, distinct positions to carry, not the degenerate
`width == 1` case the design's own §7 step 3 names as blind
(`docs/D-SLM503`)."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

import sslm_format as F
import sslm_model_writer as W
from unicode_tables import Unicode

# --- geometry -- see kv_saturation_fixture.py's own docstring for the -----
# per-site hand derivation this reuses verbatim; only these constants change.
HIDDEN_SIZE = 2
NUM_HIDDEN_LAYERS = 6
NUM_ATTENTION_HEADS = 1
NUM_KEY_VALUE_HEADS = 1
HEAD_DIM = 2
INTERMEDIATE_SIZE = 2
CONTEXT_CAP = 4
VOCAB_SIZE = 4

CANONICAL = (1073741824, -30)  # {m=2^30, e=-30} -- kv_saturation_fixture.py's own CANONICAL.
MLP_ACT_CONST = CANONICAL
SOFTMAX_KHEAD_CONST = (1073741824, -60)  # same derivation, layer-independent (see module docstring).
EMBED_CONST = CANONICAL
KV_LANDING_E_T = -14
KV_LANDING_R_T = 4294967296  # 2^32 -- kv_saturation_fixture.py's own derivation.

_IDENTITY2X2 = np.array([[1, 0], [0, 1]], dtype=np.int8)
_ZERO2X2 = np.array([[0, 0], [0, 0]], dtype=np.int8)


def _identity_fold(channels: int) -> np.ndarray:
    return np.array([(1, 0, 0) for _ in range(channels)], dtype=np.int32)


def build_kv_context_trailer_model() -> bytes:
    """Builds the NUM_HIDDEN_LAYERS-layer model artifact. Every layer uses
    the identical identity-K/V / all-zero-attention-and-MLP-branch / identity
    down_proj construction kv_saturation_fixture.py's single layer already
    validated against the real loader (D-SLM721) -- repeated per layer."""
    cfg = W.write_cfg1(
        hidden_size=HIDDEN_SIZE, num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS, num_key_value_heads=NUM_KEY_VALUE_HEADS,
        head_dim=HEAD_DIM, intermediate_size=INTERMEDIATE_SIZE, vocab_size=VOCAB_SIZE,
        context_cap=CONTEXT_CAP, tie_word_embeddings=True, kv_precision=0, kv_block_size=1,
        unicode_major=0, unicode_minor=0, unicode_patch=0, rope_theta=10000.0, rms_norm_eps=1e-6)

    embed = np.zeros((VOCAB_SIZE, HIDDEN_SIZE), dtype=np.int8)
    embed[0] = [5, -5]
    embed[1] = [5, -5]
    embed[2] = [5, -5]
    gain = np.ones((HIDDEN_SIZE,), dtype=np.int8)

    weight_tensors = {"embed": embed, "final_norm.gain": gain}
    scale_tensors = {}
    composition_constants = {"embed": EMBED_CONST, "final_norm": CANONICAL}
    kv_landing_reciprocals = {}

    for layer in range(NUM_HIDDEN_LAYERS):
        p = f"layer{layer}"
        weight_tensors.update({
            f"{p}.q_proj": _ZERO2X2,
            f"{p}.k_proj": _IDENTITY2X2,
            f"{p}.v_proj": _IDENTITY2X2,
            f"{p}.o_proj": _ZERO2X2,
            f"{p}.gate_proj": _ZERO2X2,
            f"{p}.up_proj": _ZERO2X2,
            f"{p}.down_proj": _IDENTITY2X2,
            f"{p}.attn_norm.gain": gain,
            f"{p}.mlp_norm.gain": gain,
        })
        scale_tensors.update({
            f"{p}.q_proj": _identity_fold(HIDDEN_SIZE),
            f"{p}.k_proj": _identity_fold(NUM_KEY_VALUE_HEADS * HEAD_DIM),
            f"{p}.v_proj": _identity_fold(NUM_KEY_VALUE_HEADS * HEAD_DIM),
            f"{p}.o_proj": _identity_fold(HIDDEN_SIZE),
            f"{p}.gate_proj": _identity_fold(INTERMEDIATE_SIZE),
            f"{p}.up_proj": _identity_fold(INTERMEDIATE_SIZE),
            f"{p}.down_proj": _identity_fold(HIDDEN_SIZE),
            f"{p}.ctx_fold": _identity_fold(NUM_ATTENTION_HEADS),
        })
        composition_constants.update({
            f"{p}.attn_norm": CANONICAL,
            f"{p}.q_proj": CANONICAL,
            f"{p}.o_proj": CANONICAL,
            f"{p}.attn_ctx": CANONICAL,
            f"{p}.attn_residual": CANONICAL,
            f"{p}.mlp_norm": CANONICAL,
            f"{p}.gate_proj": CANONICAL,
            f"{p}.up_proj": CANONICAL,
            f"{p}.mlp_act": MLP_ACT_CONST,
            f"{p}.down_proj": CANONICAL,
            f"{p}.mlp_residual": CANONICAL,
            f"{p}.softmax_khead0": SOFTMAX_KHEAD_CONST,
        })
        kv_landing_reciprocals.update({
            f"{p}.k_head0": (1, KV_LANDING_E_T, KV_LANDING_R_T),
            f"{p}.v_head0": (1, KV_LANDING_E_T, KV_LANDING_R_T),
        })

    weights = W.write_tensor_manifest(W.WGT1, np.int8, weight_tensors)
    weight_scales = W.write_tensor_manifest(W.WSC1, np.int32, scale_tensors)
    # ROP1: identity rotation (cos=2^30 ["1.0"], sin=0) at every one of the
    # CONTEXT_CAP positions -- row-major [context_cap, head_dim/2] (module
    # docstring; forward_sites.cpp's own ReadRopeTableEntryI64 indexing).
    # An identity rotation at every position leaves the just-landed K
    # unchanged regardless of position, matching kv_saturation_fixture.py's
    # own single-position derivation, extended across CONTEXT_CAP positions.
    rope = W.write_tensor_manifest(W.ROP1, np.int64, {
        "cos": np.array([1073741824] * CONTEXT_CAP, dtype=np.int64),
        "sin": np.array([0] * CONTEXT_CAP, dtype=np.int64),
    })
    kvc_composition = W.write_kvc1(2, composition_constants)
    kvc_landing = W.write_kvc1(3, kv_landing_reciprocals)

    sections = [
        F.Section(F.SectionType.CONFIG, cfg),
        F.Section(F.SectionType.SIGMOID_LUT, W.write_sil1()),
        F.Section(F.SectionType.WEIGHTS, weights),
        F.Section(F.SectionType.WEIGHT_SCALES, weight_scales),
        F.Section(F.SectionType.ROPE_TABLES, rope),
        F.Section(F.SectionType.COMPOSITION_CONSTANTS, kvc_composition),
        F.Section(F.SectionType.KV_LANDING_RECIPROCALS, kvc_landing),
    ]
    data, _fingerprint = F.build_artifact(sections)
    return data


def build_three_token_tokenizer() -> bytes:
    """Three one-character tokens: "a"->0, "b"->1, "c"->2 -- enough to
    tokenize the fixture's own 3-character prompt "abc" into three distinct
    positions (kv_saturation_fixture.py's own tokenizer supports exactly
    one character; this fixture needs a real multi-position prefill)."""
    id_to_bytes = [b"a", b"b", b"c"]
    byte_to_id = [0] * 256
    byte_to_id[ord("a")] = 0
    byte_to_id[ord("b")] = 1
    byte_to_id[ord("c")] = 2

    tok = bytearray()
    tok += b"TOK1"
    tok += struct.pack("<IIIII", 1, len(id_to_bytes), 0, 0, 0)
    for b in byte_to_id:
        tok += struct.pack("<I", b)
    offs, blob = [0], bytearray()
    for by in id_to_bytes:
        blob += by
        offs.append(len(blob))
    for o in offs:
        tok += struct.pack("<I", o)
    tok += struct.pack("<I", len(blob))
    tok += blob
    tok += struct.pack("<I", 0)  # special_count == 0
    tok += struct.pack("<I", 0)  # special_blob_len

    u = Unicode.build()
    sections = [
        F.Section(F.SectionType.CONFIG, b'{"tokenizer":"t1691-kv-trailer-fixture"}'),
        F.Section(F.SectionType.TOKENIZER, bytes(tok)),
        F.Section(F.SectionType.UNICODE_TABLES, u.serialize()),
        F.Section(F.SectionType.SIGMOID_LUT, W.write_sil1()),
    ]
    data, _fingerprint = F.build_artifact(sections)
    return data


def write_fixture(out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = out_dir / "t1691_kv_trailer_fixture.tok.sslm"
    model_path = out_dir / "t1691_kv_trailer_fixture.sslm"
    tok_path.write_bytes(build_three_token_tokenizer())
    model_path.write_bytes(build_kv_context_trailer_model())
    return tok_path, model_path


PROMPT = "abc"
