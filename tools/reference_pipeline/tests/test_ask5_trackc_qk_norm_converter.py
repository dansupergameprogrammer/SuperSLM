"""Ask 5, Track C — the converter learns the candidate's checkpoint conventions and its
QK-norm tensors (T-2539).

Specification: `Claude/Vitruvius/t2408-superslm-ask5-qwen3-arch-design-2026-08-29.md` §6
Track C — converter (piece c), frozen as a casebook (D-SLM5792). Two red states this suite
proves closed, both confirmed red on the unmodified tree before this round's build steps
landed (build log `Claude/Brunel/t2539-ask5-trackc-build-2026-09-02.md`):

- **Mechanism red state** — a bare-convention, bias-free fixture (built by
  `build_parameterized_fixture_checkpoint`, this round's own step 8) rejected by the
  unmodified engine with every tensor unmapped (the map assumed `model.`-prefixed names
  and unconditional q/k/v biases).
- **Product red state** — the real, pinned Qwen3-Embedding-0.6B checkpoint rejected with
  310 of 310 tensors unmapped, 338 of 338 map entries missing.

Every cell below is a fixture-scale MECHANISM check (§5.4's own distinction): fast,
isolates Track C from Track A/B, and exercises the identical defect class the real
candidate hit. The real candidate's own end-to-end conversion-and-load is a separate,
one-off product proof recorded in the build log — no fixture cell here substitutes for it.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import api, require

MODULE = "reference_pipeline.pipeline"
VALIDATE_MODULE = "sslm_convert_validate"

_TOOLS_DIR = Path(__file__).resolve().parents[2]
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))


def _fixture_builder():
    import _calibrate_checkpoint_fixture as fixture_mod
    return fixture_mod


def _build(tmp_path, name, **kwargs):
    fixture_mod = _fixture_builder()
    builder = api("_calibrate_checkpoint_fixture", "build_parameterized_fixture_checkpoint")
    return builder(tmp_path / name, **kwargs)


def _totality(pipeline, ckpt):
    cfg = pipeline.load_config(ckpt / "config.json")
    tensors = pipeline._open_checkpoint_tensors(ckpt)
    present = set(tensors.keys())
    names = pipeline._upstream_names(cfg, present)
    unmapped = sorted(present - set(names))
    missing = sorted(set(names) - present)
    return names, unmapped, missing


# ==============================================================================
# CKN-01/02/04 -- the mechanism fixture's own totality check, both directions
# ==============================================================================


def test_the_bare_bias_free_mechanism_fixture_closes_the_totality_check_to_zero(tmp_path):
    """The mechanism red state's own green twin: a bare-convention (`prefix=""`),
    bias-free fixture -- rejected by the unmodified engine (every tensor unmapped, the
    build log's own red-state proof) -- converts with zero unmapped and zero missing
    tensors once CKN-01/02/04 and the QK-norm steps (5-7) land."""
    pipeline = require(MODULE)
    ckpt = _build(tmp_path, "mechanism", prefix="", biased=False,
                  tie_word_embeddings=True, qk_norm=True)
    names, unmapped, missing = _totality(pipeline, ckpt)
    assert unmapped == [], f"unmapped: {unmapped}"
    assert missing == [], f"missing: {missing}"


def test_the_mechanism_fixtures_qk_norm_wgt1_codes_match_a_hand_recomputation(tmp_path):
    """`layer0.q_norm.gain`/`layer0.k_norm.gain`'s int8 codes, independently recomputed by
    hand from the fixture's own float values (an all-ones `(head_dim,)` gain, RoPE-pair-
    permuted -- a no-op on an all-identical vector -- then peak-scaled int8), match the
    converter's own emitted codes byte-for-byte, per the acceptance text's own oracle."""
    pipeline = require(MODULE)
    ckpt = _build(tmp_path, "mechanism_wgt1", prefix="", biased=False,
                  tie_word_embeddings=True, qk_norm=True)
    model = pipeline.load_model(ckpt)
    head_dim = model.config.head_dim
    peak = 1.0
    hand_scale = peak / 127.0
    hand_codes = np.clip(np.round(np.ones(head_dim) / hand_scale), -128, 127).astype(np.int8)
    for leaf in ("q_norm", "k_norm"):
        key = f"layer0.{leaf}.gain"
        codes = np.asarray(model.weights[key])
        scale = model.weight_scales[key][0]
        assert np.array_equal(codes, hand_codes), f"{key}: codes {codes} != hand {hand_codes}"
        assert abs(scale - hand_scale) < 1e-15, f"{key}: scale {scale} != hand {hand_scale}"


# ==============================================================================
# Backward compatibility -- executed, not assumed
# ==============================================================================


def test_the_legacy_convention_fixture_converts_byte_identically_to_the_pre_fold_build(tmp_path):
    """A `model.`-prefixed, biased, no-QK-norm fixture -- matching every existing
    incumbent exactly -- built by this round's own parameterized builder is
    byte-identical on disk, and converts to identical weight codes, against the SAME
    fixture built by the pre-existing, unparameterized `build_fixture_checkpoint`
    (unchanged by this round). Proves checkpoint-driven detection is not merely
    permissive for the new convention: it correctly re-derives the old one."""
    fixture_mod = _fixture_builder()
    pipeline = require(MODULE)

    old_ckpt = fixture_mod.build_fixture_checkpoint(tmp_path / "old")
    new_ckpt = fixture_mod.build_parameterized_fixture_checkpoint(
        tmp_path / "new_bc", prefix="model.", biased=True,
        tie_word_embeddings=True, qk_norm=False)

    old_bytes = (old_ckpt / "model.safetensors").read_bytes()
    new_bytes = (new_ckpt / "model.safetensors").read_bytes()
    assert old_bytes == new_bytes, "legacy-convention fixture bytes differ from the pre-fold build"

    old_model = pipeline.load_model(old_ckpt)
    new_model = pipeline.load_model(new_ckpt)
    assert set(old_model.weights) == set(new_model.weights)
    for key in old_model.weights:
        assert np.array_equal(np.asarray(old_model.weights[key]), np.asarray(new_model.weights[key])), (
            f"{key}: converted codes differ between the pre-fold build and this round's builder"
        )
    assert old_model.weight_scales == new_model.weight_scales


# ==============================================================================
# Namespace-detection rejections
# ==============================================================================


def test_a_checkpoint_matching_neither_known_convention_is_rejected_by_name(tmp_path):
    """A checkpoint whose tensor names match neither `model.embed_tokens.weight` nor
    `embed_tokens.weight` is a hard, named rejection -- never a silent default to
    either convention."""
    pipeline = require(MODULE)
    ckpt = _build(tmp_path, "neither", prefix=None, biased=False, tie_word_embeddings=True)
    with pytest.raises(pipeline.UnsupportedOpSet, match="neither"):
        pipeline.load_model(ckpt)


def test_a_checkpoint_matching_both_known_conventions_is_rejected_by_name(tmp_path):
    """A checkpoint carrying BOTH anchor tensors is a distinct, named rejection from the
    "matches neither" case -- the map has no way to know which convention the rest of
    the checkpoint follows once both anchors are present, so it does not guess."""
    pipeline = require(MODULE)
    ckpt = _build(tmp_path, "both", prefix="both", biased=False, tie_word_embeddings=True)
    with pytest.raises(pipeline.UnsupportedOpSet, match="both"):
        pipeline.load_model(ckpt)


# ==============================================================================
# CKN-04 -- the lm_head presence gate, both directions
# ==============================================================================


def test_ckn04_accepts_and_consumes_lm_head_when_present_on_an_untied_checkpoint(tmp_path):
    """`tie_word_embeddings=False`, `lm_head_present=True`: the entry is added and the
    real tensor is consumed -- the accept-side twin proving the fix does not merely
    suppress the entry unconditionally once untied."""
    pipeline = require(MODULE)
    ckpt = _build(tmp_path, "lm_present", prefix="model.", biased=False,
                  tie_word_embeddings=False, lm_head_present=True, qk_norm=True)
    model = pipeline.load_model(ckpt)
    assert "lm_head" in model.weights


def test_ckn04_accepts_an_untied_checkpoint_with_no_separate_lm_head(tmp_path):
    """`tie_word_embeddings=False`, `lm_head_present=False`: a legitimate untied-no-
    separate-head architecture is accepted, not rejected with a missing-tensor KeyError."""
    pipeline = require(MODULE)
    ckpt = _build(tmp_path, "lm_absent", prefix="model.", biased=False,
                  tie_word_embeddings=False, lm_head_present=False, qk_norm=True)
    model = pipeline.load_model(ckpt)   # must not raise
    assert "lm_head" not in model.weights


# ==============================================================================
# New in fold round 9 (T-2455) -- QK-norm's own composition constant
# ==============================================================================


def test_qk_norms_composition_constant_is_gain_derived_matching_the_attn_norm_formula(tmp_path):
    """`_derive_composition_constants`'s new `q_norm`/`k_norm` loop applies the identical
    formula the existing `attn_norm`/`mlp_norm` loop uses: `gain_scale = gain_of(...) /
    (1 << NORM_FRAC_BITS)`, `canonical_scale(Fraction(gain_scale) / 127)`. Independently
    recomputed by hand from the fixture's own calibrated `weight_scales`, matching
    byte-for-byte -- mutation-decisive against a reverted or wrong-formula
    implementation, the same discipline
    `test_dynamic_forward_composition.py::test_a_norm_sites_carried_scale_is_gain_derived_not_forwarded`
    already applies to `attn_norm`/`mlp_norm`."""
    from fractions import Fraction

    pipeline = require(MODULE)
    ckpt = _build(tmp_path, "compconst", prefix="", biased=False,
                  tie_word_embeddings=True, qk_norm=True)
    model = pipeline.load_model(ckpt)
    norm_frac_bits = pipeline.NORM_FRAC_BITS
    for leaf in ("q_norm", "k_norm"):
        gain_key = f"layer0.{leaf}.gain"
        gain_of = model.weight_scales[gain_key][0]
        expected = pipeline.canonical_scale(Fraction(gain_of / (1 << norm_frac_bits)) / 127)
        actual = model.composition_constants[f"layer0.{leaf}"]
        assert actual == expected, f"layer0.{leaf}: {actual} != hand-computed {expected}"


def test_qk_norms_composition_constant_gate_is_absent_on_a_pre_ask5_checkpoint(tmp_path):
    """Must-accept: a checkpoint carrying no `q_norm`/`k_norm` tensors (every pre-ask-5
    incumbent) completes `_derive_composition_constants` without exception and adds no
    `q_norm`/`k_norm` entries -- the presence gate does not `KeyError` on an unconditional
    lookup."""
    pipeline = require(MODULE)
    ckpt = _build(tmp_path, "no_qk_norm", prefix="model.", biased=True,
                  tie_word_embeddings=True, qk_norm=False)
    model = pipeline.load_model(ckpt)   # must not raise
    assert "layer0.q_norm" not in model.composition_constants
    assert "layer0.k_norm" not in model.composition_constants


def test_qk_norms_composition_constant_gate_would_keyerror_if_unconditional(tmp_path):
    """Must-reject twin (D-SLM5551): the presence gate's own mutant is "remove the `if
    gain_key in weight_scales` check and look the key up unconditionally." On the
    identical pre-ask-5 fixture the must-accept cell above uses, `weight_scales`
    genuinely carries no `q_norm`/`k_norm` key -- confirmed directly against the real,
    calibrated `model.weight_scales` this build produces, not a hand-built stand-in --
    so that unconditional lookup raises `KeyError`, proving the accept-side cell's own
    "no exception" result depends on the gate actually being present, not on the key
    coincidentally existing."""
    pipeline = require(MODULE)
    ckpt = _build(tmp_path, "no_qk_norm_mutant", prefix="model.", biased=True,
                  tie_word_embeddings=True, qk_norm=False)
    model = pipeline.load_model(ckpt)
    assert "layer0.q_norm.gain" not in model.weight_scales
    with pytest.raises(KeyError):
        model.weight_scales["layer0.q_norm.gain"][0]   # the gate's own mutant: no `if` guard


# ==============================================================================
# Regression: the existing 15-caller contract-caller census (CC-01, T-2455/D-SLM5271)
# ==============================================================================


def test_upstream_names_now_requires_the_present_argument():
    """`_upstream_names`'s signature gained a required second parameter -- calling it
    with only `cfg`, the pre-fold shape, is a `TypeError`, not a silent default. Every
    real call site in this tree is fixed in the same commit set (build log); this cell
    guards the signature itself against a future caller reverting to the one-argument
    form silently."""
    pipeline = require(MODULE)
    cfg = pipeline.ModelConfig(
        hidden_size=8, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=4, intermediate_size=16, vocab_size=48, rope_theta=10000.0,
        rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=256,
    )
    with pytest.raises(TypeError):
        pipeline._upstream_names(cfg)
