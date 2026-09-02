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

import hashlib
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
    hand from the fixture's own float values THROUGH the stated RoPE-pair permutation, then
    peak-scaled int8, match the converter's own emitted codes byte-for-byte -- per the
    acceptance text's own oracle, and per T-2543 S-2, with a fixture gain that actually
    discriminates the permutation branch: the fixture's own `[0.1, 0.2, 0.3, 0.4]` (q_norm)
    and `[0.4, 0.3, 0.2, 0.1]` (k_norm) are each strictly non-uniform, so reordering their
    elements changes which value lands in which int8 code slot -- unlike an all-ones gain,
    invariant under any permutation of itself (Poirot
    2a46a85-t2540-ask5-trackc-review.md S-2).

    This cell is itself the must-fire proof S-2 asked for: the UNPERMUTED hand codes,
    computed below from the identical raw float values with no reordering applied, are
    asserted DIFFERENT from the converter's real output -- so a revert of
    `_permuted_if_rope`'s own q_norm.gain/k_norm.gain branch (which would make the
    converter emit the unpermuted codes instead) is exactly the change this assertion
    would catch, without needing a separate mutant run."""
    pipeline = require(MODULE)
    ckpt = _build(tmp_path, "mechanism_wgt1", prefix="", biased=False,
                  tie_word_embeddings=True, qk_norm=True)
    model = pipeline.load_model(ckpt)
    head_dim = model.config.head_dim
    assert head_dim == 4, "the hand-derived permutation order below is pinned to head_dim=4"

    # Built as float32 first, matching exactly what `_parameterized_tensors` writes to disk
    # (safetensors F32) and what `_SafeTensors.tensor()` reads back (an EXACT float32->
    # float64 widening, per that reader's own docstring -- no rounding on the promotion
    # itself), so this hand computation carries no extra float64-literal rounding error
    # against the real on-disk bytes.
    raw = {
        "q_norm": np.array([0.1 * (i + 1) for i in range(head_dim)],
                            dtype=np.float32).astype(np.float64),
        "k_norm": np.array([0.1 * (head_dim - i) for i in range(head_dim)],
                            dtype=np.float32).astype(np.float64),
    }
    # The stated permutation (pipeline._rope_pair_permutation(head_dim), head_dim=4):
    # order[0::2] = arange(half); order[1::2] = arange(half, head_dim) -> [0, 2, 1, 3].
    order = np.array([0, 2, 1, 3])

    for leaf in ("q_norm", "k_norm"):
        key = f"layer0.{leaf}.gain"
        values = raw[leaf]
        permuted = values[order]
        peak = float(np.abs(permuted).max())
        hand_scale = peak / 127.0
        hand_codes = np.clip(np.round(permuted / hand_scale), -128, 127).astype(np.int8)

        unpermuted_peak = float(np.abs(values).max())
        unpermuted_scale = unpermuted_peak / 127.0
        unpermuted_codes = np.clip(
            np.round(values / unpermuted_scale), -128, 127).astype(np.int8)

        codes = np.asarray(model.weights[key])
        scale = model.weight_scales[key][0]
        assert np.array_equal(codes, hand_codes), (
            f"{key}: codes {codes} != permuted hand recomputation {hand_codes}")
        assert abs(scale - hand_scale) < 1e-15, f"{key}: scale {scale} != hand {hand_scale}"
        assert not np.array_equal(codes, unpermuted_codes), (
            f"{key}: converter output equals the UNPERMUTED hand codes {unpermuted_codes} -- "
            f"this fixture gain does not discriminate the permutation branch"
        )


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


def _hash_value(v):
    """A stable sha256 over one model field's own content -- arrays hashed by their raw
    bytes, dicts/lists by their sorted-key/ordered composition, everything else by repr.
    Used only to compress a whole QuantizedModel group into one comparable digest; not a
    numeric or cryptographic claim about the model itself."""
    if isinstance(v, dict):
        h = hashlib.sha256()
        for key in sorted(v.keys(), key=str):
            h.update(str(key).encode())
            h.update(_hash_value(v[key]))
        return h.digest()
    if isinstance(v, (list, tuple)):
        h = hashlib.sha256()
        for item in v:
            h.update(_hash_value(item))
        return h.digest()
    if isinstance(v, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(v).tobytes()).digest()
    if hasattr(v, "tobytes"):
        return hashlib.sha256(np.ascontiguousarray(v).tobytes()).digest()
    return hashlib.sha256(repr(v).encode()).digest()


def _hash_group(group):
    return hashlib.sha256(_hash_value(group)).hexdigest()


# T-2543 M-2: pinned against the base engine (`fdd4739`, this branch's own pre-fold tip),
# computed once by loading that commit's own `pipeline.py` standalone (its unchanged
# sibling modules -- intmath/rope/constrain -- resolved normally) and running it against
# the identical legacy fixture `build_fixture_checkpoint` still builds today. The prior
# form of this cell (above) compared the TIP against itself under two different builders,
# which cannot answer the brief's own exit condition ("converts byte-identically before
# AND after") because the reference was produced by the artifact under test (Poirot
# 2a46a85-t2540-ask5-trackc-review.md M-2). These are golden values, not derived at test
# time -- a change to any of these eleven groups' own content for the unchanged legacy
# fixture is exactly the regression this pin exists to catch.
_BASE_ENGINE_GOLDEN_HASHES = {
    "weights": "50f12149241434d15825c64495d5a5ebf1a5c1e66e10fbca2df6b19009e8ac8f",
    "weight_scales": "164ef153488bd10d39ceb25cc7ec2deedbe1d3393279dbef4867622f8f43136f",
    "composition_constants": "7adfdf093298bd5e26ed950a8037196ae5c66f3854d0d80b93e75e359b555041",
    "scales": "352fb2f285c8e666d340e73586e3a909543ad8ea383ad3dcd16f2d175adbf5f7",
    "residual_scales": "7afd4b346692f91e4e744df0fbad773200b6b86ba1d98e4780656f042965a40d",
    "biases": "f285db8c86cd3b66b1d99fce37ed7a7a276f783bb8d10274a10c06c345fc4be7",
    "dynamic_biases": "1f8474413794b2f8ed9355f7ba1a1c977b6c1e2a31d9b6f8ba0d2c765ed666fc",
    "kv_landing_scales": "b611b6dfe6cb4ad256a6ac8bddb12ae1eb60f10477780aeda578ddbe41efba1d",
    "kv_landing_reciprocals": "a400467118aa7b9f0512aa6513d5a2748a9b643ff1ff75976da0a9f3b1b6d537",
    "rope_tables": "fb5ca6c71c517b5395a7e8c218f90ece5c9de4642f12ec89a0ccbdb981c3306d",
    "calibration": "31d5ebbe3faba788f000bff7d0fff8dfc467e930e53f6b895c3a5502297cf4c8",
}


def test_the_legacy_fixture_converts_identically_to_the_pinned_base_engine_golden(tmp_path):
    """T-2543 M-2, the exit condition itself: the SAME, unchanged legacy fixture
    (`build_fixture_checkpoint`, untouched by this round) converted by THIS tip's own
    `pipeline.load_model` is compared group by group against `_BASE_ENGINE_GOLDEN_HASHES`
    -- values computed once from `fdd4739`'s own pipeline.py, not from this round's code.
    Every one of the eleven groups the brief's own comparison names (weights,
    weight_scales, composition_constants, scales, residual_scales, biases,
    dynamic_biases, kv_landing_scales, kv_landing_reciprocals, rope_tables, calibration)
    is included -- the cell above this one keeps the on-disk-bytes and tip-vs-tip checks,
    which are real properties in their own right; this cell is the base-vs-tip proof
    those cannot substitute for."""
    fixture_mod = _fixture_builder()
    pipeline = require(MODULE)
    ckpt = fixture_mod.build_fixture_checkpoint(tmp_path / "golden_legacy")
    model = pipeline.load_model(ckpt)

    mismatched = []
    for group_name, golden_hash in _BASE_ENGINE_GOLDEN_HASHES.items():
        tip_hash = _hash_group(getattr(model, group_name))
        if tip_hash != golden_hash:
            mismatched.append((group_name, golden_hash, tip_hash))
    assert not mismatched, (
        f"{len(mismatched)} of {len(_BASE_ENGINE_GOLDEN_HASHES)} groups diverged from the "
        f"base-engine (fdd4739) golden: {mismatched}"
    )


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


def test_a_checkpoint_carrying_q_norm_without_k_norm_is_rejected_by_name(tmp_path):
    """T-2543 C-1 (Poirot 2a46a85-t2540-ask5-trackc-review.md S-3(c)): design Sec4
    (`t2408` #4-gating) makes asymmetric QK-norm presence "a defined rejection, not two
    independent null checks." Before this fix, a checkpoint carrying `q_norm.weight` for
    a layer without `k_norm.weight` converted silently, emitting the one gain and its
    composition constant with no counterpart. `_upstream_names` now rejects it by name at
    the point it already reads the checkpoint's own key set -- earlier than, and
    independent of, the C++ layer marshal's own (unbuilt, Track B) enforcement of the
    identical invariant at load time."""
    pipeline = require(MODULE)
    ckpt = _build(tmp_path, "q_only", prefix="model.", biased=False,
                  tie_word_embeddings=True, qk_norm="q_only")
    with pytest.raises(pipeline.UnsupportedOpSet, match="asymmetric"):
        pipeline.load_model(ckpt)


def test_a_checkpoint_carrying_k_norm_without_q_norm_is_rejected_by_name(tmp_path):
    """The mirror of the cell above -- `k_norm` present, `q_norm` absent, proving the
    check is symmetric in which tensor is missing, not only which is present."""
    pipeline = require(MODULE)
    ckpt = _build(tmp_path, "k_only", prefix="model.", biased=False,
                  tie_word_embeddings=True, qk_norm="k_only")
    with pytest.raises(pipeline.UnsupportedOpSet, match="asymmetric"):
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
    """`tie_word_embeddings=False`, `lm_head_present=False`: `load_model` accepts, per the
    frozen design's own accept-side cell (`t2408` Sec6 Track C, fold round 3, CKN-04's
    second acceptance cell) -- not rejected with a missing-tensor KeyError AT LOAD TIME.

    **T-2543 S-3 correction.** This is NOT "a legitimate untied-no-separate-head
    architecture", as this cell's own docstring claimed before this round: no real
    checkpoint is known to take this shape (HuggingFace's own `AutoModelForCausalLM`
    always materializes a separate `lm_head` when embeddings are untied), and the
    pipeline's OWN downstream consumers of the head (`forward_float_reference`,
    `lm_head_weight`) both raise `KeyError('lm_head')` on a model built this way -- the
    rejection did not go away, it moved from a named message here to a bare KeyError at
    the first forward (Poirot 2a46a85-t2540-ask5-trackc-review.md S-3(a)). This cell pins
    the converter-level accept the frozen design specifies, honestly: a load-time accept
    of a hypothetical shape, not a runnable architecture."""
    pipeline = require(MODULE)
    ckpt = _build(tmp_path, "lm_absent", prefix="model.", biased=False,
                  tie_word_embeddings=False, lm_head_present=False, qk_norm=True)
    model = pipeline.load_model(ckpt)   # must not raise
    assert "lm_head" not in model.weights
    # The degradation this cell pins: the head's own float source raises where load_model
    # did not -- confirmed directly against the real, live model rather than merely
    # asserted, so a future change that makes this shape genuinely runnable (and this
    # docstring stale) is caught here too.
    with pytest.raises(KeyError):
        model.float_source("lm_head")


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


class _AlwaysContains(dict):
    """A dict that reports EVERY key as present (`__contains__` always True) while a
    genuine lookup of a key it does not actually hold still raises `KeyError` -- exactly
    the behavior `weight_scales` would have inside `_derive_composition_constants` if
    its own presence gate (`if gain_key in weight_scales:`) were removed and the lookup
    ran unconditionally. Wrapping the REAL, calibrated `weight_scales` this way and
    calling the REAL, unmutated `_derive_composition_constants` against it runs the
    actual production function under a simulated mutant environment, rather than
    re-implementing the mutant's own logic by hand (T-2543 M-6: the prior form of this
    cell asserted the same absence twice and never ran the mutant at all)."""

    def __contains__(self, key):
        return True


def test_qk_norms_composition_constant_gate_would_keyerror_if_unconditional(tmp_path):
    """Must-reject twin (D-SLM5551): the presence gate's own mutant is "remove the `if
    gain_key in weight_scales` check and look the key up unconditionally." Confirmed by
    actually running that mutant's own effective code path (via `_AlwaysContains`, above)
    against the REAL `_derive_composition_constants`, on the identical pre-ask-5 fixture
    the must-accept cell above uses: `weight_scales` genuinely carries no
    `q_norm`/`k_norm` key, so the neutralized-gate call raises `KeyError`, proving the
    accept-side cell's own "no exception" result depends on the gate actually running,
    not on the key coincidentally existing."""
    pipeline = require(MODULE)
    ckpt = _build(tmp_path, "no_qk_norm_mutant", prefix="model.", biased=True,
                  tie_word_embeddings=True, qk_norm=False)
    model = pipeline.load_model(ckpt)
    assert "layer0.q_norm.gain" not in model.weight_scales   # the must-accept cell's own premise

    # Confirm the REAL function, unmutated, against the REAL weight_scales: no exception
    # (this is the must-accept cell's own claim, re-confirmed here as the control).
    pipeline._derive_composition_constants(model.config, model.weight_scales, model.scales)

    # Now the simulated mutant: the SAME real function, the SAME real weight_scales
    # content, wrapped so the presence gate always reports "present."
    mutant_weight_scales = _AlwaysContains(model.weight_scales)
    with pytest.raises(KeyError):
        pipeline._derive_composition_constants(model.config, mutant_weight_scales, model.scales)


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
