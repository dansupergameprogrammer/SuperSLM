"""T-1899 -- Curie's red suite for T-1894 (T-1822 design Sec12 "Option G's
own coverage"): the Python-reference halves of two cells --

  - "Engine/reference bit-parity": the fused K-landing output vs. a
    re-derived `dynamic_engine.py` reference computed at the coordinated
    (post-Option-G, land-after-RoPE) ordering.
  - "Microstep/whole-token parity, Python": `dynamic_engine.py`'s own
    `begin_token`/`decode_step` path and `forward_dynamic_vec` agree on
    which ordering runs for the same loaded model.

WHY THIS FILE IS RED BY IMPORT, NOT BY ASSERTION. `dynamic_engine.py` does
not exist on main@c6cfa03 at all -- verified this session
(`git log --all --oneline -- tests/reference/superslm_spike/dynamic_engine.py`
shows it introduced on the T-1519/T-1520 lineage and present today only on
branches descending from T-1891's own spike work, never merged to `main`).
The design's own Sec31.2's text reads it directly from that path ("Verified
at source this session, tests/reference/superslm_spike/dynamic_engine.py"),
naming it as an existing asset without stating that main lacks it -- porting
or authoring this module for production is T-1894's own build obligation
(the same "engine/tooling-integration tier, gated on infrastructure that
does not exist yet" disposition Claude/Curie/t1832-...-red-suite-test-
design-2026-08-08.md Sec1 already establishes for this suite's C++-side
sibling cells), not a gap in the Coverage Model itself.

This module is therefore authored to the SAME red-cell convention this
suite's C++ side uses for a declared-but-undefined symbol (`RopeApplyPairWide`
et al. in test_main.cpp's own T-1899 section): it imports the real reference
module by its real, final name and location, and is RED BY IMPORT FAILURE
until T-1894's build lands it. `pytest.importorskip` is deliberately NOT used
here -- that would SKIP the cell, not fail it, and a skipped cell reads as
"not yet run" rather than "red for a missing mechanism", the exact
distinction StandardsDocument.md Sec4's structural-enforcement law exists to
preserve (a check that can be silently skipped is not a check).

Test-design record: Claude/Curie/t1899-optionG-red-suite-2026-08-11.md
(records worktree, D:\\Wizard).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.normpath(os.path.join(_THIS_DIR, os.pardir))
_REFERENCE_DIR = os.path.join(_TESTS_DIR, "reference")
_SPIKE_DIR = os.path.join(_REFERENCE_DIR, "superslm_spike")
# T-1894 (build fix): two entries are required, not one -- `import dynamic_engine`
# below resolves against _SPIKE_DIR (the package directory itself), while
# dynamic_engine.py's OWN internal `from superslm_spike import intmath` etc. resolve
# against _REFERENCE_DIR (the package's parent). This file originally inserted
# _TESTS_DIR (tests/) in place of _REFERENCE_DIR (tests/reference/), which cannot
# satisfy `from superslm_spike import ...` at all -- the established, precedented
# two-entry convention this repo already uses (tests/gen_intmath_fixtures.py,
# tests/reference/precompute_pinned.py) inserts the package directory AND its
# parent, never the grandparent. Not caught while dynamic_engine.py itself was
# absent (ModuleNotFoundError: No module named 'dynamic_engine' fired first, before
# this bug could surface); found by execution once the module existed to import.
if _SPIKE_DIR not in sys.path:
    sys.path.insert(0, _SPIKE_DIR)
if _REFERENCE_DIR not in sys.path:
    sys.path.insert(0, _REFERENCE_DIR)


def _import_dynamic_engine():
    """Imports the real production reference module by its real name.

    Deliberately not wrapped in try/except -- an ImportError here IS this
    cell's own red state, and pytest reports it as a collection/test error
    (a hard failure), not a pass. This function exists only so both test
    functions below share one import point and one failure message.
    """
    import dynamic_engine  # noqa: F401 (import-for-its-own-sake; ImportError is the cell)

    return dynamic_engine


def _import_pipeline():
    from superslm_spike import pipeline  # noqa: F401
    return pipeline


def _minimal_two_head_model(pipeline, *, num_hidden_layers=1, option_g_fused_k_landing=False):
    """A self-contained, calibration-corpus-free `QuantizedModel` for T-1894's
    own build-time tests (test_dynamic_engine_microstep_and_whole_token_paths_
    agree_under_flag, below).

    `pipeline.fixture_model` is NOT used here: it calls `_calibrate`, which
    reads `CALIBRATION_CORPUS_PATH`
    (`Claude/Docs/spike/shopkeeper_corpus_v1.jsonl`, resolved relative to this
    package) -- a file that exists in `D:\\Wizard`, never vendored into this
    repo (verified this session: absent anywhere under `D:\\SuperSLM`). This
    helper instead reuses `pipeline`'s own real, corpus-independent steps
    (`_pinned_weights` for deterministic int8 codes, `_derive_scales` +
    `_derive_composition_constants` for the offline composition constants,
    `_build_rope_tables` for the real ROP1-equivalent tables) with an EMPTY
    calibration observation (`_ConstMaxima` below) standing in for a real
    corpus pass -- `_output_scale`'s own documented fallback
    ("never below its inputs") is exactly the mechanism this substitutes for.

    `_pinned_weights`'s own per-channel weight-scale spread (up to 64x per
    tensor) drives the derived attn_residual/mlp_residual multipliers out of
    `RoundingDivideByPOT`'s representable domain ([0,31]) once compounded
    through this fixture's own tiny, uncalibrated scale chain -- confirmed by
    execution this session, not assumed. Flattening every tensor's own
    per-channel scale to a single uniform value (every channel identical)
    removes the channel-scale axis as a compounding variable; a constant,
    generous per-site calibration floor (`_ConstMaxima`, in place of a real
    corpus's actual maxima) keeps every derived site scale in a comparable
    range instead of drifting toward zero through the attention chain's own
    1/127 and 1/2^15 divisions -- also confirmed by execution: without it,
    the SAME domain-overflow fires at `layer0.attn_residual.branch`.
    """
    cfg = pipeline.ModelConfig(
        hidden_size=4,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=2,
        intermediate_size=4,
        vocab_size=4,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        context_cap=8,
    )
    _codes, raw_weight_scales, floats = pipeline._pinned_weights(cfg)
    weight_scales = {
        name: tuple(pipeline._UNIT_INT8_SCALE for _ in per_channel)
        for name, per_channel in raw_weight_scales.items()
    }

    class _ConstMaxima(dict):
        """Stands in for a real corpus's per-site max-abs observations
        (`pipeline._calibrate`'s own return shape) -- every site reads the
        SAME generous floor rather than the near-zero one an empty dict's
        `.get(name, 0.0)` would give through `_output_scale`'s input-product
        fallback alone (see this function's own docstring)."""

        def get(self, key, default=0.0):
            return 100.0

    scales, residual_scales, biases = pipeline._derive_scales(
        cfg, _ConstMaxima(), weight_scales, {})
    composition_constants, kv_landing_scales, kv_landing_reciprocals = (
        pipeline._derive_composition_constants(cfg, weight_scales, scales))

    return pipeline.QuantizedModel(
        config=cfg, scales=scales, weights=_codes, weight_scales=weight_scales,
        residual_scales=residual_scales, rope_tables=pipeline._build_rope_tables(cfg),
        biases=biases, float_source=pipeline._dict_float_source(floats),
        calibration=pipeline.CalibrationRecord(
            corpus_sha256="test-only, no real corpus", tokenization="test", classes=()),
        gemm_weights={}, tokenize_prompt=lambda *_a, **_k: [],
        composition_constants=composition_constants,
        kv_landing_scales=kv_landing_scales,
        kv_landing_reciprocals=kv_landing_reciprocals,
        dynamic_biases={},
        option_g_fused_k_landing=option_g_fused_k_landing,
    )


def test_option_g_fused_k_landing_matches_dynamic_engine_reference():
    """Sec12 "Engine/reference bit-parity" -- Python-reference half.

    Drives `dynamic_engine`'s own real K-landing composite -- the module's
    exported `landing_rescale_vec` (A-8 Sec17.2's own "unit surface") and its
    `_rotate_wide_pair_row` (the reference-side mirror of the engine's
    `RopeApplyPairWide`) -- on the IDENTICAL intermediate values the compiled
    engine's own fixture derivation used (`tests/sslm_t1899_optionG_fixtures.h`'s
    own header comment: `kacc=[127,-64]`, `normed_scale={m=1358184448,e=0}`,
    `kv_landing_r_t_k[0]=DynamicScaleReciprocal(2^30)=4294967296`,
    `kv_landing_e_t_k[0]=0`, the 45-degree table row `cos=sin=759250125`), and
    asserts the result equals the compiled engine's own golden value
    (`test_main.cpp`'s `TestOptionGFusedKLanding_NonNullConfiguration_
    DivergesFromLegacy`, `kOptionGFusedK_Pos1_NonNullConfiguration=[127,57]`).

    Driven at the vec-kernel level rather than through the full
    `forward_dynamic_vec` entry point: reproducing this exact fixture through
    a real embed lookup and the full attn_norm/GEMM/fold chain would require
    engineering an entire model whose derived intermediate values coincide
    with the compiled fixture's own hand-chosen ones -- this way drives the
    SAME production functions `forward_dynamic_vec`'s own K-landing block
    calls (`_rotate_wide_pair_row`, `landing_rescale_vec`) on the identical
    inputs, which is what "the reference's own construction matches the
    engine's construction" actually asks.

    Mutation-pinned two ways, matching the design's own "a cell that only
    moves when both change together is not a parity cell": (1) the NULL
    (identity-rotation) configuration must equal the NON-null configuration's
    own un-rotated landing exactly (both orders coincide at identity, by
    construction); (2) the non-null, fused result must NOT equal the
    un-rotated landing of the SAME `kacc` -- if `_rotate_wide_pair_row` were a
    no-op (the reference-side twin of the engine's own "flag exists, does
    nothing" defect class), this cell would still read as a pass on (1) alone
    but fail (2).
    """
    dynamic_engine = _import_dynamic_engine()

    kacc = [127, -64]
    m_a, e_a = 1358184448, 0
    r_t, e_t = 4294967296, 0
    cos_45 = sin_45 = 759250125
    cos_identity, sin_identity = 1073741824, 0

    def landed(seg):
        return [max(-127, min(127, v)) for v in
                dynamic_engine.landing_rescale_vec(seg, m_a, r_t, e_a, e_t)]

    # Null configuration (identity rotation): fused == un-rotated landing.
    null_rotated = dynamic_engine._rotate_wide_pair_row(kacc, [cos_identity], [sin_identity])
    null_landed = landed(null_rotated)
    unrotated_landed = landed(kacc)
    assert null_landed == unrotated_landed == [127, -81], (
        f"identity-rotation landing must equal the un-rotated landing exactly "
        f"-- got null={null_landed}, unrotated={unrotated_landed}"
    )

    # Non-null (45-degree) configuration: the cross-language parity value.
    non_null_rotated = dynamic_engine._rotate_wide_pair_row(kacc, [cos_45], [sin_45])
    non_null_landed = landed(non_null_rotated)
    assert non_null_landed == [127, 57], (
        f"the Python reference's own fused K-landing at the 45-degree "
        f"configuration must equal the compiled engine's own golden value "
        f"[127, 57] (test_main.cpp's kOptionGFusedK_Pos1_NonNullConfiguration) "
        f"-- got {non_null_landed}"
    )
    # Mutation pin: must genuinely differ from the un-rotated landing of the
    # SAME kacc, or _rotate_wide_pair_row could be a no-op and this cell would
    # not catch it.
    assert non_null_landed != unrotated_landed, (
        "the 45-degree fused result must differ from the un-rotated landing "
        "of the identical kacc -- a build where the reference's own rotation "
        "silently does nothing would still pass here otherwise"
    )


def test_dynamic_engine_microstep_and_whole_token_paths_agree_under_flag():
    """Sec12 "Microstep/whole-token parity, Python".

    Builds two twin models (`_minimal_two_head_model`, above), identical
    except `option_g_fused_k_landing` (True / False), and asserts:

    1. Under EACH model, the resumable path (`begin_token` + `decode_step`,
       `layer_budget=1`) produces the SAME `k_proj.requant` trace codes and
       the SAME final logits as the whole-token path (`forward_dynamic_vec`)
       -- design Sec31.2.3's own text: "both `forward_dynamic_vec` and
       `_forward_dynamic_vec_layers`... read the identical, single source of
       truth" (`model.option_g_fused_k_landing`, not a caller-supplied
       default).
    2. The fused model's own K-landing trace genuinely DIFFERS from the
       unfused model's own -- confirmed by execution this session (the final
       int32 logits alone do not reliably diverge at this fixture's tiny
       scale; a small per-element K-code difference can wash out through the
       softmax/context/o_proj chain before reaching the output, so the
       K-landing trace, not the logits, is this cell's own discriminating
       signal) -- proving the flag is genuinely exercised, not vacuously
       inert for this fixture.

    This is the actual gap T-1894 closes: the spike's own copy of
    `MicroStepState.__init__` never forwarded `option_g_fused_k_landing` to
    `_forward_dynamic_vec_layers` at all, so a resumed decode always ran the
    legacy order regardless of the model. A build that reintroduces that gap
    makes the resumable path's own trace equal the UNFUSED model's trace even
    under the fused model, failing assertion 1 above for the fused model
    specifically.
    """
    dynamic_engine = _import_dynamic_engine()
    pipeline = _import_pipeline()

    model_fused = _minimal_two_head_model(pipeline, option_g_fused_k_landing=True)
    model_unfused = _minimal_two_head_model(pipeline, option_g_fused_k_landing=False)
    tokens = [1, 2, 3]

    def k_trace(trace):
        return [r["codes"] for r in trace if r["site"] == "layer0.k_proj.requant"]

    def whole_token(model):
        trace = []
        logits = dynamic_engine.forward_dynamic_vec(model, tokens, trace=trace)
        return logits, k_trace(trace)

    def microstep(model):
        trace = []
        state = dynamic_engine.begin_token(model, tokens, trace=trace)
        while not dynamic_engine.decode_step(state, layer_budget=1):
            pass
        return state.logits, k_trace(trace)

    whole_logits_fused, whole_k_fused = whole_token(model_fused)
    micro_logits_fused, micro_k_fused = microstep(model_fused)
    whole_logits_unfused, whole_k_unfused = whole_token(model_unfused)
    micro_logits_unfused, micro_k_unfused = microstep(model_unfused)

    assert micro_k_fused == whole_k_fused, (
        f"under the FUSED model, the resumable path's own K-landing trace "
        f"must equal the whole-token path's -- got micro={micro_k_fused}, "
        f"whole={whole_k_fused}"
    )
    assert (np.asarray(micro_logits_fused) == np.asarray(whole_logits_fused)).all(), (
        "under the FUSED model, the resumable path's own final logits must "
        "equal the whole-token path's"
    )
    assert micro_k_unfused == whole_k_unfused, (
        f"under the UNFUSED model, the resumable path's own K-landing trace "
        f"must equal the whole-token path's -- got micro={micro_k_unfused}, "
        f"whole={whole_k_unfused}"
    )
    assert (np.asarray(micro_logits_unfused) == np.asarray(whole_logits_unfused)).all(), (
        "under the UNFUSED model, the resumable path's own final logits must "
        "equal the whole-token path's"
    )
    # The discriminating check: the fused and unfused models' own K-landing
    # traces must genuinely differ -- confirms the flag is exercised at all,
    # not merely self-consistent regardless of its own value.
    assert whole_k_fused != whole_k_unfused, (
        "the fused and unfused models' own K-landing traces must differ -- "
        "identical traces would mean this fixture's own flag has no "
        "observable effect, which would make assertion 1 above pass "
        "vacuously even under the spike's own un-fixed defect (MicroStepState "
        "silently ignoring the model's flag)"
    )
