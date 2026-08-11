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


def _composed_path_cross_language_model(pipeline, *, option_g_fused_k_landing):
    """The Python-side TWIN of `tests/sslm_t1899_optionG_fixtures.h`'s own
    `OptionGComposedPathFixture` (T-1894 build round 4, T-1901 Significant
    3/D-SLM2420, D-SLM2428) -- hidden_size=2, head_dim=2, one layer, one
    (kv-)head, identity 2x2 q/k/v/o/gate/up/down projections, embed weight
    [127,-64] for token 0 (vocab_size=1), gain=[16384,16384], rope table row
    0 identity / row 1 the 45-degree entry. Every composition constant is
    IDENTICAL to the C++ fixture's own (transcribed from this same
    derivation, executed once, both sides read the same numbers) -- see
    `tests/sslm_t1899_optionG_fixtures.h`'s own header comment for the full
    derivation trail, including why the K/V landing reciprocal/exponent is
    overridden to a finer pair after `_derive_composition_constants`'s own
    auto-derivation (the auto-derived one was too coarse to discriminate
    fused from legacy at this kacc magnitude -- found by execution).
    """
    cfg = pipeline.ModelConfig(
        hidden_size=2, num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
        head_dim=2, intermediate_size=2, vocab_size=1, rope_theta=10000.0, rms_norm_eps=1e-6,
        tie_word_embeddings=True, context_cap=2,
    )
    identity2x2 = np.array([[1, 0], [0, 1]], dtype=np.int8)
    weights = {
        "embed": np.array([[127, -64]], dtype=np.int8),
        "layer0.attn_norm.gain": np.array([16384, 16384], dtype=np.int32),
        "layer0.q_proj": identity2x2,
        "layer0.k_proj": identity2x2,
        "layer0.v_proj": identity2x2,
        "layer0.o_proj": identity2x2,
        "layer0.mlp_norm.gain": np.array([16384, 16384], dtype=np.int32),
        "layer0.gate_proj": identity2x2,
        "layer0.up_proj": identity2x2,
        "layer0.down_proj": identity2x2,
        "final_norm.gain": np.array([16384, 16384], dtype=np.int32),
    }
    weight_scales = {name: (1.0 / 127.0,) * (w.shape[0] if w.ndim == 2 else 1)
                     for name, w in weights.items()}

    class _ConstMaxima(dict):
        def get(self, key, default=0.0):
            return 100.0

    scales, residual_scales, biases = pipeline._derive_scales(cfg, _ConstMaxima(), weight_scales, {})
    composition_constants, kv_landing_scales, kv_landing_reciprocals = (
        pipeline._derive_composition_constants(cfg, weight_scales, scales))
    # Finer landing scale -- see this function's own docstring.
    r_t_fine = pipeline.intmath.dynamic_scale_reciprocal(1073741824)
    kv_landing_reciprocals = dict(kv_landing_reciprocals)
    kv_landing_reciprocals["layer0.k_head0"] = (1073741824, -30, r_t_fine)
    kv_landing_reciprocals["layer0.v_head0"] = (1073741824, -30, r_t_fine)

    model = pipeline.QuantizedModel(
        config=cfg, scales=scales, weights=weights, weight_scales=weight_scales,
        residual_scales=residual_scales,
        rope_tables=(np.zeros((2, 1), dtype=np.int64), np.zeros((2, 1), dtype=np.int64)),
        biases=biases, float_source=lambda name: None,
        calibration=pipeline.CalibrationRecord(corpus_sha256="test-only", tokenization="test", classes=()),
        gemm_weights={}, tokenize_prompt=lambda *_a, **_k: [],
        composition_constants=composition_constants,
        kv_landing_scales=kv_landing_scales,
        kv_landing_reciprocals=kv_landing_reciprocals,
        dynamic_biases={},
        option_g_fused_k_landing=option_g_fused_k_landing,
    )
    cos_flat = np.array([[1073741824], [759250125]], dtype=np.int64)  # row0 identity, row1 45deg
    sin_flat = np.array([[0], [759250125]], dtype=np.int64)
    return pipeline.with_rope_tables(model, (cos_flat, sin_flat))


def test_option_g_fused_k_landing_matches_dynamic_engine_reference():
    """Sec12 "Engine/reference bit-parity" -- Python-reference half.
    RESPECIFIED 2026-08-11 round 4 (T-1901 Significant 3/D-SLM2420,
    D-SLM2428): the round-3 version of this cell composed
    `_rotate_wide_pair_row` and `landing_rescale_vec` directly in the test
    body, so reverting the real ordering inside `_forward_dynamic_vec_layers`
    -- the design's own required half of the mutation pin -- left it green
    (T-1901's own finding, confirmed by this session's own reproduction, see
    this file's own build log).

    This cell instead builds `_composed_path_cross_language_model`
    (the Python twin of `tests/sslm_t1899_optionG_fixtures.h`'s own
    `OptionGComposedPathFixture`) and drives it through
    `dynamic_engine.forward_dynamic_vec` -- `dynamic_engine`'s own composed
    path, never a hand-assembled call to the two kernels -- asserting the
    resulting `k_proj.requant` trace codes equal the compiled engine's own
    golden values (`tests/sslm_t1899_optionG_fixtures.h`'s
    `kOptionGComposedFusedK_Pos1=[127,58]`, execution-derived and
    cross-checked by `TestOptionGSelectionDispatch_EndToEndProductionPath`,
    test_main.cpp).

    Mutation-pinned by construction: the assertions below read the K-landing
    codes from the TRACE `forward_dynamic_vec` itself produces while running
    `_forward_dynamic_vec_layers`'s own real `if fused_k_landing: seg =
    _rotate_wide_pair_row(...)` branch (`dynamic_engine.py`) -- a revert of
    that ordering changes what value lands in the trace, which this cell
    reads directly, not a value this test's own body independently computed.
    Verified by execution this session (not merely reasoned): reverting the
    rotate-before-land order to land-before-rotate inside
    `_forward_dynamic_vec_layers` on a scratch copy makes the fused-model
    trace read [127,-82] (the un-rotated landing) instead of [127,58] at
    token_index=1, failing this cell's own assertion -- see the build log's
    own record of this executed demonstration.
    """
    dynamic_engine = _import_dynamic_engine()
    pipeline = _import_pipeline()

    model = _composed_path_cross_language_model(pipeline, option_g_fused_k_landing=True)
    trace = []
    dynamic_engine.forward_dynamic_vec(model, [0, 0], trace=trace)

    k_trace = {r["token_index"]: r["codes"] for r in trace if r["site"] == "layer0.k_proj.requant"}

    assert k_trace.get(0) == (127, -82), (
        f"the NULL (identity-rotation, token_index=0) configuration must land kacc=[127,-64] "
        f"unrotated -- got {k_trace.get(0)}, want (127, -82)"
    )
    assert k_trace.get(1) == (127, 58), (
        f"the fused K-landing at the 45-degree (token_index=1) configuration, computed through "
        f"dynamic_engine's own composed path (_forward_dynamic_vec_layers), must equal the "
        f"compiled engine's own golden value (127, 58) -- got {k_trace.get(1)}"
    )
    # Mutation pin: the non-null, fused result must differ from the null
    # configuration's own landing -- if the composed path's own rotation
    # were a no-op (or if the ordering reverted to land-then-rotate, which
    # for THIS kacc/rope-table pair reduces to the un-rotated value at every
    # position), this assertion is what catches it.
    assert k_trace.get(1) != k_trace.get(0), (
        "the 45-degree fused result must differ from the null-configuration landing of the "
        "identical kacc -- a build where the composed path's own rotation silently does "
        "nothing (or reverts to land-then-rotate) would still pass the two assertions above "
        "only by coincidence; this is the discriminating check"
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
