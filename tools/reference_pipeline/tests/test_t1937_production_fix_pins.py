"""T-1937 -- mutation pins for the three PRODUCTION remedies landed against
`Claude/Poirot/ecc820fd4f-armd-arme-converter-matrix.md` (T-1936's FIX-THEN-SHIP review
of the T-1934 Arm D/E converter build).

Design of record: `Claude/Vitruvius/t1822-activation-scale-remedy-design-2026-08-07.md`
§31.4.1-§31.4.4. Review of record: `Claude/Poirot/ecc820fd4f-armd-arme-converter-matrix.md`
findings 5.3 (Q-scale), 5.4 (refusal gates), 5.5 (NaN classification), D-SLM2624-2626.

This file pins the PRODUCTION-side findings only (T-1937's scope). The three test-author
findings (cells 3/4/5's vacuous sweep, cell 7's label-vs-behavior gap, the presence/absence
naming defect) are a separate, later round's scope and are not touched or pinned here.

Every pin below was verified red-then-green by execution during authoring, per the
same-round pin rule (`Claude/CLAUDE.md`): the reverse mutation was applied in-process,
the suite was run, the failure observed, and the mutation reverted. See
`Claude/Brunel/t1937-production-fix-round-2026-08-11.md` for the quoted executions.
"""

import math

import pytest

from conftest import api, require

MODULE = "reference_pipeline.pipeline"


def fixture_config(pipeline):
    """Identical to T-1933's own `fixture_config` -- two kv_heads, two layers, the same
    §11 fixture shape every sibling test file in this suite uses."""
    return pipeline.ModelConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=32,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=16,
    )


# ==============================================================================
# Finding 5.3 -- the selection metric's Q scale is the production Q path
# (D-SLM2624; §31.4.1's pinned metric, D-SLM2555)
# ==============================================================================

def test_layer_q_scale_uses_the_weight_dominance_floor_not_bare_maxabs():
    """`_layer_q_scale` must compute D-SLM2555's pinned production path
    (`_output_scale`'s dominance floor: `max(activation_term, weight_channel_product *
    headroom)`), not the prior round's bare `max-abs/127` (T-1936 §5.3). A hand-built
    `maxima`/`weight_scales` fixture where the weight-channel product dominates the
    activation term makes the two formulas diverge by >100x -- unmistakable under either
    a real fixture's rounding or a mutation back to the old formula.
    """
    pipeline = require(MODULE)
    maxima = {"layer0.attn_norm.out": 1.0, "layer0.q": 2.0}
    # attn_norm.gain quantized at Q(16) with value 65536 -> norm_in = 1.0.
    # q_proj's own channel scales: max is 10.0, far above maxima["layer0.q"]/127 (~0.0157).
    weight_scales = {
        "layer0.attn_norm.gain": (65536.0,),
        "layer0.q_proj": [10.0, 5.0, 1.0],
    }
    q_scale = pipeline._layer_q_scale(maxima, weight_scales, "layer0")

    naive_bare_maxabs = maxima["layer0.q"] / pipeline._INT8_MAX
    assert q_scale > 100 * naive_bare_maxabs, (
        f"the dominance-floor-driven scale ({q_scale}) must be far above the bare "
        f"max-abs/127 figure ({naive_bare_maxabs}) on a fixture built specifically so "
        f"the weight-channel term dominates -- if this fails, `_layer_q_scale` has "
        f"regressed to the prior round's simplification"
    )
    # The correct value is dominance-floor driven: attn_norm_scale * max(q_proj scales)
    # * headroom, where attn_norm_scale itself is dominance-floor driven by norm_in=1.0.
    expected = 1.0 * pipeline._SCALE_HEADROOM * 10.0 * pipeline._SCALE_HEADROOM
    assert q_scale == pytest.approx(expected, rel=1e-9)


def test_arm_d_q_scale_matches_the_shipped_production_q_path():
    """Arm D/E's own Q-scale (`_layer_q_scale`, run over `_kv_calibration_capture`'s own
    `maxima`) must equal the shipped per-layer path's own `projection_scale` chain
    (`_derive_scales`'s own computation, over `_calibrate`'s own `maxima`), computed
    independently over the identical calibration corpus and weights. D-SLM2555: "the
    metric's definition is 'the production Q path', never hardcoded to a simplified
    stand-in." Both sides now call the identical module-level `_attn_norm_scale`/
    `_projection_scale` functions, so this is an equality, not merely a close reading.
    """
    pipeline = require(MODULE)
    cfg = fixture_config(pipeline)
    weights, weight_scales, floats = pipeline._pinned_weights(cfg)
    float_weight = pipeline._dict_float_source(floats)
    records = pipeline.calibration_records()
    tokenize = pipeline._fixture_tokenize_prompt(cfg)
    record_tokenize = pipeline._bridge_record_tokenizer(tokenize)

    # The shipped per-layer path's own Q scale (Arm A/B's own machinery).
    shipped_maxima = pipeline._calibrate(cfg, float_weight, records, record_tokenize)

    # Arm D/E's own capture-derived Q scale, over the identical corpus and weights.
    capture, capture_maxima = pipeline._kv_calibration_capture(cfg, float_weight, records, record_tokenize)

    # Every layer, not layer0 alone: on this fixture the pre- and post-RoPE observations
    # coincide at layer0 and differ at layer1, so a layer0-only pin cannot fail on the
    # dropped-pre-RoPE defect it exists to catch (T-1939 §5.2).
    for layer in range(cfg.num_hidden_layers):
        prefix = f"layer{layer}"
        shipped_attn_norm_scale = pipeline._attn_norm_scale(shipped_maxima, weight_scales, prefix)
        shipped_q_scale = pipeline._projection_scale(
            shipped_maxima, weight_scales, f"{prefix}.q", f"{prefix}.q_proj", shipped_attn_norm_scale)
        built_q_scale = pipeline._layer_q_scale(capture_maxima, weight_scales, prefix)

        assert built_q_scale == pytest.approx(shipped_q_scale, rel=1e-9), (
            f"{prefix}: Arm D/E's own Q-scale ({built_q_scale}) must equal the shipped production "
            f"`projection_scale` chain ({shipped_q_scale}) -- a divergence here means the "
            f"sweep is quantizing Q against a different scale than the artifact will ship"
        )


# ==============================================================================
# Finding 5.4 -- the two refusal gates are structural, not merely stated
# (D-SLM2625; §31.4.3's own "structural rather than a discipline")
# ==============================================================================

def test_grade_arithmetic_report_refuses_a_duplicate_semantic_key():
    """T-1936 §5.4's own executed demonstration, reproduced against the fixed code:
    two phrasings of `12+15` sharing one `("+", (12, 15))` key must now be REFUSED by
    `grade_arithmetic_report` itself, not silently graded as two independent trials
    (D-SLM2345's own kill, D-SLM2625)."""
    pipeline = require(MODULE)
    problems = [
        {"problem_id": "p1", "prompt": "12+15=", "operation": "+",
         "operand_tuple": (12, 15), "ground_truth": 27},
        {"problem_id": "p2", "prompt": "What is 12 plus 15?", "operation": "+",
         "operand_tuple": (12, 15), "ground_truth": 27},
    ]
    model_answers = {"p1": 27, "p2": 27}
    float_answers = {"p1": 27, "p2": 27}
    with pytest.raises(pipeline.DuplicateArithmeticProblem) as excinfo:
        pipeline.grade_arithmetic_report(problems, model_answers, float_answers)
    assert "('+', (12, 15))" in str(excinfo.value) or "(12, 15)" in str(excinfo.value)


def test_grade_arithmetic_report_still_grades_a_distinct_corpus():
    """Negative companion: a genuinely distinct corpus must not be refused, and the two
    columns (truth-correct, float-agreement) must still be computed independently
    (D-SLM2346) -- the gate refuses collisions, it does not block ordinary grading."""
    pipeline = require(MODULE)
    problems = [
        {"problem_id": "p1", "prompt": "12+15=", "operation": "+",
         "operand_tuple": (12, 15), "ground_truth": 27},
        {"problem_id": "p2", "prompt": "17*23=", "operation": "*",
         "operand_tuple": (17, 23), "ground_truth": 391},
    ]
    model_answers = {"p1": 27, "p2": 400}
    float_answers = {"p1": 27, "p2": 391}
    report = pipeline.grade_arithmetic_report(problems, model_answers, float_answers)
    by_id = {row["problem_id"]: row for row in report}
    assert by_id["p1"]["truth_correct"] is True
    assert by_id["p1"]["float_agreement"] is True
    assert by_id["p2"]["truth_correct"] is False
    assert by_id["p2"]["float_agreement"] is False


def test_calibrate_kv_landing_arm_refuses_when_calibration_and_eval_populations_collide():
    """`calibrate_kv_landing_arm` is the actual capture path Arm D/E's per-head candidate
    sweep runs through (§31.4.1's own calibration population consumer). Given `eval_ids`
    (the grading corpus's own document IDs), a `records` population whose own `"id"`
    collides with `eval_ids` must be refused by this function itself, before any capture
    or sweep work -- D-SLM2557's "structural, not merely stated" requirement, D-SLM2625.

    Records carry a well-formed `"utterance"` (not just an `"id"`) deliberately: the
    refusal must happen because the collision is refused, not because a malformed
    fixture crashes downstream tokenization for an unrelated reason -- a mutation that
    neuters the gate must be caught by "the collision silently passed through," not by
    an accidental `KeyError` that would fail this test for the wrong reason either way.
    """
    pipeline = require(MODULE)
    cfg = fixture_config(pipeline)
    weights, weight_scales, floats = pipeline._pinned_weights(cfg)
    float_weight = pipeline._dict_float_source(floats)
    tokenize = pipeline._fixture_tokenize_prompt(cfg)

    records = [{"id": f"held_out_{i}", "utterance": "test utterance"} for i in range(3)]
    eval_ids = [f"tuned_on_{i}" for i in range(5)]
    eval_ids[2] = records[1]["id"]  # the constructed collision

    with pytest.raises(pipeline.CalibrationPopulationOverlap) as excinfo:
        pipeline.calibrate_kv_landing_arm(
            cfg, float_weight, records, tokenize, arm="D", eval_ids=eval_ids)
    assert records[1]["id"] in str(excinfo.value)


def test_calibrate_kv_landing_arm_proceeds_when_populations_are_disjoint():
    """Negative companion: a genuinely disjoint `eval_ids` must not be refused, and the
    capture must still run and return the ordinary Arm D result shape -- the gate refuses
    collisions, it does not block ordinary calibration, and omitting `eval_ids` entirely
    (every one of T-1933's own frozen §0 call sites) must remain unaffected."""
    pipeline = require(MODULE)
    cfg = fixture_config(pipeline)
    weights, weight_scales, floats = pipeline._pinned_weights(cfg)
    float_weight = pipeline._dict_float_source(floats)
    tokenize = pipeline._fixture_tokenize_prompt(cfg)
    records = [dict(record, id=f"held_out_{i}")
               for i, record in enumerate(pipeline.calibration_records())]
    eval_ids = [f"tuned_on_{i}" for i in range(5)]

    result = pipeline.calibrate_kv_landing_arm(
        cfg, float_weight, records, tokenize, arm="D", eval_ids=eval_ids)
    assert "layer0.k_head0" in result["kv_landing"]

    # Omitting eval_ids entirely -- every T-1933 call site's own shape -- must still work
    # with no check performed (no second population to be disjoint from).
    result_no_eval_ids = pipeline.calibrate_kv_landing_arm(
        cfg, float_weight, records, tokenize, arm="D")
    assert "layer0.k_head0" in result_no_eval_ids["kv_landing"]


# ==============================================================================
# Finding 5.5 -- an undefined ratio refuses rather than reports UNDERPOWERED
# (D-SLM2626; StandardsDocument.md §5.4)
# ==============================================================================

def test_classify_signed_ratio_resolution_refuses_every_nan_input():
    """T-1936 §5.5's own executed demonstration, reproduced against the fixed code: a NaN
    in any of the four arguments must return `"NOT RESOLVED"` (the genuinely-undefined
    label), never `"UNDERPOWERED"` (a measurement conclusion) or `"RESOLVED"`."""
    pipeline = require(MODULE)
    classify = pipeline.classify_signed_ratio_resolution
    nan = float("nan")
    assert classify(candidate=nan, reference=100.0, se=0.5, z_crit=1.96) == "NOT RESOLVED"
    assert classify(candidate=100.0, reference=nan, se=0.5, z_crit=1.96) == "NOT RESOLVED"
    assert classify(candidate=100.0, reference=100.0, se=nan, z_crit=1.96) == "NOT RESOLVED"
    assert classify(candidate=100.0, reference=100.0, se=0.5, z_crit=nan) == "NOT RESOLVED"


def test_classify_signed_ratio_resolution_still_classifies_defined_inputs():
    """Negative companion: the pre-existing, non-NaN behavior is unchanged -- a resolved
    delta still reports RESOLVED, an underpowered one still reports UNDERPOWERED, and a
    non-positive `se`/`z_crit` still reports NOT RESOLVED (T-1890, D-SLM2324-2327)."""
    pipeline = require(MODULE)
    classify = pipeline.classify_signed_ratio_resolution
    assert classify(candidate=100.001, reference=100.0, se=0.0005, z_crit=1.96) == "RESOLVED"
    assert classify(candidate=100.0001, reference=100.0, se=0.5, z_crit=1.96) == "UNDERPOWERED"
    assert classify(candidate=100.0, reference=100.0, se=0.0, z_crit=1.96) == "NOT RESOLVED"
    assert classify(candidate=100.0, reference=100.0, se=0.5, z_crit=0.0) == "NOT RESOLVED"
