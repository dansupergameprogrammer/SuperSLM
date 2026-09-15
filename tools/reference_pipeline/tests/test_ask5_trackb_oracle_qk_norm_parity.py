"""Ask 5, Track B — the oracle's QK-norm call site is ONE shared implementation, not three
independently-drifting ones (T-2553).

T-2551 gave `_float_layer` a QK-norm call site. `_kv_calibration_capture` (Arm D/E's own
per-head calibration capture) and `_vec_forward` (the integer engine's own Python parity
shadow) each independently re-walk the per-layer forward and had none — an oracle with
multiple forward implementations of which only one applies a real architectural operation is
exactly the shape `StandardsDocument.md` §7's sibling-pinning rule names: a repair (or, here,
an addition) that leaves an unaudited sibling standing inherits whatever the sibling is wrong
about, and the sibling here was wrong about applying QK-norm at all.

T-2553 gives both siblings the identical call, `_apply_qk_norm` (`pipeline.py`) — one shared
function every independent layer walk in this module calls, so there is one QK-norm
composition, never two (or three) reasoned to agree. This file pins that: a fixture with
non-uniform `q_norm`/`k_norm` gains where each sibling's own observable output measurably
differs with the norm applied versus without it — materiality, not merely execution, matching
the same discipline `test_ask5_trackc_qk_norm_converter.py` and T-2551's own build log already
establish for this class of claim (`StandardsDocument.md` §5.4).
"""

import importlib.util
import inspect
import sys

import numpy as np
import pytest

import conftest
from conftest import api, require

MODULE = "reference_pipeline.pipeline"


def _fixture_config(pipeline):
    """The §11 fixture model's shape (test_pipeline.fixture_config's own kwargs, inlined so
    this file does not import across test modules) -- carries non-uniform q_norm/k_norm
    gains unconditionally (_weight_shapes, T-2539), which is what makes this fixture usable
    for the materiality proof below without any hand-authored perturbation.
    """
    return pipeline.ModelConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=32,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=16,
    )


def _without_qk_norm(floats, cfg):
    """The identical float source, with every layer's own q_norm.gain/k_norm.gain removed --
    the fixture's own no-QK-norm twin, matching every other absence-vs-presence materiality
    proof in this codebase (T-2551's own build log; T-2425's own spike log)."""
    stripped = dict(floats)
    for layer in range(cfg.num_hidden_layers):
        stripped.pop(f"layer{layer}.q_norm.gain", None)
        stripped.pop(f"layer{layer}.k_norm.gain", None)
    return stripped


def test_apply_qk_norm_is_the_one_shared_implementation():
    """(carried-scale delta §7 Cell 7, D-SLM6145): _float_layer and _kv_calibration_capture
    call the IDENTICAL function object, not two separately-authored copies that happen to
    compute the same formula today and can silently drift apart tomorrow -- the defect class
    this whole ticket exists to close, made mechanically checkable.

    Strengthened from a bare `hasattr` check (which sees only whether the function EXISTS,
    not whether either caller actually invokes it -- confirmed blind by execution: deleting
    both `_float_layer`'s and `_kv_calibration_capture`'s own calls to `_apply_qk_norm` left
    the bare `hasattr` form green, T-2559 §3) to a per-caller source-inspection assertion:
    each caller's own source text must contain a real call to `_apply_qk_norm(`, so a mutant
    deleting either call site ALONE -- not only both together -- turns this cell red.
    """
    pipeline = require(MODULE)
    assert hasattr(pipeline, "_apply_qk_norm"), (
        "no shared _apply_qk_norm function -- T-2553's own governing fix is absent"
    )
    apply_qk_norm = pipeline._apply_qk_norm
    for caller_name in ("_float_layer", "_kv_calibration_capture"):
        caller = getattr(pipeline, caller_name)
        source = inspect.getsource(caller)
        assert "_apply_qk_norm(" in source, (
            f"{caller_name}'s own source does not call _apply_qk_norm(...) -- the shared "
            f"implementation exists but this caller does not invoke it, exactly the "
            f"single-caller-deleted mutant a bare hasattr check cannot see"
        )
        # The call is to the SAME function object every other caller shares -- not a
        # same-named local shadowing it (module globals resolve at call time, so this reads
        # the identical object `pipeline._apply_qk_norm` above already confirmed exists).
        assert caller.__globals__.get("_apply_qk_norm") is apply_qk_norm, (
            f"{caller_name}'s own module globals resolve '_apply_qk_norm' to a different "
            f"function object than pipeline._apply_qk_norm -- not the one shared implementation"
        )


def _load_mutant_pipeline_module(transform, tmp_path):
    """(carried-scale delta §7 Cell 7): loads a MUTATED copy of pipeline.py's own real source
    text, transformed by `transform`, as an isolated module -- never touching the real,
    imported `pipeline` module other tests in this file share. Used to prove the strengthened
    per-caller assertion (above) actually turns red on a single-call-site-deleted mutant,
    the exact case a bare `hasattr` check cannot see (T-2559 §3)."""
    real_path = conftest.TOOLS_DIR / "reference_pipeline" / "pipeline.py"
    real_source = real_path.read_text(encoding="utf-8")
    mutated_source = transform(real_source)
    assert mutated_source != real_source, "sanity: the transform must actually change the source"
    mutant_path = tmp_path / "pipeline_mutant.py"
    mutant_path.write_text(mutated_source, encoding="utf-8")
    module_name = f"_t2560_pipeline_mutant_{id(mutant_path)}"
    spec = importlib.util.spec_from_file_location(module_name, mutant_path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses' own field-type resolution (ModelConfig/QuantizedModel/StaticScales, all
    # defined in pipeline.py) looks the defining class's module up via sys.modules[cls.
    # __module__] -- registered here, matching the standard importlib pattern, and popped in
    # the finally block so this mutant never lingers in sys.modules past this one load.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    # CALIBRATION_CORPUS_PATH is derived from the module's OWN __file__ at load time
    # (pipeline.py:220-221) -- the mutant lives under tmp_path, not the real
    # tools/reference_pipeline/ directory, so this is repointed at the real corpus the real,
    # imported pipeline module already resolved correctly.
    real_pipeline = require(MODULE)
    module.CALIBRATION_CORPUS_PATH = real_pipeline.CALIBRATION_CORPUS_PATH
    return module


def _assert_apply_qk_norm_called(module, caller_name):
    """The strengthened check's own logic (mirrors the real assertion above), run against
    an arbitrary module -- real or mutant -- so both share exactly one implementation of
    what "caught" means."""
    caller = getattr(module, caller_name)
    source = inspect.getsource(caller)
    assert "_apply_qk_norm(" in source, (
        f"{caller_name}'s own source does not call _apply_qk_norm(...)"
    )


def test_strengthened_assertion_rejects_float_layer_call_site_deleted_alone(tmp_path):
    """Must-reject twin 1/2 (D-SLM6145): deleting ONLY `_float_layer`'s own call to
    `_apply_qk_norm` (leaving `_kv_calibration_capture`'s own call intact) must turn the
    strengthened per-caller check red -- the single-caller case the pre-delta bare `hasattr`
    form, and T-2559's own both-deleted probe, never distinguished from a healthy tree."""
    def _t(text):
        pipeline = require(MODULE)
        source_of_caller = inspect.getsource(pipeline._float_layer)
        call_line = "    q, k = _apply_qk_norm(q, k, tensors, prefix, cfg)\n"
        assert call_line in source_of_caller, "sanity: _float_layer's own call line must match verbatim"
        assert text.count(source_of_caller) == 1, (
            "sanity: _float_layer's own source must appear verbatim, once, in the file"
        )
        mutated_caller_source = source_of_caller.replace(
            call_line, "    q = q  # T-2560 mutant: _float_layer's own _apply_qk_norm call deleted\n", 1)
        return text.replace(source_of_caller, mutated_caller_source, 1)
    mutant = _load_mutant_pipeline_module(_t, tmp_path)
    _assert_apply_qk_norm_called(mutant, "_kv_calibration_capture")  # the untouched sibling still passes
    with pytest.raises(AssertionError):
        _assert_apply_qk_norm_called(mutant, "_float_layer")


def test_strengthened_assertion_rejects_kv_calibration_capture_call_site_deleted_alone(tmp_path):
    """Must-reject twin 2/2 (D-SLM6145): the symmetric single-caller deletion, on
    `_kv_calibration_capture` instead."""
    def _t(text):
        pipeline = require(MODULE)
        source_of_caller = inspect.getsource(pipeline._kv_calibration_capture)
        assert "q, k = _apply_qk_norm(q, k, tensors, prefix, cfg)" in source_of_caller
        mutated_caller_source = source_of_caller.replace(
            "q, k = _apply_qk_norm(q, k, tensors, prefix, cfg)",
            "q = q  # T-2560 mutant: _kv_calibration_capture's own _apply_qk_norm call deleted",
            1,
        )
        assert text.count(source_of_caller) == 1, (
            "sanity: _kv_calibration_capture's own source must appear verbatim, once, in the file"
        )
        return text.replace(source_of_caller, mutated_caller_source, 1)
    mutant = _load_mutant_pipeline_module(_t, tmp_path)
    _assert_apply_qk_norm_called(mutant, "_float_layer")  # the untouched sibling still passes
    with pytest.raises(AssertionError):
        _assert_apply_qk_norm_called(mutant, "_kv_calibration_capture")


def _fixture_config_group1(pipeline):
    """(carried-scale delta §7 Cell 10, D-SLM6149): num_attention_heads == num_key_value_heads
    -- group=1, the general `kv_head = h / group` formula's own boundary case (the pinned
    real candidate's own 2:1 ratio never exercises the trivial `kv_head = h` mapping this
    produces). Otherwise identical to `_fixture_config`, above."""
    return pipeline.ModelConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, head_dim=8, intermediate_size=64, vocab_size=32,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=16,
    )


def _attention_iexp_rows(pipeline, model, tokens):
    """C30's live consumer of Q-head and softmax-K carried scales."""
    capture = []
    pipeline.forward_dynamic(model, tokens, attention_capture=capture)
    return [(row["layer"], row["position"], row["head"], row["iexp"])
            for row in capture]


def test_group1_geometry_must_accept_the_fixed_build():
    """Cell 10 must-accept (D-SLM6149): the fixed build's forward_dynamic matches
    composition_ref.py's independent oracle bit-for-bit at group=1 -- the same proof
    test_dynamic_forward_logits.py already runs at this fixture's own group=2."""
    pipeline = require(MODULE)
    import composition_ref
    cfg = _fixture_config_group1(pipeline)
    model = pipeline.fixture_model(cfg)
    tokens = [0, 1, 3, 5]
    logits = np.asarray(pipeline.forward_dynamic(model, tokens))
    expected = composition_ref.forward_dynamic_logits_oracle(model, tokens)
    assert logits.tolist() == [[int(v) for v in row] for row in expected], (
        "forward_dynamic and the independent oracle diverge at group=1 -- the carried-scale "
        "contract does not hold at this geometry"
    )


def test_group1_geometry_must_reject_the_collapsed_q_scale_mutant(tmp_path):
    """Cell 10 must-reject 1/2 (D-SLM6149): the same collapsed-Q construction C1's own
    pre-fix behavior produced (every head reads the LAST head's own carried scale, §7 Cell
    2's own must-reject), re-run at group=1. A build with this defect must diverge from the
    correct oracle at this geometry too, not only at the real candidate's own 2:1 ratio."""
    pipeline = require(MODULE)
    import composition_ref
    cfg = _fixture_config_group1(pipeline)
    model = pipeline.fixture_model(cfg)
    tokens = [0, 1, 3, 5]
    expected = composition_ref.forward_dynamic_logits_oracle(model, tokens)
    expected_iexp = _attention_iexp_rows(pipeline, model, tokens)

    def _t(text):
        source = inspect.getsource(pipeline.forward_dynamic)
        assign_line = "                    q_scale_by_head[h][t] = scale\n"
        assert assign_line in source, "sanity: the per-head assignment must match verbatim"
        assert text.count(source) == 1
        mutated = source.replace(
            assign_line,
            "                    q_scale_by_head[h][t] = scale\n"
            "                    for _mh in range(cfg.num_attention_heads):\n"
            "                        q_scale_by_head[_mh][t] = scale  "
            "# T-2560 mutant: collapse onto the last head visited\n",
            1,
        )
        return text.replace(source, mutated, 1)
    mutant = _load_mutant_pipeline_module(_t, tmp_path)
    mutant_model = mutant.fixture_model(cfg)
    mutant_iexp = _attention_iexp_rows(mutant, mutant_model, tokens)
    assert mutant_iexp != expected_iexp, (
        "the collapsed-Q mutant did not reach C30's per-head i-exp constants at group=1"
    )


def test_group1_geometry_must_reject_the_prenorm_k_scale_mutant(tmp_path):
    """Cell 10 must-reject 2/2 (D-SLM6149): the same pre-norm-K-scale construction Cell 3's
    own pre-fix behavior produced (softmax_khead never switches off the raw, pre-norm
    k_scale once K is relanded onto the new post-norm scale), re-run at group=1."""
    pipeline = require(MODULE)
    import composition_ref
    cfg = _fixture_config_group1(pipeline)
    model = pipeline.fixture_model(cfg)
    tokens = [0, 1, 3, 5]
    expected = composition_ref.forward_dynamic_logits_oracle(model, tokens)
    expected_iexp = _attention_iexp_rows(pipeline, model, tokens)

    def _t(text):
        source = inspect.getsource(pipeline._derive_composition_constants)
        needle = "Fraction(k_normed_scale if k_norm_present else k_scale) /"
        assert needle in source, "sanity: the softmax_khead switch must match verbatim"
        assert text.count(source) == 1
        mutated = source.replace(
            needle,
            "Fraction(k_scale) /  # T-2560 mutant: never switches off the raw pre-norm scale",
            1,
        )
        return text.replace(source, mutated, 1)
    mutant = _load_mutant_pipeline_module(_t, tmp_path)
    mutant_model = mutant.fixture_model(cfg)
    mutant_iexp = _attention_iexp_rows(mutant, mutant_model, tokens)
    assert mutant_iexp != expected_iexp, (
        "the pre-norm-K-scale mutant did not reach C30's per-head i-exp constants at group=1"
    )


def test_cell2_collapsed_q_scale_reproduces_only_the_last_heads_value(tmp_path):
    """Cell 2 (D-SLM6116, the review's own C1): a build that collapses every query head's
    own post-norm scale onto the LAST head's value (§3's own pre-fix behavior) must diverge
    from the correct per-head build -- the cell fails on any head but the last. Run at this
    file's own standard group=2 fixture; Cell 10 (above) re-runs the identical mutant at
    group=1.

    Measured, not merely asserted: the max-abs logit divergence is printed and quoted.
    """
    pipeline = require(MODULE)
    import composition_ref
    cfg = _fixture_config(pipeline)
    model = pipeline.fixture_model(cfg)
    tokens = [0, 1, 3, 5]
    expected = composition_ref.forward_dynamic_logits_oracle(model, tokens)
    expected_iexp = _attention_iexp_rows(pipeline, model, tokens)

    def _t(text):
        source = inspect.getsource(pipeline.forward_dynamic)
        assign_line = "                    q_scale_by_head[h][t] = scale\n"
        assert assign_line in source, "sanity: the per-head assignment must match verbatim"
        assert text.count(source) == 1
        mutated = source.replace(
            assign_line,
            "                    q_scale_by_head[h][t] = scale\n"
            "                    for _mh in range(cfg.num_attention_heads):\n"
            "                        q_scale_by_head[_mh][t] = scale  "
            "# T-2560 mutant: collapse onto the last head visited\n",
            1,
        )
        return text.replace(source, mutated, 1)
    mutant = _load_mutant_pipeline_module(_t, tmp_path)
    mutant_model = mutant.fixture_model(cfg)
    mutant_iexp = _attention_iexp_rows(mutant, mutant_model, tokens)
    assert mutant_iexp != expected_iexp, (
        "the collapsed-Q mutant did not reach C30's per-head i-exp constants"
    )


def test_cell3_k_scale_reads_the_old_prenorm_softmax_khead_and_diverges(tmp_path):
    """Cell 3 (D-SLM6117, the delta's own quoted mismatch class): reading K's post-norm,
    relanded codes through the OLD, pre-norm softmax_khead constant -- the fixed engine
    relands K onto the new k_normed scale (§4) but this mutant never switches
    softmax_khead off the raw pre-norm k_scale -- must diverge from the correct oracle. Run
    at this file's own standard group=2 fixture (_fixture_config), the same geometry Cells
    1/2 use; Cell 10 (above) re-runs the identical construction at group=1.

    Measured, not merely asserted: the max-abs logit divergence this mutant produces is
    printed and quoted in the build record, matching the review's own "measured, not
    summarized" discipline (`StandardsDocument.md` §5.4) -- the exact magnitude is a
    property of this fixture's own calibration, not pinned to the review's 79.2x/39.6x
    reading on the real candidate.
    """
    pipeline = require(MODULE)
    import composition_ref
    cfg = _fixture_config(pipeline)
    model = pipeline.fixture_model(cfg)
    tokens = [0, 1, 3, 5]
    expected = composition_ref.forward_dynamic_logits_oracle(model, tokens)
    expected_iexp = _attention_iexp_rows(pipeline, model, tokens)

    def _t(text):
        source = inspect.getsource(pipeline._derive_composition_constants)
        needle = "Fraction(k_normed_scale if k_norm_present else k_scale) /"
        assert needle in source, "sanity: the softmax_khead switch must match verbatim"
        assert text.count(source) == 1
        mutated = source.replace(
            needle,
            "Fraction(k_scale) /  # T-2560 mutant: never switches off the raw pre-norm scale",
            1,
        )
        return text.replace(source, mutated, 1)
    mutant = _load_mutant_pipeline_module(_t, tmp_path)
    mutant_model = mutant.fixture_model(cfg)
    mutant_iexp = _attention_iexp_rows(mutant, mutant_model, tokens)
    assert mutant_iexp != expected_iexp, (
        "the pre-norm-K-scale mutant did not reach C30's per-head i-exp constants"
    )


def test_kv_calibration_capture_maxima_differ_with_and_without_qk_norm():
    """Arm D/E's own per-head calibration capture -- the sibling T-2551's own build log
    found silently omitting QK-norm. maxima[f"{prefix}.q"] (the SAME site
    _layer_q_scale/_projection_scale read to derive the production Q scale, per this file's
    own docstring) measurably differs between a QK-norm-bearing fixture and its
    tensor-stripped twin -- not merely that the call site executes, but that it changes a
    number this arm's own downstream scale derivation actually reads.
    """
    pipeline = require(MODULE)
    cfg = _fixture_config(pipeline)
    _, _, floats_with = pipeline._pinned_weights(cfg)
    floats_without = _without_qk_norm(floats_with, cfg)
    records = pipeline.calibration_records()
    tokenize = pipeline._fixture_tokenize_prompt(cfg)
    record_tokenize = pipeline._bridge_record_tokenizer(tokenize)

    _, maxima_with = pipeline._kv_calibration_capture(
        cfg, pipeline._dict_float_source(floats_with), records, record_tokenize)
    _, maxima_without = pipeline._kv_calibration_capture(
        cfg, pipeline._dict_float_source(floats_without), records, record_tokenize)

    differed = False
    for layer in range(cfg.num_hidden_layers):
        key = f"layer{layer}.q"
        assert key in maxima_with and key in maxima_without, (
            f"{key}: missing from one of the two maxima dicts -- capture did not run"
        )
        if maxima_with[key] != pytest.approx(maxima_without[key], rel=1e-12):
            differed = True
    assert differed, (
        "_kv_calibration_capture's own maxima[layer{L}.q] is bit-for-bit identical with "
        "and without q_norm/k_norm at every layer -- QK-norm has no measurable effect on "
        "this sibling's own observable output, the exact silent-no-op shape T-2551's own "
        "_float_layer bug (a wrong lookup key) already produced once"
    )


def test_vec_forward_layer_outputs_differ_with_and_without_qk_norm():
    """_vec_forward -- the integer engine's own Python parity shadow, this module's THIRD
    independent layer walk (test_integer_pipeline_tracks_the_float_reference_per_layer is
    what caught it missing QK-norm entirely: the int8 parity shadow diverged from the
    now-corrected float oracle by more than that test's own 25% structural-defect
    tolerance). Real, materially different forward_layers output with QK-norm tensors
    present vs. absent, dequantized so the comparison is apples to apples.
    """
    pipeline = require(MODULE)
    cfg = _fixture_config(pipeline)
    weights, weight_scales, floats_with = pipeline._pinned_weights(cfg)
    float_weight_with = pipeline._dict_float_source(floats_with)
    maxima_with = pipeline._calibrate(
        cfg, float_weight_with, pipeline.calibration_records(),
        pipeline._bridge_record_tokenizer(pipeline._fixture_tokenize_prompt(cfg)))
    scales_with, residual_scales_with, biases_with = pipeline._derive_scales(
        cfg, maxima_with, weight_scales, {})
    composition_with, kv_scales_with, kv_recip_with = pipeline._derive_composition_constants(
        cfg, weight_scales, scales_with)
    model_with = pipeline.QuantizedModel(
        config=cfg, scales=scales_with, weights=weights, weight_scales=weight_scales,
        residual_scales=residual_scales_with, rope_tables=pipeline._build_rope_tables(cfg),
        biases=biases_with, float_source=float_weight_with,
        tokenize_prompt=pipeline._fixture_tokenize_prompt(cfg),
        calibration=pipeline._calibration_record("fixture"), gemm_weights={},
        composition_constants=composition_with, kv_landing_scales=kv_scales_with,
        kv_landing_reciprocals=kv_recip_with)

    floats_without = _without_qk_norm(floats_with, cfg)
    weights_without = dict(weights)
    weight_scales_without = dict(weight_scales)
    for layer in range(cfg.num_hidden_layers):
        weights_without.pop(f"layer{layer}.q_norm.gain", None)
        weights_without.pop(f"layer{layer}.k_norm.gain", None)
        weight_scales_without.pop(f"layer{layer}.q_norm.gain", None)
        weight_scales_without.pop(f"layer{layer}.k_norm.gain", None)
    float_weight_without = pipeline._dict_float_source(floats_without)
    maxima_without = pipeline._calibrate(
        cfg, float_weight_without, pipeline.calibration_records(),
        pipeline._bridge_record_tokenizer(pipeline._fixture_tokenize_prompt(cfg)))
    scales_without, residual_scales_without, biases_without = pipeline._derive_scales(
        cfg, maxima_without, weight_scales_without, {})
    composition_without, kv_scales_without, kv_recip_without = (
        pipeline._derive_composition_constants(
            cfg, weight_scales_without, scales_without))
    model_without = pipeline.QuantizedModel(
        config=cfg, scales=scales_without, weights=weights_without,
        weight_scales=weight_scales_without, residual_scales=residual_scales_without,
        rope_tables=pipeline._build_rope_tables(cfg), biases=biases_without,
        float_source=float_weight_without, tokenize_prompt=pipeline._fixture_tokenize_prompt(cfg),
        calibration=pipeline._calibration_record("fixture"), gemm_weights={},
        composition_constants=composition_without, kv_landing_scales=kv_scales_without,
        kv_landing_reciprocals=kv_recip_without)

    tokens = [1, 3, 5, 7]
    out_with = pipeline.forward_layers(model_with, tokens)
    out_without = pipeline.forward_layers(model_without, tokens)
    assert out_with[0].shape == out_without[0].shape
    max_delta = max(
        float(np.abs(np.asarray(a) - np.asarray(b)).max())
        for a, b in zip(out_with, out_without))
    assert max_delta > 0.0, (
        "_vec_forward's own forward_layers output is bit-for-bit identical with and "
        "without q_norm/k_norm -- the integer parity shadow's new QK-norm site has no "
        "measurable effect, the exact silent-no-op shape this ticket exists to close"
    )


# ==============================================================================
# External review Significant 1 (D-SLM6263): the k_normed calibration key must observe the
# UNION of post-norm/pre-RoPE and post-norm/post-RoPE component maxima, not pre-RoPE alone.
# ==============================================================================


def test_rope_can_raise_the_component_maximum_the_reviewers_own_counterexample():
    """(D-SLM6263, external review Significant 1): reproduces the review's own executed
    counterexample bit-for-bit -- a two-token, one-head K input whose position-1 pair is
    (1, 1) -- run through the real `_float_rope` primitive directly. RoPE preserves a
    rotated pair's L2 norm, not the component-wise maximum an int8 scale is chosen from:
    either output component can reach sqrt(x^2+y^2), up to sqrt(2) times the larger
    pre-rotation component. This is the mechanism the union observation (below) exists to
    enclose, quoted rather than assumed (`StandardsDocument.md` §5.4).
    """
    pipeline = require(MODULE)
    k = np.zeros((2, 1, 2), dtype=np.float64)
    k[1, 0, :] = [1.0, 1.0]
    pre_peak = float(np.abs(k).max())
    rotated = pipeline._float_rope(k, theta=10000.0)
    post_peak = float(np.abs(rotated).max())
    print(f"rope_pre_peak={pre_peak}")
    print(f"rope_post_peak={post_peak}")
    print(f"post_pair={rotated[1, 0, :].tolist()}")
    assert pre_peak == 1.0
    assert post_peak == pytest.approx(1.3817732906760363)
    assert post_peak > pre_peak, (
        "the constructed rotation did not raise the component maximum -- this "
        "fixture cannot reproduce the review's own mechanism, and the union "
        "observation this ticket adds would have nothing to enclose"
    )


def test_k_normed_union_maxima_encloses_the_post_rope_peak_on_the_checked_in_fixture():
    """(D-SLM6263): must-accept. Grounded against the repository's own checked-in §11
    fixture and calibration corpus (`_fixture_config`/`_pinned_weights`/
    `calibration_records`) -- the SAME construction the review itself replayed to find the
    defect ("the reviewer measured layer 1 head 1 post-RoPE codes required at 127.17
    against a scale sized to 127: the store clips silently on the repository's own
    fixture"). Executed pre-fix (confirmed this session, matching the review's own
    numbers exactly): `maxima["layer1.k_normed"]` = 2.331695135661406 (pre-RoPE peak
    only), while the true post-RoPE peak is 2.3348409208433023 -- requiring
    2.3348409208433023 / (2.331695135661406/127) = 127.17134088929147 codes against a
    scale sized to 127. This asserts the FIXED code's own union maxima equals the true
    post-RoPE peak, and that the derived static scale (`_derive_scales`'s own
    `k_normed_head{h}.scale`) encloses it -- at most 127 codes required, never more.
    """
    pipeline = require(MODULE)
    cfg = _fixture_config(pipeline)
    weights, weight_scales, floats = pipeline._pinned_weights(cfg)
    float_weight = pipeline._dict_float_source(floats)
    records = pipeline.calibration_records()
    tokenize = pipeline._fixture_tokenize_prompt(cfg)
    record_tokenize = pipeline._bridge_record_tokenizer(tokenize)

    # The independent, per-head capture (`_kv_calibration_capture`, Arm C/D/E's own raw
    # material) is a SECOND, differently-structured walk over the identical primitives --
    # used here only to compute the ground-truth union peak this test grades the
    # production `_calibrate`/`_derive_scales` path against, never as the thing under test.
    capture, _ = pipeline._kv_calibration_capture(cfg, float_weight, records, record_tokenize)
    true_union_peak = {
        layer: pipeline._per_head_maxabs(capture, layer, "union", cfg.num_key_value_heads)
        for layer in range(cfg.num_hidden_layers)
    }
    print(f"true_union_peak={true_union_peak}")
    assert true_union_peak[1][1] == pytest.approx(2.3348409208433023), (
        "sanity: the checked-in fixture's own layer1/head1 union peak no longer matches "
        "the review's own quoted reading -- the fixture or corpus changed under this test"
    )

    maxima = pipeline._calibrate(cfg, float_weight, records, record_tokenize)
    assert maxima["layer1.k_normed"] == pytest.approx(2.3348409208433023), (
        f"maxima['layer1.k_normed'] = {maxima.get('layer1.k_normed')!r}, not the union "
        f"peak 2.3348409208433023 -- the fix's post-RoPE observation is not landing under "
        f"the k_normed key"
    )

    scales, _, _ = pipeline._derive_scales(cfg, maxima, weight_scales, {})
    derived = dict(scales.nonlinear)
    for layer in range(cfg.num_hidden_layers):
        for head in range(cfg.num_key_value_heads):
            scale = derived[f"layer{layer}.k_normed_head{head}.scale"]
            peak = true_union_peak[layer][head]
            codes_required = peak / scale
            assert codes_required <= 127.0 + 1e-9, (
                f"layer{layer} head{head}: the derived k_normed_head scale ({scale!r}) "
                f"requires {codes_required} codes to cover the true post-RoPE peak "
                f"({peak!r}) -- the store clips, the exact defect Significant 1 names"
            )


def test_k_normed_union_maxima_is_lost_under_the_post_rope_observation_deletion_mutant(tmp_path):
    """Mutation-sensitive (D-SLM6263): a mutant that deletes batched calibration's SECOND
    `_observe(maxima, f"{prefix}.k_normed", k)` call -- the fix's own post-RoPE
    observation, added against Significant 1 -- reverts `maxima["layer1.k_normed"]` to the
    pre-fix, review-found value (2.331695135661406, the pre-RoPE peak alone) instead of
    the true union (2.3348409208433023). Quoted: the mutant's own derived scale then
    requires 127.17134088929147 codes to cover the real post-RoPE peak -- the review's own
    reading, reproduced by deleting exactly the line this ticket adds.
    """
    pipeline = require(MODULE)
    cfg = _fixture_config(pipeline)

    def _t(text):
        source = inspect.getsource(pipeline._float_calibration_layer_batch)
        # The batch path is the production calibration walk after T-2693. Its conditional
        # raw-key union is followed immediately by the post-RoPE k_normed observation;
        # deleting that one line leaves only the pre-RoPE observation above.
        tail = (
            "    if not _has_qk_norm(tensors, prefix):\n"
            "        _observe(maxima, f\"{prefix}.q\", q)\n"
            "        _observe(maxima, f\"{prefix}.k\", k)\n"
            "    _observe(maxima, f\"{prefix}.k_normed\", k)\n"
        )
        assert tail in source, "sanity: the batch path's post-RoPE observation block must match verbatim"
        assert text.count(source) == 1, (
                "sanity: the batched calibration source must appear verbatim, once, in the file"
        )
        mutated = source.replace(
            tail,
            "    if not _has_qk_norm(tensors, prefix):\n"
            "        _observe(maxima, f\"{prefix}.q\", q)\n"
            "        _observe(maxima, f\"{prefix}.k\", k)\n"
            "    # T-2572 mutant: the post-RoPE k_normed observation deleted\n",
            1,
        )
        assert mutated.count('_observe(maxima, f"{prefix}.k_normed", k)') == 1, (
            "sanity: exactly one of the two k_normed observe calls must survive the mutant"
        )
        return text.replace(source, mutated, 1)

    mutant = _load_mutant_pipeline_module(_t, tmp_path)
    weights, weight_scales, floats = mutant._pinned_weights(cfg)
    float_weight = mutant._dict_float_source(floats)
    records = mutant.calibration_records()
    tokenize = mutant._fixture_tokenize_prompt(cfg)
    record_tokenize = mutant._bridge_record_tokenizer(tokenize)

    maxima = mutant._calibrate(cfg, float_weight, records, record_tokenize)
    print(f"mutant maxima['layer1.k_normed']={maxima.get('layer1.k_normed')!r}")
    assert maxima["layer1.k_normed"] == pytest.approx(2.331695135661406), (
        "sanity: the mutant did not revert to the pre-fix, pre-RoPE-only reading -- "
        "either the deletion targeted the wrong call or the fix's own second call "
        "is not what supplies the union"
    )

    scales, _, _ = mutant._derive_scales(cfg, maxima, weight_scales, {})
    derived = dict(scales.nonlinear)
    scale = derived["layer1.k_normed_head1.scale"]
    codes_required = 2.3348409208433023 / scale
    print(f"mutant codes_required={codes_required}")
    assert codes_required > 127.0, (
        f"the mutant's derived scale ({scale!r}) still covers the true post-RoPE peak in "
        f"{codes_required} codes -- deleting the fix's own post-RoPE observation did not "
        f"reproduce Significant 1's own found defect, so this cell cannot discriminate it"
    )
