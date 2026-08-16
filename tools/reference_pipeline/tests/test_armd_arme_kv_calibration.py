"""T-1933 -- Arm D/E's red suite: T-1822 §31.4.4's 11-cell Coverage Model.

Design of record: `Claude/Vitruvius/t1822-activation-scale-remedy-design-2026-08-07.md`
§31.4.1 (the converter change), §31.4.2 (the five-arm matrix), §31.4.3 (the arithmetic
corpus), §31.4.4 (the 11-cell Coverage Model table this suite realizes) -- as corrected
by `Claude/Vitruvius/t1919-armd-correction-fusedq-fold-2026-08-11.md` (13-candidate
enumeration, Arm E, structural population disjointness). This suite realizes the model;
it does not amend it -- no cell here required the design to be re-interpreted (§6).

**At this suite's original authoring (T-1933), none of Arm D/E's own machinery existed on
this tree.** `pipeline.py`'s legacy converter (`_calibrate`/`_observe`/`_derive_scales`/
`_derive_composition_constants`) computed one K/V scale per LAYER (not per head) and wrote
it into both `kv_head` slots identically (`pipeline.py:1809-2082` -- see each cell below
for the exact lines; the standing pins below cells 1/2 confirm this legacy path is still
what Arms A/B exercise today). At authoring time there was no calibration-policy argument,
no candidate-scale sweep, no five-arm harness, no arithmetic-distinctness gate, and no
calibration-policy provenance marker anywhere in this file or elsewhere under
`Tools/superslm_spike/`; every cell below was therefore authored against symbols this
suite itself named as the contract the build had to satisfy (§0) -- red by an absent
symbol, an absent dataclass field, or, where the property was reachable through code that
already existed, a real assertion failure against the then-current, degenerate behavior.

**The build has since landed** (T-1934/T-1937/T-1938, `Claude/Poirot/7f99a81048-t1937-
t1938-confirmation.md`): every §0 symbol now exists on `reference_pipeline.pipeline`, and
every cell in this file runs green against the real, shipped implementation. Each cell's
own docstring below states what made it RED at authoring time; where a T-1936
confirmation-review finding showed a cell passing for the wrong reason, the re-authored
cell (cells 3-5, 7) states the mutation it discriminates now instead.

## §0 -- The contract this suite specifies (declared, not built)

New symbols on `reference_pipeline.pipeline`, named here so the build round has no interface
question left to resolve on its own initiative (T-1899's own declared-not-defined
convention, adapted to Python's red-unimplemented shape via `conftest.api`/`require`):

- `calibrate_kv_landing_arm(cfg, float_weight, records, tokenize, *, arm)` -- `arm` one of
  `"A","B","C","D","E"` (the design's own letters, T-1822 §31.4.2). Returns a `dict`:
  `"ordering"` (`"legacy"|"fused"`), `"granularity"` (`"per_layer"|"per_head"`),
  `"observation_domain"` (`"union"|"post_rope_only"|None`), `"kv_landing"`
  (`{f"{prefix}.k_head{h}": (m, e)}`), `"kv_reciprocals"`, `"softmax_khead"`
  (`{f"{prefix}.softmax_khead{h}": (m, e)}`), `"candidates"` (arms D/E only:
  `{f"{prefix}.k_head{h}": [{"offset_eighths_bit": k, "attention_score_error": float,
  "saturation_rate": float}, ...13 entries, k in {-8..-1,0,8,16,24,32}]}`),
  `"selected_offset"` (arms D/E only: `{f"{prefix}.k_head{h}": k}`). Realizes cells 1-5, 7.
- `classify_signed_ratio_resolution(candidate, reference, se, z_crit)` -- returns
  `"RESOLVED"|"NOT RESOLVED"|"UNDERPOWERED"` from the bare ratio alone (T-1890,
  D-SLM2324-2327), no `δ_min`-shaped term. Realizes cell 6.
- `CalibrationPopulationOverlap(ValueError)` and
  `check_calibration_eval_disjoint(calibration_ids, eval_ids)` -- raises on any collision.
  Realizes cell 8.
- `DuplicateArithmeticProblem(ValueError)` and
  `check_arithmetic_corpus_distinct(problems)` -- `problems` a list of
  `{"operation": str, "operand_tuple": tuple, ...}`; raises on a duplicate
  `(operation, operand_tuple)` key. Realizes cell 9.
- `grade_arithmetic_report(problems, model_answers, float_answers)` -- returns a list of
  `{"problem_id", "truth_correct": bool, "float_agreement": bool}`, the two columns
  computed independently (D-SLM2346). Realizes cell 10.
- `QuantizedModel` gains a new frozen-dataclass field,
  `kv_calibration_provenance: dict = dataclasses.field(default_factory=dict)`, alongside
  the existing `composition_constants`/`kv_landing_scales`/`dynamic_biases` fields
  (`pipeline.py:874-882`) -- `{"arm": str, "ordering": str, "observation_domain": str|None}`
  per layer/head or per model, distinguishing a per-head-calibrated build from a
  legacy-union one. `artifact_cache.save_artifact`/`load_artifact` gain the matching
  serialize/deserialize lines (`artifact_cache.py:328-341` write side,
  `artifact_cache.py:588-599` reconstruction side). Realizes cell 11.

Every new symbol lives in `reference_pipeline.pipeline` (the converter this design's own
§31.4.1 reads from, `D:\\Wizard\\Tools\\superslm_spike\\pipeline.py`), matching the
project's own citation of this file as Arm D/E's home.
"""

import dataclasses
import json

import numpy as np
import pytest

from conftest import api, require

MODULE = "reference_pipeline.pipeline"
ARTIFACT_MODULE = "reference_pipeline.artifact_cache"


def fixture_config(pipeline):
    """The suite's own §11 fixture shape (identical to `test_pipeline.py`/
    `test_artifact_cache.py`'s own `fixture_config` -- two kv_heads is the minimum needed
    to state a per-head distinctness claim at all)."""
    return pipeline.ModelConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=32,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, context_cap=16,
    )


def _expect_field_settable(build, *, field, why):
    """Attempt a `dataclasses.replace` that sets `field`, converting a `TypeError` (the
    field not existing on the dataclass) into an ordinary red-unimplemented
    `AssertionError` -- the dataclass-field analogue of `conftest.api`'s module-attribute
    check, kept local to this file rather than added to the shared harness (T-1933's own
    scope is this one suite, not a change to every test file's fixtures). Renamed from
    `_expect_absent_field` (T-1939 §5.4/D-SLM2643, Minor): the field it once documented as
    absent has since been added by the build round, so a name and docstring built around
    "the field does not exist" would misdescribe every call site below. What the helper
    does is unchanged and still useful once the field exists: it converts a genuine
    regression (the field removed again, `TypeError` on `dataclasses.replace`) into the
    suite's own red-unimplemented shape rather than an unrelated crash."""
    try:
        return build()
    except TypeError as exc:
        raise AssertionError(
            f"red-unimplemented: {field} ({why}). Observed: {exc}"
        ) from None


# ==============================================================================
# Cell 1 -- Per-head scale distinctness (§31.4.4 row 1; pipeline.py:1828-1834, 2056-2065)
# ==============================================================================

def test_today_kv_landing_is_a_degenerate_duplicate_across_both_heads_standing_pin():
    """GREEN standing pin: the CURRENT converter's own degenerate baseline, so the red
    cell below is provably a change from something rather than an assertion against
    nothing.

    `_derive_composition_constants` (`pipeline.py:2056-2058`) writes `kv_landing[...]`
    from ONE layer-level `k_scale` into BOTH `kv_head` slots, `for head in
    range(cfg.num_key_value_heads)` reusing the identical `(m_t_k, e_t_k)` pair every
    iteration -- exactly the gap `pipeline.py:1828-1834`'s own comment names ("the SURFACE
    is per-head... the granularity of the fixture's calibration is not").
    """
    pipeline = require(MODULE)
    cfg = fixture_config(pipeline)
    model = pipeline.fixture_model(cfg)
    assert model.kv_landing_scales["layer0.k_head0"] == model.kv_landing_scales["layer0.k_head1"], (
        "today's converter is expected to duplicate the layer-level K scale into both "
        "kv_head slots -- if this now fails, the degenerate baseline changed and this "
        "pin (and the red cell below) needs re-reading against the new baseline"
    )


def test_fixture_validity_real_per_head_k_activations_genuinely_differ():
    """Fixture-validity precondition (T-1899's own convention, D-SLM2399): before trusting
    a red cell that claims per-head distinctness is achievable, confirm by EXECUTION --
    not by construction -- that this fixture's own two kv_heads produce genuinely
    different real K activations. If they coincided, a red cell built on this fixture
    would be vacuous: any per-head implementation, correct or a duplicate-preserving stub,
    would pass it by accident.

    Computed directly through the REAL, already-shipped `_float_project`/`_float_rope`/
    `_float_rmsnorm` (the same primitives `_float_layer` itself calls,
    `pipeline.py:2133-2156`), over the first 20 records of the real 600-record
    calibration corpus, on the fixture's own real pinned weights. This pass measured:
    head0 max-abs (pre-RoPE 1.1451, post-RoPE 1.1169), head1 max-abs (pre-RoPE 2.9996,
    post-RoPE 2.9997) -- head1 is ~2.6x head0 throughout, non-degenerate by construction
    of the fixture's own per-channel-ranged weights (`_pinned_float_weight`,
    `pipeline.py:956-971`), not by coincidence of this particular corpus slice.
    """
    pipeline = require(MODULE)
    cfg = fixture_config(pipeline)
    weights, weight_scales, floats = pipeline._pinned_weights(cfg)
    float_weight = pipeline._dict_float_source(floats)
    records = pipeline.calibration_records()[:20]
    tokenize_prompt = pipeline._fixture_tokenize_prompt(cfg)

    per_head_pre = {0: [], 1: []}
    per_head_post = {0: [], 1: []}
    for record in records:
        tokens = tokenize_prompt(pipeline.run_prompt_messages(record))[: cfg.context_cap]
        hidden = floats["embed"][list(tokens), :]
        tensors = pipeline._layer_tensors(float_weight, "layer0")
        normed = pipeline._float_rmsnorm(hidden, cfg.rms_norm_eps) * tensors["layer0.attn_norm.gain"]
        k = pipeline._float_project(tensors.__getitem__, "layer0.k_proj", normed).reshape(
            -1, cfg.num_key_value_heads, cfg.head_dim)
        for head in range(2):
            per_head_pre[head].append(float(np.abs(k[:, head, :]).max()))
        k_rope = pipeline._float_rope(k, cfg.rope_theta)
        for head in range(2):
            per_head_post[head].append(float(np.abs(k_rope[:, head, :]).max()))

    head0_max = max(max(per_head_pre[0]), max(per_head_post[0]))
    head1_max = max(max(per_head_pre[1]), max(per_head_post[1]))
    assert head0_max != head1_max, (
        f"the fixture's own two kv_heads produced coincidentally IDENTICAL real K "
        f"activations (head0={head0_max}, head1={head1_max}) -- this fixture cannot "
        f"discriminate a genuine per-head calibration from a duplicate-preserving stub "
        f"and must not be used for the cell below without a different fixture"
    )
    assert abs(head0_max - head1_max) / max(head0_max, head1_max) > 0.1, (
        "the two heads' real activations differ by less than 10%% -- too close to "
        "confidently discriminate a per-head implementation from rounding noise"
    )


def test_arm_d_per_head_calibration_produces_distinct_kv_head_scales():
    """§31.4.4 row 1: "For at least one converted artifact, the two `kv_head` slots of
    at least one layer carry different `(m_t, e_t)` pairs under any per-head calibration
    policy (Arms C, D, or E) -- a test asserts non-identity, refuting the
    degenerate-duplicate default." (source: `pipeline.py:1828-1834`; §31.4.1.)

    The fixture is confirmed non-vacuous by the preceding real-execution cell. Originally
    RED via the absent contract symbol (§0): `calibrate_kv_landing_arm` did not exist on
    `reference_pipeline.pipeline` at authoring time. The build has since landed; this cell now
    runs green against the real, shipped `calibrate_kv_landing_arm`.
    """
    pipeline = require(MODULE)
    calibrate_kv_landing_arm = api(MODULE, "calibrate_kv_landing_arm")
    cfg = fixture_config(pipeline)
    weights, weight_scales, floats = pipeline._pinned_weights(cfg)
    float_weight = pipeline._dict_float_source(floats)
    records = pipeline.calibration_records()
    tokenize = pipeline._fixture_tokenize_prompt(cfg)

    result = calibrate_kv_landing_arm(cfg, float_weight, records, tokenize, arm="D")
    assert result["kv_landing"]["layer0.k_head0"] != result["kv_landing"]["layer0.k_head1"], (
        "Arm D's per-head calibration must not duplicate the same (m_t, e_t) pair into "
        "both kv_head slots -- that is the legacy default this cell exists to refute"
    )


# ==============================================================================
# Cell 2 -- softmax_khead regeneration (§31.4.4 row 2; pipeline.py:1917, 2061-2065)
# ==============================================================================

def test_today_softmax_khead_is_derived_from_the_layer_level_k_scale_standing_pin():
    """GREEN standing pin: today's `softmax_khead{head}` constants (`pipeline.py:2064-2065`)
    are both derived from the SAME layer-level `k_scale`, so they are identical across
    heads -- the exact defect §31.4.1 names as "the single highest-value correctness
    obligation this converter change carries"."""
    pipeline = require(MODULE)
    cfg = fixture_config(pipeline)
    model = pipeline.fixture_model(cfg)
    assert (model.composition_constants["layer0.softmax_khead0"]
            == model.composition_constants["layer0.softmax_khead1"])


def test_arm_d_softmax_khead_moves_with_only_its_own_head_scale():
    """§31.4.4 row 2: "Every `softmax_khead{head}` constant is derived from *that head's
    own* `k_scale`, not a layer-collapsed one, once any per-head calibration runs -- a
    test perturbs one head's candidate-scale selection and asserts only that head's
    `softmax_khead` constant moves." (source: `pipeline.py:1880-1881`
    [D-SLM2352/§31.4.1's own citation of the composition constant]; §31.4.1's own fourth
    dependent constant.)

    Mutation fixture: two `float_weight` sources identical in every tensor except
    `layer0.k_proj`'s own head-1 rows (rows `[head_dim:2*head_dim)` of the K weight
    matrix), scaled 2.5x. Head 0's own rows, and every Q/norm input feeding both heads,
    are byte-identical between the two builds -- a correct per-head implementation must
    therefore leave head 0's own `softmax_khead0` unchanged and move only
    `softmax_khead1`. Originally RED via the absent contract symbol (§0); the build has
    since landed and this cell now runs green against the real, shipped
    `calibrate_kv_landing_arm`.
    """
    pipeline = require(MODULE)
    calibrate_kv_landing_arm = api(MODULE, "calibrate_kv_landing_arm")
    cfg = fixture_config(pipeline)
    weights, weight_scales, floats_a = pipeline._pinned_weights(cfg)
    floats_b = dict(floats_a)
    k_proj = np.array(floats_a["layer0.k_proj"], copy=True)
    k_proj[cfg.head_dim:2 * cfg.head_dim, :] *= 2.5   # head 1's rows only
    floats_b["layer0.k_proj"] = k_proj

    records = pipeline.calibration_records()
    tokenize = pipeline._fixture_tokenize_prompt(cfg)
    result_a = calibrate_kv_landing_arm(
        cfg, pipeline._dict_float_source(floats_a), records, tokenize, arm="D")
    result_b = calibrate_kv_landing_arm(
        cfg, pipeline._dict_float_source(floats_b), records, tokenize, arm="D")

    assert result_a["softmax_khead"]["layer0.softmax_khead0"] == \
        result_b["softmax_khead"]["layer0.softmax_khead0"], (
        "perturbing ONLY head 1's own K weight rows moved head 0's own softmax_khead "
        "constant -- the composition constant was regenerated from a layer-collapsed "
        "value, not that head's own"
    )
    assert result_a["softmax_khead"]["layer0.softmax_khead1"] != \
        result_b["softmax_khead"]["layer0.softmax_khead1"], (
        "perturbing head 1's own K weight rows did not move head 1's own softmax_khead "
        "constant at all -- softmax_khead1 was not regenerated from the per-head scale"
    )


# ==============================================================================
# Cells 3-5 -- the 13-candidate sweep: score-error selection, finer reachability,
# saturation reported not gated (§31.4.4 rows 3-5; §31.4.1, T-1919 §2(a))
# ==============================================================================
#
# T-1936 §5.1/D-SLM2622 (Critical): the prior authoring of these three cells built an
# independent reference sweep over a hand-authored Q/K fixture and asserted properties of
# THAT reference alone, never calling `calibrate_kv_landing_arm` -- with the entire
# 13-candidate sweep replaced in-process by "always max-abs, empty report", the full
# suite stayed green (1874 passed, 13 deselected, identical to shipped). Re-authored here
# to DRIVE THE REAL SWEEP: all three cells below read a single, module-scoped execution
# of the real, end-to-end `calibrate_kv_landing_arm` (production forward pass, production
# per-head capture, production 13-candidate score-error sweep) on the suite's OWN
# already-shared §11 fixture (`fixture_config`, `_pinned_weights`, the full 600-record
# calibration corpus -- the identical fixture
# `test_fixture_validity_real_per_head_k_activations_genuinely_differ` above already
# confirmed by execution is non-degenerate). No synthetic Q/K array is constructed
# anywhere in this section; every number below is read from a real executed production
# result (StandardsDocument.md §5.4: verified at source or by execution, never by
# construction).
#
# EXECUTED this pass: on this fixture, Arm D's real sweep at `layer0.k_head1` selects
# offset -3 (finer than max-abs's offset 0) with attention-score error 1172.115 against
# max-abs's own 1348.644 in the SAME executed report, and saturation_rate 0.03125 against
# max-abs's own 0.015625 -- the winner clips MORE than max-abs and is still selected. This
# divergence is naturally present in the suite's existing shared fixture; it did not need
# to be manufactured.

_ARMD_REAL_SWEEP_KEY = "layer0.k_head1"
_ARMD_REAL_SWEEP_MAXABS_OFFSET = 0


@pytest.fixture(scope="module")
def _armd_arme_real_calibration():
    """Drive the REAL, end-to-end `calibrate_kv_landing_arm` for arms B, C, D, and E, on
    the suite's shared §11 fixture -- the full 600-record calibration corpus through the
    real float forward pass, the real per-head capture, and (for D/E) the real
    13-candidate score-error sweep. Computed ONCE per module and shared by cells 3, 4, 5,
    and 7 below (each arm computed once, not once per cell), so this fixture is the
    single source every cell in this section reads from -- no cell recomputes or
    reimplements any part of the sweep.
    """
    pipeline = require(MODULE)
    calibrate_kv_landing_arm = api(MODULE, "calibrate_kv_landing_arm")
    cfg = fixture_config(pipeline)
    weights, weight_scales, floats = pipeline._pinned_weights(cfg)
    float_weight = pipeline._dict_float_source(floats)
    records = pipeline.calibration_records()
    tokenize = pipeline._fixture_tokenize_prompt(cfg)
    return {
        arm: calibrate_kv_landing_arm(cfg, float_weight, records, tokenize, arm=arm)
        for arm in ("B", "C", "D", "E")
    }


def test_arm_d_selection_is_score_error_minimizing_not_max_abs(_armd_arme_real_calibration):
    """§31.4.4 row 3: "The candidate-scale sweep's selected value is not, in general, the
    finest legal scale or the max-abs value -- a fixture where max-abs and score-error-
    minimal diverge asserts the selection picks the latter, over the full 13-candidate
    enumeration." (source: T-1880 §6.2, D-SLM2360, D-SLM2554.)

    Calls the real `calibrate_kv_landing_arm` (via the module-scoped fixture above, arm
    D) and reads its actual returned `selected_offset`/`candidates` -- not a
    recomputation. On this fixture, `layer0.k_head1`'s real sweep selects offset -3, not
    max-abs's offset 0, and that offset's own reported attention-score error is strictly
    lower than max-abs's own reported error in the SAME executed report, so the selection
    genuinely minimizes error rather than landing on a non-max-abs offset by accident.

    RED under the always-max-abs mutation (`_armd_arme_candidate_sweep` ->
    `return 0, base_scale, []`, T-1936 §4): `selected_offset` collapses to 0 for every
    head, so the first assertion below fails directly.

    The third assertion below pins the property §31.4.4 row 3 actually names --
    minimization, not merely "lower error than max-abs" -- closing T-1939 §5.1/D-SLM2640
    (Significant): a sweep rebound to select the SECOND-lowest-error candidate
    (`layer0.k_head1`: offset -4, error 1225.397915, against the true winner's offset -3,
    error 1172.115035) is still not max-abs, still finer than max-abs, and still
    lower-error than max-abs -- so the first two assertions above stayed green under that
    mutation and the full 1881-test suite reported no failure. Confirmed by execution this
    round (`Claude/Curie/t1933-armd-arme-red-suite-2026-08-11.md`'s T-1940 addendum): RED
    under the second-best mutation, GREEN after revert.
    """
    result = _armd_arme_real_calibration["D"]
    reports = {r["offset_eighths_bit"]: r for r in result["candidates"][_ARMD_REAL_SWEEP_KEY]}
    winner_offset = result["selected_offset"][_ARMD_REAL_SWEEP_KEY]
    assert winner_offset != _ARMD_REAL_SWEEP_MAXABS_OFFSET, (
        f"the real 13-candidate sweep selected max-abs (offset "
        f"{_ARMD_REAL_SWEEP_MAXABS_OFFSET}) for {_ARMD_REAL_SWEEP_KEY} on a fixture whose "
        f"own report shows a lower-error candidate exists -- selection is not minimizing "
        f"attention-score error"
    )
    assert (reports[winner_offset]["attention_score_error"]
            < reports[_ARMD_REAL_SWEEP_MAXABS_OFFSET]["attention_score_error"]), (
        "the selected candidate's own reported attention-score error is not lower than "
        "max-abs's own reported error in the same executed report -- the selection did "
        "not actually minimize error"
    )
    # Tie-break must match the sweep's own documented rule: ties broken toward the
    # candidate closer to max-abs (lower abs(k)), not toward enumeration order — the two
    # disagree exactly when all candidates score identically (T-1939 §11.2.1's all-zero head).
    true_argmin_offset = min(
        reports, key=lambda k: (reports[k]["attention_score_error"], abs(k)))
    assert winner_offset == true_argmin_offset, (
        f"the real sweep selected offset {winner_offset} for {_ARMD_REAL_SWEEP_KEY}, but "
        f"the report's own minimum-attention-score-error candidate is offset "
        f"{true_argmin_offset} (error "
        f"{reports[true_argmin_offset]['attention_score_error']}) -- the selection is "
        f"directionally right (lower error than max-abs) but is not the score-error-"
        f"minimal candidate §31.4.4 row 3 requires"
    )


def test_arm_d_finer_than_max_abs_candidates_are_reachable(_armd_arme_real_calibration):
    """§31.4.4 row 4 (new, round 5, D-SLM2554): "For at least one `(layer, kv_head)`, the
    sweep's selected candidate is one of the eight `k<0` (finer-than-max-abs) candidates
    -- a fixture where a finer candidate genuinely minimizes attention-score error
    asserts the selection reaches it, refuting a coarser-only implementation that
    silently ignores the finer half of the enumeration."

    Reads the real sweep's own `selected_offset` (module-scoped fixture above, arm D).
    `layer0.k_head1`'s real winner (-3) is one of the eight finer-than-max-abs candidates.

    RED under BOTH mutations named in T-1936 §4: the always-max-abs mutation collapses
    `selected_offset` to 0 (caught by `winner_offset < 0` below); the coarser-only
    enumeration mutation (`_ARMD_ARME_SWEEP_OFFSETS` -> `(0, 8, 16, 24, 32)`) also forces
    this head's own winner to 0, because on this fixture every coarser candidate's own
    reported error exceeds max-abs's own -- confirmed by execution, this round's own
    handoff.
    """
    result = _armd_arme_real_calibration["D"]
    winner_offset = result["selected_offset"][_ARMD_REAL_SWEEP_KEY]
    assert winner_offset < 0, (
        f"{_ARMD_REAL_SWEEP_KEY}'s real sweep selected offset {winner_offset}, not one of "
        f"the eight k<0 finer-than-max-abs candidates -- the finer half of the "
        f"enumeration is unreachable"
    )


def test_arm_d_saturation_is_reported_and_does_not_gate_selection(_armd_arme_real_calibration):
    """§31.4.4 row 5 (corrected round 5): "Every candidate's saturation rate is computed
    and included in the sweep's own report (alongside attention-score error), for at
    least one candidate that clips more than max-abs -- a test asserts the report carries
    that candidate's saturation figure and that the candidate is still selectable on
    attention-score error alone, refuting both silent omission and a reintroduced hard
    reject." (source: §31.4.1, D-SLM2555.)

    Reads the real sweep's own report (module-scoped fixture above, arm D). All 13
    candidates are present with their own saturation figures, and the WINNER itself
    (`layer0.k_head1`, offset -3) clips more codes than max-abs (saturation_rate 0.03125
    vs. 0.015625) and is still the selected candidate -- the strongest form of "still
    selectable": it was actually selected, not merely eligible.

    RED under the always-max-abs mutation (`_armd_arme_candidate_sweep` ->
    `return 0, base_scale, []`): `candidates` becomes an empty list, so
    `len(reports) == 13` fails directly.
    """
    result = _armd_arme_real_calibration["D"]
    reports = {r["offset_eighths_bit"]: r for r in result["candidates"][_ARMD_REAL_SWEEP_KEY]}
    assert len(reports) == 13, (
        f"expected all 13 candidates reported for {_ARMD_REAL_SWEEP_KEY}, got "
        f"{len(reports)} -- the sweep's own report is not carrying every candidate's "
        f"saturation figure"
    )
    winner_offset = result["selected_offset"][_ARMD_REAL_SWEEP_KEY]
    winner_saturation = reports[winner_offset]["saturation_rate"]
    maxabs_saturation = reports[_ARMD_REAL_SWEEP_MAXABS_OFFSET]["saturation_rate"]
    assert winner_saturation > maxabs_saturation, (
        f"the selected (winning) candidate's own saturation_rate ({winner_saturation}) is "
        f"not higher than max-abs's own ({maxabs_saturation}) -- this fixture does not "
        f"exercise 'the winner clips more than max-abs and is still selected'"
    )
    assert winner_saturation > 0.0, "the winning candidate's own saturation_rate must be nonzero"


# ==============================================================================
# Cell 6 -- Five-arm resolution independence from worthwhileness
# (§31.4.4 row 6; T-1890, D-SLM2324-2327, §31.4.2)
# ==============================================================================

def test_signed_ratio_resolves_a_statistically_resolved_but_tiny_delta():
    """§31.4.4 row 6: "The `signed_ratio` resolution verdict for any cell does not
    reference a `δ_min`-shaped floor; a test constructs a resolved-but-small delta and
    asserts it prints `RESOLVED`, not `NOT RESOLVED`, distinct from the separate
    worthwhileness judgment." (source: T-1890, D-SLM2324-2327; §31.4.2.)

    Fixture: `candidate=100.001`, `reference=100.0`, `se=0.0005`, `z_crit=1.96` ->
    `signed_ratio = 0.001 / (1.96*0.0005) = 1.0204`, |ratio| >= 1 -- statistically
    resolved by T-1870 §6's own bare-ratio rule. The raw delta (0.001) is far below
    T-1870's own named `δ_min` defaults (0.05, 0.06) cited at §31.4.2 -- a rule that folds
    a worthwhileness floor into the resolution verdict would print NOT RESOLVED here;
    this design's own rule must not. Originally RED via the absent contract symbol; the
    build has since landed and this cell now runs green against the real, shipped
    `classify_signed_ratio_resolution`.
    """
    classify_signed_ratio_resolution = api(MODULE, "classify_signed_ratio_resolution")
    candidate, reference, se, z_crit = 100.001, 100.0, 0.0005, 1.96
    signed_ratio = (candidate - reference) / (z_crit * se)
    assert abs(signed_ratio) >= 1.0, "fixture sanity: this delta must cross the z-critical threshold"
    assert abs(candidate - reference) < 0.05, "fixture sanity: the raw delta must be smaller than either named δ_min default"
    verdict = classify_signed_ratio_resolution(candidate, reference, se, z_crit)
    assert verdict == "RESOLVED", (
        f"a statistically resolved delta (signed_ratio={signed_ratio:.4f}) that is far "
        f"smaller than either named δ_min default printed {verdict!r}, not RESOLVED -- "
        f"a worthwhileness floor has leaked into the resolution verdict"
    )


# ==============================================================================
# Cell 7 -- Arm E variable-separation
# (§31.4.4 row 7, new round 5, D-SLM2556; §31.4.1/§31.4.2)
# ==============================================================================

def test_arm_to_arm_deltas_isolate_exactly_one_variable_each(_armd_arme_real_calibration):
    """§31.4.4 row 7: "Each of B->E (granularity), E->D (observation domain), and C->E
    (ordering) moves exactly one of the matrix's three named variables while holding the
    other two constant -- a test asserts, for each of the three deltas, that the two
    arms' own recorded (ordering, granularity, observation-domain) triples differ in
    exactly one coordinate." (source: §31.4.1/§31.4.2, corrected round 5.)

    Uses the module-scoped real-calibration fixture above (arms B, C, D, E, computed
    once).

    Label check (unchanged in substance from the original authoring): the three arms'
    own recorded (ordering, granularity, observation_domain) triples differ in exactly
    one coordinate per named delta.

    Behaviour pin (T-1936 §5.2/D-SLM2623, Significant): the label check alone passes even
    if an arm computes the WRONG observation domain while still reporting the RIGHT
    label -- verified by execution, forcing `_per_head_maxabs`'s own domain argument to
    `"union"` regardless of the arm (collapsing Arm D's post-RoPE-only restriction into
    Arm E's) leaves the label check green, because `_ARM_SPECS`'s own strings never
    change. The behaviour pin below closes that gap: `layer0.k_head0` is a fixture where
    union max-abs (1.1451) and post-RoPE-only max-abs (1.1169) genuinely diverge
    (T-1933 §4.1, confirmed again by execution this round), so a correct implementation
    must land Arms D and E on DIFFERENT computed `kv_landing` scales at that key, not
    merely different label strings.

    RED under the label-vs-behaviour mutation named above (`_per_head_maxabs` domain
    forced to `"union"`): with `observation_domain` no longer affecting the computation
    at all, Arm D's own capture becomes byte-identical to Arm E's, so
    `results["D"]["kv_landing"]["layer0.k_head0"] == results["E"]["kv_landing"]["layer0.k_head0"]`
    and the behaviour-pin assertion below fails, while the label check above stays green
    -- exactly the gap this cell existed to close.
    """
    triples = {
        arm: (result["ordering"], result["granularity"], result["observation_domain"])
        for arm, result in _armd_arme_real_calibration.items()
    }

    def differs_in_exactly_one_coordinate(a, b):
        return sum(1 for x, y in zip(triples[a], triples[b]) if x != y) == 1

    assert differs_in_exactly_one_coordinate("B", "E"), "B->E must isolate granularity alone"
    assert differs_in_exactly_one_coordinate("E", "D"), "E->D must isolate observation domain alone"
    assert differs_in_exactly_one_coordinate("C", "E"), "C->E must isolate ordering alone"

    behavior_key = "layer0.k_head0"
    kv_landing_d = _armd_arme_real_calibration["D"]["kv_landing"][behavior_key]
    kv_landing_e = _armd_arme_real_calibration["E"]["kv_landing"][behavior_key]
    assert kv_landing_d != kv_landing_e, (
        f"Arm D (observation_domain='post_rope_only') and Arm E (observation_domain="
        f"'union') produced IDENTICAL kv_landing at {behavior_key} despite declaring "
        f"different observation_domain labels -- the E->D delta the label claims to "
        f"isolate has no effect on the actual computed scale, so an arm could compute "
        f"the WRONG domain while still reporting the RIGHT label and the assertions "
        f"above would not notice"
    )


# ==============================================================================
# Cell 8 -- Calibration/evaluation population disjointness
# (§31.4.4 row 8, new round 5, D-SLM2557; §31.4.1)
# ==============================================================================

def test_calibration_and_grading_populations_must_not_overlap():
    """§31.4.4 row 8: "No document ID in the 30-document held-out-reserve calibration
    population collides with any document ID in the 239-document `tuned_on` grading
    corpus -- a test constructs one collision and asserts the grading tool refuses to
    emit a verdict, reusing T-1870 §9.3's own mechanism." (source: §31.4.1, D-SLM2557.)

    Originally RED via the absent contract symbol; the build has since landed and this
    cell now runs green against the real, shipped `check_calibration_eval_disjoint`.
    """
    check_calibration_eval_disjoint, overlap_error = api(
        MODULE, "check_calibration_eval_disjoint", "CalibrationPopulationOverlap")
    calibration_ids = [f"held_out_{i}" for i in range(30)]
    eval_ids = [f"tuned_on_{i}" for i in range(239)]
    eval_ids[100] = calibration_ids[5]  # the constructed collision

    with pytest.raises(overlap_error) as excinfo:
        check_calibration_eval_disjoint(calibration_ids, eval_ids)
    assert calibration_ids[5] in str(excinfo.value), (
        "the refusal must name the colliding document ID, not merely report that a "
        "collision occurred"
    )

    # Negative companion: a genuinely disjoint pair must NOT be refused.
    eval_ids_clean = [f"tuned_on_{i}" for i in range(239)]
    check_calibration_eval_disjoint(calibration_ids, eval_ids_clean)  # must not raise


# ==============================================================================
# Cell 9 -- Arithmetic distinctness gate (§31.4.4 row 9; D-SLM2345; §31.4.3)
# ==============================================================================

def test_arithmetic_corpus_distinctness_gate_refuses_a_duplicate_semantic_key():
    """§31.4.4 row 9: "The grading tool refuses a corpus manifest containing two problems
    with the same `(operation, operand_tuple)` key; a test constructs one collision and
    asserts refusal." (source: D-SLM2345; §31.4.3.)

    D-SLM2345's own kill: 8 of 13 phrasings collapsed to 2 semantic keys undetected. The
    fixture reproduces exactly that shape at small scale -- two DIFFERENT phrasings of
    `12+15` (`"12+15="`, `"What is 12 plus 15?"`), same `("+", (12, 15))` key. Originally
    RED via the absent contract symbol; the build has since landed and this cell now runs
    green against the real, shipped `check_arithmetic_corpus_distinct`.
    """
    check_arithmetic_corpus_distinct, duplicate_error = api(
        MODULE, "check_arithmetic_corpus_distinct", "DuplicateArithmeticProblem")
    problems = [
        {"prompt": "12+15=", "operation": "+", "operand_tuple": (12, 15), "ground_truth": 27},
        {"prompt": "What is 12 plus 15?", "operation": "+", "operand_tuple": (12, 15), "ground_truth": 27},
        {"prompt": "17*23=", "operation": "*", "operand_tuple": (17, 23), "ground_truth": 391},
    ]
    with pytest.raises(duplicate_error) as excinfo:
        check_arithmetic_corpus_distinct(problems)
    assert "('+', (12, 15))" in str(excinfo.value) or "(12, 15)" in str(excinfo.value), (
        "the refusal must name the colliding semantic-identity key"
    )

    # Negative companion: three genuinely distinct problems must NOT be refused.
    distinct_problems = [
        {"prompt": "12+15=", "operation": "+", "operand_tuple": (12, 15), "ground_truth": 27},
        {"prompt": "17*23=", "operation": "*", "operand_tuple": (17, 23), "ground_truth": 391},
        {"prompt": "40-9=", "operation": "-", "operand_tuple": (40, 9), "ground_truth": 31},
    ]
    check_arithmetic_corpus_distinct(distinct_problems)  # must not raise


# ==============================================================================
# Cell 10 -- Truth/float-agreement separation (§31.4.4 row 10; D-SLM2346; §31.4.2)
# ==============================================================================

def test_truth_correct_and_float_agreement_are_independent_and_can_disagree():
    """§31.4.4 row 10: "The arithmetic report carries two distinct columns
    (truth-correct, float-agreement) for every problem and every arm, never merged; a
    test asserts both columns are present and independently computed." (source:
    D-SLM2346; §31.4.2.)

    D-SLM2346's own finding: "correcting toward the reference" and "getting the right
    answer" are not the same claim -- on two of T-1891's own discordant items, the FLOAT
    REFERENCE ITSELF answered wrong. The fixture reproduces exactly that shape: one
    problem where the model's answer matches the float reference's answer (agreement)
    but the float reference itself is wrong against ground truth (not correct) -- a
    single merged column could not represent this case at all. Originally RED via the
    absent contract symbol; the build has since landed and this cell now runs green
    against the real, shipped `grade_arithmetic_report`.
    """
    grade_arithmetic_report = api(MODULE, "grade_arithmetic_report")
    problems = [
        {"problem_id": "p1", "operation": "+", "operand_tuple": (12, 15), "ground_truth": 27},
        {"problem_id": "p2", "operation": "*", "operand_tuple": (17, 23), "ground_truth": 391},
    ]
    # p1: float reference is WRONG (26, not 27); model agrees with the (wrong) float
    # reference. p2: both float reference and model are correct.
    model_answers = {"p1": 26, "p2": 391}
    float_answers = {"p1": 26, "p2": 391}

    report = grade_arithmetic_report(problems, model_answers, float_answers)
    by_id = {row["problem_id"]: row for row in report}

    assert set(by_id["p1"].keys()) >= {"problem_id", "truth_correct", "float_agreement"}
    assert by_id["p1"]["truth_correct"] is False, "p1's model answer (26) does not match ground truth (27)"
    assert by_id["p1"]["float_agreement"] is True, "p1's model answer (26) matches the float reference's own (26)"
    assert by_id["p1"]["truth_correct"] != by_id["p1"]["float_agreement"], (
        "the fixture's whole point is a case where truth-correct and float-agreement "
        "DISAGREE -- a merged single column could not represent p1 at all"
    )
    assert by_id["p2"]["truth_correct"] is True
    assert by_id["p2"]["float_agreement"] is True


# ==============================================================================
# Cell 11 -- Artifact metadata / calibration-policy marker
# (§31.4.4 row 11, new, T-1897 C17/D-SLM2369; §31.4.1's fourth dependent constant)
# ==============================================================================

def test_kv_calibration_provenance_field_exists_on_quantized_model():
    """§31.4.4 row 11: "A per-head-calibrated artifact's (Arm C, D, or E)
    `Provenance`/`Calibration` section carries a policy marker identifying its own
    calibration policy (union vs. post-RoPE-only) and ordering, distinguishable from a
    legacy-union-calibrated artifact and from each other; a test omits the marker and
    asserts the cell then fails to distinguish the builds." (source: §31.4.1's fourth
    dependent constant.)

    Renamed from `..._is_absent_from_quantized_model` (T-1936 §5.6/finding 6, Minor):
    that name was correct only while the field genuinely did not exist. The build round
    (T-1934) added `kv_calibration_provenance` to `QuantizedModel`
    (`pipeline.py:842-882`), so the helper's own `build()` now SUCCEEDS instead of
    raising, and the test as previously named asserted nothing explicitly on the result --
    a presence pin wearing an absence name. The helper (renamed `_expect_field_settable`,
    T-1939 §5.4/D-SLM2643, Minor -- it was still `_expect_absent_field` when this
    paragraph was first written) still converts a `TypeError` into a red-unimplemented
    `AssertionError`, so this cell stays vital against a REGRESSION that removes the
    field; what changed is the name and the explicit assertion below, not the underlying
    discipline.

    Pins presence via `dataclasses.replace` -- the same construction `with_rope_tables`
    (`pipeline.py:906-913`) already uses for every other post-build field change on this
    frozen dataclass -- and asserts the built model actually carries the value passed in,
    not merely that construction did not raise.
    """
    pipeline = require(MODULE)
    cfg = fixture_config(pipeline)
    model = pipeline.fixture_model(cfg)
    provenance = {"layer0": {"arm": "D", "ordering": "fused",
                              "observation_domain": "post_rope_only"}}

    def build():
        return dataclasses.replace(model, kv_calibration_provenance=provenance)

    built = _expect_field_settable(
        build, field="QuantizedModel.kv_calibration_provenance",
        why="regression guard -- the build round added this field; a TypeError here "
            "means it was removed again",
    )
    assert built.kv_calibration_provenance == provenance, (
        "QuantizedModel.kv_calibration_provenance did not carry the value passed to "
        "dataclasses.replace -- the field exists but does not hold what was set"
    )


def test_kv_calibration_provenance_round_trips_through_save_and_load_artifact(tmp_path):
    """The provenance marker's own point is to survive to a LOADED artifact, distinct
    across two builds that differ only in calibration policy -- otherwise a later reader
    (an auditor, or a re-derivation check) cannot tell which policy built a given saved
    artifact apart from re-running the whole calibration.

    At this suite's original authoring, `save_artifact`'s own metadata dict
    (`artifact_cache.py:328-341`) had no `kv_calibration_provenance` key, and
    `load_artifact`'s own `QuantizedModel(...)` reconstruction (`artifact_cache.py:588-599`)
    had no matching kwarg -- so even a model carrying the field in memory would have
    silently lost it on a round trip, a second, independent gap from cell 11's own field-
    existence precondition above. The build round wired both the write side
    (`artifact_cache.py:336-350`) and the reconstruction side (`artifact_cache.py:565-625`):
    this cell now runs green as a standing regression pin on the round trip, not merely on
    field existence (T-1939 §5.4/D-SLM2643, Minor -- this paragraph previously still
    claimed the round-trip gap as an open RED precondition after the build closed it).
    """
    pipeline = require(MODULE)
    save_artifact, load_artifact = api(ARTIFACT_MODULE, "save_artifact", "load_artifact")
    cfg = fixture_config(pipeline)
    model = pipeline.fixture_model(cfg)

    def build_d():
        return dataclasses.replace(
            model, kv_calibration_provenance={"layer0": {"arm": "D", "ordering": "fused",
                                                           "observation_domain": "post_rope_only"}})

    def build_e():
        return dataclasses.replace(
            model, kv_calibration_provenance={"layer0": {"arm": "E", "ordering": "fused",
                                                           "observation_domain": "union"}})

    model_d = _expect_field_settable(
        build_d, field="QuantizedModel.kv_calibration_provenance",
        why="regression guard -- the build round added this field; a TypeError here "
            "means it was removed again")
    model_e = _expect_field_settable(
        build_e, field="QuantizedModel.kv_calibration_provenance",
        why="regression guard -- the build round added this field; a TypeError here "
            "means it was removed again")

    save_artifact(model_d, tmp_path / "arm_d")
    save_artifact(model_e, tmp_path / "arm_e")
    loaded_d = load_artifact(tmp_path / "arm_d")
    loaded_e = load_artifact(tmp_path / "arm_e")

    assert loaded_d.kv_calibration_provenance == model_d.kv_calibration_provenance, (
        "the provenance marker did not survive a save/load round trip"
    )
    assert loaded_d.kv_calibration_provenance != loaded_e.kv_calibration_provenance, (
        "two builds with different calibration policies produced INDISTINGUISHABLE "
        "loaded provenance -- the marker cannot tell the builds apart, which is the "
        "whole point of this cell"
    )


# ==============================================================================
# T-1935 -- pin owed by T-1934's own build log §6: `_compute_fingerprint`
# (`artifact_cache.py`) hashes `kv_calibration_provenance`, but no cell above checks
# the FINGERPRINT VALUE for a provenance-carrying build -- the round-trip test above
# (cell 11) only checks field equality after load, never `manifest.json`'s own
# `source_fingerprint`. Per the field's own stated purpose ("distinguishable... from
# each other", `QuantizedModel`'s own docstring), two builds differing only in
# `kv_calibration_provenance` must fingerprint differently; without the hashing this
# build landed, they would not.
# ==============================================================================

def test_source_fingerprint_differs_when_only_kv_calibration_provenance_differs(tmp_path):
    """Mutation pin for T-1934's pin-owed item (`Claude/Brunel/t1934-armd-arme-build-
    2026-08-11.md` §6): `_compute_fingerprint` (`artifact_cache.py`) hashes
    `model.kv_calibration_provenance`, so two `QuantizedModel`s that differ ONLY in that
    field produce DIFFERENT `manifest.json` `source_fingerprint` values from
    `save_artifact`.

    Isolates the condition the way cell 11's round-trip test does not: `model_d` and
    `model_e` are both `dataclasses.replace(model, kv_calibration_provenance=...)` from
    the SAME base fixture model, so every other field (weights, scales, config, rope
    tables, biases) is identical between them -- the fingerprint's own determinism
    property (same model twice -> same hash, `test_artifact_cache.py`'s own
    `test_the_manifest_carries_the_format_version_and_source_fingerprint`) rules out
    every other field as the source of a divergence, isolating provenance as the one
    variable under test.

    If `_compute_fingerprint`'s hashing of `kv_calibration_provenance` is removed, this
    test goes RED: `manifest_d["source_fingerprint"] == manifest_e["source_fingerprint"]`,
    because nothing else differs between `model_d` and `model_e`. Confirmed by executing
    the mutation (removing the `for key in sorted(model.kv_calibration_provenance.keys())`
    block from `_compute_fingerprint`) -- see `Claude/Curie/t1933-armd-arme-red-suite-
    2026-08-11.md`'s T-1935 addendum for the quoted red output. Reverted before commit.
    """
    save_artifact = api(ARTIFACT_MODULE, "save_artifact")
    pipeline = require(MODULE)
    cfg = fixture_config(pipeline)
    model = pipeline.fixture_model(cfg)

    model_d = dataclasses.replace(
        model, kv_calibration_provenance={"layer0": {"arm": "D", "ordering": "fused",
                                                       "observation_domain": "post_rope_only"}})
    model_e = dataclasses.replace(
        model, kv_calibration_provenance={"layer0": {"arm": "E", "ordering": "fused",
                                                       "observation_domain": "union"}})

    path_d = tmp_path / "arm_d"
    path_e = tmp_path / "arm_e"
    save_artifact(model_d, path_d)
    save_artifact(model_e, path_e)

    manifest_d = json.loads((path_d / "manifest.json").read_text())
    manifest_e = json.loads((path_e / "manifest.json").read_text())

    assert manifest_d["source_fingerprint"] != manifest_e["source_fingerprint"], (
        "two QuantizedModels differing ONLY in kv_calibration_provenance produced the "
        "SAME source_fingerprint -- the fingerprint is not sensitive to the provenance "
        "field, so a per-head-calibrated build (Arm D) and a differently-calibrated "
        "build (Arm E) saved from otherwise-identical model state would be silently "
        "indistinguishable by fingerprint alone"
    )


# ==============================================================================
# T-1940 addendum -- Q-scale equality pin extended to every layer
# (T-1939 §5.2, Significant, D-SLM2639/D-SLM2643; confirmation review of T-1937's
# production-fix round)
# ==============================================================================

def test_arm_d_q_scale_matches_shipped_production_q_path_at_every_layer():
    """T-1939 §5.2 (Significant): `test_arm_d_q_scale_matches_the_shipped_production_
    q_path` (`Tools/superslm_spike/tests/test_t1937_production_fix_pins.py`) checks the
    equality between Arm D/E's own captured Q scale and the shipped `_derive_scales`
    Q-scale chain at `"layer0"` ONLY -- the one layer of two on the §11 fixture where the
    pre-RoPE half of `_kv_calibration_capture`'s Q observation union has ZERO effect
    (union and post-RoPE-only max-abs coincide there: `4.59892462371` both ways, confirmed
    by execution this round). At `"layer1"` the two diverge (`3.41027051431` union vs.
    `3.39751225947` post-RoPE-only), so a mutation that drops the pre-RoPE
    `_observe(maxima, "{prefix}.q", q)` call from `_kv_calibration_capture` reproduces the
    ORIGINAL T-1936 defect's own measured `layer1` Q scale digit-for-digit
    (`0.026752065035162028`, confirmed by execution this round). When this cell was
    authored the sibling file's then-layer0-only pin stayed green under that mutation;
    commit `3cb010510e` has since extended the sibling pin to every layer, so the same
    mutation is now caught by BOTH tests (re-confirmed by execution, T-1939 §11).

    That sibling file is outside T-1940's own writable scope (this suite's own file is
    `test_armd_arme_kv_calibration.py`; the flagged pin lives in `test_t1937_production_
    fix_pins.py`, authored and owned by a different round -- T-1933's own casebook already
    states "no T-1937 pin file" among what this suite's rounds do not touch). This test
    closes the coverage gap at the SUITE level instead of editing that file: it re-derives
    the identical equality the sibling pin states, but loops over every layer, so the full
    suite discriminates the dropped-pre-RoPE mutation regardless of which file's pin
    catches it. The sibling pin's layer0-only scope was closed separately at `3cb010510e`
    (every-layer loop, proven red at layer1 under the same mutation).

    RED under the mutation: `layer1`'s built Q scale diverges from shipped by ~0.37%,
    caught by the loop below. Confirmed by execution this round
    (`Claude/Curie/t1933-armd-arme-red-suite-2026-08-11.md`'s T-1940 addendum): RED under
    the dropped-pre-RoPE mutation, GREEN after revert.
    """
    pipeline = require(MODULE)
    cfg = fixture_config(pipeline)
    weights, weight_scales, floats = pipeline._pinned_weights(cfg)
    float_weight = pipeline._dict_float_source(floats)
    records = pipeline.calibration_records()
    tokenize = pipeline._fixture_tokenize_prompt(cfg)
    record_tokenize = pipeline._bridge_record_tokenizer(tokenize)

    shipped_maxima = pipeline._calibrate(cfg, float_weight, records, record_tokenize)
    capture, capture_maxima = pipeline._kv_calibration_capture(
        cfg, float_weight, records, record_tokenize)

    for layer in range(cfg.num_hidden_layers):
        prefix = f"layer{layer}"
        shipped_attn_norm_scale = pipeline._attn_norm_scale(shipped_maxima, weight_scales, prefix)
        shipped_q_scale = pipeline._projection_scale(
            shipped_maxima, weight_scales, f"{prefix}.q", f"{prefix}.q_proj", shipped_attn_norm_scale)
        built_q_scale = pipeline._layer_q_scale(capture_maxima, weight_scales, prefix)
        assert built_q_scale == pytest.approx(shipped_q_scale, rel=1e-9), (
            f"Arm D/E's own Q-scale at {prefix} ({built_q_scale}) must equal the shipped "
            f"production `projection_scale` chain ({shipped_q_scale}) -- a divergence "
            f"here means the sweep is quantizing Q against a different scale than the "
            f"artifact will ship, at a layer the sibling file's own layer0-only pin "
            f"cannot see"
        )
