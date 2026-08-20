"""T-2208: pins for the repaired B3 pooled gate (`run_b3_multi_pair_check`) -- the min-form
`Delta_final[C] = min(Delta_relative[C], Delta_absolute[C])` over `composed_mean`/`composed_tail`
(T-2204 design §4/§5, D-SLM3731/D-SLM3741/D-SLM3744), the removal of `effect_mean`/`effect_tail`
from `accepted`'s AND (D-SLM3741, reasoning corrected D-SLM3749), and the UNRESOLVED disposition
for a fired `Delta_relative` floor (D-SLM3222).

Every cell here is algebraic: `pair_draws` are hand-built two-arrays-of-known-mean/se/p95
constructions (the same convention `test_sslm_convert_adapter_b3_diagnostic.py`'s own
`_two_point_draws` already uses), not a real model forward pass -- this file exercises
`run_b3_multi_pair_check`'s own arithmetic directly. The real-adapter, real-collector proof that
the repaired gate accepts a genuine adapter and rejects five genuinely corrupted ones is a
separate commissioning battery (`Claude/Brunel/t2208-pooled-gate-commissioning/`, Wizard repo),
required before this gate's readings are load-bearing (`StandardsDocument.md` §5.4).

Cell 1 (`test_absolute_anchor_rejects_a_uniformly_scaled_candidate_the_relative_only_form_
accepted`) is written RED-THEN-GREEN against the OLD (pre-T-2204) ratio-only statistic: the
construction is a PILOT/VALIDATION pair scaled together by the same factor (exactly D-SLM3217's
own named defect -- a corrupted adapter's PILOT and VALIDATION partitions drawn from the same
corrupted distribution, so a ratio of the two never moves regardless of how corrupted the
adapter is) -- the OLD `val < 1.5 * pilot` comparison, computed inline below from the same raw
values, ACCEPTS it; the CURRENT code, which additionally requires `val < reference_anchors[C]`,
REJECTS it. Confirmed by inlining the pre-repair formula (`sslm_convert_adapter.py`@`071c5f3`,
before this file's own branch) rather than asserting it from memory.
"""

import numpy as np
import pytest

import sslm_convert_adapter as A


def _pair_draws(*, composed_pilot, composed_val, effect_pilot=(0.01, 0.01),
                effect_val=(0.005, 0.005), name="pinned_pair"):
    """One synthetic (layer, proj) pair's own raw draws, built from plain arrays -- the SAME
    shape `_b3_collect_pair_raw_draws` returns, just hand-constructed instead of derived from a
    real forward pass. `run_b3_multi_pair_check` pools this list exactly as it pools real pairs."""
    raw = {
        "composed_pilot": np.asarray(composed_pilot, dtype=np.float64),
        "composed_val": np.asarray(composed_val, dtype=np.float64),
        "effect_pilot": np.asarray(effect_pilot, dtype=np.float64),
        "effect_val": np.asarray(effect_val, dtype=np.float64),
    }
    return [(name, raw)]


# --- The architectural property: `reference_anchors` is never derived from `pair_draws` --------

def test_reference_anchors_is_a_keyword_parameter_never_derived_from_pair_draws():
    """T-2204 design §9 dimension 7's own contract claim, made checkable by inspection rather
    than by convention: `run_b3_multi_pair_check` accepts `reference_anchors` as an explicit
    parameter (defaulting to the frozen module constant `_B3_REFERENCE_ANCHORS`, itself two plain
    float literals with zero references to `pair_draws`, `raw`, or any per-adapter quantity)."""
    import inspect
    sig = inspect.signature(A.run_b3_multi_pair_check)
    assert "reference_anchors" in sig.parameters
    assert sig.parameters["reference_anchors"].default is None  # resolved to the frozen constant
                                                                  # inside the function body, never
                                                                  # computed from an argument.
    assert set(A._B3_REFERENCE_ANCHORS) == {"composed_mean", "composed_tail"}
    for v in A._B3_REFERENCE_ANCHORS.values():
        assert isinstance(v, float) and v > 0.0


def test_reference_anchors_override_is_honored_and_echoed_back():
    """A caller-supplied `reference_anchors` dict is what the gate actually grades against -- not
    merely accepted and ignored -- and is echoed back on the result for auditability."""
    pair_draws = _pair_draws(composed_pilot=(1e-5, 1e-5), composed_val=(1e-4, 1e-4))
    tight = {"composed_mean": 1e-9, "composed_tail": 1e-9}
    result = A.run_b3_multi_pair_check(pair_draws, reference_anchors=tight)
    assert result["reference_anchors"] == tight
    assert result["delta_composed_mean_absolute"] == 1e-9
    assert result["composed_mean_disposition"] == "reject"  # val 1e-4 >> anchor 1e-9


# --- The min-form: the absolute anchor rejects what the ratio-only form could never reject -----

def test_absolute_anchor_rejects_a_uniformly_scaled_candidate_the_relative_only_form_accepted():
    """D-SLM3217's own named defect: PILOT and VALIDATION scaled together by the same factor make
    a ratio-only comparison invariant to the corruption's own magnitude. `composed_pilot=1.0`,
    `composed_val=1.2` -- both far above the frozen `composed_mean`/`composed_tail` anchors
    (3.452072e-4 / 1.693989e-3), simulating a whole-adapter corruption whose PILOT and VALIDATION
    partitions moved together."""
    pair_draws = _pair_draws(composed_pilot=(1.0, 1.0), composed_val=(1.2, 1.2))

    # RED (pre-repair): the OLD ratio-only comparison, computed inline from the same raw pilot/
    # val values this cell feeds the current code -- `sslm_convert_adapter.py`@`071c5f3`'s own
    # `delta_composed_mean = _B3_SAFETY_INFLATION * pilot_stat["upper_ci"]` collapses to
    # `1.5 * pilot_value` here (se=0 for a two-identical-point array), and `val < delta` ACCEPTS.
    old_delta_composed_mean = A._B3_SAFETY_INFLATION * 1.0
    old_delta_composed_tail = A._B3_SAFETY_INFLATION * 1.0
    assert 1.2 < old_delta_composed_mean, "the OLD ratio-only form must accept this construction"
    assert 1.2 < old_delta_composed_tail, "the OLD ratio-only form must accept this construction"

    # GREEN (current code): the min-form's absolute anchor is far below 1.2, so both surviving
    # conjuncts reject regardless of how the ratio-only term reads.
    result = A.run_b3_multi_pair_check(pair_draws)
    assert result["composed_mean_disposition"] == "reject"
    assert result["composed_tail_disposition"] == "reject"
    assert result["disposition"] == "reject"
    assert result["accepted"] is False
    assert result["delta_composed_mean"] == pytest.approx(A._B3_REFERENCE_ANCHORS["composed_mean"])
    assert result["delta_composed_tail"] == pytest.approx(A._B3_REFERENCE_ANCHORS["composed_tail"])


def test_relative_term_still_binds_for_a_healthy_candidate_near_its_own_reference_scale():
    """Design §4's own stated property: `Delta_relative` is the TIGHTER (more sensitive) bound
    for a healthy adapter close to its own reference scale -- the anchor should not be the binding
    term for an honest candidate. `composed_pilot=1e-5`, `composed_val=1.05e-5` (a 5% pilot-to-
    validation drift, well inside ordinary sampling noise) stays under BOTH terms, and the min of
    the two equals the (tighter) relative term, not the (looser, for this scale) absolute one."""
    pair_draws = _pair_draws(composed_pilot=(1e-5, 1e-5), composed_val=(1.05e-5, 1.05e-5))
    result = A.run_b3_multi_pair_check(pair_draws)
    assert result["composed_mean_disposition"] == "accept"
    assert result["composed_tail_disposition"] == "accept"
    assert result["accepted"] is True
    # the relative term (1.5e-5) is tighter than the frozen anchor (3.452072e-4) at this scale.
    assert result["delta_composed_mean"] == pytest.approx(result["delta_composed_mean_relative"])
    assert result["delta_composed_mean_relative"] < result["delta_composed_mean_absolute"]


# --- effect_mean/effect_tail are removed from `accepted`'s AND -----------------------------

def test_effect_conjuncts_no_longer_gate_accepted():
    """T-2204 design §4 (D-SLM3741, D-SLM3749): `effect_mean`/`effect_tail` are computed (still
    returned, informational) but no longer part of `accepted`'s AND. A construction whose
    `composed_*` conjuncts cleanly accept and whose `effect_*` conjuncts would have failed the
    OLD AND-of-4 must accept under the current code -- the informational fields still read False,
    proving they were graded and simply not consulted, not silently skipped."""
    pair_draws = _pair_draws(composed_pilot=(1e-5, 1e-5), composed_val=(1.05e-5, 1.05e-5),
                             effect_pilot=(0.01, 0.01), effect_val=(100.0, 100.0))
    result = A.run_b3_multi_pair_check(pair_draws)
    assert result["effect_mean_accepts"] is False
    assert result["effect_tail_accepts"] is False
    old_style_and_of_4 = (result["composed_mean_accepts"] and result["composed_tail_accepts"]
                          and result["effect_mean_accepts"] and result["effect_tail_accepts"])
    assert old_style_and_of_4 is False, "sanity: this construction WOULD have failed the old AND-of-4"
    assert result["accepted"] is True, (
        "the current gate must accept: composed_mean/composed_tail both clear, and effect_* no "
        "longer gates"
    )
    assert result["disposition"] == "accept"


# --- UNRESOLVED: a fired Delta_relative floor is distinguished from both ACCEPT and REJECT -----

def test_composed_mean_floor_firing_reads_unresolved_not_silently_rounded_to_a_pass_or_fail():
    """D-SLM3222/D-SLM2846: when the pooled PILOT population's own `mean + z*se` (pre-clamp) is
    negative, the pre-repair code clamped it to 0.0 and graded VALIDATION against that clamped
    floor; this repair instead reads the conjunct UNRESOLVED. Construction: 90% of the pilot
    population at a mild negative value, 10% at a larger positive value, at n=10,000 -- large
    enough that the bootstrap/parametric SE is small relative to the negative mean (floor fires)
    while the P95 (driven by the top 10%) still reads positive, so `composed_tail` is unaffected
    -- proving the floor is scoped to `composed_mean` alone, exactly as `_B3_SAFETY_INFLATION *
    (pooled_c_pilot_stat["mean"] + ...)` has no analogous floor for the P95-based tail term."""
    n = 10000
    composed_pilot = np.asarray([-0.05] * int(n * 0.9) + [0.3] * int(n * 0.1), dtype=np.float64)
    composed_val = np.asarray([0.0001] * 100, dtype=np.float64)
    pair_draws = _pair_draws(composed_pilot=composed_pilot, composed_val=composed_val)

    # Sanity: the pre-clamp relative term is genuinely negative at this construction.
    pilot_stat = A._b3_stat(composed_pilot)
    raw = A._B3_SAFETY_INFLATION * (pilot_stat["mean"] + A._B3_Z_95_ONE_SIDED * pilot_stat["se"])
    assert raw < 0.0, "fixture must reach the floor -- the pre-clamp relative term must be negative"
    assert pilot_stat["p95"] > 0.0, "fixture's own P95 must stay positive so composed_tail is unaffected"

    result = A.run_b3_multi_pair_check(pair_draws)
    assert result["composed_mean_disposition"] == "unresolved"
    assert result["composed_mean_accepts"] is False, "an unresolved conjunct is never rounded to a pass"
    assert result["composed_tail_disposition"] == "accept"
    assert result["disposition"] == "unresolved", (
        "the whole-gate disposition must surface UNRESOLVED, distinct from both ACCEPT and REJECT, "
        "when no surviving conjunct rejects but one is unresolved"
    )
    assert result["accepted"] is False


def test_a_genuine_reject_takes_priority_over_an_unresolved_conjunct_in_the_overall_disposition():
    """When one surviving conjunct genuinely rejects and the other is unresolved, the whole-gate
    disposition reads REJECT, not UNRESOLVED -- a concrete finding is not downgraded to "we
    couldn't tell" because an unrelated conjunct's own pilot sample happened to be thin."""
    n = 10000
    composed_pilot = np.asarray([-0.05] * int(n * 0.9) + [0.3] * int(n * 0.1), dtype=np.float64)
    # composed_tail's own relative Delta uses the SAME pooled p95 (0.3-driven); force a reject by
    # setting VALIDATION's own p95 far above both the relative and absolute tail anchors.
    composed_val = np.asarray([50.0] * 100, dtype=np.float64)
    pair_draws = _pair_draws(composed_pilot=composed_pilot, composed_val=composed_val)

    result = A.run_b3_multi_pair_check(pair_draws)
    assert result["composed_mean_disposition"] == "unresolved"
    assert result["composed_tail_disposition"] == "reject"
    assert result["disposition"] == "reject"
    assert result["accepted"] is False
