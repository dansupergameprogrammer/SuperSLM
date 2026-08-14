"""Standing cells for the corrected freeze-health gate (design Sec26.15 as corrected, T-2099).

These promote the T-2099 probe's own findings to permanent coverage, per D-SLM3297. The probe runs
the real collector and takes minutes; these run on synthetic arrays in under a second and pin the
DECISION RULE rather than any particular measured number -- so a future edit that reintroduces the
one-sided form, or that quietly turns the three-valued verdict back into a boolean, reddens here.

The finding these exist for (external review, 2026-08-14; D-SLM3295): Sec26.15 derived resolving
power from the REFERENCE sample alone and escalated the reference alone, which understates the
ratio's uncertainty always and can drive any cell to a false RESOLVED. `test_reference_only_
escalation_cannot_manufacture_resolution` is that finding, as a cell.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sslm_convert_adapter as A  # noqa: E402


def _sample(n, *, scale=1.0, loc=1.0, seed=0):
    """A positive-valued sample with a stable shape -- the gate's own populations are
    non-negative L2-gap magnitudes, and both statistics it reads (sd, p95) need a spread."""
    rng = np.random.default_rng(seed)
    return np.abs(rng.normal(loc=loc, scale=0.25 * scale, size=n)) + 0.05


# --------------------------------------------------------------------- the joint SE composition

def test_ratio_relative_se_composes_both_samples():
    a = _sample(400, seed=1)
    b = _sample(400, seed=2)
    rel_a = A.freeze_relative_bootstrap_se(a, A._freeze_p95, n_resamples=400, seed=7)
    rel_b = A.freeze_relative_bootstrap_se(b, A._freeze_p95, n_resamples=400, seed=8)
    joint = A.freeze_ratio_relative_se(a, b, A._freeze_p95, n_resamples=400, seed=7)
    # The joint term is strictly larger than EITHER side alone -- the property Sec26.15 lacked.
    assert joint > rel_a
    assert joint > rel_b
    assert joint == pytest.approx(math.sqrt(rel_a ** 2 + rel_b ** 2), rel=0.35)


def test_joint_margin_is_materially_smaller_than_the_one_sided_margin_it_replaces():
    """At comparable sample sizes the correction costs about a factor of sqrt(2). Asserted as a
    MATERIAL gap rather than a strict inequality: `joint < one_sided` alone is satisfied by
    bootstrap seed luck if the joint term silently degenerates back to the one-sided one, so it
    would not discriminate the defect it exists for."""
    cand = _sample(300, seed=3)
    ref = _sample(300, seed=4)
    band = A._FREEZE_BANDS["p95_pooled"]
    one_sided = band / A.freeze_relative_bootstrap_se(ref, A._freeze_p95, n_resamples=400, seed=9)
    joint = band / A.freeze_ratio_relative_se(cand, ref, A._freeze_p95, n_resamples=400, seed=9)
    assert one_sided / joint > 1.2, (
        f"one-sided={one_sided:.4f} joint={joint:.4f} -- the joint term is not composing both "
        "samples; it has degenerated to the reference-only form Sec26.15 shipped")
    assert one_sided / joint == pytest.approx(math.sqrt(2.0), rel=0.30)


def test_reference_only_escalation_cannot_manufacture_resolution():
    """THE FINDING. Growing the REFERENCE alone drives the one-sided margin up without bound while
    the joint margin plateaus at the candidate's own contribution. Sec26.15's specified escalation
    was therefore a false-RESOLVED generator."""
    band = A._FREEZE_BANDS["p95_pooled"]
    cand = _sample(40, seed=5)                       # the production candidate, never escalated
    one_sided, joint = [], []
    for n_ref in (40, 400, 4000, 40000):
        ref = _sample(n_ref, seed=6)
        one_sided.append(band / A.freeze_relative_bootstrap_se(
            ref, A._freeze_p95, n_resamples=300, seed=11))
        joint.append(band / A.freeze_ratio_relative_se(
            cand, ref, A._freeze_p95, n_resamples=300, seed=11))
    # The one-sided figure grows without limit as the reference grows...
    assert one_sided[-1] > 5 * one_sided[0]
    # ...while the joint figure is bounded by the candidate's own resolving power.
    cand_only = band / A.freeze_relative_bootstrap_se(cand, A._freeze_p95, n_resamples=300, seed=11)
    assert joint[-1] <= cand_only * 1.05
    assert joint[-1] < one_sided[-1] / 3, "the gap the finding is about must be large, not marginal"


def test_joint_escalation_of_both_sides_does_raise_the_margin():
    """The corrective direction: escalating BOTH sides together is what actually buys resolution."""
    band = A._FREEZE_BANDS["p95_pooled"]
    margins = []
    for n in (40, 400, 4000):
        margins.append(band / A.freeze_ratio_relative_se(
            _sample(n, seed=12), _sample(n, seed=13), A._freeze_p95, n_resamples=300, seed=14))
    assert margins[0] < margins[1] < margins[2]


# ------------------------------------------------------------------------ the three-valued rule

def test_unresolved_is_checked_first_even_when_out_of_band():
    """D-SLM3288: UNRESOLVED is evaluated BEFORE the band comparison. A wildly out-of-band
    candidate at low power reads UNRESOLVED, never REFUSE -- an underpowered sample cannot convict."""
    cand = _sample(8, scale=3.0, loc=4.0, seed=15)
    ref = _sample(8, seed=16)
    tier = A.freeze_tier_verdict(cand, ref, band_sd=0.20, band_p95=0.10, n_resamples=200, seed=17)
    assert tier["margin"] < A._FREEZE_MARGIN_SE
    assert tier["in_band"] is False
    assert tier["verdict"] == A.FREEZE_UNRESOLVED


def test_in_band_but_underpowered_is_unresolved_and_never_pass():
    """The ordering's DANGEROUS direction, and the one the out-of-band cell above cannot see: a
    candidate that lands INSIDE the band on an underpowered sample. Checking the band first turns
    that into a PASS, which is exactly D-SLM2846's own rule ("a measured delta falling inside the
    achieved resolving power is UNRESOLVED, never PASSED") inverted.

    Found by mutation: reordering the two checks left every other cell in this file green."""
    cand = _sample(10, seed=29)
    ref = _sample(10, seed=29)          # same stream -> in band by construction
    tier = A.freeze_tier_verdict(cand, ref, band_sd=0.20, band_p95=0.10, n_resamples=200, seed=30)
    assert tier["in_band"] is True, "fixture precondition: the candidate must land inside the band"
    assert tier["margin"] < A._FREEZE_MARGIN_SE, "fixture precondition: the sample must be underpowered"
    assert tier["verdict"] == A.FREEZE_UNRESOLVED, (
        "an in-band reading on an underpowered sample is UNRESOLVED -- reporting it as PASS is "
        "reading 'no difference detected' as 'no difference exists'")


def test_the_gate_does_not_allow_a_freeze_on_an_underpowered_in_band_reading():
    """The same rule at the gate's own boundary rather than the tier's: an honest, in-band
    candidate whose sample is too small must not produce freeze_allowed=True on step one."""
    def collect_pair(_name, pilot_n):
        n = max(6, pilot_n // 5)
        return _sample(n, seed=31), _sample(n, seed=31)
    result = A.run_freeze_health_gate(
        collect_pair, ["layer0.q_proj"], projection_type_of=lambda n: n.split(".")[1],
        pilot_n=30, pilot_n_ceiling=30, budget_seconds=1e9, n_resamples=200,
        time_fn=_FakeClock(1.0))
    assert result["history"][0]["tier1"]["in_band"] is True
    assert result["history"][0]["tier1"]["margin"] < A._FREEZE_MARGIN_SE
    assert result["freeze_allowed"] is False
    assert result["overall"] == A.FREEZE_UNRESOLVED


def test_resolved_and_out_of_band_is_refuse():
    cand = _sample(6000, loc=3.0, seed=18)
    ref = _sample(6000, loc=1.0, seed=19)
    tier = A.freeze_tier_verdict(cand, ref, band_sd=0.20, band_p95=0.10, n_resamples=300, seed=20)
    assert tier["margin"] >= A._FREEZE_MARGIN_SE
    assert tier["verdict"] == A.FREEZE_REFUSE


def test_resolved_and_in_band_is_pass_and_is_not_vacuous():
    """The non-vacuous must-accept: candidate and reference are INDEPENDENT draws of the same
    distribution, so no ratio is exactly 1.0 -- unlike Sec26.14.3's retired same-array control,
    which returned 1.0000 under any band and could not fail."""
    cand = _sample(6000, seed=21)
    ref = _sample(6000, seed=22)
    tier = A.freeze_tier_verdict(cand, ref, band_sd=0.20, band_p95=0.10, n_resamples=300, seed=23)
    assert tier["verdict"] == A.FREEZE_PASS
    assert tier["per_stat"]["p95"]["ratio"] != 1.0
    assert tier["per_stat"]["sd"]["ratio"] != 1.0


def test_the_same_array_control_is_vacuous_and_that_is_why_it_was_retired():
    """Kept as a measured property rather than a claim: passing the identical array as both sides
    reads a ratio of exactly 1.0 under ANY band, which is a control that cannot fail."""
    arr = _sample(200, seed=24)
    tier = A.freeze_tier_verdict(arr, arr, band_sd=0.0001, band_p95=0.0001,
                                 n_resamples=200, seed=25)
    assert tier["per_stat"]["p95"]["ratio"] == 1.0
    assert tier["per_stat"]["sd"]["ratio"] == 1.0
    assert tier["in_band"] is True, "in-band under a band of 0.0001 -- the definition of vacuous"


def test_tier_margin_is_the_minimum_over_the_statistics_that_gate_it():
    """Sec26.15 read resolving power from the p95 alone while the tier also gates on sd. A tier is
    no better resolved than the worst statistic that can reject it."""
    cand = _sample(500, seed=26)
    ref = _sample(500, seed=27)
    tier = A.freeze_tier_verdict(cand, ref, band_sd=0.20, band_p95=0.10, n_resamples=300, seed=28)
    assert tier["margin"] == min(s["margin"] for s in tier["per_stat"].values())
    assert tier["margin"] <= tier["margin_p95_only"]
    assert tier["limiting_stat"] in ("sd", "p95")


# ------------------------------------------------------------------------------- composition

@pytest.mark.parametrize("verdicts,expected", [
    ([A.FREEZE_PASS], A.FREEZE_PASS),
    ([A.FREEZE_PASS, A.FREEZE_PASS], A.FREEZE_PASS),
    ([A.FREEZE_PASS, A.FREEZE_UNRESOLVED], A.FREEZE_UNRESOLVED),
    ([A.FREEZE_PASS, A.FREEZE_REFUSE], A.FREEZE_REFUSE),
    ([A.FREEZE_UNRESOLVED, A.FREEZE_REFUSE], A.FREEZE_REFUSE),
    ([A.FREEZE_REFUSE, A.FREEZE_UNRESOLVED, A.FREEZE_PASS], A.FREEZE_REFUSE),
])
def test_across_tier_precedence_refuse_dominates_unresolved_dominates_pass(verdicts, expected):
    assert A.compose_freeze_verdict(verdicts)["overall"] == expected


def test_freeze_allowed_is_true_only_on_pass_and_never_conflates_refuse_with_unresolved():
    assert A.compose_freeze_verdict([A.FREEZE_PASS])["freeze_allowed"] is True
    refuse = A.compose_freeze_verdict([A.FREEZE_REFUSE])
    unresolved = A.compose_freeze_verdict([A.FREEZE_UNRESOLVED])
    assert refuse["freeze_allowed"] is False and unresolved["freeze_allowed"] is False
    # Both read False and they are NOT the same disposition -- the three-valued verdict must
    # survive alongside the boolean, or the operator cannot tell "collect more" from "reject".
    assert refuse["overall"] != unresolved["overall"]


# ------------------------------------------------------------------------------ the budget

def test_adapter_level_budget_is_not_a_per_pair_ceiling_multiplied_out():
    """D-SLM3296. Sec26.15.3 derived a 30-minute ceiling for ONE pair on ONE side. Charging that
    per pair, on both sides, over a 196-pair adapter authorises 196 hours, which is not a ceiling."""
    per_pair_multiplied = 196 * 2 * 1800.0
    assert A._FREEZE_ESCALATION_BUDGET_SECONDS < per_pair_multiplied / 100.0


def test_predicted_step_cost_charges_both_sides():
    one_side = 196 * A._FREEZE_COST_A * (4800.0 ** A._FREEZE_COST_B)
    assert A.freeze_predicted_step_seconds(4800, 196) == pytest.approx(2.0 * one_side)


# --------------------------------------------------------------- the escalation loop end to end

class _FakeClock:
    def __init__(self, per_call=1.0):
        self.t, self.per_call = 0.0, per_call

    def __call__(self):
        self.t += self.per_call
        return self.t


def _stable_name_seed(name: str) -> int:
    """A deterministic per-name offset. NOT `hash(name)`: Python randomises string hashing per
    process (PYTHONHASHSEED), which made these cells pass or fail depending on the interpreter
    run -- caught by re-running the same mutation twice and getting two different failure sets."""
    return sum((i + 1) * b for i, b in enumerate(name.encode("utf-8"))) % 97


def _collector(*, cand_loc=1.0, seed=100):
    """A collector whose sample size grows with pilot_n, exactly as the real one does -- so the
    loop's escalation genuinely buys resolving power on BOTH sides."""
    calls = []

    def collect_pair(name, pilot_n):
        calls.append((name, pilot_n))
        n = max(8, pilot_n // 5)
        off = _stable_name_seed(name)
        return (_sample(n, loc=cand_loc, seed=seed + off),
                _sample(n, loc=1.0, seed=seed + 1000 + off))

    return collect_pair, calls


def test_gate_escalates_both_sides_at_the_same_step():
    collect_pair, calls = _collector()
    result = A.run_freeze_health_gate(
        collect_pair, ["layer0.q_proj"], projection_type_of=lambda n: n.split(".")[1],
        pilot_n=50, n_resamples=200, time_fn=_FakeClock(1.0))
    # One call per pair per step returns BOTH sides at the same pilot_n -- there is no code path
    # that grows the reference without growing the candidate.
    assert len(calls) == result["escalation_steps"]
    assert [pn for _n, pn in calls] == [50 * (2 ** i) for i in range(result["escalation_steps"])]


def test_gate_resolves_an_honest_candidate_and_allows_the_freeze():
    collect_pair, _calls = _collector(cand_loc=1.0)
    result = A.run_freeze_health_gate(
        collect_pair, ["layer0.q_proj"], projection_type_of=lambda n: n.split(".")[1],
        pilot_n=200, n_resamples=200, time_fn=_FakeClock(1.0))
    assert result["overall"] == A.FREEZE_PASS
    assert result["freeze_allowed"] is True
    assert result["stop_reason"] == "resolved"


def test_gate_refuses_a_corrupted_candidate():
    collect_pair, _calls = _collector(cand_loc=4.0)
    result = A.run_freeze_health_gate(
        collect_pair, ["layer0.q_proj"], projection_type_of=lambda n: n.split(".")[1],
        pilot_n=200, n_resamples=200, time_fn=_FakeClock(1.0))
    assert result["overall"] == A.FREEZE_REFUSE
    assert result["freeze_allowed"] is False


def test_gate_stops_on_the_adapter_budget_and_says_so():
    collect_pair, _calls = _collector()
    result = A.run_freeze_health_gate(
        collect_pair, ["layer0.q_proj"], projection_type_of=lambda n: n.split(".")[1],
        pilot_n=8, budget_seconds=3.0, n_resamples=100, time_fn=_FakeClock(1.0))
    assert result["stop_reason"] == "adapter_budget_exhausted"
    assert result["overall"] == A.FREEZE_UNRESOLVED
    assert result["freeze_allowed"] is False


def test_gate_stops_on_the_pilot_n_ceiling_and_says_so():
    collect_pair, _calls = _collector()
    result = A.run_freeze_health_gate(
        collect_pair, ["layer0.q_proj"], projection_type_of=lambda n: n.split(".")[1],
        pilot_n=8, pilot_n_ceiling=16, budget_seconds=1e9, n_resamples=100,
        time_fn=_FakeClock(1.0))
    assert result["stop_reason"] == "pilot_n_ceiling"
    assert result["overall"] == A.FREEZE_UNRESOLVED


def test_a_budget_stop_is_never_reported_as_a_pass():
    """'We ran out of budget' and 'no difference detected' must not be the same output --
    StandardsDocument.md Sec5.4's own null-result rule, at the gate's own boundary."""
    collect_pair, _calls = _collector()
    result = A.run_freeze_health_gate(
        collect_pair, ["layer0.q_proj"], projection_type_of=lambda n: n.split(".")[1],
        pilot_n=8, budget_seconds=3.0, n_resamples=100, time_fn=_FakeClock(1.0))
    assert result["freeze_allowed"] is False
    assert result["overall"] != A.FREEZE_PASS
    assert result["stop_reason"] != "resolved"


def test_history_records_every_step_for_the_operator_facing_log():
    collect_pair, _calls = _collector()
    result = A.run_freeze_health_gate(
        collect_pair, ["layer0.q_proj"], projection_type_of=lambda n: n.split(".")[1],
        pilot_n=100, n_resamples=200, time_fn=_FakeClock(1.0))
    assert result["history"], "the escalation history is the operator-facing record"
    for step in result["history"]:
        assert step["overall"] in (A.FREEZE_PASS, A.FREEZE_REFUSE, A.FREEZE_UNRESOLVED)
        assert "tier1" in step and "tier2" in step
        assert step["tier1"]["margin"] == min(
            s["margin"] for s in step["tier1"]["per_stat"].values())
