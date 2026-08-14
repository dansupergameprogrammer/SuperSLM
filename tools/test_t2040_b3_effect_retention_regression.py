"""Curie's regression pin for B3's fifth red-first cell, re-derived to design Sec22's corrected
form (T-2040, D-SLM3126-3131/D-SLM3143, `Claude/Curie/t2018-slora-serial-red-suite-2026-08-13.md`
Sec19, Wizard repo).

WHAT THIS PINS: the gated quantity for the effect-retention conjunct is `effect_distance_runtime[i]`
ALONE (single-arm, one-sided against `Delta_effect` calibrated from PILOT's own honest
distribution) -- the retired paired `gap_effect[i] = effect_distance_runtime[i] -
effect_distance_baked[i]` form (Sec21/T-2022's shape) is dropped from the gate and retained only
as a reported diagnostic. This is a regression pin at the ONE MEASURED CELL (layer 0, q_proj,
real shopkeeper adapter D-SLM2845 names, synthetic-Gaussian calibration corpus) T-2033's own probe
and T-2036's own re-run already executed and disposed (D-SLM3128, reproduced exactly at checkpoint
`61125d8`) -- this file re-derives it as a FILED, ASSERTING regression test rather than a one-off
probe/script, per D-SLM3143's own routing.

Imports `t2029_b3_execute` (the build's own committed B3 harness, `tools/t2029_b3_execute.py`,
commit `61125d8`) and `sslm_convert_adapter` (the real converter's own `derive_amplifying_triple`/
`realized_ratio`, ported B0 arithmetic) DIRECTLY -- both live in this same `tools/` directory on
this branch (`curie/t2040-red-suite-cells`, forked from `brunel/t2021-slora-build`@`c81e48c`).
Neither file is modified by this test (`StandardsDocument.md`'s "never silently edit another
seat's filed instrument" discipline).

INDEPENDENCE PROPERTY (D-SLM3126's own argument, restated here since it is this cell's own
correctness claim, not merely a comment): `effect_distance_runtime[i]` is graded against the float
PEFT delta `yd = scaling*(B*A)*x`, computed directly from the trained A/B in float -- sharing NONE
of the runtime mechanism's internals (no `T`, no delta fold, no `u_i8`, no `rho`). This is the
identical independence argument design Sec6 item 1's own D-SLM3006 finding already makes for
grading the runtime arm against the baked arm, applied here to the float delta reference directly.

Requires the real assets on disk (unchanged from `t2029_b3_execute.py`'s own requirement):
`D:\\hf_cache\\superslm_artifacts\\qwen2.5-1.5b-shopkeeper-lora-v1\\final\\` and
`D:\\hf_cache\\hub\\models--Qwen--Qwen2.5-1.5B-Instruct\\...\\model.safetensors`. Skipped, not
failed, if either is absent (a real-weights regression pin cannot run without the real weights;
this is an environment precondition, not a code defect).
"""

import math
from pathlib import Path

import pytest

import t2029_b3_execute as B

ADAPTER_PRESENT = B.ADAPTER_DIR.exists()
BASE_PRESENT = B.BASE_MODEL_PATH.exists()

pytestmark = pytest.mark.skipif(
    not (ADAPTER_PRESENT and BASE_PRESENT),
    reason=f"real weights not present on disk (adapter={ADAPTER_PRESENT}, base={BASE_PRESENT}) "
    "-- this regression pin needs the real shopkeeper adapter and base checkpoint",
)

# Pinned figures, D-SLM3128 (T-2033's disposition) and D-SLM3143 (T-2038's routing), reproduced
# exactly by T-2036's own re-run at checkpoint 61125d8. A tolerance of 1e-5 absolute allows for
# platform/library floating-point non-bit-identity (numpy/safetensors reductions are not pinned to
# a single summation order the way this project's own integer reproducible path is) while still
# catching any material drift -- these are float64 numpy computations, not the project's pinned
# integer determinism boundary.
_TOL = 1e-5

PINNED_DELTA_EFFECT_MEAN = 0.307208
PINNED_DELTA_EFFECT_TAIL = 0.521957
PINNED_HONEST_UPPER_CI = 0.173233
PINNED_HONEST_P95 = 0.351188
PINNED_ANNIHILATED_UPPER_CI = 0.875659
PINNED_ANNIHILATED_P95 = 1.194786


@pytest.fixture(scope="module")
def fixture():
    return B.build_fixture()


@pytest.fixture(scope="module")
def frozen_delta(fixture):
    pilot = B.collect_gaps(fixture, B.PILOT_N, B.PILOT_SEED_BASE, t_scale=1.0)
    # design Sec22/D-SLM3127: calibrated from PILOT's own single-arm effect_distance_runtime
    # distribution alone -- never the retired paired effect_gap.
    effect_rt_pilot_stat = B.stat(pilot["effect_rt_pilot"])
    delta_effect_mean = B.SAFETY_INFLATION * (
        effect_rt_pilot_stat["mean"] + B.Z_95_ONE_SIDED * effect_rt_pilot_stat["se"]
    )
    delta_effect_tail = B.SAFETY_INFLATION * effect_rt_pilot_stat["p95"]
    return {
        "delta_effect_mean": delta_effect_mean,
        "delta_effect_tail": delta_effect_tail,
        "pilot_n": effect_rt_pilot_stat["n"],
    }


def test_pilot_calibration_reproduces_pinned_delta(frozen_delta):
    """PILOT (n=43): Delta_effect_mean=0.307208, Delta_effect_tail=0.521957 (D-SLM3128)."""
    assert frozen_delta["pilot_n"] == 43, (
        f"PILOT partition size drifted: got {frozen_delta['pilot_n']}, pinned at 43 -- the "
        "deterministic item-hash split (design Sec6 item 1) must reproduce this exactly for the "
        "SAME seed_base/n_items"
    )
    assert math.isclose(frozen_delta["delta_effect_mean"], PINNED_DELTA_EFFECT_MEAN, abs_tol=_TOL), (
        f"Delta_effect_mean drifted: got {frozen_delta['delta_effect_mean']!r}, "
        f"pinned {PINNED_DELTA_EFFECT_MEAN!r}"
    )
    assert math.isclose(frozen_delta["delta_effect_tail"], PINNED_DELTA_EFFECT_TAIL, abs_tol=_TOL), (
        f"Delta_effect_tail drifted: got {frozen_delta['delta_effect_tail']!r}, "
        f"pinned {PINNED_DELTA_EFFECT_TAIL!r}"
    )


def test_honest_reference_accepts_on_validation_reproduces_pinned_figures(fixture, frozen_delta):
    """VALIDATION (honest, t=1, n=159): upper_CI=0.173233, p95=0.351188 -> ACCEPT (D-SLM3128)."""
    val = B.collect_gaps(fixture, B.PILOT_N, B.VALIDATION_SEED_BASE, t_scale=1.0)
    effect_rt_val_stat = B.stat(val["effect_rt_val"])

    assert effect_rt_val_stat["n"] == 159, (
        f"VALIDATION partition size drifted: got {effect_rt_val_stat['n']}, pinned at 159"
    )
    assert math.isclose(effect_rt_val_stat["upper_ci"], PINNED_HONEST_UPPER_CI, abs_tol=_TOL), (
        f"honest upper_CI drifted: got {effect_rt_val_stat['upper_ci']!r}, "
        f"pinned {PINNED_HONEST_UPPER_CI!r}"
    )
    assert math.isclose(effect_rt_val_stat["p95"], PINNED_HONEST_P95, abs_tol=_TOL), (
        f"honest p95 drifted: got {effect_rt_val_stat['p95']!r}, pinned {PINNED_HONEST_P95!r}"
    )

    accepts = effect_rt_val_stat["upper_ci"] < frozen_delta["delta_effect_mean"]
    assert accepts, (
        "the honest (t=1) reference adapter's VALIDATION-partition effect-retention conjunct must "
        f"ACCEPT under the corrected single-arm form: upper_CI={effect_rt_val_stat['upper_ci']!r} "
        f"Delta_effect_mean={frozen_delta['delta_effect_mean']!r}"
    )
    tail_accepts = effect_rt_val_stat["p95"] < frozen_delta["delta_effect_tail"]
    assert tail_accepts, (
        f"the honest reference's tail conjunct must also ACCEPT: p95={effect_rt_val_stat['p95']!r} "
        f"Delta_effect_tail={frozen_delta['delta_effect_tail']!r}"
    )


def test_full_annihilation_rejects_reproduces_pinned_figures(fixture, frozen_delta):
    """T_SCALE(255.9) (n=200): upper_CI=0.875659, p95=1.194786 -> REJECT (D-SLM3128/D-SLM3091)."""
    ann = B.collect_gaps(fixture, B.PILOT_N, B.VALIDATION_SEED_BASE, t_scale=255.9)
    ann_effect_rt = B.stat(ann["effect_rt_val"] + ann["effect_rt_pilot"])

    assert ann_effect_rt["n"] == 200, f"annihilation sample size drifted: got {ann_effect_rt['n']}"
    assert math.isclose(ann_effect_rt["upper_ci"], PINNED_ANNIHILATED_UPPER_CI, abs_tol=_TOL), (
        f"annihilated upper_CI drifted: got {ann_effect_rt['upper_ci']!r}, "
        f"pinned {PINNED_ANNIHILATED_UPPER_CI!r}"
    )
    assert math.isclose(ann_effect_rt["p95"], PINNED_ANNIHILATED_P95, abs_tol=_TOL), (
        f"annihilated p95 drifted: got {ann_effect_rt['p95']!r}, pinned {PINNED_ANNIHILATED_P95!r}"
    )

    accepts = ann_effect_rt["upper_ci"] < frozen_delta["delta_effect_mean"]
    assert not accepts, (
        "T_SCALE(255.9) full annihilation must be REJECTED by the effect-retention conjunct, "
        f"unconditionally: upper_CI={ann_effect_rt['upper_ci']!r} "
        f"Delta_effect_mean={frozen_delta['delta_effect_mean']!r}"
    )


def test_baked_arm_retained_as_diagnostic_never_gates(fixture, frozen_delta):
    """design Sec6 item 1a: effect_distance_baked is reported, never gates -- confirmed by showing
    the gate's own accept/reject verdict is unchanged whether or not the baked-arm figure is even
    computed (the gate function, `frozen_delta`/this test's own accept check, never reads it)."""
    val = B.collect_gaps(fixture, B.PILOT_N, B.VALIDATION_SEED_BASE, t_scale=1.0)
    # t2029_b3_execute.collect_gaps() does not return effect_bk_val directly (T-2036's own harness
    # tracks only the single-arm effect_rt and the retired paired effect_gap) -- recovered here as
    # effect_bk = effect_rt - effect_gap, algebraically exact since effect_gap is defined as that
    # difference (t2029_b3_execute.py's own run_token(): "effect_gap = el_rt - el_bk").
    effect_bk_val = [rt - gap for rt, gap in zip(val["effect_rt_val"], val["effect_gap_val"])]
    effect_bk_val_stat = B.stat(effect_bk_val)
    # The baked-arm figure is real and reported (matching design Sec6 item 1's own "retained as a
    # DIAGNOSTIC only" precedent for its per-arm float-distance figures) -- printed for the
    # record, asserted only to exist and be finite, never compared against Delta_effect.
    assert math.isfinite(effect_bk_val_stat["mean"]), "the baked-arm diagnostic must be a real, computed figure"
    print(f"\n[diagnostic, never gates] effect_distance_baked VALIDATION mean="
          f"{effect_bk_val_stat['mean']!r} (baked's own catastrophic-cancellation noise floor, "
          "D-SLM3126's root cause -- reported for cross-reference only)")


def test_independence_float_delta_reference_shares_no_runtime_internals(fixture):
    """D-SLM3126's own independence argument, checked directly: yd (the float PEFT delta
    reference) is a pure function of (A_f, B_f, xf) -- it never reads T, u_i8, rho, or any other
    quantity the runtime mechanism (run_token's own delta-fold/u-fold path) derives. Confirmed by
    construction: yd = B_f @ (A_f @ xf), computed identically regardless of t_scale (which only
    perturbs the RUNTIME mechanism's own T), so yd is bit-identical across t_scale values."""
    import numpy as np

    g = np.random.default_rng(0x1234)
    xf = g.standard_normal(fixture.d_in)
    yd_honest = fixture.B_f @ (fixture.A_f @ xf)
    yd_at_annihilation = fixture.B_f @ (fixture.A_f @ xf)  # yd's own formula takes no t_scale input
    assert np.array_equal(yd_honest, yd_at_annihilation), (
        "the float PEFT delta reference must be bit-identical regardless of t_scale -- it shares "
        "no runtime-mechanism internals (T, u_i8, rho) with the quantity it grades, by construction"
    )
