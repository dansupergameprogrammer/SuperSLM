"""T-2213 (D-SLM3783): Brunel's red-first suite for the B3 magnitude sanity check that replaces
the retired pooled ACCEPT/REJECT gate (`run_b3_pooled_report`, `tools/sslm_convert_adapter.py`).

Five adversary strikes (`Claude/Loki/t2205`, `t2207`, `t2209`, `t2210`, `t2211`,
`Claude/Vitruvius/t2204`'s own fold round 4, Wizard repo) established that the retired pooled
gate's accept boundary was one frozen reference adapter's own idiosyncrasy: honest in-domain
adapters rejected on magnitude alone (9 of 18 at >=2.5x the reference) and on pure structure alone
(20 of 60 at the reference's own magnitude), and no in-band corruption construction ever elevated
the statistic once magnitude was intercepted. Dan's ruling (D-SLM3783): delete the arithmetic, ship
the per-pair diagnostics (unchanged, tested elsewhere -- `test_sslm_convert_adapter_b3_diagnostic
.py`) plus a wide, non-blocking magnitude sanity WARNING.

This is the builder's own red-first suite, authored in the same context as the check it tests --
NOT the independent, blind-to-the-builder's-own-controls commissioning `StandardsDocument.md`
§5.4 requires before an instrument's verdicts are load-bearing. That independent pass is owed, not
claimed here (`Claude/Poirot/8af620a-t2214-gate-retirement-confirmation.md` finding S6): this
file's own must-accept sweep (Cell 2, 1.0x-6.0x) is entirely inside the shipped band and is
structurally unable to surface the honest candidates the arc's own executed census already
recorded above the band (~25x, ~100x -- see `_B3_MAGNITUDE_WARN_RATIO`'s own derivation comment),
which is exactly the blind spot the independence clause exists to catch. The check's readings are
not load-bearing (they never gate emission), which is what keeps this suite legitimate as a
mechanism check in the meantime.

Cells below pin the mechanism `_b3_collect_pair_raw_draws`/`run_b3_pooled_report` actually
implement: the must-accept sweep confirms the check does not repeat the retired gate's own
false-REJECT (a candidate T-2210's own strike showed the OLD gate wrongly refusing), the must-fire
cells confirm a grossly rescaled candidate is at least flagged for visibility, and the boundary
cells (Cell 3b) pin the literal shipped constant with numeric literals rather than reading it back
from the module, so the suite cannot silently tolerate the constant drifting to a materially
different value. None of this establishes that the check discriminates honest from corrupted
adapters in general -- the derivation comment above `_B3_MAGNITUDE_WARN_RATIO` states plainly
that the full executed census shows it does not. Both constructions below are producible by the
real production data path: `_b3_collect_pair_raw_draws` is the real per-pair collector, called
here with hand-built `(a_f, b_scaled)` pairs exactly as
`Claude/Loki/t2210-probe/t2210_guard_probe.py`'s own `collect()` calls it, not a reimplementation.
"""

import numpy as np
import pytest

import sslm_convert_adapter as A

# The threshold under test, read from the module so this file never drifts from what actually
# ships if `_B3_MAGNITUDE_WARN_RATIO` is re-derived later.
_RATIO = A._B3_MAGNITUDE_WARN_RATIO


def _collect(w_f, a_f, b_scaled, pilot_n=8):
    """The real per-pair preparation + collection path (`sslm_convert_adapter.py`'s own
    `build_runtime_additive_sections` loop, mirrored here exactly as
    `Claude/Loki/t2210-probe/t2210_guard_probe.py`'s own `collect()` does), at a small pilot_n so
    this suite runs fast -- the arithmetic under test (delta_norm_sq, the magnitude-ratio band) is
    independent of pilot_n; only `_b3_pair_diagnostic`'s own bootstrap SEs consume the draw count,
    and this file does not assert anything about those margins."""
    pipeline = A._load_spike()
    d_out, d_in = w_f.shape
    Wc, w_scales = pipeline.quantize_weight_per_channel(w_f, output_axis=0)
    w = np.asarray(w_scales, dtype=np.float64)
    S = float(np.max(w))
    Ac, alpha_scales = pipeline.quantize_weight_per_channel(a_f, output_axis=0)
    alpha = np.asarray(alpha_scales, dtype=np.float64)
    Bc, beta_scales = pipeline.quantize_weight_per_channel(b_scaled, output_axis=0)
    beta = np.asarray(beta_scales, dtype=np.float64)

    g = np.random.default_rng(0xC0FFEE)
    xf = g.standard_normal(d_in)
    xmax = float(np.max(np.abs(xf)))
    X = xmax / 127.0 if xmax > 0.0 else 1.0
    xc = np.clip(np.round(xf / X), -127, 127).astype(np.int64)
    u_acc = Ac.astype(np.int64) @ xc
    T_value = A.compute_t(list(alpha), list(np.abs(u_acc)))

    return A._b3_collect_pair_raw_draws(w_f, a_f, b_scaled, Wc, w, S, Ac, alpha, Bc, beta, T_value,
                                        pilot_n=pilot_n)


def _honest_pair(rng, d=8, rank=2, scale=0.02):
    w_f = rng.normal(scale=0.02, size=(d, d))
    a_f = rng.normal(scale=scale, size=(rank, d))
    b_scaled = rng.normal(scale=scale, size=(d, rank))
    return w_f, a_f, b_scaled


# --- Cell 1: `_b3_collect_pair_raw_draws` reports the real Frobenius-norm-squared of `b_scaled @
#     a_f`, not a placeholder -----------------------------------------------------------------

def test_delta_norm_sq_matches_the_real_composed_delta():
    rng = np.random.default_rng(0)
    w_f, a_f, b_scaled = _honest_pair(rng)
    raw = _collect(w_f, a_f, b_scaled)
    expected = float(np.sum((b_scaled @ a_f) ** 2))
    assert raw["delta_norm_sq"] == pytest.approx(expected, rel=1e-12)


# --- Cell 2: must-not-fire -- an honest sweep, including candidates at and above the >=2.5x
#     magnitude T-2210's own strike named, produces no warning ---------------------------------

@pytest.mark.parametrize("magnitude_ratio", [1.0, 1.5, 2.0, 2.5, 4.0, 6.0])
def test_honest_sweep_up_to_and_beyond_2_5x_reference_magnitude_never_warns(magnitude_ratio):
    """T-2210's own fracture (`Claude/Loki/t2210-t2204-fold3-strike-2026-08-20.md`, D-SLM3769):
    honest, uncorrupted, in-domain candidates were falsely REJECTed by the retired pooled gate at
    magnitude >=2.5x the reference -- T-2210's own strike varied magnitude along the SAME axis
    production reads (`lora_alpha`/`r`, folded once into `b_scaled` before the collector ever
    runs, `AdapterMeta.scaling`/`read_peft_lora_pair`, `sslm_convert_adapter.py:401-424,482`),
    which scales the pooled composed delta LINEARLY (`delta = b_scaled @ a_f`, linear in
    `b_scaled`). This cell reproduces that exact axis: the candidate is the SAME honest `(w_f,
    a_f)` as the reference, with `b_scaled` scaled by `magnitude_ratio` -- an ordinary honest
    adapter trained (or declared) at a different `lora_alpha`, not a corrupted weight tensor. The
    replacement check must never repeat T-2210's false-REJECT -- this is the direct must-accept
    commissioning cell `StandardsDocument.md` §5.4 requires, run across the observed honest range
    (up to the ~6.2-6.3x ceiling the fold-round-4 probe measured on its own [quadratic] axis,
    `Claude/Vitruvius/t2204-fold-round4-probe/t2204r4_magnitude_domain_output.json`)."""
    rng = np.random.default_rng(1)
    ref_w_f, ref_a_f, ref_b_scaled = _honest_pair(rng, scale=0.02)
    ref_raw = _collect(ref_w_f, ref_a_f, ref_b_scaled)
    reference_delta_norm = ref_raw["delta_norm_sq"] ** 0.5

    cand_raw = _collect(ref_w_f, ref_a_f, ref_b_scaled * magnitude_ratio)
    result = A.run_b3_pooled_report([("cand", cand_raw)], reference_delta_norm=reference_delta_norm)
    assert result["magnitude_warning"] is None, (
        f"an honest candidate at {magnitude_ratio}x the reference's own composed-delta magnitude "
        f"must not trip the magnitude sanity warning -- got {result['magnitude_warning']}"
    )


def test_honest_sweep_multi_pair_pooled_never_warns():
    """The must-not-fire cell above at pair granularity; this cell pools several honest pairs
    (the real production shape -- one adapter has many adapted (layer, proj) pairs) and confirms
    the pooling itself introduces no false warning."""
    rng = np.random.default_rng(2)
    pairs = []
    ref_norm_sq_total = 0.0
    for i in range(5):
        w_f, a_f, b_scaled = _honest_pair(rng, scale=0.02)
        raw = _collect(w_f, a_f, b_scaled)
        pairs.append((f"pair{i}", raw))
        ref_norm_sq_total += raw["delta_norm_sq"]
    reference_delta_norm = ref_norm_sq_total ** 0.5

    result = A.run_b3_pooled_report(pairs, reference_delta_norm=reference_delta_norm)
    assert result["magnitude_warning"] is None
    assert result["delta_norm"] == pytest.approx(reference_delta_norm, rel=1e-9), (
        "a candidate pooled from the SAME draws the reference was frozen from must read ratio 1.0"
    )


# --- Cell 3: must-fire -- a x50-scaled corrupted adapter trips the warning --------------------

def test_x50_scaled_corruption_trips_the_magnitude_warning():
    """The T-2210 probe census's own `scaled_hot`/uniform B-scale(k) construction at k=50
    (`Claude/Loki/t2210-probe/t2210_magnitude_axis.py`'s own `CORRUPT k=50` cell) -- a
    production-feasible corruption (the real collector, real quantization path, `B` scaled
    post-hoc exactly as a corrupted adapter file would arrive). This is the must-reject
    commissioning cell `StandardsDocument.md` §5.4 requires: genuinely large in the decided
    quantity (delta_norm ratio), and producible by the real conversion path."""
    rng = np.random.default_rng(3)
    ref_w_f, ref_a_f, ref_b_scaled = _honest_pair(rng, scale=0.02)
    ref_raw = _collect(ref_w_f, ref_a_f, ref_b_scaled)
    reference_delta_norm = ref_raw["delta_norm_sq"] ** 0.5

    corrupt_b_scaled = ref_b_scaled * 50.0  # same base weight/A, B scaled x50 post-hoc
    corrupt_raw = _collect(ref_w_f, ref_a_f, corrupt_b_scaled)

    result = A.run_b3_pooled_report([("corrupt", corrupt_raw)], reference_delta_norm=reference_delta_norm)
    assert result["magnitude_warning"] is not None, (
        "a x50-scaled corrupted adapter (T-2210 probe census's own construction) must trip the "
        "magnitude sanity warning"
    )
    assert result["magnitude_warning"]["disposition"] == "unresolved"
    assert result["magnitude_warning"]["ratio_to_reference"] == pytest.approx(50.0, rel=1e-9)


@pytest.mark.parametrize("k", [15, 50, 1000])
def test_uniform_b_scale_corruptions_above_the_ratio_all_trip_the_warning(k):
    """Every uniform B-scale(k) corruption at k strictly above `_B3_MAGNITUDE_WARN_RATIO` must
    warn -- the boundary is monotone in k for this construction (delta_norm scales linearly with a
    uniform B-scale, `Claude/Vitruvius/t2204-fold-round4-probe`'s own `ratio_to_ref` field
    confirms this by execution: k=10 -> ratio 10.0, k=50 -> ratio 50.0, k=1000 -> ratio 1000.0)."""
    rng = np.random.default_rng(4)
    ref_w_f, ref_a_f, ref_b_scaled = _honest_pair(rng, scale=0.02)
    ref_raw = _collect(ref_w_f, ref_a_f, ref_b_scaled)
    reference_delta_norm = ref_raw["delta_norm_sq"] ** 0.5

    corrupt_raw = _collect(ref_w_f, ref_a_f, ref_b_scaled * float(k))
    result = A.run_b3_pooled_report([("corrupt", corrupt_raw)], reference_delta_norm=reference_delta_norm)
    assert result["magnitude_warning"] is not None, f"k={k} (ratio {k}x > {_RATIO}x) must warn"


# --- Cell 3a: the lower band -- fix for D-SLM3787 finding S5 -----------------------------------
# Two executed mutations against the pre-fix suite left 32/32 green: deleting the lower band
# entirely (`if not (lo <= ratio <= hi)` -> `if not (ratio <= hi)`), and moving
# `_B3_MAGNITUDE_WARN_RATIO` to 6.5 or 14.9. No cell constructed a candidate SMALLER than its
# reference, and no cell asserted a numeric boundary rather than reading the shipped constant back
# from the module. These two cells close both gaps.

def test_a_candidate_at_a_twentieth_of_the_reference_warns_on_the_lower_band():
    """A candidate far BELOW the reference (ratio ~0.05x) must warn -- this is the only shape in
    the executed census (`Claude/Vitruvius/t2204-fold-round4-probe`'s "scale=0.002 CORRUPT k=10",
    ratio 0.1x) the lower band exists to catch (S2/S5). Deleting `lo <= ratio` from the shipped
    predicate leaves this cell red."""
    rng = np.random.default_rng(8)
    ref_w_f, ref_a_f, ref_b_scaled = _honest_pair(rng, scale=0.02)
    ref_raw = _collect(ref_w_f, ref_a_f, ref_b_scaled)
    reference_delta_norm = ref_raw["delta_norm_sq"] ** 0.5

    small_raw = _collect(ref_w_f, ref_a_f, ref_b_scaled * 0.05)
    result = A.run_b3_pooled_report([("small", small_raw)], reference_delta_norm=reference_delta_norm)
    assert result["magnitude_warning"] is not None, (
        "a candidate at ~0.05x the reference must trip the lower-band warning"
    )
    assert result["magnitude_warning"]["ratio_to_reference"] == pytest.approx(0.05, rel=1e-9)


def test_a_candidate_at_half_the_reference_does_not_warn_on_the_lower_band():
    """A candidate moderately below the reference (ratio 0.5x, well inside `[1/10, 10]`) must NOT
    warn -- pairs with the cell above to confirm the lower band has a real inside as well as a
    real outside, not just an always-fire predicate."""
    rng = np.random.default_rng(9)
    ref_w_f, ref_a_f, ref_b_scaled = _honest_pair(rng, scale=0.02)
    ref_raw = _collect(ref_w_f, ref_a_f, ref_b_scaled)
    reference_delta_norm = ref_raw["delta_norm_sq"] ** 0.5

    half_raw = _collect(ref_w_f, ref_a_f, ref_b_scaled * 0.5)
    result = A.run_b3_pooled_report([("half", half_raw)], reference_delta_norm=reference_delta_norm)
    assert result["magnitude_warning"] is None


# --- Cell 3b: literal-boundary pin on the shipped constant -- fix for D-SLM3787 finding S5 ------
# Written as numeric literals, not read from `_RATIO`/`A._B3_MAGNITUDE_WARN_RATIO`, so a future
# re-derivation of the shipped constant to a materially different value (this file's own sibling
# cells read the module and so would silently "pass" a drifted constant) is caught here instead.

def test_ratio_just_inside_ten_x_does_not_warn_and_just_outside_does():
    """Boundary pin at literal 9.0x/11.0x around the shipped `10.0` -- M1 also names the exact
    inclusive boundary (10.0x itself) as untested; covered by the second assertion below via the
    corruption census's own k=10 reading, replicated with a hand-built pair."""
    rng = np.random.default_rng(10)
    ref_w_f, ref_a_f, ref_b_scaled = _honest_pair(rng, scale=0.02)
    ref_raw = _collect(ref_w_f, ref_a_f, ref_b_scaled)
    reference_delta_norm = ref_raw["delta_norm_sq"] ** 0.5

    inside = _collect(ref_w_f, ref_a_f, ref_b_scaled * 9.0)
    r_inside = A.run_b3_pooled_report([("inside", inside)], reference_delta_norm=reference_delta_norm)
    assert r_inside["magnitude_warning"] is None, "9.0x is inside the shipped [1/10, 10] band"

    outside = _collect(ref_w_f, ref_a_f, ref_b_scaled * 11.0)
    r_outside = A.run_b3_pooled_report([("outside", outside)], reference_delta_norm=reference_delta_norm)
    assert r_outside["magnitude_warning"] is not None, "11.0x is outside the shipped [1/10, 10] band"


def test_ratio_at_exactly_the_boundary_is_silent_on_the_inclusive_compare():
    """M1: the census's own two exact-10.0x corruptions (`t2204r4_magnitude_domain_output.json`'s
    "scale=0.02 CORRUPT k=10" and "scale=0.002 CORRUPT k=1000") pass silently because the shipped
    predicate is `lo <= ratio <= hi`, inclusive. Pinned directly, not read back from the module, so
    a future switch to an exclusive compare is a visible behavior change here."""
    rng = np.random.default_rng(11)
    ref_w_f, ref_a_f, ref_b_scaled = _honest_pair(rng, scale=0.02)
    ref_raw = _collect(ref_w_f, ref_a_f, ref_b_scaled)
    reference_delta_norm = ref_raw["delta_norm_sq"] ** 0.5

    at_boundary = _collect(ref_w_f, ref_a_f, ref_b_scaled * 10.0)
    result = A.run_b3_pooled_report([("boundary", at_boundary)], reference_delta_norm=reference_delta_norm)
    assert result["magnitude_warning"] is None, (
        "a candidate at EXACTLY 10.0x the reference is silent on the shipped inclusive `<=` -- "
        "this is the same comparison that let a real executed corruption through (S2/M1)"
    )


# --- Cell 3c: `magnitude_warn_ratio` validation -- fix for D-SLM3787 finding M3 -----------------

@pytest.mark.parametrize("bad_ratio", [1.0, 0.5, 0.0, -3.0])
def test_magnitude_warn_ratio_at_or_below_one_is_rejected(bad_ratio):
    """A ratio of exactly 1.0 collapses the tolerance band to a single point; below 1.0 it
    inverts (`lo > hi`, everything warns); 0.0 previously raised `ZeroDivisionError` instead of a
    clear error. All four are rejected up front with a `ValueError`."""
    rng = np.random.default_rng(12)
    w_f, a_f, b_scaled = _honest_pair(rng, scale=0.02)
    raw = _collect(w_f, a_f, b_scaled)
    with pytest.raises(ValueError):
        A.run_b3_pooled_report([("p0", raw)], reference_delta_norm=1.0, magnitude_warn_ratio=bad_ratio)


# --- Cell 4: never a REJECT -- the warning carries no field an emission decision reads ---------

def test_magnitude_warning_never_sets_an_emission_deciding_field():
    """The ticket's own hard requirement: 'it fires a named WARNING with an UNRESOLVED
    disposition -- never a REJECT.' `build_runtime_additive_sections`'s own `margin_exceeded` is
    hardcoded False (unconditionally, independent of this function's return) -- this cell pins
    that the returned dict itself carries no `accepted`/`rejected`/`margin_exceeded` field that a
    future caller could mistakenly wire into an emission decision."""
    rng = np.random.default_rng(5)
    ref_w_f, ref_a_f, ref_b_scaled = _honest_pair(rng, scale=0.02)
    ref_raw = _collect(ref_w_f, ref_a_f, ref_b_scaled)
    reference_delta_norm = ref_raw["delta_norm_sq"] ** 0.5
    corrupt_raw = _collect(ref_w_f, ref_a_f, ref_b_scaled * 1000.0)

    result = A.run_b3_pooled_report([("corrupt", corrupt_raw)], reference_delta_norm=reference_delta_norm)
    assert result["magnitude_warning"] is not None
    assert result["magnitude_warning"]["disposition"] == "unresolved"
    for forbidden in ("accepted", "rejected", "margin_exceeded", "reject"):
        assert forbidden not in result, f"pooled report must not carry an '{forbidden}' field"
        assert forbidden not in result["magnitude_warning"], (
            f"magnitude_warning must not carry an '{forbidden}' field"
        )


# --- Cell 5: no reference configured -> the check is reported but inert -----------------------

def test_no_reference_configured_never_warns_regardless_of_candidate_magnitude():
    """`reference_delta_norm=None` (the production default, T-2213: no live reference adapter
    checkpoint exists on disk, O1) means the check is present but inert -- confirms the magnitude
    warning cannot fire on an unconfigured reference even for a wildly corrupted candidate."""
    rng = np.random.default_rng(6)
    w_f, a_f, b_scaled = _honest_pair(rng, scale=0.02)
    raw = _collect(w_f, a_f, b_scaled * 1_000_000.0)
    result = A.run_b3_pooled_report([("cand", raw)], reference_delta_norm=None)
    assert result["magnitude_warning"] is None
    assert result["delta_norm"] > 0.0  # delta_norm is still reported, just not compared


# --- Cell 6: retirement pin -- the deleted pooled verdict fields do not reappear ---------------

def test_run_b3_pooled_report_has_no_accept_reject_arithmetic():
    """T-2213 (D-SLM3783) retired the pooled gate's own accept/reject arithmetic. This is the
    diff pin: the returned dict from a real (non-degenerate) call carries none of the fields the
    retired verdict used to return, and the function itself does not exist under its retired
    name."""
    rng = np.random.default_rng(7)
    w_f, a_f, b_scaled = _honest_pair(rng, scale=0.02)
    raw = _collect(w_f, a_f, b_scaled)
    result = A.run_b3_pooled_report([("p0", raw)])
    for retired_field in (
        "accepted", "composed_mean_accepts", "composed_tail_accepts",
        "effect_mean_accepts", "effect_tail_accepts",
        "delta_composed_mean", "delta_composed_tail",
        "delta_effect_mean", "delta_effect_tail", "pooled_validation",
    ):
        assert retired_field not in result, f"retired field '{retired_field}' must not reappear"
    assert not hasattr(A, "run_b3_multi_pair_check"), (
        "the retired pooled-gate function name must not exist any more -- "
        "run_b3_pooled_report is its replacement"
    )
