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

This file commissions that warning in both directions, per `StandardsDocument.md` §5.4: a
must-accept (an honest sweep, including the >=2.5x magnitude candidates T-2210's own strike named,
must never warn) and a must-reject (a x50-scaled corrupted adapter -- the same construction shape
the T-2210 probe census used -- must warn). Both constructions are producible by the real
production data path: `_b3_collect_pair_raw_draws` is the real per-pair collector, called here with
hand-built `(a_f, b_scaled)` pairs exactly as `Claude/Loki/t2210-probe/t2210_guard_probe.py`'s own
`collect()` calls it, not a reimplementation.
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
