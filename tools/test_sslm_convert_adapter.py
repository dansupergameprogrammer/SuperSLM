"""Curie's red suite for sslm_convert_adapter.py, ported to green (T-2021/T-2029, B0).

Each test below ports one of `tests/t2018-slora-serial/t2018_offline_red.cpp`'s eight B0 cells
(`Claude/Curie/t2018-slora-serial-red-suite-2026-08-13.md` Sec4, Wizard repo) to exercise this
module's own derivation directly, at the RATIO level rather than by reproducing
`t2018_offline_red.cpp`'s own `QuantizedBase`/`BuildAdapter` random-quantization fixture bit-for-
bit — each C++ cell's own claim is a property of `derive_amplifying_triple`/`gate_check`'s
arithmetic over a stated ratio (or a ratio derived from stated `T`/`beta`/`S`/`alpha` values), not
a property of the C++ suite's own PRNG stream, so a hand-constructed ratio establishes the
identical claim `t2018_offline_red.cpp` proves by random sampling. Every hand-computed numeric
witness (`rho=4.5`, `delta_raw=1,000,000`, the fracture cell's `26.0%`/`0.02413` figures) is cited
from the design document (`Claude/Vitruvius/t1977-...md` Sec4/Sec11 B0) or from
`t2018_offline_red.cpp`'s own comments, not re-derived independently.
"""

import math

import pytest

import sslm_convert_adapter as A


# --- Cell 1: a zero-effect adapter folds to exactly zero contribution ----------------------

def test_zero_effect_adapter_folds_to_exactly_zero_contribution():
    # gain=0.0 => every Bf element is exactly 0.0 => beta[i] == 0.0 for every channel => rho == 0.0
    # for every channel, which `derive_amplifying_triple` must signal infeasible (rho <= 0), not
    # silently accept -- the caller's own contract (B1a/B2, design Sec8) is that a zero-effect
    # adapter is represented by the GATE being skipped entirely (bound(sequence) AND
    # adapted(sequence.adapter, this_projection) both true, but delta_wide[i] computed as an
    # explicit, correctly-derived zero), not by feeding rho=0 into this derivation. This cell
    # asserts the boundary explicitly: rho=0 is out of this derivation's domain (identity=1,
    # rho=1.0, is the only exact pass-through; rho=0 is a distinct, infeasible case).
    triple = A.derive_amplifying_triple(0.0)
    assert triple is None, "rho=0 (a fully zero-effect channel) is out of derive_amplifying_triple's own domain -- the caller represents a zero-effect channel by skipping the delta-add step entirely (design Sec8), not by deriving a triple for rho=0"

    # The identity (rho=1.0, exact pass-through) case IS representable, and reproduces the input
    # unchanged when applied (realized_ratio(triple) == 1.0 exactly) -- the genuine "no scaling
    # needed" case this design's own identity bit exists for.
    identity_triple = A.derive_amplifying_triple(1.0)
    assert identity_triple == A.AmplifyingTriple(identity=1, mult=0, exponent=0)
    assert A.realized_ratio(identity_triple) == 1.0


# --- Cell 2: hand-computed nonzero delta -- correct triple matches, mis-derived diverges ---

def test_hand_computed_nonzero_delta_correct_triple_matches_misderived_diverges():
    # Design Sec11 B0's own executed witness: "executed at rho=4.5... rel_err = 0 for the correct
    # triple, rel_err = 0.778 for the mis-derived one."
    triple = A.derive_amplifying_triple(4.5)
    assert triple is not None, "rho=4.5 must derive a legal amplifying triple (well inside [2^-31,2^31])"

    delta_raw = 1_000_000  # exact integer, per the design's own text
    expected = 4.5 * delta_raw

    # This module derives the TRIPLE; applying it (ApplyAmplifyingWeightScaleFold) is the real
    # engine's own job (B1a/B2). The triple's OWN fixed-point precision is verified here via
    # realized_ratio's algebraic decode, which is exactly the quantity ApplyAmplifyingWeightScaleFold
    # reproduces to within its own rounding (design Sec4's "correct triple reproduces the exact
    # expected value... to rel_err=0 at printed precision").
    realized = A.realized_ratio(triple)
    correct_value = realized * delta_raw
    rel_err_correct = abs(correct_value - expected) / expected
    assert rel_err_correct < 1e-5, f"correct triple's rel_err={rel_err_correct:.8f}, expected ~1e-6 scale (the primitive's own fixed-point rounding)"

    # Deliberately mis-derived: invert the ratio direction (1/4.5 instead of 4.5) -- the suite's
    # own WRONG_DIRECTION mutation family (mech=7).
    wrong_triple = A.derive_amplifying_triple(1.0 / 4.5)
    assert wrong_triple is not None
    wrong_realized = A.realized_ratio(wrong_triple)
    wrong_value = wrong_realized * delta_raw
    rel_err_wrong = abs(wrong_value - expected) / expected
    assert rel_err_wrong > 0.5, f"mis-derived (inverted-direction) triple must diverge measurably; rel_err={rel_err_wrong:.6f}"
    assert rel_err_wrong > rel_err_correct * 100.0


# --- Cell 3: WIRING AND DERIVATION gate at the fracture point -- OLD fails, NEW passes -----

def test_wiring_derivation_gate_at_the_fracture_cell_old_fails_new_passes():
    # T-1990's exact fracture point (design Sec4): d=896, r=8, gain=0.4. Rather than reproducing
    # the C++ suite's own random (base, adapter) draw bit-for-bit, this cell constructs a channel
    # whose required ratio is amplifying (rho > 1) -- the exact condition the OLD mechanism
    # (ApplyWeightScaleFold's own (0,1]-only domain, T-1990's fracture) cannot represent, and the
    # NEW mechanism (ApplyAmplifyingWeightScaleFold's [2^-31,2^31] domain, this module's own
    # `derive_amplifying_triple`) represents exactly.
    t_value, beta_i, s_value = 2.0, 0.9, 1.0  # rho = T*beta/S = 1.8 -- amplifying (>1)
    rho = A.required_ratio_independent(t_value, beta_i, s_value)
    assert rho > 1.0, "fixture must be amplifying (rho>1), matching T-1990's own fracture condition"

    # OLD mechanism: ApplyWeightScaleFold's own representable ratio set is (0,1] only (design
    # Sec4's own T-1990 finding) -- a ratio above 1 is infeasible under that domain by construction
    # (mirrored here as an explicit domain check, since the OLD primitive itself is not this
    # module's to reimplement -- it is the real, unmodified engine primitive, untouched by this
    # design, design Sec4: "WSC1's own base fold... still calls ApplyWeightScaleFold unmodified").
    old_domain_ok = 0.0 < rho <= 1.0
    assert not old_domain_ok, "OLD mechanism's (0,1]-only domain must reject this amplifying ratio (T-1990's own fracture)"

    # NEW mechanism: derive_amplifying_triple represents it, and the two-part gate passes.
    triple = A.derive_amplifying_triple(rho)
    assert triple is not None, "NEW mechanism must derive a legal triple for this amplifying ratio"
    result = A.gate_check(triple, required_rho=rho, t_value=t_value, beta_i=beta_i, s_value=s_value)
    assert result.gate_passes, f"NEW mechanism (correct repair) must PASS the WIRING AND DERIVATION gate: {result}"
    assert result.derivation_rel_err <= A.DERIVATION_REL_BOUND, (
        f"NEW mechanism's own derivation rel_err ({result.derivation_rel_err:.3e}) must sit within the constructive 2^-31 bound"
    )


# --- Cell 4: WIRING refuses every PARTIAL(f), f>0 -------------------------------------------

def test_wiring_refuses_partial_repair_at_every_nonzero_fraction():
    t_value, beta_i, s_value = 2.0, 0.9, 1.0
    rho = A.required_ratio_independent(t_value, beta_i, s_value)
    correct = A.derive_amplifying_triple(rho)
    assert correct is not None

    # A PARTIAL(f) defect: the channel is left on the OLD mechanism's own clamped triple
    # (identity=1, a pass-through substituting rho=1 for the true amplifying rho) instead of the
    # correctly-derived amplifying triple -- WIRING must refuse it (it does not match the triple
    # independently re-derived from the SAME required_rho), regardless of how "close" identity=1
    # happens to score on any other metric.
    partial_wrong = A.AmplifyingTriple(identity=1, mult=0, exponent=0)
    assert partial_wrong != correct, "fixture sanity: the PARTIAL defect's triple must differ from the correct one"
    result = A.gate_check(partial_wrong, required_rho=rho, t_value=t_value, beta_i=beta_i, s_value=s_value)
    assert not result.wiring_matches, "WIRING must refuse a PARTIAL(f>0) defect (channel left unrepaired)"
    assert not result.gate_passes


# --- Cell 5: DERIVATION refuses RHO_SCALE(c), c != 1 ----------------------------------------

def test_derivation_refuses_rho_scale_misderivation():
    t_value, beta_i, s_value = 2.0, 0.9, 1.0
    rho_indep = A.required_ratio_independent(t_value, beta_i, s_value)
    for c in (1.0001, 1.0725, 2.0, 16.0):
        # A constant-factor mis-derivation: the mechanism derives its triple from c*rho_indep
        # instead of rho_indep, then applies it -- WIRING alone cannot see this (the triple IS
        # correctly derived from ITS OWN, wrong, required_rho), but DERIVATION's independent
        # recombination (T, beta[i], S, never the mechanism's own combined rho) catches it.
        mis_derived_rho = c * rho_indep
        applied = A.derive_amplifying_triple(mis_derived_rho)
        assert applied is not None
        result = A.gate_check(applied, required_rho=mis_derived_rho, t_value=t_value, beta_i=beta_i,
                               s_value=s_value)
        assert result.wiring_matches, f"c={c}: WIRING must PASS (the triple IS honestly derived from its own, wrong, rho)"
        assert not result.derivation_matches, f"c={c}: DERIVATION must trip (independent recombination disagrees)"
        assert not result.gate_passes, f"RHO_SCALE({c}) must be refused by the gate"

    # c = 1.0 (no mis-derivation) must NOT trip DERIVATION -- the two-sided calibration
    # (StandardsDocument.md Sec5.4: the instrument must be inert at its own null point).
    inert = A.gate_check(A.derive_amplifying_triple(rho_indep), required_rho=rho_indep,
                          t_value=t_value, beta_i=beta_i, s_value=s_value)
    assert inert.derivation_matches, "RHO_SCALE(1.0) (the inert null) must NOT trip DERIVATION"
    assert inert.gate_passes


# --- Cell 6: DERIVATION refuses RHO_PERM(p), p>0 --------------------------------------------

def test_derivation_refuses_rho_perm_misderivation():
    t_value, s_value = 2.0, 1.0
    beta = [0.9, 0.3]  # two channels, distinct betas -- an off-by-one read swaps them
    rho_correct = A.required_ratio_independent(t_value, beta[0], s_value)
    rho_swapped = A.required_ratio_independent(t_value, beta[1], s_value)  # off-by-one: reads beta[1] for channel 0
    assert rho_correct != rho_swapped, "fixture sanity: the two channels' betas must differ"

    applied = A.derive_amplifying_triple(rho_swapped)  # the mechanism derives from the WRONG beta
    assert applied is not None
    # DERIVATION recombines with the CORRECT beta[0] (the independent reference reads the
    # adapter's own per-channel scale array correctly; only the mechanism under test misindexed).
    result = A.gate_check(applied, required_rho=rho_swapped, t_value=t_value, beta_i=beta[0],
                           s_value=s_value)
    assert result.wiring_matches, "WIRING must PASS (the triple IS honestly derived from the swapped rho it computed)"
    assert not result.derivation_matches, "DERIVATION must trip on the off-by-one index (recombination reads the correct beta[0])"
    assert not result.gate_passes


# --- Cell 7: the DERIVATION bound is constructive (2^-31), not fitted -----------------------

def test_derivation_bound_is_constructive_across_gains():
    s_value = 1.0
    for gain in (0.05, 0.4, 0.8, 1.6):
        t_value, beta_i = 2.0, gain
        rho = A.required_ratio_independent(t_value, beta_i, s_value)
        triple = A.derive_amplifying_triple(rho)
        assert triple is not None, f"gain={gain}: rho={rho} must derive a legal triple"
        result = A.gate_check(triple, required_rho=rho, t_value=t_value, beta_i=beta_i, s_value=s_value)
        assert result.derivation_rel_err <= A.DERIVATION_REL_BOUND, (
            f"gain={gain}: correct mechanism's derivation rel_err {result.derivation_rel_err:.3e} exceeds the 2^-31 bound"
        )
        assert result.gate_passes


# --- Cell 8: the gate does NOT and CANNOT catch a mis-derived T, by design ------------------

def test_gate_does_not_and_cannot_catch_t_scale_misderivation_by_design():
    # Design Sec4/Sec19/D-SLM3018: T is a converter-derived output, not a primary input.
    # gate_check's own DERIVATION half reads whatever t_value the CALLER passes -- if the
    # mechanism under test AND the gate both read the mechanism's own (possibly wrong) T local,
    # a mis-derived T moves both sides of the comparison by the identical factor and leaves no
    # footprint. This is the documented, accepted scope limit this cell asserts explicitly.
    beta_i, s_value = 0.9, 1.0
    honest_t = 2.0
    defect_t = 255.9 * honest_t  # T-2007's own T_SCALE(255.9) construction, scaled onto this fixture

    honest_rho = A.required_ratio_independent(honest_t, beta_i, s_value)
    honest_triple = A.derive_amplifying_triple(honest_rho)
    assert honest_triple is not None
    honest_result = A.gate_check(honest_triple, required_rho=honest_rho, t_value=honest_t,
                                  beta_i=beta_i, s_value=s_value)
    assert honest_result.gate_passes, "the honest (t=1) mechanism must pass WIRING AND DERIVATION"

    # The mis-derived-T mechanism: it derives ITS OWN rho from ITS OWN (wrong) T, applies the
    # resulting triple, and the gate's DERIVATION half is (per this scope limit) handed the SAME
    # wrong T -- because the gate has no other source for T than the converter's own local.
    defect_rho = A.required_ratio_independent(defect_t, beta_i, s_value)
    defect_triple = A.derive_amplifying_triple(defect_rho)
    assert defect_triple is not None
    defect_result = A.gate_check(defect_triple, required_rho=defect_rho, t_value=defect_t,
                                  beta_i=beta_i, s_value=s_value)
    assert defect_result.gate_passes, (
        "T-2007's own finding: a converter that mis-derives T STILL passes WIRING AND DERIVATION "
        "-- this is the documented, accepted scope limit (design Sec4/Sec19), not a bug"
    )

    # The defect is real and material even though the gate cannot see it: the realized ratio
    # under the mis-derived T diverges sharply from the honest one -- a composed-quality defect
    # this gate structurally cannot catch, exactly the residual B3's baked-adapter comparator
    # (this module does not implement) exists to close.
    assert A.realized_ratio(defect_triple) != pytest.approx(A.realized_ratio(honest_triple), rel=1e-3), (
        "the T-mis-derivation must produce a materially different realized ratio, even though the "
        "gate cannot see it"
    )


# --- AmplifyingScaleRatioOutOfDomain: explicit infeasibility signal, never a silent clamp ---

def test_genuinely_out_of_domain_ratio_signals_explicit_infeasibility():
    # Design Sec6 item 2 / B0's own third red-first cell: a ratio far outside [2^-31,2^31] must
    # signal infeasibility explicitly, not silently derive a load-legal-but-wrong triple.
    impossible = math.exp2(40.0)  # 2^40, far outside the domain
    assert A.derive_amplifying_triple(impossible) is None, (
        "a ratio of 2^40 must be signaled genuinely infeasible, not silently accepted"
    )
    # Negative control: a ratio at the domain's own structural ceiling must still derive legally.
    assert A.derive_amplifying_triple(math.exp2(30.0)) is not None, (
        "a ratio of 2^30 (inside the domain) must derive a legal triple"
    )


# --- compute_t: the D-SLM2916 definition, a converter-derived output, not a free parameter ---

def test_compute_t_is_the_dslm2916_definition():
    alpha = [0.1, 0.2, 0.05]
    u_acc_max = [1000.0, 200.0, 5000.0]
    expected = max(a * m / 127.0 for a, m in zip(alpha, u_acc_max))
    assert A.compute_t(alpha, u_acc_max) == pytest.approx(expected)
    # Degenerate corpus (all-zero magnitudes) must not divide by zero or return <=0 -- the
    # calibration corpus is assumed nonempty and nonzero by construction (a real corpus always
    # has SOME nonzero activation), but this function stays total rather than crashing.
    assert A.compute_t([], []) == 1.0
    assert A.compute_t([0.1], [0.0]) == 1.0


# --- derive_delta_fold_triples / derive_u_fold_triples: the per-array derivation surface ----

def test_derive_delta_fold_triples_matches_per_channel_gate():
    s_value = 1.0
    beta = [0.9, 0.3, 1.0]  # channel 2's beta=1.0 with T=1.0 => rho=1.0 => exact identity
    t_value = 1.0
    triples, required = A.derive_delta_fold_triples([0.9, 0.3, 1.0], s_value, beta, t_value)
    assert len(triples) == len(beta) == len(required)
    assert triples[2] == A.AmplifyingTriple(identity=1, mult=0, exponent=0)
    for triple, rho, b in zip(triples, required, beta):
        assert triple is not None
        result = A.gate_check(triple, required_rho=rho, t_value=t_value, beta_i=b, s_value=s_value)
        assert result.gate_passes


def test_derive_u_fold_triples_matches_per_rank_gate():
    t_value = 2.0
    alpha = [0.1, 4.0, 1.0]  # rank 1's alpha=4.0 with T=2.0 => rho=2.0, amplifying
    triples, required = A.derive_u_fold_triples(alpha, t_value)
    assert len(triples) == len(alpha) == len(required)
    for triple, rho, a in zip(triples, required, alpha):
        assert triple is not None
        assert rho == pytest.approx(a / t_value)  # u-fold's own rho_u = alpha[k] / T (design Sec4 extension)
        rel_err = abs(A.realized_ratio(triple) - rho) / rho if rho != 0 else 0.0
        assert rel_err <= A.DERIVATION_REL_BOUND, f"rank alpha={a}: derived triple's realized ratio diverges from rho={rho} by {rel_err:.3e}"
