#!/usr/bin/env python3
"""T-1954 (Brunel spike, disposable, never merges; T-1822 design Sec32 "Fused
Q"). Turns t1954_calibrate_q.exe's own calibration dump into per-layer
q_landing_m_out/e_out/e_t/r_t -- the four LayerWeights fields the spike's
fused-Q branch consumes (forward_sites.h/.cpp).

Derivation, transplanted directly from `_derive_composition_constants`'s own
K/V formula (tests/reference/superslm_spike/pipeline.py, the SAME function
this build's own construction cites at Sec32.2/Sec32.7), reusing that
module's real `canonical_scale`/`intmath.dynamic_scale_reciprocal` --
nothing here re-implements that arithmetic independently.

For K: `k_scale = scales.output_scale(...)` (a real-valued, units-per-code
target scale from float calibration); `kv_landing[...] = canonical_scale(k_scale)`
(the absolute target, KvLandingScales' own (m,e)); `ratio_m, ratio_e =
canonical_scale(k_scale / k_s_ref)` where `k_s_ref` is the q/k/v projection's
own reference-fold scale (already an artifact-carried constant); `r_t =
dynamic_scale_reciprocal(ratio_m)`; `kv_reciprocals[...] = (ratio_m, ratio_e, r_t)`.

Q has no float-calibration pipeline available in this spike (out of scope --
this is a build-and-gate ticket, no graded capture). The substitute: derive
a target scale directly from the observed INTEGER peak of the rotated wide
accumulator (t1954_calibrate_q.exe's own dump), chosen so that peak lands at
TARGET_CODE (comfortably under the +/-127 clamp, leaving headroom for
tokens this small calibration set did not cover) rather than from a
separately-calibrated float statistic. This is a spike-tier substitution,
not the design's own literal calibration policy (Sec32.2's own "canonical_scale
of the observed post-RoPE peak") -- named explicitly here and in the build log.

CarriedScale convention (checked_chain_funnel.h): value == m * 2**e, m
canonical in [2**30, 2**31). `q_site_constant = canonical_scale(s_ref/127)`
(ProjectAndFunnel's own site_constant, already loaded from the real
artifact), so `s_ref = 127 * m * 2**e` recovers the projection's own
reference-fold scale exactly, with no second artifact-parsing pass.

Usage: python tools/t1954_derive_q_landing.py <calibration_csv_or_stdin>
Prints one C++ initializer line per layer plus a JSON array (for the
self-check tool to parse) to stdout.
"""
import csv
import json
import sys
from fractions import Fraction

sys.path.insert(0, "tests/reference")
from superslm_spike import intmath  # noqa: E402
from superslm_spike.pipeline import canonical_scale  # noqa: E402

TARGET_CODE = 40  # under the +/-127 clamp; headroom for tokens this
                   # calibration set's own 5 prompts did not cover, AND
                   # headroom against the fused-site exponent floor (a
                   # larger TARGET_CODE landed one layer's own e_t one step
                   # below FUSED_Q_LANDING_EXPONENT_MIN, executed and caught
                   # by this script's own domain check before any C++ ever
                   # saw the value -- see the build log's own derivation
                   # note).

# KvLandingReciprocals' own legal domain (model.cpp, ValidateKvLandingReciprocalsDomain) --
# checked here too so a violation is caught before the self-check tool ever
# loads these constants, matching this build's own "verify at source, not by
# construction" discipline (StandardsDocument.md Sec5.4).
KV_LANDING_RECIPROCAL_MIN = (1 << 31) + 1
KV_LANDING_RECIPROCAL_MAX = 1 << 32
# T-1954's own re-derived floor for the FUSED site (build log's own S1-style
# derivation: RopeApplyPairWide's output is int64_t-bounded, identical operand
# class to K's own fused branch_code, same formula, same -25 floor -- not
# assumed equal to K/V's legacy -60).
FUSED_Q_LANDING_EXPONENT_MIN = -25


def derive_one_layer(peak_abs_branch_code: int, m_a: int, e_a: int, site_m: int, site_e: int):
    if peak_abs_branch_code <= 0:
        raise ValueError("peak_abs_branch_code must be positive (a layer with zero observed "
                          "activity across the whole calibration set is a calibration-set gap, "
                          "not a value to derive a scale from)")
    s_ref = Fraction(127 * site_m) * (Fraction(2) ** site_e)
    # The real value the peak accumulator element represents, PRE weight-scale
    # reference fold (matching `qacc[i] * m_a * 2**e_a`, the normed-activation
    # scale applied to the wide accumulator's own integer magnitude -- see
    # this script's own module docstring for the full derivation chain).
    real_peak_pre_sref = Fraction(peak_abs_branch_code) * Fraction(m_a) * (Fraction(2) ** e_a)
    real_peak = real_peak_pre_sref * s_ref

    q_scale_target = real_peak / TARGET_CODE
    m_out, e_out = canonical_scale(q_scale_target)

    ratio = q_scale_target / s_ref  # == real_peak_pre_sref / TARGET_CODE; s_ref cancels exactly
    ratio_m, ratio_e = canonical_scale(ratio)
    r_t = intmath.dynamic_scale_reciprocal(ratio_m)

    if not (KV_LANDING_RECIPROCAL_MIN <= r_t <= KV_LANDING_RECIPROCAL_MAX):
        raise ValueError(f"r_t={r_t} outside legal [{KV_LANDING_RECIPROCAL_MIN},"
                          f"{KV_LANDING_RECIPROCAL_MAX}] -- derivation invalid for this layer")
    if ratio_e < FUSED_Q_LANDING_EXPONENT_MIN:
        raise ValueError(f"e_t={ratio_e} below the fused-site floor "
                          f"{FUSED_Q_LANDING_EXPONENT_MIN} -- derivation invalid for this layer")

    return {
        "q_landing_m_out": m_out,
        "q_landing_e_out": e_out,
        "q_landing_e_t": ratio_e,
        "q_landing_r_t": r_t,
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    text = open(path, "r", encoding="ascii") if path else sys.stdin
    reader = csv.DictReader(text)
    results = []
    for row in reader:
        layer = int(row["layer"])
        peak = int(row["peak_abs_branch_code"])
        m_a = int(row["m_a_at_peak"])
        e_a = int(row["e_a_at_peak"])
        site_m = int(row["q_site_constant_m"])
        site_e = int(row["q_site_constant_e"])
        derived = derive_one_layer(peak, m_a, e_a, site_m, site_e)
        derived["layer"] = layer
        results.append(derived)

    results.sort(key=lambda r: r["layer"])
    for r in results:
        print(f'{{{r["layer"]}, {{{r["q_landing_m_out"]}, {r["q_landing_e_out"]}, '
              f'{r["q_landing_e_t"]}, {r["q_landing_r_t"]}}}}},')
    print(json.dumps(results), file=sys.stderr)


if __name__ == "__main__":
    main()
