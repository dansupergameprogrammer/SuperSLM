"""T-1898 strike -- independent truth reference. DISPOSABLE.

Written from LandingRescale's own documented formula (forward_sites.cpp:592-594):

    round_half_away_from_zero((branch_code * m_a * r_t) / 2^(62 - (e_a - e_t)))
    "with a negative composite exponent an EXACT left shift (no rounding)"

in exact Python integers, taking no input from the C++ path. The C++ probe
emits only inputs, the returned `raw`, and the `out_magnitude_exceeded_int64`
flag; every judgement below is made here.

The engine's caller wraps the return in ClampRopeCode (forward_sites.cpp:735):
clamp to [-127, 127] -- NOT [-128, 127].
"""

import csv
import sys
from collections import Counter

SPIKE_FLOOR = -25   # kOptionGFusedKvLandingExponentMin, spike forward_sites.cpp:68
LOAD_FLOOR = -60    # kKvLandingExponentMin, model.cpp:810
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1


def truth_raw(bc, m_a, r_t, e_a, e_t):
    """The exact value LandingRescale is specified to return, unbounded."""
    k = 62 - (e_a - e_t)
    num = bc * m_a * r_t
    if k < 0:
        return num << (-k)                      # exact left shift, no rounding
    if num == 0:
        return 0
    sign = 1 if num > 0 else -1
    mag = abs(num)
    div = 1 << k
    # round half away from zero
    q = (2 * mag + div) // (2 * div)
    return sign * q


def clamp_code(v):
    return 127 if v > 127 else (-127 if v < -127 else v)


def main(path):
    rows = 0
    q1_counterexamples = []            # wrong code while flag says clean
    flag_true = 0
    wrong_code = 0
    band_wrong = Counter()             # e_t band -> wrong-code count
    band_total = Counter()
    witness = None                     # smallest |bc| wrong inside [-60,-25)

    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows += 1
            bc = int(r["bc"]); m_a = int(r["m_a"]); r_t = int(r["r_t"])
            e_a = int(r["e_a"]); e_t = int(r["e_t"])
            raw = int(r["raw"]); exceeded = r["exceeded"] == "1"

            t = truth_raw(bc, m_a, r_t, e_a, e_t)
            t_code = clamp_code(t)
            got_code = clamp_code(raw)
            bad = (got_code != t_code)

            if exceeded:
                flag_true += 1
            if bad:
                wrong_code += 1
                if not exceeded:
                    if len(q1_counterexamples) < 10:
                        q1_counterexamples.append((bc, m_a, r_t, e_a, e_t, raw, t, t_code, got_code))

            if LOAD_FLOOR <= e_t < SPIKE_FLOOR:
                band = "design-admits / spike-REFUSES  [-60,-25)"
                if bad and (witness is None or abs(bc) < abs(witness[0])):
                    witness = (bc, m_a, r_t, e_a, e_t, raw, t, t_code, got_code)
            elif e_t >= SPIKE_FLOOR:
                band = "both admit                      [-25, +)"
            else:
                band = "neither (below load floor)"
            band_total[band] += 1
            if bad:
                band_wrong[band] += 1

    print(f"rows observed: {rows}")
    print(f"rows where out_magnitude_exceeded_int64 == true : {flag_true}")
    print(f"rows where the emitted K code != truth          : {wrong_code}")
    print()
    print("Q1 -- is the dynamic gate authoritative?")
    print(f"  wrong code WITH the flag false (counterexamples): {len(q1_counterexamples)}"
          + (" (capped at 10)" if len(q1_counterexamples) == 10 else ""))
    for c in q1_counterexamples:
        print(f"    bc={c[0]} m_a={c[1]} r_t={c[2]} e_a={c[3]} e_t={c[4]} "
              f"raw={c[5]} truth={c[6]} truth_code={c[7]} got_code={c[8]}")
    if not q1_counterexamples:
        print("    NONE. Every wrong-code row is flagged. Remedy 2, threaded and run")
        print("    unconditionally, refuses every one of them -- it is sound as a gate.")
    print()
    print("Q2 -- what the exponent bands contain")
    for b in sorted(band_total):
        print(f"  {b}: {band_wrong[b]} wrong of {band_total[b]}")
    print()
    if witness:
        c = witness
        print("Smallest-|branch_code| wrong-code witness inside [-60,-25):")
        print(f"  branch_code = {c[0]}   (|bc| = 2^{abs(c[0]).bit_length()-1}-ish, fits int64:"
              f" {INT64_MIN <= c[0] <= INT64_MAX})")
        print(f"  m_a = {c[1]}   r_t = {c[2]}   e_a = {c[3]}   e_t = {c[4]}")
        print(f"  load-legal (e_t >= -60): yes")
        print(f"  design's re-derived pre-filter (-60): ADMITS, with {c[4]-LOAD_FLOOR} bits of margin")
        print(f"  spike's own fused floor (-25):        REFUSES"
              f" (OptionGFusedLandingExponentOutOfDomain)")
        print(f"  K code written = {c[8]}    K code truth = {c[7]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "t1898_observations.csv")
