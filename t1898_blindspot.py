"""T-1898 strike -- characterise the blind spot. DISPOSABLE.

Reads the confirmation CSV and reports, using the same independent exact-integer
truth reference, exactly where `out_magnitude_exceeded_int64` is FALSE while the
emitted K code is wrong: the operand threshold, the exponent region, and the
minimal witness.
"""

import csv
import sys
from t1898_truth_reference import truth_raw, clamp_code


def main(path):
    blind = []          # wrong code, flag false
    caught = 0          # wrong code, flag true
    rows = 0
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows += 1
            bc, m_a, r_t = int(r["bc"]), int(r["m_a"]), int(r["r_t"])
            e_a, e_t, raw = int(r["e_a"]), int(r["e_t"]), int(r["raw"])
            exceeded = r["exceeded"] == "1"
            t_code = clamp_code(truth_raw(bc, m_a, r_t, e_a, e_t))
            if clamp_code(raw) != t_code:
                if exceeded:
                    caught += 1
                else:
                    blind.append((bc, m_a, r_t, e_a, e_t, raw, t_code, clamp_code(raw)))

    print(f"rows: {rows}")
    print(f"wrong code, gate FIRED (refused by remedy 2)      : {caught}")
    print(f"wrong code, gate SILENT (remedy 2 says clean)     : {len(blind)}")
    if not blind:
        print("no blind spot found in this region")
        return

    print()
    print("Blind-spot region:")
    print(f"  |branch_code| range : 2^{min(abs(b[0]) for b in blind).bit_length()-1} .. "
          f"2^{max(abs(b[0]) for b in blind).bit_length()-1}")
    print(f"  |m_a| values        : {sorted({abs(b[1]) for b in blind})}")
    print(f"  r_t values          : {sorted({b[2] for b in blind})}")
    print(f"  e_a range           : {min(b[3] for b in blind)} .. {max(b[3] for b in blind)}")
    print(f"  e_t range           : {min(b[4] for b in blind)} .. {max(b[4] for b in blind)}")
    print(f"  composed k range    : {min(62-(b[3]-b[4]) for b in blind)} .. "
          f"{max(62-(b[3]-b[4]) for b in blind)}")
    mags = [abs(b[0]) * abs(b[1]) * b[2] for b in blind]
    print(f"  |bc*m_a*r_t| range  : 2^{min(mags).bit_length()-1} .. 2^{max(mags).bit_length()-1}")
    print(f"  emitted codes       : {sorted({b[7] for b in blind})}")
    print(f"  truth codes         : {sorted({b[6] for b in blind})}")

    print()
    smallest = min(blind, key=lambda b: abs(b[0]) * abs(b[1]) * b[2])
    bc, m_a, r_t, e_a, e_t, raw, t_code, got = smallest
    mag = abs(bc) * abs(m_a) * r_t
    k = 62 - (e_a - e_t)
    print("Minimal witness (smallest |bc*m_a*r_t|):")
    print(f"  branch_code = {bc}        (fits int64: RopeApplyPairWide's only guarantee)")
    print(f"  m_a = {m_a}   (|m_a| <= kCompositionScaleMaxAbsM = 2147483647)")
    print(f"  r_t = {r_t}   (in [2^31+1, 2^32], ValidateKvLandingReciprocalsDomain)")
    print(f"  e_a = {e_a}   (in [-80, 39], model.cpp's own derivation)")
    print(f"  e_t = {e_t}   (>= -60, ValidateKvLandingReciprocalsDomain) -- above BOTH floors")
    print(f"  composed k = {k}  -> rounding-divide branch")
    print(f"  |bc*m_a*r_t| = 2^{mag.bit_length()-1}   (U128MulSmall's stated precondition: ~2^91)")
    print(f"  2*mag + 2^k  = 2^{(2*mag + (1<<k)).bit_length()-1}  -> exceeds the 128-bit carry,"
          f" U128Add wraps")
    print(f"  LandingRescale raw = {raw}   out_magnitude_exceeded_int64 = FALSE")
    print(f"  K code written = {got}     K code truth = {t_code}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "t1898_confirm.csv")
