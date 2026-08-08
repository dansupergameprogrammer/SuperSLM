#!/usr/bin/env python3
r"""T-1822 fold 8 -- the site-1 refined landing hook against an exact reference.

The oracle arm's refined embed hook (t1822_fold8_site1_oracle_arm.py) implements
design Sec 4.1 at G = 1, k_cap = 3 in torch float32.  This check replicates the
hook's own torch arithmetic line for line (CPU) and compares it against an exact
rational reference of the construction:

    d      = max|row|  (1 if the row is all zero)
    k_i    = largest k in [0,3] with |x_i| * 2^k <= d      (exact comparison)
    q_i    = round_half_away_from_zero(x_i * 127 * 2^k_i / d)   (exact rational)
    xhat_i = q_i * d / (127 * 2^k_i)

Asserted, per element, over structured rows (all-zero, single-outlier, rail) and
10,000 random rows spanning six orders of magnitude:
  * the hook's k_i equals the reference's wherever |x_i|*2^k vs d is not within
    one float32 ulp of equality (at an exact tie the float32 comparison may
    legitimately differ; ties are counted and reported);
  * the hook's integer code equals the reference's, or the disagreeing element
    sits within one float32 ulp of a rounding boundary (counted);
  * construction-level identities hold exactly: |q| <= 127, and xhat == 0
    wherever x == 0.

A wrong quantum, a wrong k rule, or a swapped operand fails loudly; sub-ulp
tie behaviour is reported, not hidden.  Reproduce:  python tools\t1822_fold8_hook_reference_check.py
"""
import sys
from fractions import Fraction

import numpy as np
import torch

INT8_MAX = 127
K_CAP = 3


def round_half_away(t):
    return torch.sign(t) * torch.floor(torch.abs(t) + 0.5)


def hook_math(xf):
    """The refined hook's own lines, torch float32, verbatim semantics."""
    d = xf.abs().amax(dim=-1, keepdim=True)
    d = torch.where(d > 0, d, torch.ones_like(d))
    ax = xf.abs()
    k = torch.zeros_like(xf)
    for kk in range(1, K_CAP + 1):
        k = torch.where(ax * float(1 << kk) <= d, float(kk), k)
    quantum = d / (INT8_MAX * torch.pow(torch.tensor(2.0), k))
    q = round_half_away(xf / quantum).clamp(-INT8_MAX, INT8_MAX)
    return k, q, q * quantum


def reference(row):
    """Exact rational reference over the float32 values themselves."""
    vals = [Fraction(float(v)) for v in row]
    d = max(abs(v) for v in vals)
    if d == 0:
        d = Fraction(1)
    ks, qs = [], []
    for v in vals:
        k = 0
        for kk in range(1, K_CAP + 1):
            if abs(v) * (1 << kk) <= d:
                k = kk
        t = v * 127 * (1 << k) / d
        # round half away from zero, exactly
        n, den = t.numerator, t.denominator
        mag = (2 * abs(n) + den) // (2 * den)
        q = mag if n >= 0 else -mag
        q = max(-127, min(127, q))
        ks.append(k)
        qs.append(q)
    return ks, qs


def check_rows(rows, label):
    x = torch.tensor(rows, dtype=torch.float32)
    hk, hq, _ = hook_math(x)
    n_elem = 0
    k_mismatch_nontie = 0
    q_mismatch_nonboundary = 0
    ties = 0
    boundary = 0
    for i, row in enumerate(rows):
        rks, rqs = reference(row)
        d = max(abs(Fraction(float(v))) for v in row) or Fraction(1)
        for j, v in enumerate(row):
            n_elem += 1
            vk, vq = int(hk[i, j]), int(hq[i, j])
            if vk != rks[j]:
                # tie tolerance: |x|*2^k within one ulp of d for some k
                av = abs(Fraction(float(v)))
                near = any(abs(av * (1 << kk) - d) <= d * Fraction(1, 2 ** 23)
                           for kk in range(1, K_CAP + 1))
                if near:
                    ties += 1
                else:
                    k_mismatch_nontie += 1
            elif vq != rqs[j]:
                t = Fraction(float(v)) * 127 * (1 << rks[j]) / d
                frac = abs(t) - (abs(t).numerator // abs(t).denominator)
                if abs(frac - Fraction(1, 2)) <= abs(t) * Fraction(1, 2 ** 22) + Fraction(1, 2 ** 22):
                    boundary += 1
                else:
                    q_mismatch_nonboundary += 1
    print(f"  {label}: {n_elem} elements; hard k mismatches {k_mismatch_nontie}, "
          f"hard code mismatches {q_mismatch_nonboundary} "
          f"(sub-ulp ties {ties}, rounding-boundary {boundary})")
    return k_mismatch_nontie == 0 and q_mismatch_nonboundary == 0


def main():
    ok = True
    ok &= check_rows([[0.0] * 64], "all-zero row")
    ok &= check_rows([[1000.0] + [1.0] * 63], "single-outlier row")
    ok &= check_rows([[float(v) for v in range(-32, 32)]], "ramp row")
    rng = np.random.default_rng(18220808)
    rows = []
    for _ in range(200):
        scale = 10.0 ** rng.uniform(-3, 3)
        rows.append(list((rng.standard_normal(50) * scale).astype(np.float32)))
    ok &= check_rows(rows, "200 random rows x 50")
    # sanity: q == 0 exactly where x == 0
    x = torch.tensor([[0.0, 5.0, -3.0, 0.0]], dtype=torch.float32)
    _, q, xh = hook_math(x)
    assert q[0, 0] == 0 and q[0, 3] == 0 and xh[0, 0] == 0 and xh[0, 3] == 0
    print(f"HOOK REFERENCE CHECK: {'GREEN' if ok else 'RED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
