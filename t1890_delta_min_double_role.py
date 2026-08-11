"""
T-1890 adversary probe -- the delta_min double-role fracture.

Target promise
--------------
Claude/Vitruvius/t1870-encoder-retrieval-instrument-design-2026-08-09.md

  §3.2  "For a paired binary metric (recall@1), detecting a true proportion difference
         delta at two-sided significance alpha and power 1-beta needs
             n = (z_{alpha/2} + z_beta)^2 * sigma_d^2 / delta^2"
        + an "Executed table" of ten (power, delta, N) rows.

  §3.3  "The corpus size, executed at delta_min = 0.05 ... n = 784.888 -> n = 785"
        "785 documents ... sized for a single-cell test ... 80% power"

  §6    RESOLVED-RECOVERY  iff  signed_ratio > 1  AND  (candidate - base) >= +delta_min
        with signed_ratio = (candidate - base) / (z_crit * SE_paired)

The sizing formula solves for the N at which a test rejecting at  z_crit * SE  has
power 1-beta against a true effect delta.  The rule in §6 does not reject at
z_crit * SE.  It rejects at  max(z_crit * SE, delta_min).  This probe measures the
power the §6 rule actually achieves at the N the §3 formula selects.

Everything below uses the design's own planning inputs, unchanged:
  sigma_d = 0.50            (§3.2, "conservative (upper) end" of the measured 0.399-0.531)
  z_beta  = 0.841621 (80%), 1.281552 (90%)
  z_crit  = 3.038074 (§3.2 / §6, family-wise m=21)
  z_crit  = 1.959964 (§3.3, m=1 for Stage 1)
"""

import numpy as np
from scipy.stats import norm

SIGMA_D = 0.50
Z_B80 = 0.841621
Z_B90 = 1.281552
Z_FAMILY = 3.038074   # §3.2/§6, Bonferroni m=21
Z_SINGLE = 1.959964   # §3.3, m=1


def size_corpus(z_crit, z_beta, delta, sigma_d=SIGMA_D):
    """§3.2's formula, verbatim."""
    return (z_crit + z_beta) ** 2 * sigma_d ** 2 / delta ** 2


def power_naive(n, z_crit, delta_true, sigma_d=SIGMA_D):
    """Power of the test the FORMULA describes: reject iff  mean > z_crit * SE."""
    se = sigma_d / np.sqrt(n)
    return 1.0 - norm.cdf((z_crit * se - delta_true) / se)


def power_compound(n, z_crit, delta_min, delta_true, sigma_d=SIGMA_D):
    """Power of the rule §6 actually states: reject iff mean > z_crit*SE AND mean >= delta_min."""
    se = sigma_d / np.sqrt(n)
    thresh = max(z_crit * se, delta_min)
    return 1.0 - norm.cdf((thresh - delta_true) / se)


print("=" * 78)
print("PART 1 -- reproduce the design's own N figures from its own formula")
print("=" * 78)

rows_32 = [  # (power label, z_beta, delta_min, N as printed in §3.2's executed table)
    ("80%", Z_B80, 0.05, 1505), ("80%", Z_B80, 0.06, 1045), ("80%", Z_B80, 0.07, 768),
    ("80%", Z_B80, 0.08, 588),  ("80%", Z_B80, 0.10, 376),
    ("90%", Z_B90, 0.05, 1866), ("90%", Z_B90, 0.06, 1296), ("90%", Z_B90, 0.07, 952),
    ("90%", Z_B90, 0.08, 729),  ("90%", Z_B90, 0.10, 467),
]
print(f"{'stage':<10}{'power':<7}{'d_min':<8}{'N printed':<11}{'N recomputed':<14}{'match'}")
for lbl, zb, d, n_doc in rows_32:
    n_calc = size_corpus(Z_FAMILY, zb, d)
    print(f"{'§3.2':<10}{lbl:<7}{d:<8.2f}{n_doc:<11}{n_calc:<14.3f}"
          f"{'YES' if int(np.ceil(n_calc)) == n_doc else 'NO'}")
n785 = size_corpus(Z_SINGLE, Z_B80, 0.05)
print(f"{'§3.3':<10}{'80%':<7}{0.05:<8.2f}{785:<11}{n785:<14.3f}"
      f"{'YES' if int(np.ceil(n785)) == 785 else 'NO'}")

print()
print("=" * 78)
print("PART 2 -- which of §6's two thresholds actually binds, at the design's own N")
print("=" * 78)
print(f"{'stage':<8}{'power':<7}{'d_min':<8}{'N':<8}{'z_crit*SE':<12}{'binding thresh':<16}{'which binds'}")
allrows = [("§3.2", lbl, zb, d, n, Z_FAMILY) for lbl, zb, d, n in rows_32]
allrows.append(("§3.3", "80%", Z_B80, 0.05, 785, Z_SINGLE))
for stage, lbl, zb, d, n, zc in allrows:
    se = SIGMA_D / np.sqrt(n)
    stat = zc * se
    binds = "delta_min" if d > stat else "z_crit*SE"
    print(f"{stage:<8}{lbl:<7}{d:<8.2f}{n:<8}{stat:<12.6f}{max(stat, d):<16.6f}{binds}")

print()
print("=" * 78)
print("PART 3 -- CLOSED FORM: why delta_min always binds, for any target power > 50%")
print("=" * 78)
print("""
  The formula sets   sqrt(n) = (z_crit + z_beta) * sigma / delta_min
  so the statistical threshold at that n is

      z_crit * sigma / sqrt(n)  =  delta_min * z_crit / (z_crit + z_beta)

  which is strictly LESS than delta_min whenever z_beta > 0, i.e. whenever the
  target power exceeds 50%.  The larger the power asked for, the further below
  delta_min the statistical threshold falls -- so delta_min binds by construction,
  in every row, at every effect size, in both stages.
""")
print(f"{'stage':<8}{'power':<7}{'ratio z/(z+zb)':<18}{'-> stat thresh':<17}{'d_min':<9}{'< d_min?'}")
for stage, lbl, zb, d, n, zc in allrows:
    ratio = zc / (zc + zb)
    print(f"{stage:<8}{lbl:<7}{ratio:<18.6f}{ratio * d:<17.6f}{d:<9.2f}{'YES' if ratio < 1 else 'NO'}")

print()
print("=" * 78)
print("PART 4 -- the power §6's rule ACTUALLY achieves at the design's own N and delta")
print("=" * 78)
print(f"{'stage':<8}{'claimed':<10}{'d_min':<8}{'N':<8}{'formula-rule pwr':<19}{'§6-rule pwr':<14}{'shortfall'}")
for stage, lbl, zb, d, n, zc in allrows:
    pn = power_naive(n, zc, d)
    pc = power_compound(n, zc, d, d)
    print(f"{stage:<8}{lbl:<10}{d:<8.2f}{n:<8}{pn:<19.4f}{pc:<14.4f}{pn - pc:+.4f}")

print()
print("=" * 78)
print("PART 5 -- MONTE CARLO, paired-binary generative model, §6's rule implemented verbatim")
print("=" * 78)
print("""per-document paired delta d_i in {-1,0,+1};  P(+1)=p10, P(-1)=p01
chosen so that  mean = p10 - p01 = delta_true  and  sd = sigma_d = 0.50
SE_paired is computed from the sample, as a grading tool would.""")

rng = np.random.default_rng(20260810)


def monte_carlo(n, z_crit, delta_min, delta_true, sigma_d=SIGMA_D, trials=200_000):
    var = sigma_d ** 2
    s = var + delta_true ** 2          # p10 + p01
    p10 = (s + delta_true) / 2.0
    p01 = (s - delta_true) / 2.0
    assert p10 >= 0 and p01 >= 0 and p10 + p01 <= 1.0, (p10, p01)
    probs = [p01, 1.0 - p10 - p01, p10]
    draws = rng.choice(np.array([-1.0, 0.0, 1.0]), size=(trials, n), p=probs)
    mean = draws.mean(axis=1)
    se = draws.std(axis=1, ddof=1) / np.sqrt(n)
    signed_ratio = mean / (z_crit * se)
    resolved_recovery = (signed_ratio > 1) & (mean >= delta_min)
    formula_rule = signed_ratio > 1
    rp = 1.959964 * se                                  # campaign RP convention
    underpowered = np.abs(mean) < rp
    return resolved_recovery.mean(), formula_rule.mean(), underpowered.mean()

print()
print(f"{'stage':<8}{'d_min':<8}{'N':<8}{'d_true':<9}{'MC §6 rule':<13}{'MC formula rule':<18}{'MC underpowered'}")
mc_cases = [
    ("§3.3", 0.05, 785, 0.05, Z_SINGLE),
    ("§3.3", 0.05, 785, 0.0460, Z_SINGLE),   # the arm's own measured point estimate
    ("§3.2", 0.06, 1045, 0.06, Z_FAMILY),
    ("§3.2", 0.05, 1505, 0.05, Z_FAMILY),
    ("§3.2", 0.10, 376, 0.10, Z_FAMILY),
]
for stage, d, n, dt, zc in mc_cases:
    rr, fr, up = monte_carlo(n, zc, d, dt)
    print(f"{stage:<8}{d:<8.2f}{n:<8}{dt:<9.4f}{rr:<13.4f}{fr:<18.4f}{up:.4f}")

print()
print("=" * 78)
print("PART 6 -- Stage 1 at the ONLY measured estimate of this arm's effect")
print("=" * 78)
print("""T-1835 §2 / T-1870 §11.2: off04_k_proj_landing measured +0.0460 recall@1 (N=239).
That is the sole measurement of this hypothesis in existence, and it sits BELOW
delta_min = 0.05.  Under §6's rule the observed delta must reach 0.05, so:""")
print()
print(f"{'N':<9}{'z_crit*SE':<12}{'binding':<11}{'P(RESOLVED-RECOVERY) at d_true=0.0460'}")
for n in [239, 384, 500, 785, 1200, 2000, 5000, 20000]:
    se = SIGMA_D / np.sqrt(n)
    stat = Z_SINGLE * se
    p = power_compound(n, Z_SINGLE, 0.05, 0.0460)
    print(f"{n:<9}{stat:<12.6f}{('d_min' if 0.05 > stat else 'z*SE'):<11}{p:.4f}")
print("""
Growing the corpus drives this DOWN, not up: once delta_min binds (N >= 385) the
gate demands the sample mean exceed a value the true effect does not reach, and
every added document shrinks the sampling noise that was the only route to it.""")

print()
print("=" * 78)
print("PART 7 -- N at which power stops being bought, and what 80% would actually cost")
print("=" * 78)
n_cross = (Z_SINGLE * SIGMA_D / 0.05) ** 2
print(f"§3.3 crossover N (z_crit*SE == delta_min): {n_cross:.1f} -> N = {int(np.ceil(n_cross))}")
print(f"  design's N = 785;  documents past the crossover = {785 - int(np.ceil(n_cross))}")
print(f"  §3.3's composition: 460 held-out `id` + 325 NEWLY AUTHORED `ood` (§12.6 cost)")
print()
print("Power of §6's rule at d_true = d_min = 0.05, swept over N:")
for n in [239, 385, 500, 785, 1500, 10000]:
    print(f"  N={n:<7} power = {power_compound(n, Z_SINGLE, 0.05, 0.05):.4f}")
print("""
It is pinned at exactly 0.5000 for every N at or past the crossover.  No corpus
size restores the stated 80%, because at d_true == d_min the rule asks the sample
mean to land above its own expectation, which is a coin flip at any precision.""")
print()
print("For 80% power under §6's rule you need d_true strictly above d_min:")
print(f"{'d_true':<10}{'d_true - d_min':<17}{'N for 80% power'}")
for dt in [0.055, 0.06, 0.07, 0.08, 0.10]:
    n_req = (Z_B80 * SIGMA_D / (dt - 0.05)) ** 2
    print(f"{dt:<10.3f}{dt - 0.05:<17.3f}{int(np.ceil(n_req))}")
print("  (at d_true <= d_min = 0.05, no N achieves 80% -- the column is undefined)")
