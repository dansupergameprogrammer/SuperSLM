#!/usr/bin/env python3
"""T-1893 DEBUNK PROBE (disposable). Leg 3's null, re-derived paired.

Three things to check against the raw stored sequences:

1. PROVENANCE. Both arms are scored against a `float_output_ids` reference that
   each run stored for itself. If the two runs' float references are not
   identical sequences, the two agreement rates are not measured against the
   same reference and the delta means nothing. Checked byte-for-byte.

2. THE VARIANCE CONSTRUCTION. The packet combines leg 3's two arms as
   sqrt(hw_old^2 + hw_fused^2) -- two INDEPENDENT Wilson CIs -- while leg 1
   correctly uses a paired construction for the same shape of question (both
   arms answer the identical per-item question against the identical
   reference). Independent CIs are wider than paired ones whenever the arms are
   positively correlated, so this choice makes a null EASIER to declare. The
   paired resolving power is computed here.

3. WHAT valid_prefix ACTUALLY COUNTS. By construction (t1818_rescore
   valid_prefix: for first divergence d, agree=d out of n=d+1) every prompt
   contributes EXACTLY ONE disagreement if it differs at all and ZERO
   otherwise, and the d agreements it contributes are agreements by definition
   of the prefix -- they carry no independent information. So the numerator's
   information content is exactly "how many prompts differ", i.e. 32 Bernoulli
   trials, not 159/146 of them, and the Wilson half-width on 139/159 is
   computed over a denominator whose size is itself an outcome of the arm being
   measured. Both the effective-N correction and the paired test on the 32
   prompts are computed here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

BASE = Path(r"D:\SuperSLM\.worktrees\t1891-optionG-spike\out\t1891_capture")


def wilson_hw(k, n, z=1.96):
    if n == 0:
        return float("nan")
    p = k / n
    den = 1 + z * z / n
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return margin / den


def exact_binom_two_sided(k, m):
    """two-sided exact test that a split k of m is 50/50 (McNemar exact)."""
    if m == 0:
        return 1.0
    lo = min(k, m - k)
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(lo + 1)) / 2 ** m)


def main() -> int:
    old = json.loads((BASE / "t1800_old.json").read_text())
    fused = json.loads((BASE / "t1800_fused.json").read_text())
    ro, rf = old["per_prompt"], fused["per_prompt"]
    assert len(ro) == len(rf) == 32

    print("=== 1. provenance: is the float reference identical across the two runs? ===")
    mismatch = [i for i, (a, b) in enumerate(zip(ro, rf))
                if a["float_output_ids"] != b["float_output_ids"]]
    qmis = [i for i, (a, b) in enumerate(zip(ro, rf)) if a.get("question") != b.get("question")]
    print(f"prompts whose stored float reference differs between the runs: {len(mismatch)} {mismatch}")
    print(f"prompts whose question text differs between the runs: {len(qmis)}")
    same_arm = sum(1 for a, b in zip(ro, rf) if a["int8_output_ids"] == b["int8_output_ids"])
    print(f"prompts where OLD and FUSED produced identical int8 output: {same_arm}/32")

    print("\n=== 2. paired analysis on the 32 prompts (the real unit) ===")
    # a prompt "agrees" if the int8 sequence equals the float reference exactly
    ao = [a["int8_output_ids"] == a["float_output_ids"] for a in ro]
    af = [b["int8_output_ids"] == b["float_output_ids"] for b in rf]
    n = 32
    b01 = sum(1 for o, f in zip(ao, af) if (not o) and f)   # old wrong, fused right
    b10 = sum(1 for o, f in zip(ao, af) if o and (not f))   # old right, fused wrong
    b11 = sum(1 for o, f in zip(ao, af) if o and f)
    b00 = sum(1 for o, f in zip(ao, af) if (not o) and (not f))
    delta = (sum(af) - sum(ao)) / n
    p01, p10 = b01 / n, b10 / n
    hw_paired = 1.96 * math.sqrt(max(p01 + p10 - (p01 - p10) ** 2, 0.0) / n)
    hw_indep = math.sqrt(wilson_hw(sum(ao), n) ** 2 + wilson_hw(sum(af), n) ** 2)
    step = 1.0 / n
    rp_paired = max(step, hw_paired)
    print(f"full-sequence exact match: old={sum(ao)}/32={sum(ao)/n:.4f}  fused={sum(af)}/32={sum(af)/n:.4f}")
    print(f"discordant: old-wrong/fused-right={b01}  old-right/fused-wrong={b10}  "
          f"concordant: both-right={b11} both-wrong={b00}")
    print(f"paired delta = {delta:+.4f}")
    print(f"resolving power, PAIRED (packet's leg-1 construction) = {rp_paired:.4f} "
          f"-> {'DETECTABLE' if abs(delta) > rp_paired else 'NOT DISTINGUISHABLE FROM ZERO'}")
    print(f"resolving power, INDEPENDENT (packet's leg-3 construction) = {hw_indep:.4f} "
          f"-> {'DETECTABLE' if abs(delta) > hw_indep else 'NOT DISTINGUISHABLE FROM ZERO'}")
    print(f"exact McNemar two-sided p on ({b01},{b10}) = {exact_binom_two_sided(b01, b01+b10):.4f}")

    print("\n=== 3. what valid_prefix's denominator is made of ===")
    rs = json.loads((BASE / "t1818_rescore.json").read_text())
    for lab in ("old", "fused"):
        pp = rs[lab]["per_prompt"]
        vp = rs[lab]["semantics"]["valid_prefix"]
        disagreements = sum(p["valid_prefix_n"] - p["valid_prefix_agree"] for p in pp)
        differing = sum(1 for p in pp if p["valid_prefix_n"] != p["valid_prefix_agree"])
        per_prompt_max = max(p["valid_prefix_n"] - p["valid_prefix_agree"] for p in pp)
        print(f"{lab:<6} valid_prefix {vp['agree']}/{vp['n']} = {vp['rate']:.4f} "
              f"+/-{vp['wilson_95_halfwidth']:.4f} (Wilson, n={vp['n']} treated as independent trials)")
        print(f"       total disagreeing positions = {disagreements}; prompts that differ at all "
              f"= {differing}; max disagreements contributed by any one prompt = {per_prompt_max}")
        print(f"       => numerator information = {differing} prompt-level events, not {vp['n']} trials")
        # effective-N Wilson: same rate, but n = number of prompts
        k_eff = 32 - differing
        print(f"       Wilson half-width recomputed at the prompt level "
              f"({k_eff}/32 prompts fully agreeing) = {wilson_hw(k_eff, 32):.4f} "
              f"(vs {vp['wilson_95_halfwidth']:.4f} claimed)")
        # design-effect style effective n for a ratio-of-sums with cluster size varying
        mean_len = vp["n"] / 32
        print(f"       mean comparable positions per prompt = {mean_len:.2f}")

    print("\n=== 4. are leg 3's three reported rows independent evidence? ===")
    for lab in ("old", "fused"):
        pp = rs[lab]["per_prompt"]
        differing = sum(1 for p in pp if p["valid_prefix_n"] != p["valid_prefix_agree"])
        fse = rs[lab]["full_sequence_exact_match"]
        print(f"{lab:<6} prompts differing (drives valid_prefix numerator) = {differing}; "
              f"full_seq_exact = {fse['agree']}/{fse['n']} -> 32-{fse['agree']} = {32-fse['agree']} differing")
        print(f"       identical prompt-level event set: {differing == 32 - fse['agree']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
