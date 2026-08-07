#!/usr/bin/env python3
"""T-1819 attack on claims 5 and 6.

Claim 5: "nine of eleven chain-funnel sites, plus the K/V landing, match a uniform null."
Claim 6: "the residual sites sit 21-30% farther from a boundary than uniform, and are
          5-20x more channel-structured."

A null is only a null against a stated resolving power (StandardsDocument.md 5.4).  This
script reads T-1796's own per-site-per-prompt SITE records straight out of the boundary
dumps and, for every site, reports its deviation from the uniform null IN UNITS OF ITS OWN
ACHIEVED RESOLVING POWER -- the inter-prompt standard deviation at n=3, which is the
resolving-power measure T-1796 itself uses.  A site whose deviation is many multiples of
that spread has NOT matched the null; it has been placed in the null bucket by eye.

For claim 6 it additionally computes the sampling-noise floor of the per-channel
coefficient of variation.  Per-channel within-5% counts are binomial with n elements per
channel, so even a site with ZERO real channel structure shows CV = sqrt((1-p)/(n*p)).
The "5-20x more channel-structured" ratio is a ratio against that floor, not against a
measured zero, and its numerator and denominator are therefore not the same quantity.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SITE_RE = re.compile(r"^SITE (\S+) n=(\d+) mean=(\S+) min=\S+ within1pct=(\d+) "
                     r"within5pct=(\d+) within10pct=(\d+)")
CHAN_RE = re.compile(r"^CHANNEL (\S+) (\d+) n=(\d+) mean=(\S+) within5pct=(\d+)")


LAYER_PREFIX = re.compile(r"^layer\.layer\d+\.")


def kind(name: str) -> str:
    """'layer.layer17.attn_ctx' -> 'attn_ctx'; 'kv_landing_k' -> 'kv_landing_k'."""
    return LAYER_PREFIX.sub("", name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    # per prompt: site-kind -> [n, w1, w5, w10, sum(mean*n)]
    per_prompt = []
    chan = defaultdict(dict)  # prompt -> kind -> {channel_index: [n, w5]}
    for pi, d in enumerate(a.dumps):
        acc = defaultdict(lambda: [0, 0, 0, 0, 0.0])
        with open(d) as f:
            for line in f:
                if line.startswith("SITE "):
                    m = SITE_RE.match(line)
                    name, n, mean, w1, w5, w10 = m.group(1), int(m.group(2)), float(m.group(3)), \
                        int(m.group(4)), int(m.group(5)), int(m.group(6))
                    k = kind(name)
                    r = acc[k]
                    r[0] += n; r[1] += w1; r[2] += w5; r[3] += w10; r[4] += mean * n
                elif line.startswith("CHANNEL "):
                    m = CHAN_RE.match(line)
                    # aggregate a channel index across all 28 layers, which is what a
                    # site-level "per-channel CV" pools over
                    key = (kind(m.group(1)), int(m.group(2)))
                    slot = chan[pi].setdefault(key[0], {})
                    cur = slot.get(key[1], [0, 0])
                    cur[0] += int(m.group(3)); cur[1] += int(m.group(5))
                    slot[key[1]] = cur
        per_prompt.append(acc)

    kinds = sorted(per_prompt[0].keys())
    print("=== claim 5: deviation from the uniform null, in units of achieved resolving power ===")
    print("null: mean 0.25, %within-1% = 2, %within-5% = 10, %within-10% = 20")
    print(f"{'site':>20} {'n/prompt':>10} | {'mean':>7} {'sd':>7} {'dev/sd':>8} | "
          f"{'%<5%':>7} {'sd':>6} {'dev':>7} {'dev/sd':>9} | {'verdict':>14}")
    rows = []
    for k in kinds:
        means = np.array([p[k][4] / p[k][0] for p in per_prompt])
        w5 = np.array([100.0 * p[k][2] / p[k][0] for p in per_prompt])
        w1 = np.array([100.0 * p[k][1] / p[k][0] for p in per_prompt])
        n = int(np.mean([p[k][0] for p in per_prompt]))
        sd_m = means.std(ddof=1)
        sd5 = w5.std(ddof=1)
        dev_m = means.mean() - 0.25
        dev5 = w5.mean() - 10.0
        z_m = abs(dev_m) / sd_m if sd_m > 0 else float("inf")
        z5 = abs(dev5) / sd5 if sd5 > 0 else float("inf")
        verdict = "MATCHES null" if max(z_m, z5) < 3 else "DEVIATES"
        rows.append(dict(site=k, n_per_prompt=n, mean=means.mean(), mean_sd=sd_m,
                         mean_dev=dev_m, mean_z=z_m, w1=w1.mean(),
                         w5=w5.mean(), w5_sd=sd5, w5_dev=dev5, w5_z=z5, verdict=verdict))
        print(f"{k:>20} {n:>10} | {means.mean():>7.4f} {sd_m:>7.4f} {z_m:>8.1f} | "
              f"{w5.mean():>7.3f} {sd5:>6.3f} {dev5:>+7.3f} {z5:>9.1f} | {verdict:>14}")

    n_match = sum(1 for r in rows if r["verdict"] == "MATCHES null" and r["site"] not in
                  ("kv_landing_k", "kv_landing_v", "prob_quant"))
    n_chain = sum(1 for r in rows if r["site"] not in
                  ("kv_landing_k", "kv_landing_v", "prob_quant"))
    print(f"\nchain-funnel sites matching the uniform null at |dev| < 3 sd: "
          f"{n_match} of {n_chain}  (T-1796 claims 9 of 11)")

    print()
    print("=== claim 6: per-channel CV against its own binomial sampling floor ===")
    print(f"{'site':>20} {'chans':>7} {'n/chan':>8} {'CV observed':>12} {'CV noise floor':>15} "
          f"{'excess (obs/floor)':>19} {'structural CV':>15}")
    cv_rows = []
    for k in kinds:
        cvs, floors, ncs, nch = [], [], [], []
        for pi in range(len(a.dumps)):
            recs = chan[pi].get(k)
            if not recs:
                continue
            recs = list(recs.values())
            n_arr = np.array([r[0] for r in recs], dtype=float)
            w_arr = np.array([r[1] for r in recs], dtype=float)
            frac = w_arr / n_arr
            p = frac.mean()
            cvs.append(frac.std(ddof=1) / p if p > 0 else float("nan"))
            nbar = n_arr.mean()
            floors.append(math.sqrt((1 - p) / (nbar * p)) if p > 0 else float("nan"))
            ncs.append(nbar); nch.append(len(recs))
        if not cvs:
            continue
        cv, fl = float(np.mean(cvs)), float(np.mean(floors))
        struct = math.sqrt(max(0.0, cv ** 2 - fl ** 2))
        cv_rows.append(dict(site=k, channels=int(np.mean(nch)), n_per_channel=float(np.mean(ncs)),
                            cv=cv, cv_floor=fl, excess=cv / fl, structural_cv=struct))
        print(f"{k:>20} {int(np.mean(nch)):>7} {np.mean(ncs):>8.0f} {cv:>12.4f} "
              f"{fl:>15.4f} {cv/fl:>19.2f} {struct:>15.4f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"sites": rows, "channel_cv": cv_rows}, indent=2))
    print("written:", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
