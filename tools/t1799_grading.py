#!/usr/bin/env python3
"""T-1799: grade the widened-accumulate-only product candidate (t1797_residual_arms' "k8"
and "k4" arms, unmodified) against the verified float32 reference, per state and per
position, over 9 held-out prompts (E0's p7-p15 -- distinct from every prompt used to tune
or grade the T-1797 solve's or T-1798 debunk's arms, both of which used only p1-p6).

Inputs: out/t1799/p{i}_float.bin (copied verbatim from the existing T-1797 E0 capture, not
recomputed), out/t1799/p{i}_eng.bin (production, copied verbatim, used only for the k0
self-check), out/t1799arms/p{i}_{k0,k4,k8}.bin (this ticket's own runs).

Outputs: per-state table at the last prompt token (comparable to T-1797/T-1798's own
tables) and per-state table pooled over ALL prompt positions (this ticket's primary axis,
per the dispatch brief), for both k8 (primary candidate) and k4 (lighter-cost sibling),
plus paired t-statistics against production.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HELD_OUT = list(range(7, 16))
STATES = list(range(29))
HIDDEN = 1536


def load(path):
    return np.fromfile(path, dtype=np.float32).astype(np.float64).reshape(-1, 29, HIDDEN)


def rel_l2_cos(a, b):
    d = a - b
    rel = np.linalg.norm(d) / np.linalg.norm(b)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return rel, cos


def main() -> int:
    arms = ["eng", "k4", "k8"]
    # last-token cell
    last = {a: {s: [] for s in STATES} for a in arms}
    # all-position cell: per (prompt) pooled rel L2 needs per-position vectors; store
    # per-position (rel, cos) tuples per state, tagged by prompt for pairing.
    allpos = {a: {s: [] for s in STATES} for a in arms}  # list of (prompt_id, pos, rel, cos)

    for i in HELD_OUT:
        flt = load(f"out/t1799/p{i}_float.bin")
        n_pos = flt.shape[0]
        for arm in arms:
            path = (f"out/t1799/p{i}_eng.bin" if arm == "eng"
                    else f"out/t1799arms/p{i}_{arm}.bin")
            e = load(path)
            assert e.shape == flt.shape, (i, arm, e.shape, flt.shape)
            t_last = e.shape[0] - 1
            for s in STATES:
                rel, cos = rel_l2_cos(e[t_last, s], flt[t_last, s])
                last[arm][s].append((i, rel, cos))
                for t in range(n_pos):
                    rel_t, cos_t = rel_l2_cos(e[t, s], flt[t, s])
                    allpos[arm][s].append((i, t, rel_t, cos_t))

    def paired_t(arm_vals, base_vals):
        # arm_vals, base_vals: aligned lists of (key..., rel, cos) with same keys/order
        d = np.array([av[-2] - bv[-2] for av, bv in zip(arm_vals, base_vals)])
        n = d.size
        if n < 2:
            return d.mean(), float("nan"), float("nan")
        sd = d.std(ddof=1)
        sem = sd / np.sqrt(n)
        t = d.mean() / sem if sem > 0 else float("inf")
        return d.mean(), sem, t

    print("== T-1799: last-prompt-token cell (9 held-out prompts, per state) ==")
    print(f"{'st':>3} | {'prod':>7} | {'k4':>7} | {'k8':>7} | {'k4-prod(t)':>14} | {'k8-prod(t)':>14} | prod cos | k8 cos")
    for s in STATES:
        prod = np.array([v[1] for v in last["eng"][s]])
        k4v = np.array([v[1] for v in last["k4"][s]])
        k8v = np.array([v[1] for v in last["k8"][s]])
        prod_cos = np.array([v[2] for v in last["eng"][s]]).mean()
        k8_cos = np.array([v[2] for v in last["k8"][s]]).mean()
        d4, sem4, t4 = paired_t(last["k4"][s], last["eng"][s])
        d8, sem8, t8 = paired_t(last["k8"][s], last["eng"][s])
        print(f"{s:>3} | {prod.mean():.3f} | {k4v.mean():.3f} | {k8v.mean():.3f} | "
              f"{d4:+.4f}(t={t4:+.1f}) | {d8:+.4f}(t={t8:+.1f}) | {prod_cos:.4f} | {k8_cos:.4f}")

    print()
    print("== T-1799: ALL-POSITIONS cell (9 held-out prompts, all positions per prompt, per state) ==")
    print(f"{'st':>3} | {'n':>5} | {'prod':>7} | {'k4':>7} | {'k8':>7} | {'k4-prod(t)':>14} | {'k8-prod(t)':>14} | worse-than-prod(k8)")
    for s in STATES:
        prod = np.array([v[2] for v in allpos["eng"][s]])
        k4v = np.array([v[2] for v in allpos["k4"][s]])
        k8v = np.array([v[2] for v in allpos["k8"][s]])
        d4, sem4, t4 = paired_t(allpos["k4"][s], allpos["eng"][s])
        d8, sem8, t8 = paired_t(allpos["k8"][s], allpos["eng"][s])
        worse = int((k8v > prod).sum())
        print(f"{s:>3} | {prod.size:>5} | {prod.mean():.3f} | {k4v.mean():.3f} | {k8v.mean():.3f} | "
              f"{d4:+.4f}(t={t4:+.1f}) | {d8:+.4f}(t={t8:+.1f}) | {worse}/{prod.size}")

    # summary totals across all states/positions
    total_n = sum(len(allpos["eng"][s]) for s in STATES)
    total_worse_k8 = sum(int((np.array([v[2] for v in allpos["k8"][s]]) >
                               np.array([v[2] for v in allpos["eng"][s]])).sum()) for s in STATES)
    total_worse_k4 = sum(int((np.array([v[2] for v in allpos["k4"][s]]) >
                               np.array([v[2] for v in allpos["eng"][s]])).sum()) for s in STATES)
    print()
    print(f"pooled (all states x all positions, n={total_n}): "
          f"k8 worse than production at {total_worse_k8} cells; "
          f"k4 worse than production at {total_worse_k4} cells")

    # final-state headline, all-positions pooled (single number, like T-1798's F4)
    s28_prod = np.array([v[2] for v in allpos["eng"][28]])
    s28_k8 = np.array([v[2] for v in allpos["k8"][28]])
    s28_k4 = np.array([v[2] for v in allpos["k4"][28]])
    print()
    print(f"final state (28), all positions pooled, n={s28_prod.size}: "
          f"prod {s28_prod.mean():.3f} -> k8 {s28_k8.mean():.3f} "
          f"({100*(1-s28_k8.mean()/s28_prod.mean()):.1f}% relative reduction); "
          f"k4 {s28_k4.mean():.3f} ({100*(1-s28_k4.mean()/s28_prod.mean()):.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
