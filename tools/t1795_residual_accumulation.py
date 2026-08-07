#!/usr/bin/env python3
"""T-1795: per-layer residual-stream accumulation analysis.

Reads the engine's real 29-state residual dump (tools/t1795_residual_probe.cpp's own
"*_residual.txt", self-checked bit-for-bit against production at 6 checkpoint layers and
carrying an executed budget-invariance cross-check on its own final state) and the
independent float32-verified reference (tools/t1795_float_residual_reference.py's own
"*_float.txt", precision verified by execution, argmax-cross-checked against the
model's own standard forward), both at the SAME 29 boundaries (embedding output, then
each of the 28 decoder layers' own output), last prompt token only, and computes per
state:

  - relative L2 distance:  ||engine - float|| / ||float||           (diffusion/magnitude)
  - cosine similarity:     (engine . float) / (||engine|| ||float||) (rotation/direction)
  - norm ratio:            ||engine|| / ||float||                    (scaling)

reported per layer, per prompt, and pooled -- with resolving power computed directly
(inter-prompt spread at n=3) per StandardsDocument.md 5.4, not asserted.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_states(path: Path):
    states = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            n = int(parts[0])
            vals = [float(x) for x in parts[1 : 1 + n]]
            assert len(vals) == n, (len(vals), n)
            states.append(vals)
    assert len(states) == 29, len(states)
    return states


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, nargs="+")
    parser.add_argument("--float", required=True, nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    per_prompt = []
    for eng_path, float_path in zip(args.engine, args.float):
        eng_states = read_states(Path(eng_path))
        float_states = read_states(Path(float_path))
        rows = []
        for idx in range(29):
            e = eng_states[idx]
            f = float_states[idx]
            ne = norm(e)
            nf = norm(f)
            d = dot(e, f)
            cos = d / (ne * nf) if ne > 0 and nf > 0 else float("nan")
            rel_l2 = norm([a - b for a, b in zip(e, f)]) / nf if nf > 0 else float("nan")
            rows.append({
                "state_index": idx,  # 0 = embedding output, k = layer (k-1)'s own output
                "engine_norm": ne,
                "float_norm": nf,
                "norm_ratio": ne / nf if nf > 0 else float("nan"),
                "cosine": cos,
                "rel_l2": rel_l2,
            })
        per_prompt.append(rows)

    n_prompts = len(per_prompt)
    pooled = []
    for idx in range(29):
        rel_l2s = [per_prompt[p][idx]["rel_l2"] for p in range(n_prompts)]
        cosines = [per_prompt[p][idx]["cosine"] for p in range(n_prompts)]
        ratios = [per_prompt[p][idx]["norm_ratio"] for p in range(n_prompts)]
        def meanstd(xs):
            m = sum(xs) / len(xs)
            var = sum((x - m) ** 2 for x in xs) / len(xs)
            return m, math.sqrt(var)
        rel_l2_mean, rel_l2_std = meanstd(rel_l2s)
        cos_mean, cos_std = meanstd(cosines)
        ratio_mean, ratio_std = meanstd(ratios)
        pooled.append({
            "state_index": idx,
            "rel_l2_mean": rel_l2_mean, "rel_l2_std": rel_l2_std,
            "cosine_mean": cos_mean, "cosine_std": cos_std,
            "norm_ratio_mean": ratio_mean, "norm_ratio_std": ratio_std,
        })

    out = {"per_prompt": per_prompt, "pooled": pooled}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"{'state':>5} {'rel_l2':>9} {'std':>7} {'cosine':>9} {'std':>8} {'norm_ratio':>10} {'std':>7}")
    for row in pooled:
        print(f"{row['state_index']:>5} {row['rel_l2_mean']:>9.4f} {row['rel_l2_std']:>7.4f} "
              f"{row['cosine_mean']:>9.6f} {row['cosine_std']:>8.6f} "
              f"{row['norm_ratio_mean']:>10.4f} {row['norm_ratio_std']:>7.4f}")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
