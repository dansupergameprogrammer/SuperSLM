#!/usr/bin/env python3
"""T-1778: compare the engine's REAL Q15 attention probabilities
(`tools/t1778_engine_attn_probe.cpp`'s own dump) against the INDEPENDENT float reference
(`tools/t1778_float_attn_reference.py`'s own dump) -- the comparison this ticket exists to
run. Answers, per checkpoint layer: is layer 0's attention genuinely worse against ground
truth, or only against a reference (IdealProbsQ15) that shares its construction?

Reads two text dumps for the SAME prompt (same chat-templated text on both sides -- checked
here by comparing declared width, not re-derived) and reports, per (layer, head) row and
pooled per layer:

  - total variation distance (TVD = 0.5 * sum_k |p_engine[k] - p_float[k]|) -- a bounded
    [0,1] distance between two probability distributions, insensitive to which side is
    "reference", unlike a directed KL divergence.
  - mean absolute per-element probability difference.
  - max absolute per-element probability difference.
  - elements differing by more than the engine's own Q15 resolution floor (1/32768).

This script performs NO fitting, no thresholding decision, and computes nothing derived from
`IExpScaleConstants`/`q_ln2`/`kIExpClipN` -- it reads two already-computed probability arrays
and reports plain distance statistics between them.

Usage
-----
    python tools\\t1778_compare_probs.py out\\t1778\\p1.txt out\\t1778\\p1_float.txt
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict


def read_engine_dump(path: str):
    """Format: `<num_rows> <width>` then `<layer> <head> <width> <p_0> ... <p_{width-1}>`
    (p_k integer Q15, 0..32768) per row."""
    rows = {}
    with open(path, "r", encoding="ascii") as f:
        header = f.readline().split()
        num_rows, width = int(header[0]), int(header[1])
        for _ in range(num_rows):
            parts = f.readline().split()
            layer, head, w = int(parts[0]), int(parts[1]), int(parts[2])
            probs_q15 = [int(x) for x in parts[3 : 3 + w]]
            rows[(layer, head)] = [p / 32768.0 for p in probs_q15]
    return rows, width


def read_float_dump(path: str):
    """Format: `<num_rows> <width>` then `<layer> <head> <width> <p_0> ... <p_{width-1}>`
    (p_k float64 repr) per row."""
    rows = {}
    with open(path, "r", encoding="ascii") as f:
        header = f.readline().split()
        num_rows, width = int(header[0]), int(header[1])
        for _ in range(num_rows):
            parts = f.readline().split()
            layer, head, w = int(parts[0]), int(parts[1]), int(parts[2])
            probs = [float(x) for x in parts[3 : 3 + w]]
            rows[(layer, head)] = probs
    return rows, width


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("engine_dump")
    parser.add_argument("float_dump")
    args = parser.parse_args(argv)

    engine_rows, engine_width = read_engine_dump(args.engine_dump)
    float_rows, float_width = read_float_dump(args.float_dump)

    if engine_width != float_width:
        print(
            f"FAILED: width mismatch -- engine width={engine_width}, float width={float_width} "
            f"(the two dumps are not the same prompt/position)",
            file=sys.stderr,
        )
        return 1

    engine_keys = set(engine_rows.keys())
    float_keys = set(float_rows.keys())
    if engine_keys != float_keys:
        print(
            f"FAILED: (layer,head) key mismatch -- engine only: {sorted(engine_keys - float_keys)}, "
            f"float only: {sorted(float_keys - engine_keys)}",
            file=sys.stderr,
        )
        return 1

    Q15_FLOOR = 1.0 / 32768.0

    per_row = {}
    for key in sorted(engine_keys):
        pe = engine_rows[key]
        pf = float_rows[key]
        if len(pe) != len(pf):
            print(f"FAILED: row {key} width mismatch: engine={len(pe)} float={len(pf)}", file=sys.stderr)
            return 1
        abs_diffs = [abs(a - b) for a, b in zip(pe, pf)]
        tvd = 0.5 * sum(abs_diffs)
        mean_abs = sum(abs_diffs) / len(abs_diffs)
        max_abs = max(abs_diffs)
        n_above_floor = sum(1 for d in abs_diffs if d > Q15_FLOOR)
        per_row[key] = {
            "tvd": tvd,
            "mean_abs": mean_abs,
            "max_abs": max_abs,
            "n_above_floor": n_above_floor,
            "width": len(pe),
        }

    print(f"width={engine_width}  rows={len(per_row)}\n")
    print(f"{'layer':>5} {'head':>4} {'TVD':>10} {'mean|d|':>10} {'max|d|':>10} {'>Q15floor':>10}")
    for key in sorted(per_row.keys()):
        r = per_row[key]
        print(
            f"{key[0]:>5} {key[1]:>4} {r['tvd']:>10.6f} {r['mean_abs']:>10.6f} {r['max_abs']:>10.6f} "
            f"{r['n_above_floor']:>4}/{r['width']:<5}"
        )

    print("\nPer-layer pooled (mean over heads):")
    per_layer = defaultdict(list)
    for (layer, head), r in per_row.items():
        per_layer[layer].append(r)
    print(f"{'layer':>5} {'mean TVD':>10} {'mean mean|d|':>13} {'mean max|d|':>12} {'total >floor':>13} {'total elems':>12}")
    layer_summary = {}
    for layer in sorted(per_layer.keys()):
        rs = per_layer[layer]
        mean_tvd = sum(r["tvd"] for r in rs) / len(rs)
        mean_mean_abs = sum(r["mean_abs"] for r in rs) / len(rs)
        mean_max_abs = sum(r["max_abs"] for r in rs) / len(rs)
        total_above = sum(r["n_above_floor"] for r in rs)
        total_elems = sum(r["width"] for r in rs)
        layer_summary[layer] = (mean_tvd, mean_mean_abs, mean_max_abs, total_above, total_elems)
        print(
            f"{layer:>5} {mean_tvd:>10.6f} {mean_mean_abs:>13.6f} {mean_max_abs:>12.6f} "
            f"{total_above:>13} {total_elems:>12}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
