#!/usr/bin/env python3
"""T-1778: pool the per-element engine-vs-float-reference comparison across all three
held-out prompts, weighted by element count (not an average of per-prompt averages), and
report per-layer and grand-pooled TVD/mean|delta|/max|delta| plus the achieved resolving
power. Reuses read_engine_dump/read_float_dump from t1778_compare_probs.py unchanged."""

from __future__ import annotations

import sys
from collections import defaultdict

sys.path.insert(0, "tools")
from t1778_compare_probs import read_engine_dump, read_float_dump  # noqa: E402

PAIRS = [
    ("out/t1778/p1.txt", "out/t1778/p1_float.txt"),
    ("out/t1778/p2.txt", "out/t1778/p2_float.txt"),
    ("out/t1778/p3.txt", "out/t1778/p3_float.txt"),
]

Q15_FLOOR = 1.0 / 32768.0


def main():
    per_layer_elems = defaultdict(list)  # layer -> list of abs_diff (every element, every prompt, every head)
    per_layer_row_tvds = defaultdict(list)  # layer -> list of per-row TVD (one per (prompt,layer,head))

    for engine_path, float_path in PAIRS:
        engine_rows, ew = read_engine_dump(engine_path)
        float_rows, fw = read_float_dump(float_path)
        assert ew == fw, (engine_path, ew, fw)
        assert set(engine_rows) == set(float_rows)
        for (layer, head), pe in engine_rows.items():
            pf = float_rows[(layer, head)]
            assert len(pe) == len(pf)
            abs_diffs = [abs(a - b) for a, b in zip(pe, pf)]
            per_layer_elems[layer].extend(abs_diffs)
            per_layer_row_tvds[layer].append(0.5 * sum(abs_diffs))

    print(f"{'layer':>5} {'n_elems':>8} {'mean|d|':>10} {'max|d|':>10} {'n>Q15floor':>11} {'pct>floor':>10} {'mean_row_TVD':>13} {'n_rows':>7}")
    layer0_mean = None
    others = []
    for layer in sorted(per_layer_elems.keys()):
        diffs = per_layer_elems[layer]
        n = len(diffs)
        mean_d = sum(diffs) / n
        max_d = max(diffs)
        n_above = sum(1 for d in diffs if d > Q15_FLOOR)
        pct_above = 100.0 * n_above / n
        row_tvds = per_layer_row_tvds[layer]
        mean_row_tvd = sum(row_tvds) / len(row_tvds)
        print(f"{layer:>5} {n:>8} {mean_d:>10.6f} {max_d:>10.6f} {n_above:>11} {pct_above:>9.2f}% {mean_row_tvd:>13.6f} {len(row_tvds):>7}")
        if layer == 0:
            layer0_mean = mean_row_tvd
        else:
            others.append((layer, mean_row_tvd))

    print()
    print(f"layer 0 mean row TVD: {layer0_mean:.6f}")
    print(f"other checkpoint layers' mean row TVD: {sorted(others, key=lambda x: x[1])}")
    print(f"min of others: {min(o[1] for o in others):.6f}  max of others: {max(o[1] for o in others):.6f}")
    print(f"layer 0 rank among {{0}}+others by mean row TVD (1=worst): "
          f"{sorted([layer0_mean] + [o[1] for o in others], reverse=True).index(layer0_mean) + 1} of {len(others)+1}")

    # Grand pooled (all layers together) achieved resolving power context.
    all_diffs = [d for diffs in per_layer_elems.values() for d in diffs]
    print(f"\ngrand pooled: n={len(all_diffs)} elements, mean|d|={sum(all_diffs)/len(all_diffs):.6f}, "
          f"max|d|={max(all_diffs):.6f}")
    print(f"Q15 resolution floor (engine side): {Q15_FLOOR:.8f}")


if __name__ == "__main__":
    main()
