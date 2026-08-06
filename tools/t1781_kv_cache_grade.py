#!/usr/bin/env python3
"""T-1781 cell C10 -- grade Arm B's independent recompute (tools\\sslm_decode_
kv_probe.cpp's own per-position int8 dump, sslm_layer_trace.cpp's dump format)
against the independent float reference (tools\\t1781_float_raw_text_dump.py's
dump, float_reference_layer_dump.py's own dump format) at each self-generated
position -- reusing tools\\layer_bisection_report.py's own comparator
(compare_layer_row, check_provenance) UNMODIFIED, rather than a new one, so
this grading step inherits that comparator's own already-reviewed
Spearman/Pearson/max|z-diff| statistic set (design S5) and its own repeat-vs-
repeat resolving-power convention (design S10), instead of a fresh,
unreviewed implementation of either.

Usage
-----
    python tools\\t1781_kv_cache_grade.py --dump-dir out\\t1781 --label arith_12plus15 --pos 1 6 11
    python tools\\t1781_kv_cache_grade.py --dump-dir out\\t1781 --label arith_12plus15 --pos 1 \\
        --repeat-suffix repeat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_bisection_report as lbr  # noqa: E402 -- reuse, not reimplement


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--pos", type=int, nargs="+", required=True)
    parser.add_argument("--repeat-suffix", default=None,
                         help="if set, also compute repeat-vs-repeat resolving power for the FIRST "
                              "position listed, using <label>.pos<N>.<suffix>.float.bin")
    args = parser.parse_args(argv)

    out_dir = Path(args.dump_dir)
    print("=" * 100)
    print(f"T-1781 C10 grading -- label={args.label}, positions={args.pos}, "
          f"29 layer boundaries (embedding + 28 layers) each, cross-engine (int8 Arm B "
          f"recompute vs. independent float reference, teacher-forced on the engine's own "
          f"self-generated continuation)")
    print("=" * 100)

    for pos in args.pos:
        int8_path = out_dir / f"{args.label}.pos{pos}.int8.bin"
        float_path = out_dir / f"{args.label}.pos{pos}.float.bin"
        if not int8_path.exists() or not float_path.exists():
            print(f"\n--- pos={pos}: SKIPPED (missing {int8_path if not int8_path.exists() else float_path}) ---")
            continue
        int8_dump = lbr.load_int8_layer_dump(int8_path)
        float_dump = lbr.load_float_layer_dump(float_path)
        lbr.check_provenance(int8_dump, float_dump)

        row_stats = [
            lbr.compare_layer_row(int8_dump.codes[i], float_dump.values[i]) for i in range(int8_dump.rows)
        ]
        print(f"\n--- pos={pos} (fingerprint=0x{int8_dump.prompt_fingerprint:016X}, provenance OK) ---")
        for i, s in enumerate(row_stats):
            layer_name = "embed" if i == 0 else f"layer{i}"
            print(f"    {layer_name:<8} spearman={s.spearman:7.4f} pearson={s.pearson:7.4f} "
                  f"max|z-diff|={s.max_abs_z_diff:8.4f}")

    if args.repeat_suffix:
        pos = args.pos[0]
        float_path = out_dir / f"{args.label}.pos{pos}.float.bin"
        float_repeat_path = out_dir / f"{args.label}.pos{pos}.{args.repeat_suffix}.float.bin"
        if float_path.exists() and float_repeat_path.exists():
            float_dump = lbr.load_float_layer_dump(float_path)
            float_repeat_dump = lbr.load_float_layer_dump(float_repeat_path)
            lbr.check_provenance(float_dump, float_repeat_dump)
            resolving = lbr.repeat_vs_repeat_dispersion(float_dump.values, float_repeat_dump.values)
            print(f"\n--- float-side repeat-vs-repeat resolving power, pos={pos} "
                  f"({args.label}) ---")
            for i, v in enumerate(resolving):
                layer_name = "embed" if i == 0 else f"layer{i}"
                print(f"    {layer_name:<8} max|z-diff| repeat-vs-repeat = {v:8.4f}")
            print(f"    max over all {len(resolving)} boundaries = {max(resolving):.4f}")
        else:
            print(f"\n--- repeat resolving power SKIPPED: missing {float_path} or {float_repeat_path} ---")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
