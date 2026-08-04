#!/usr/bin/env python3
"""T-1704 Part 4, follow-up -- reproduces T-1703's OWN statistic (mech2-vs-
non-mech2 argmax-CHANNEL disjointness at mlp_act's damage locus, computed
from the single captured position -- the last prompt token, which is all
T-1703 ever had) directly from this task's own all-position captures
(restricted to the last captured position, the apples-to-apples comparison),
placed side by side with the TRUE GLOBAL argmax (over every captured
position) this task's own t1704_position_analysis.py already computed.

This directly answers whether T-1703's 100% (locus) vs 13.6% (non-locus)
disjointness pattern survives once the position axis is opened up, or
whether it was an artifact of the single position T-1703's own instrument
was restricted to.

Usage: python tools\\t1704_prize_question_lastpos_vs_global.py --dir out\\t1704_population_allpos
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_saturation_report as KSR  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1704_position_analysis import analyze_prompt  # noqa: E402

LOCUS_LAYERS = list(range(21, 27))
NON_LOCUS_LAYERS = [l for l in range(28) if l not in LOCUS_LAYERS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    results = {}
    for label, _q, role in KSR.POPULATION:
        path = os.path.join(args.dir, f"{label}.sitedump")
        print(f"analyzing {label} (role={role}) ...", flush=True)
        results[label] = analyze_prompt(path)

    mech2_labels = [l for l, _q, r in KSR.POPULATION if r == "mech2"]
    nonmech2_labels = [l for l, _q, r in KSR.POPULATION if r != "mech2"]

    for kind, title in (("last", "RESTRICTED TO LAST CAPTURED POSITION (apples-to-apples with T-1703)"),
                         ("global", "TRUE GLOBAL ARGMAX (over every captured position, this task's own extension)")):
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
        disjoint_locus = 0
        disjoint_nonlocus = 0
        for layer in range(28):
            mech2_channels = set()
            nonmech2_channels = set()
            for label in mech2_labels:
                r = results[label]
                if kind == "last":
                    ch = r["per_site_layer_last_channel"][("mlp_act", layer)]
                else:
                    _pos, ch, _val = r["per_site_layer_best"][("mlp_act", layer)]
                mech2_channels.add(ch)
            for label in nonmech2_labels:
                r = results[label]
                if kind == "last":
                    ch = r["per_site_layer_last_channel"][("mlp_act", layer)]
                else:
                    _pos, ch, _val = r["per_site_layer_best"][("mlp_act", layer)]
                nonmech2_channels.add(ch)
            disjoint = mech2_channels.isdisjoint(nonmech2_channels)
            in_locus = layer in LOCUS_LAYERS
            if disjoint:
                if in_locus:
                    disjoint_locus += 1
                else:
                    disjoint_nonlocus += 1
            print(f"  layer{layer:2d} {'LOCUS' if in_locus else '     '}: mech2_channels={sorted(mech2_channels)} "
                  f"nonmech2_channels={sorted(nonmech2_channels)} disjoint={disjoint}")
        print(f"\n  {kind}: disjoint at {disjoint_locus}/{len(LOCUS_LAYERS)} locus layers "
              f"({100*disjoint_locus/len(LOCUS_LAYERS):.1f}%), {disjoint_nonlocus}/{len(NON_LOCUS_LAYERS)} "
              f"non-locus layers ({100*disjoint_nonlocus/len(NON_LOCUS_LAYERS):.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
