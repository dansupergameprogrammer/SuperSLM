#!/usr/bin/env python3
"""T-1869 attack A -- independent reproduction of T-1868's 21-cell family table
from the pooled vectors, without reading T-1868's JSON.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from t1869_probe_common import CELLS, load_cell, metrics_for, paired, ref_top1_index

TEST_ARMS = {"stageB": ["ceiling", "middle"],
             "stageC": ["m1_g32", "m1_g128", "m2_residuals", "m2_site16", "m1m2_g128"]}


def main():
    out = {}
    for stage in ("stageB", "stageC"):
        labels, mats, ref_mat, dom = load_cell(stage)
        jstar = ref_top1_index(ref_mat)
        per = {name: metrics_for(m, jstar) for name, m in mats.items()}
        out[stage] = {"n": len(labels), "arms": sorted(mats), "cells": {}}
        b = per["base"]
        for arm in TEST_ARMS[stage] + [a for a in sorted(mats) if a.startswith("basedup")]:
            if arm not in per:
                continue
            c = per[arm]
            out[stage]["cells"][arm] = {
                m: paired(c[i], b[i]) for i, m in enumerate(("recall1", "rank_top1", "margin_top1"))
            }
        # raw per-document deltas for the primary arm, kept for later probes
        out[stage]["mean_margin_base"] = float(b[2].mean())
        out[stage]["mean_recall1_base"] = float(b[0].mean())
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
