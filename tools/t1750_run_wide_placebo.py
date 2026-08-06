#!/usr/bin/env python3
"""T-1750: batch driver running T-1698's own per-arm pipeline
(`t1698_run_arm`'s build+live-effect-guard+decode+summarize sequence,
unmodified) over the 24 NEW placebo channels/alphas found by
`t1750_match_search_wide.py`, extending T-1698's own 5-sample noise-floor
calibration to 29 samples total (5 original, reproduced exactly in this
task's build log, + 24 new). Sequential, one channel at a time (real-engine
decode is CPU/IO bound, not parallelized in the source campaign either).

Usage: python tools\\t1750_run_wide_placebo.py
"""
from __future__ import annotations

import filecmp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_saturation_report as KSR  # noqa: E402
import t1697_decode_compare as DC  # noqa: E402
import t1697_outlier_migration as M  # noqa: E402
from sslm_artifact_reader import read_artifact  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
ARTIFACT_DIR = os.path.join(REPO_ROOT, "out", "t1698_artifacts")
BASELINE_PATH = str(KSR.MODEL_PATH)

CHOSEN_ALPHAS = {5: 0.175, 575: 0.18, 581: 0.18, 856: 0.17, 2430: 0.18, 2674: 0.17,
                 3193: 0.175, 3885: 0.175, 4122: 0.17, 4460: 0.17, 4821: 0.17, 5030: 0.175,
                 5108: 0.17, 5198: 0.185, 6111: 0.17, 6491: 0.17, 7658: 0.17, 7796: 0.165,
                 7918: 0.175, 8159: 0.175, 8224: 0.175, 8234: 0.18, 8715: 0.17, 8804: 0.18}


def run_one(channel: int, alpha: float) -> dict:
    run_label = f"t1750_wide_ch{channel}"
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    artifact = read_artifact(BASELINE_PATH)
    records_by_label = M.load_population_sitedumps()

    max_x = M.max_abs_activation_per_layer(channel, records_by_label)
    max_w = M.max_abs_weight_column_per_layer(artifact, channel)
    layer_scales = M.derive_layer_scales(max_x, max_w, alpha)

    out_path = os.path.join(ARTIFACT_DIR, f"ch{channel}_alpha{alpha:.4f}.sslm")
    r = M.build_migrated_artifact(BASELINE_PATH, out_path, channel, layer_scales)

    same_artifact = filecmp.cmp(BASELINE_PATH, out_path, shallow=False)
    assert not same_artifact, f"LIVE-EFFECT GUARD FAILED for channel {channel} alpha={alpha}"
    print(f"[{run_label}] GUARD: live-effect confirmed -- bytes differ from baseline -- PASS")

    results = DC.run_against_artifact(out_path, run_label)
    DC.summarize(run_label, results)

    if os.path.exists(out_path):
        os.remove(out_path)

    mech2 = [x for x in results if x["group"] == "mech2"]
    control = [x for x in results if x["group"] == "control"]
    m_agree = sum(x["n_agree"] for x in mech2)
    m_pos = sum(x["n_positions"] for x in mech2)
    m_sp = [v for x in mech2 for v in x["spearman_full"]]
    c_agree = sum(x["n_agree"] for x in control)
    c_pos = sum(x["n_positions"] for x in control)
    c_sp = [v for x in control for v in x["spearman_full"]]
    import numpy as np
    return dict(channel=channel, alpha=alpha, run_label=run_label,
                mech2_agree=m_agree, mech2_pos=m_pos, mech2_pct=100.0 * m_agree / m_pos,
                mech2_spearman=float(np.mean(m_sp)),
                control_agree=c_agree, control_pos=c_pos, control_pct=100.0 * c_agree / c_pos,
                control_spearman=float(np.mean(c_sp)))


def main() -> int:
    summary_path = os.path.join(REPO_ROOT, "out", "t1750_wide_placebo_summary.csv")
    rows = []
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("channel,alpha,run_label,mech2_agree,mech2_pos,mech2_pct,mech2_spearman,"
                "control_agree,control_pos,control_pct,control_spearman\n")
        for channel, alpha in CHOSEN_ALPHAS.items():
            print(f"\n########## channel {channel} alpha={alpha} ##########", flush=True)
            row = run_one(channel, alpha)
            rows.append(row)
            f.write(f"{row['channel']},{row['alpha']},{row['run_label']},"
                    f"{row['mech2_agree']},{row['mech2_pos']},{row['mech2_pct']:.4f},"
                    f"{row['mech2_spearman']:.4f},{row['control_agree']},{row['control_pos']},"
                    f"{row['control_pct']:.4f},{row['control_spearman']:.4f}\n")
            f.flush()
    print(f"\nALL RUNS COMPLETE. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
