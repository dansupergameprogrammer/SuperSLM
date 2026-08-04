#!/usr/bin/env python3
"""T-1697 driver: runs the float64 identity check, the alpha=0 no-op check,
the live-effect check, and the alpha sweep's artifact build + saturation/
weight-quant-error report, for one migrated channel. Does NOT run the real
engine decode (see t1697_decode_compare.py) -- this script is the offline,
no-engine-run half of the pipeline.
"""
from __future__ import annotations

import filecmp
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_saturation_report as KSR  # noqa: E402
import t1697_outlier_migration as M  # noqa: E402
from sslm_artifact_reader import read_artifact  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "out", "t1697_artifacts")
BASELINE_PATH = str(KSR.MODEL_PATH)

CHANNEL = int(sys.argv[1]) if len(sys.argv) > 1 else 609
ALPHA_GRID = [0.0, 0.15, 0.18, 0.20, 0.25, 0.30]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    artifact = read_artifact(BASELINE_PATH)
    records_by_label = M.load_population_sitedumps()

    max_x = M.max_abs_activation_per_layer(CHANNEL, records_by_label)
    max_w = M.max_abs_weight_column_per_layer(artifact, CHANNEL)
    print(f"=== channel {CHANNEL} ===")
    for L in range(28):
        print(f"  layer{L}: max|X_c|={max_x[L]:.6g} max|W_c|={max_w[L]:.6g}")

    baseline_sha = sha256_file(BASELINE_PATH)
    baseline_size = os.path.getsize(BASELINE_PATH)
    print(f"\nbaseline sha256={baseline_sha} size={baseline_size}")

    # =========================================================================
    # PHASE 1 -- the float64 identity check, BEFORE any int8 is touched.
    # Run at a handful of (layer, prompt) cells spanning the locus band and
    # a control layer, at a representative nonzero alpha (0.25).
    # =========================================================================
    print("\n=== PHASE 1: float64 identity check (alpha=0.25, before int8) ===")
    s_c_025 = M.derive_layer_scales(max_x, max_w, 0.25)
    worst_rel = 0.0
    sample_cells = [(L, label) for L in [0, 12, 21, 23, 25, 26, 27]
                     for label, _q, _r in KSR.POPULATION[:3]]
    for L, label in sample_cells:
        r = M.float_identity_check(artifact, CHANNEL, L, s_c_025[L], label, records_by_label)
        worst_rel = max(worst_rel, r.max_rel_diff)
        print(f"  layer{L} prompt={label!r} s_c={r.s_c:.4f} baseline_norm={r.baseline_y_norm:.6g} "
              f"max_abs_diff={r.max_abs_diff:.3e} max_rel_diff={r.max_rel_diff:.3e}")
    print(f"PHASE 1 RESULT: worst max_rel_diff over all sampled cells = {worst_rel:.3e} "
          f"({'PASS -- float64-roundoff level' if worst_rel < 1e-8 else 'FAIL -- not an identity'})")

    # =========================================================================
    # PHASE 2/3 -- alpha sweep: build artifact, alpha=0 no-op, live-effect,
    # saturation count, weight-quant error.
    # =========================================================================
    print("\n=== PHASE 2/3: alpha sweep -- artifact build, no-op check, saturation, weight-quant error ===")
    results = {}
    for alpha in ALPHA_GRID:
        s_c = M.derive_layer_scales(max_x, max_w, alpha)
        out_path = os.path.join(OUT_DIR, f"ch{CHANNEL}_alpha{alpha:.3f}.sslm")
        r = M.build_migrated_artifact(BASELINE_PATH, out_path, CHANNEL, s_c)
        r.alpha = alpha
        results[alpha] = r

        total_sat = sum(ls.n_saturated for ls in r.layer_stats)
        total_elem = sum(ls.n_elements for ls in r.layer_stats)
        max_err = max((ls.max_abs_weight_quant_error for ls in r.layer_stats), default=0.0)
        rmse_errs = [ls.rmse_weight_quant_error for ls in r.layer_stats if ls.s_c != 1.0]
        mean_rmse = float(np.mean(rmse_errs)) if rmse_errs else 0.0
        n_migrated_layers = sum(1 for ls in r.layer_stats if ls.s_c != 1.0)

        print(f"\nalpha={alpha:.3f}: s_c range [{min(s_c):.4f}, {max(s_c):.4f}], "
              f"{n_migrated_layers}/28 layers migrated")
        print(f"  bytes_changed={r.bytes_changed}  identical_to_baseline={r.identical_to_baseline}")
        print(f"  down_proj weight saturation events: {total_sat} / {total_elem} elements "
              f"({100.0*total_sat/total_elem:.3f}%)")
        print(f"  down_proj weight-quant error: max_abs={max_err:.6g}  mean_rmse_over_migrated_layers={mean_rmse:.6g}")

        if alpha == 0.0:
            same = filecmp.cmp(BASELINE_PATH, out_path, shallow=False)
            out_sha = sha256_file(out_path)
            print(f"  ALPHA=0 NO-OP CHECK: bit-identical to baseline = {same} "
                  f"(baseline sha={baseline_sha[:16]}... vs alpha=0 sha={out_sha[:16]}...) "
                  f"{'PASS' if same else 'FAIL'}")
        else:
            same = filecmp.cmp(BASELINE_PATH, out_path, shallow=False)
            print(f"  LIVE-EFFECT CHECK: bit-identical to baseline = {same} "
                  f"{'FAIL -- should differ' if same else 'PASS -- artifact bytes changed'}")

    print("\nRESULT: pipeline phase 1-3 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
