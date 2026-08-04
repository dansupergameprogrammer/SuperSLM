#!/usr/bin/env python3
"""T-1698 driver: builds one migrated artifact (channel, alpha), verifies the
alpha=0 no-op / live-effect guard, decodes the real engine against it and the
float reference (reusing `t1697_decode_compare.run_against_artifact`/
`summarize` unmodified), then deletes the ~1.5GB artifact file (the decode
dumps under `out/t1697_decode/<run_label>/` are kept -- small, and the only
artifact this task needs to reproduce a number from).

Usage: python tools\\t1698_run_arm.py <channel> <alpha> <run_label>
"""
from __future__ import annotations

import filecmp
import hashlib
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


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main() -> int:
    if len(sys.argv) < 4:
        print("usage: t1698_run_arm.py <channel> <alpha> <run_label>", file=sys.stderr)
        return 1
    channel = int(sys.argv[1])
    alpha = float(sys.argv[2])
    run_label = sys.argv[3]

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    artifact = read_artifact(BASELINE_PATH)
    records_by_label = M.load_population_sitedumps()

    max_x = M.max_abs_activation_per_layer(channel, records_by_label)
    max_w = M.max_abs_weight_column_per_layer(artifact, channel)
    layer_scales = M.derive_layer_scales(max_x, max_w, alpha)
    n_migrated = sum(1 for s in layer_scales if s != 1.0)

    out_path = os.path.join(ARTIFACT_DIR, f"ch{channel}_alpha{alpha:.4f}.sslm")
    r = M.build_migrated_artifact(BASELINE_PATH, out_path, channel, layer_scales)
    r.alpha = alpha

    total_sat = sum(ls.n_saturated for ls in r.layer_stats)
    total_elem = sum(ls.n_elements for ls in r.layer_stats)
    max_err = max((ls.max_abs_weight_quant_error for ls in r.layer_stats), default=0.0)
    rmse_errs = [ls.rmse_weight_quant_error for ls in r.layer_stats if ls.s_c != 1.0]
    mean_rmse = sum(rmse_errs) / len(rmse_errs) if rmse_errs else 0.0

    print(f"=== channel {channel} alpha={alpha} run_label={run_label} ===")
    print(f"n_migrated_layers={n_migrated}/28  bytes_changed={r.bytes_changed}  "
          f"identical_to_baseline={r.identical_to_baseline}")
    print(f"saturation: {total_sat}/{total_elem} ({100.0*total_sat/total_elem:.4f}%)  "
          f"max_weight_quant_error={max_err:.6g}  mean_rmse_over_migrated={mean_rmse:.6g}")
    print(f"migrated layer indices: {[ls.layer_idx for ls in r.layer_stats if ls.s_c != 1.0]}")

    if alpha == 0.0:
        same_artifact = filecmp.cmp(BASELINE_PATH, out_path, shallow=False)
        assert same_artifact, "GUARD FAILED: alpha=0 artifact is not bit-identical to baseline"
        print("GUARD: alpha=0 artifact bit-identical to baseline -- PASS")
        # Decode against the freshly-built (bit-identical) copy itself, not
        # the original baseline path -- this is the actual re-verification
        # of THIS pipeline (a fresh build through t1698's own driver), not a
        # re-assertion of T-1697's own already-established result.
        decode_path = out_path
    else:
        same_artifact = filecmp.cmp(BASELINE_PATH, out_path, shallow=False)
        assert not same_artifact, (
            f"LIVE-EFFECT GUARD FAILED: alpha={alpha} artifact is bit-identical to baseline "
            f"(migration had no effect -- would silently look like a null result)")
        print("GUARD: live-effect confirmed -- artifact bytes differ from baseline -- PASS")
        decode_path = out_path

    results = DC.run_against_artifact(decode_path, run_label)
    DC.summarize(run_label, results)

    if alpha == 0.0:
        # Decode-level no-op re-verification: byte-identical to the existing
        # T-1697 baseline decode dumps, confirmed by direct diff, not assumed.
        baseline_decode_dir = os.path.join(REPO_ROOT, "out", "t1697_decode", "baseline")
        run_decode_dir = os.path.join(REPO_ROOT, "out", "t1697_decode", run_label)
        mismatches = []
        for case in DC.ALL_CASES:
            fname = f"{case.label}.int8.bin"
            a = os.path.join(baseline_decode_dir, fname)
            b = os.path.join(run_decode_dir, fname)
            if not filecmp.cmp(a, b, shallow=False):
                mismatches.append(fname)
        if mismatches:
            print(f"GUARD FAILED: alpha=0 decode differs from baseline decode at: {mismatches}")
            return 1
        print(f"GUARD: alpha=0 decode byte-identical to baseline decode at all "
              f"{len(DC.ALL_CASES)} prompts -- PASS")

    if os.path.exists(out_path):
        os.remove(out_path)
        print(f"artifact deleted: {out_path}")

    print(f"\nRESULT: {run_label} complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
