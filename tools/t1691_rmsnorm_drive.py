#!/usr/bin/env python3
"""T-1691 parity-extension pass (2026-08-04, Brunel) -- the real drive for
site kind 1/4 (RMSNorm): drives `out/sslm_layer_trace.exe` with `--site-dump`
against the REAL Qwen2.5-1.5B-Instruct artifact, over the same nine-prompt
population `kv_saturation_report.py`/`layer_bisection_report.py` already
established, at ALL 28 layer rows (widened from the original {5, 26, 27, 28}
build-sequencing-pass band -- the coverage gap this pass closes: rows 12-20
are a reversal band, unexplained and immediately upstream of the divergence
onset at row 21-22, and were never parity-checked). For each prompt and each
target row, computes `rmsnorm_shadow_codes` for both attn_norm and mlp_norm
and compares against the real engine's own captured codes by EXACT
integer-code equality (design Sec9/Sec10's own comparison shape for this
class of check).

Per this campaign's own standing commission: STOPS at the first disagreement
and reports it (layer, site, prompt, the two code arrays) -- it does not
continue comparing further cells once one disagrees, because a single
disagreement is already the campaign's answer for this site kind.

Usage: python tools\\t1691_rmsnorm_drive.py
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_saturation_report as KSR  # noqa: E402  -- the established 9-prompt population + paths
import shadow_layer_recompute as S  # noqa: E402
from layer_bisection_report import load_int8_layer_dump  # noqa: E402
from sslm_artifact_reader import read_artifact  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER_EXE = os.path.join(REPO_ROOT, "out", "sslm_layer_trace.exe")
OUT_DIR = os.path.join(REPO_ROOT, "out", "t1691_rmsnorm_drive")

# All 28 rows (parity-extension pass, 2026-08-04): the reversal band (12-22,
# the reversal itself plus the divergence onset) is checked FIRST -- Dan's
# own stated priority for this pass -- so a run that has to stop partway
# still covers the least-inspected, most-interesting region before anything
# else. The remaining rows follow in ascending order.
_PRIORITY_ROWS = list(range(12, 23))  # 12..22
TARGET_ROWS = _PRIORITY_ROWS + [r for r in range(1, 29) if r not in _PRIORITY_ROWS]
TARGET_LAYER_INDICES = [row - 1 for row in TARGET_ROWS]  # 0-indexed


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isfile(DRIVER_EXE):
        print(f"FAILED: {DRIVER_EXE} not built -- run tools\\build_layer_trace.bat", file=sys.stderr)
        return 1
    if not os.path.isfile(KSR.MODEL_PATH):
        print(f"FAILED: real artifact not found at {KSR.MODEL_PATH}", file=sys.stderr)
        return 1

    artifact = read_artifact(str(KSR.MODEL_PATH))
    total_cells = 0
    total_mismatches = 0

    for label, question, role in KSR.POPULATION:
        prompt = KSR.build_prompt(question)
        dump_path = os.path.join(OUT_DIR, f"{label}.dump")
        site_dump_path = os.path.join(OUT_DIR, f"{label}.sitedump")
        proc = subprocess.run(
            [str(DRIVER_EXE), str(KSR.MODEL_PATH), str(KSR.TOKENIZER_PATH), prompt,
             "--dump", dump_path, "--site-dump", site_dump_path],
            capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            print(f"FAILED: driver failed for prompt {label!r} (exit {proc.returncode})\n"
                  f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}", file=sys.stderr)
            return 1
        assert "self_check: production and manual-replay paths agree" in proc.stdout, (
            f"prompt {label!r}: self-check line missing from stdout -- driver may have changed "
            f"its own success-reporting text")

        base = load_int8_layer_dump(dump_path)
        site_records = {r.site: r for r in S.load_site_dump(site_dump_path)}

        for row, layer_index in zip(TARGET_ROWS, TARGET_LAYER_INDICES):
            if layer_index >= base.rows - 1:  # rows includes the embedding row (row 0)
                continue  # geometry does not reach this row (never happens for the real 28-layer artifact)

            attn_norm_gain = artifact.layers[layer_index].attn_norm_gain.astype(np.int32)
            mlp_norm_gain = artifact.layers[layer_index].mlp_norm_gain.astype(np.int32)

            # attn_norm
            h_attn = base.codes[layer_index]
            status, shadow_codes = S.rmsnorm_shadow_codes(h_attn, attn_norm_gain)
            real_codes = site_records[f"layer{layer_index}.attn_norm"].codes
            total_cells += 1
            if status != "Ok" or not np.array_equal(shadow_codes, real_codes):
                total_mismatches += 1
                print(f"MISMATCH: prompt={label!r} role={role} row={row} layer_index={layer_index} "
                      f"site=attn_norm status={status}\n"
                      f"  shadow={shadow_codes.tolist() if shadow_codes is not None else None}\n"
                      f"  real  ={real_codes.tolist()}\n"
                      f"  h_attn={h_attn.tolist()}\n  gain  ={attn_norm_gain.tolist()}")
                print(f"RESULT: STOP -- RMSNorm disagrees with the real engine at the first cell checked "
                      f"({total_cells} cells checked, {total_mismatches} mismatch(es))")
                return 2

            # mlp_norm -- real input is THIS layer's own captured attn_residual output
            h_mlp = site_records[f"layer{layer_index}.attn_residual"].codes
            status, shadow_codes = S.rmsnorm_shadow_codes(h_mlp, mlp_norm_gain)
            real_codes = site_records[f"layer{layer_index}.mlp_norm"].codes
            total_cells += 1
            if status != "Ok" or not np.array_equal(shadow_codes, real_codes):
                total_mismatches += 1
                print(f"MISMATCH: prompt={label!r} role={role} row={row} layer_index={layer_index} "
                      f"site=mlp_norm status={status}\n"
                      f"  shadow={shadow_codes.tolist() if shadow_codes is not None else None}\n"
                      f"  real  ={real_codes.tolist()}\n"
                      f"  h_mlp ={h_mlp.tolist()}\n  gain  ={mlp_norm_gain.tolist()}")
                print(f"RESULT: STOP -- RMSNorm disagrees with the real engine at the first cell checked "
                      f"({total_cells} cells checked, {total_mismatches} mismatch(es))")
                return 2

        print(f"prompt={label!r} role={role}: attn_norm/mlp_norm exact-match at rows {TARGET_ROWS} -- OK")

    print(f"RESULT: RMSNorm agrees with the real engine EXACTLY at every checked cell "
          f"({total_cells} cells: {len(KSR.POPULATION)} prompts x {len(TARGET_ROWS)} rows x 2 sites "
          f"[attn_norm, mlp_norm], 0 mismatches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
