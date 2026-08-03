#!/usr/bin/env python3
"""T-1691 build-sequencing session (site kind 3/4, the MLP, build log
Claude/Brunel/superslm-t1691-mlp-site-comparison-build-2026-08-03.md) -- the
real drive: drives `out/sslm_layer_trace.exe` with `--site-dump` and `--dump`
against the REAL Qwen2.5-1.5B-Instruct artifact, over the same nine-prompt
population `t1691_rmsnorm_drive.py`/`t1691_attention_drive.py` already
established, at the divergence band (layers 26-28) plus the layer-5 control
(unchanged scope). For each prompt and each of the four target rows, computes
gate_proj.requant/up_proj.requant/mlp_act/down_proj.requant/mlp_residual via
`shadow_layer_recompute.py`'s own MLP-site functions and compares against the
real engine's own captured codes by EXACT integer-code equality (the same
comparison shape the RMSNorm and attention-interior drives already used).

Every input this drive feeds a shadow function is the REAL captured value at
that site (gate_proj/up_proj's shadow reads the real mlp_norm codes; mlp_act's
shadow reads the real gate_proj.requant and up_proj.requant codes; down_proj's
shadow reads the real mlp_act codes; mlp_residual's shadow reads the real
down_proj.requant codes AND the real STAGED attention-residual stream, i.e.
the real attn_residual codes for this same layer, matching `RunLayerLoop`'s
own construction that the MLP half reconciles against the staged
attention-residual output, not the pre-attention stream -- forward_sites.cpp
"Reads the STAGED attention-residual output... reconciles against the STAGED
attention-residual output") -- never a value this drive's own prior
comparison produced, so a mismatch at one site cannot itself cause a mismatch
at the next (this campaign's own established per-site-independent
methodology, D-SLM744/747/751).

This site kind carries NO ambiguity of the kind D-SLM745 named for attn_ctx:
every primitive `mlp_act_shadow_codes` uses (`SiluSigmoidQ15`,
`CheckSiluCompositionScaleDomain`) was directly validated against the real
compiled function (this pass's own red-first proof,
`tools/test_t1691_mlp_shadow.py`), and every input to every MLP-site
comparison in this drive is the real engine's own captured value -- there is
no unobserved production-code gap analogous to RoPE/K-V-landing anywhere in
this site kind's own construction.

Per this session's own commission: STOPS at the first disagreement and
reports it (layer, site, prompt, the two code arrays) -- it does not continue
comparing further cells once one disagrees, because a single disagreement is
already the campaign's answer for this site kind.

Usage: python tools\\t1691_mlp_drive.py
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

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DRIVER_EXE = os.path.join(REPO_ROOT, "out", "sslm_layer_trace.exe")
OUT_DIR = os.path.join(REPO_ROOT, "out", "t1691_mlp_drive")

# Dan's own stated scope for this build-sequencing campaign: layers 26-28
# plus the layer-5 control -- unchanged from the RMSNorm and attention-
# interior drives and the site-dump extension's own default target-row band.
TARGET_ROWS = [5, 26, 27, 28]
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
    hidden_size = artifact.config["hidden_size"]

    total_cells = 0
    total_mismatches = 0

    def stop(label, role, row, layer_index, site, extra: str) -> int:
        nonlocal total_mismatches
        total_mismatches += 1
        print(f"MISMATCH: prompt={label!r} role={role} row={row} layer_index={layer_index} "
              f"site={site}\n{extra}")
        print(f"RESULT: STOP -- the MLP disagrees with the real engine at the first cell checked "
              f"({total_cells} cells checked, {total_mismatches} mismatch(es))")
        return 2

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
            if layer_index >= base.rows - 1:
                continue  # geometry does not reach this row (never happens for the real 28-layer artifact)

            layer = artifact.layers[layer_index]

            def rec(name: str):
                return site_records[f"layer{layer_index}.{name}"]

            # --- gate_proj.requant: real mlp_norm codes/scale -> shadow ---
            mlp_norm_rec = rec("mlp_norm")
            gate_status, gate_shadow = S.project_and_funnel_shadow_codes(
                mlp_norm_rec.codes, mlp_norm_rec.m_out, mlp_norm_rec.e_out, layer.gate_weight,
                layer.gate_fold[:, 0], layer.gate_fold[:, 1], layer.gate_fold[:, 2], None)
            gate_real = rec("gate_proj.requant")
            total_cells += 1
            if gate_status != "Ok" or not np.array_equal(gate_shadow, gate_real.codes):
                return stop(label, role, row, layer_index, "gate_proj.requant",
                            f"  status={gate_status}\n"
                            f"  shadow={gate_shadow.tolist() if gate_shadow is not None else None}\n"
                            f"  real  ={gate_real.codes.tolist()}")

            # --- up_proj.requant: real mlp_norm codes/scale -> shadow -----
            up_status, up_shadow = S.project_and_funnel_shadow_codes(
                mlp_norm_rec.codes, mlp_norm_rec.m_out, mlp_norm_rec.e_out, layer.up_weight,
                layer.up_fold[:, 0], layer.up_fold[:, 1], layer.up_fold[:, 2], None)
            up_real = rec("up_proj.requant")
            total_cells += 1
            if up_status != "Ok" or not np.array_equal(up_shadow, up_real.codes):
                return stop(label, role, row, layer_index, "up_proj.requant",
                            f"  status={up_status}\n"
                            f"  shadow={up_shadow.tolist() if up_shadow is not None else None}\n"
                            f"  real  ={up_real.codes.tolist()}")

            # --- mlp_act: real gate_proj.requant + real up_proj.requant ---
            # codes/scale -> shadow. The real SIL1 table (artifact bytes, a
            # legitimate shared INPUT) is the same table this artifact's own
            # runtime materializes.
            act_status, act_shadow = S.mlp_act_shadow_codes(
                gate_real.codes, gate_real.m_out, gate_real.e_out, up_real.codes, artifact.sigmoid_lut)
            act_real = rec("mlp_act")
            total_cells += 1
            if act_status != "Ok" or not np.array_equal(act_shadow, act_real.codes):
                return stop(label, role, row, layer_index, "mlp_act",
                            f"  status={act_status}\n"
                            f"  shadow={act_shadow.tolist() if act_shadow is not None else None}\n"
                            f"  real  ={act_real.codes.tolist()}")

            # --- down_proj.requant: real mlp_act codes/scale -> shadow ----
            down_status, down_shadow = S.project_and_funnel_shadow_codes(
                act_real.codes, act_real.m_out, act_real.e_out, layer.down_weight,
                layer.down_fold[:, 0], layer.down_fold[:, 1], layer.down_fold[:, 2], None)
            down_real = rec("down_proj.requant")
            total_cells += 1
            if down_status != "Ok" or not np.array_equal(down_shadow, down_real.codes):
                return stop(label, role, row, layer_index, "down_proj.requant",
                            f"  status={down_status}\n"
                            f"  shadow={down_shadow.tolist() if down_shadow is not None else None}\n"
                            f"  real  ={down_real.codes.tolist()}")

            # --- mlp_residual: real down_proj.requant + real STAGED -------
            # attention-residual stream (site-dump's own "attn_residual"
            # record for this SAME layer, not the base dump's pre-attention
            # row -- RunLayerLoop's own construction, forward_sites.cpp) ->
            # shadow.
            attn_residual_real = rec("attn_residual")
            res_status, res_shadow = S.residual_reconcile_shadow_codes(
                down_real.codes, down_real.m_out, down_real.e_out, attn_residual_real.codes,
                attn_residual_real.m_out, attn_residual_real.e_out, hidden_size)
            res_real = rec("mlp_residual")
            total_cells += 1
            if res_status != "Ok" or not np.array_equal(res_shadow, res_real.codes):
                return stop(label, role, row, layer_index, "mlp_residual",
                            f"  status={res_status}\n"
                            f"  shadow={res_shadow.tolist() if res_shadow is not None else None}\n"
                            f"  real  ={res_real.codes.tolist()}")

        print(f"prompt={label!r} role={role}: gate_proj/up_proj/mlp_act/down_proj/mlp_residual "
              f"exact-match at rows {TARGET_ROWS} -- OK")

    print(f"RESULT: the MLP agrees with the real engine EXACTLY at every checked cell "
          f"({total_cells} cells: {len(KSR.POPULATION)} prompts x {len(TARGET_ROWS)} rows x 5 sites "
          f"[gate_proj.requant, up_proj.requant, mlp_act, down_proj.requant, mlp_residual], "
          f"{total_mismatches} mismatch(es))")
    return 0 if total_mismatches == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
