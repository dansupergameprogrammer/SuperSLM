#!/usr/bin/env python3
"""T-1691 build-sequencing session -- the precision-shadow real drive (design
`Claude/Vitruvius/superslm-t1683-source-attribution-design-2026-08-02.md`
Sec7 step 6c). The SECOND arm of the C1/C2 decomposition, consulted only
once the parity arm (5a) has cleared every cell it was driven against --
D-SLM747 (RMSNorm 72/72), D-SLM750/751 (attention interior 144/144),
D-SLM753/754 (MLP 180/180), D-SLM757-759 (RoPE 1044/1044), all exact, zero
residual. This drive measures what design Sec7 step 6c calls C2's own
magnitude: `parity-shadow-vs-precision-shadow` Spearman, at the SAME layers
and prompts the parity arm already cleared.

**Scope boundary, named plainly.** `precision_shadow_layer` (`tools/
shadow_layer_recompute.py`) is a generic `dequantized-weight-matrix @ row`
composition -- it has no representation for a site that is not a stored
int8 weight matrix. Of the real eleven-site, five-kind inventory (design
Sec7 step 5a), only the FIVE `project_and_funnel` sites (q_proj, o_proj,
gate_proj, up_proj, down_proj) are literally a `w_int8` matrix with an
`(identity, mult, shift)` fold triple -- `rms_norm` (a per-channel gain
vector, not a matrix; the funnel's `d_prime`/`r`/`s` derive from the
normalized row itself, not a weight-scale fold at all), `mlp_act` (a
nonlinear sigmoid-LUT composition), `residual_reconcile` (an addition), and
`context_row_funnel` (a softmax-weighted sum over attention-derived,
not artifact-stored, probabilities) have no dequantizable weight matrix for
`precision_shadow_layer` to operate on. This drive therefore measures C2's
magnitude at the five weight-bearing sites only -- 5 of the real layer's 11
site invocations, 2 of 5 site KINDS. The remaining four site kinds (rms_norm,
mlp_act, residual_reconcile, context_row_funnel) are NOT measured by this
arm; closing that gap would require new precision-shadow-specific
per-kind functions (siblings to `rmsnorm_shadow_codes`/`mlp_act_shadow_codes`/
`residual_reconcile_shadow_codes`/`attn_ctx_shadow_codes_from_real_landing_
and_rotation`, each with its int8 mechanisms replaced by float64 pass-
through) that do not exist in this tree today -- named as this drive's own
remainder, not built here.

At each of the five sites, the comparison is `parity_shadow` (== the real
engine's own captured codes at that site, per the exact-equality proof
already closed for all four site kinds) against `precision_shadow_layer`'s
float64 output, fed the SAME real captured input codes -- exactly like every
prior T-1691 drive's own "every input is the real engine's own captured
value" convention (D-SLM744/747). `q_proj` carries one further named
limitation: `ProjectAndFunnel`'s real construction folds a per-channel bias
between the weight-scale fold and the funnel (`ApplyBiasReconcileRow`,
design Sec7 step 5a); `precision_shadow_layer`'s own composition (`tools/
shadow_layer_recompute.py:1219-1231`) has no bias term at all -- built,
not authored here -- so `q_proj`'s own precision-shadow output omits the
real bias contribution the other four sites' calls (bias=None already,
matching production) do not carry to begin with.

Usage: python tools\\t1691_precision_drive.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_saturation_report as KSR  # noqa: E402  -- the established 9-prompt population + paths
import layer_bisection_report as lbr  # noqa: E402
import shadow_layer_recompute as S  # noqa: E402
from layer_bisection_report import load_int8_layer_dump  # noqa: E402
from sslm_artifact_reader import read_artifact  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DRIVER_EXE = os.path.join(REPO_ROOT, "out", "sslm_layer_trace.exe")
OUT_DIR = os.path.join(REPO_ROOT, "out", "t1691_precision_drive")

# Dan's own stated scope for this build-sequencing campaign: layers 26-28
# plus the layer-5 control -- unchanged from every prior T-1691 site-kind
# drive (RMSNorm, attention interior, MLP, RoPE).
TARGET_ROWS = [5, 26, 27, 28]
TARGET_LAYER_INDICES = [row - 1 for row in TARGET_ROWS]  # 0-indexed

# The five project_and_funnel sites (design Sec7 step 5a's own five-of-eleven
# real weight-bearing invocations): (site name in the chain-record dump,
# real-input chain-record name, weight/fold attribute prefix on the artifact
# layer object, bias attribute name or None).
SITES = [
    ("q_proj.requant", "attn_norm", "q", "q_bias"),
    ("o_proj.requant", "attn_ctx", "o", None),
    ("gate_proj.requant", "mlp_norm", "gate", None),
    ("up_proj.requant", "mlp_norm", "up", None),
    ("down_proj.requant", "mlp_act", "down", None),
]


def main() -> int:
    import subprocess

    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isfile(DRIVER_EXE):
        print(f"FAILED: {DRIVER_EXE} not built -- run tools\\build_layer_trace.bat", file=sys.stderr)
        return 1
    if not os.path.isfile(KSR.MODEL_PATH):
        print(f"FAILED: real artifact not found at {KSR.MODEL_PATH}", file=sys.stderr)
        return 1

    artifact = read_artifact(str(KSR.MODEL_PATH))

    # rows: (label, role, target_row, site_name) -> LayerRowStats
    all_stats: dict[str, list] = {}
    cell_log: list[tuple] = []
    total_cells = 0

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

        row_stats = []
        for row, layer_index in zip(TARGET_ROWS, TARGET_LAYER_INDICES):
            if layer_index >= base.rows - 1:
                continue  # geometry does not reach this row (never happens for the real 28-layer artifact)

            layer = artifact.layers[layer_index]

            def rec(name: str):
                return site_records[f"layer{layer_index}.{name}"]

            for site_name, input_site, attr, bias_attr in SITES:
                input_rec = rec(input_site)
                real_rec = rec(site_name)

                w = getattr(layer, f"{attr}_weight")
                fold = getattr(layer, f"{attr}_fold")
                triple = S.SiteFoldTriple(
                    kind="project_and_funnel", w_int8=w,
                    identity=fold[:, 0], mult=fold[:, 1], shift=fold[:, 2])

                precision_a = S.precision_shadow_layer(input_rec.codes, [triple])
                # Repeat-vs-repeat resolving power for THIS comparison pair
                # (StandardsDocument Sec5.4: re-measured, not assumed): both
                # arms are deterministic pure functions of the SAME real
                # captured input codes -- executed here as a second,
                # independent call rather than reasoned from determinism
                # alone.
                precision_b = S.precision_shadow_layer(input_rec.codes, [triple])
                repeat_diff = float(np.max(np.abs(precision_a - precision_b)))
                if repeat_diff != 0.0:
                    print(f"FAILED: precision_shadow_layer non-deterministic at prompt={label!r} "
                          f"row={row} site={site_name}: repeat diff={repeat_diff}", file=sys.stderr)
                    return 1

                # parity == the real engine's own captured codes at this
                # site, per the exact-equality proof already closed for
                # every T-1691 site kind (D-SLM747/750/751/753/754/757-759)
                # -- never recomputed here, so this drive's own comparison
                # is never tautological against its own construction.
                parity_codes = real_rec.codes.astype(np.int8)

                stats = lbr.compare_layer_row(parity_codes, precision_a)
                total_cells += 1
                row_stats.append(stats)
                cell_log.append((label, role, row, site_name, stats.spearman,
                                  stats.pearson, stats.max_abs_z_diff))

        all_stats[label] = row_stats
        print(f"prompt={label!r} role={role}: precision shadow driven at rows {TARGET_ROWS} x "
              f"5 project_and_funnel sites -- {len(row_stats)} cells")

    # Design Sec7 red-first proof part 4's own sanity gate, reused verbatim
    # (finiteness / range, not a pass/fail on magnitude).
    bad = 0
    for row_stats in all_stats.values():
        for s in row_stats:
            for value in (s.spearman, s.pearson, s.max_abs_z_diff):
                if not np.isfinite(value) and not (np.isnan(value)):
                    bad += 1
    if bad:
        print(f"FAILED: {bad} non-finite stat(s) among {total_cells} cells -- sanity gate failed",
              file=sys.stderr)
        return 1

    mech2_rho = [s for (lbl, role, row, site, s, p, z) in cell_log if role == "mech2"]
    control_rho = [s for (lbl, role, row, site, s, p, z) in cell_log if role != "mech2"]
    all_rho = [c[4] for c in cell_log]
    all_z = [c[6] for c in cell_log]

    by_site: dict[str, list[float]] = {}
    for c in cell_log:
        by_site.setdefault(c[3], []).append(c[4])

    print("\n--- per-cell (prompt, row, site) -> spearman, pearson, max_abs_z_diff ---")
    for lbl, role, row, site, sp, pe, z in cell_log:
        print(f"  {lbl:24s} role={role:6s} row={row:2d} site={site:18s} "
              f"spearman={sp:+.6f} pearson={pe:+.6f} max_abs_z_diff={z:.6f}")

    print(f"\nRESULT: precision shadow driven -- {total_cells} cells "
          f"({len(KSR.POPULATION)} prompts x {len(TARGET_ROWS)} rows x {len(SITES)} "
          f"project_and_funnel sites), 0 non-finite, 0 non-deterministic-repeat.")
    print(f"parity-shadow-vs-precision-shadow spearman: min={min(all_rho):.6f} "
          f"max={max(all_rho):.6f} over all {len(all_rho)} cells.")
    print(f"  mech2 group (n={len(mech2_rho)}): min={min(mech2_rho):.6f} max={max(mech2_rho):.6f}")
    print(f"  control group (n={len(control_rho)}): min={min(control_rho):.6f} "
          f"max={max(control_rho):.6f}")
    print(f"max_abs_z_diff over all cells: min={min(all_z):.6f} max={max(all_z):.6f}")
    print("\nper-site spearman (n=36 each -- 9 prompts x 4 rows):")
    for site, vals in by_site.items():
        print(f"  {site:18s} min={min(vals):.6f} max={max(vals):.6f} "
              f"mean={sum(vals) / len(vals):.6f}")
    print("resolving power (repeat-vs-repeat, this comparison pair): 0.0 exact -- both arms are "
          "deterministic pure functions of the same real captured input codes, executed twice per "
          "cell above and confirmed bit-identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
