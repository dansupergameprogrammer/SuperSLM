#!/usr/bin/env python3
"""T-1691 build-sequencing session (site kind 2/4, the attention interior,
build log Claude/Brunel/superslm-t1691-site-comparison-build2-2026-08-03.md)
-- the real drive: drives `out/sslm_layer_trace.exe` with `--site-dump` and
`--dump` against the REAL Qwen2.5-1.5B-Instruct artifact, over the same
nine-prompt population `t1691_rmsnorm_drive.py`/`kv_saturation_report.py`/
`layer_bisection_report.py` already established, at the divergence band
(layers 26-28) plus the layer-5 control (Dan's own stated scope, unchanged
from the RMSNorm drive). For each prompt and each of the four target rows,
computes q_proj.requant/attn_ctx/o_proj.requant/attn_residual via
`shadow_layer_recompute.py`'s own attention-interior functions and compares
against the real engine's own captured codes by EXACT integer-code equality
(design Sec9/Sec10's own comparison shape for this class of check, the same
shape the RMSNorm drive already used).

Every input this drive feeds a shadow function is the REAL captured value at
that site (q_proj's shadow reads the real attn_norm codes; attn_ctx's shadow
reads the real q_proj.requant codes AND the real prior-position K/V codes
from the K/V-context trailer; o_proj's shadow reads the real attn_ctx codes;
attn_residual's shadow reads the real o_proj.requant codes and the real
pre-attention hidden-state stream) -- never a value this drive's own prior
comparison produced, so a mismatch at one site cannot itself cause a
mismatch at the next (this campaign's own established per-site-independent
methodology, D-SLM744/747).

**Scope boundary named plainly (per this pass's own commission): the CURRENT
position's own K/V landing and BOTH RoPE rotations (Q's and the current
K's) are computed by this drive's own independently-coded primitives inside
`attn_ctx_shadow_codes` -- neither is traced from the real engine (D-SLM745:
`RopeApplySite` carries no trace-hook parameter, and the pre-rotation K/V
landing write is overwritten in place before any tool regains control). A
mismatch this drive reports at `attn_ctx` could therefore originate in this
drive's own RoPE/K-V-landing reproduction as well as in the real engine's
arithmetic -- it is not, by itself, evidence pointing at one side over the
other. q_proj, o_proj, and attn_residual carry no such ambiguity: every
input and the comparison codes are the real engine's own captured values.**

Per this session's own commission: STOPS at the first disagreement and
reports it (layer, site, prompt, the two code arrays) -- it does not continue
comparing further cells once one disagrees, because a single disagreement is
already the campaign's answer for this site kind.

Usage: python tools\\t1691_attention_drive.py
"""

from __future__ import annotations

import os
import struct
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
OUT_DIR = os.path.join(REPO_ROOT, "out", "t1691_attention_drive")

# Dan's own stated scope for this build-sequencing pass: layers 26-28 plus
# the layer-5 control -- matches the RMSNorm drive and the site-dump
# extension's own default target-row band.
TARGET_ROWS = [5, 26, 27, 28]
TARGET_LAYER_INDICES = [row - 1 for row in TARGET_ROWS]  # 0-indexed


def _saturation_trailer_end(dump_path: str, base) -> int:
    """The K/V-context trailer is appended after the 29-row body AND T-1689's
    own 28-uint64 saturation-delta trailer (design Sec7 step 4). Computes
    that byte offset directly from the dump's own documented layout --
    transcribed here rather than imported, so a defect in one reader cannot
    silently hide a defect in the other (`tools/test_t1691_attention_shadow.
    py`'s own identically-named, independently-transcribed helper; this
    module's own "no shared apparatus" construction, applied to
    trailer-offset arithmetic)."""
    with open(dump_path, "rb") as f:
        data = f.read()
    rows, hidden_size, _fp, _cm = struct.unpack_from("<QQQQ", data, 0)
    body_end = 32 + rows * (16 + hidden_size)
    (num_deltas,) = struct.unpack_from("<Q", data, body_end)
    return body_end + 8 + num_deltas * 8


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isfile(DRIVER_EXE):
        print(f"FAILED: {DRIVER_EXE} not built -- run tools\\build_layer_trace.bat", file=sys.stderr)
        return 1
    if not os.path.isfile(KSR.MODEL_PATH):
        print(f"FAILED: real artifact not found at {KSR.MODEL_PATH}", file=sys.stderr)
        return 1

    artifact = read_artifact(str(KSR.MODEL_PATH))
    num_heads = artifact.config["num_attention_heads"]
    num_kv_heads = artifact.config["num_key_value_heads"]
    head_dim = artifact.config["head_dim"]
    hidden_size = artifact.config["hidden_size"]

    total_cells = 0
    total_mismatches = 0

    def stop(label, role, row, layer_index, site, extra: str) -> int:
        nonlocal total_mismatches
        total_mismatches += 1
        print(f"MISMATCH: prompt={label!r} role={role} row={row} layer_index={layer_index} "
              f"site={site}\n{extra}")
        print(f"RESULT: STOP -- the attention interior disagrees with the real engine at the "
              f"first cell checked ({total_cells} cells checked, {total_mismatches} mismatch(es))")
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
        trailer = S.load_kv_context_trailer(
            dump_path, after_offset=_saturation_trailer_end(dump_path, base))
        position = trailer.context_length
        width = position + 1

        for row, layer_index in zip(TARGET_ROWS, TARGET_LAYER_INDICES):
            if layer_index >= base.rows - 1:
                continue  # geometry does not reach this row (never happens for the real 28-layer artifact)
            if row not in trailer.target_rows:
                print(f"FAILED: row {row} not among the K/V-context trailer's own captured rows "
                      f"{trailer.target_rows} -- the site-dump/trailer target-row bands have "
                      f"diverged", file=sys.stderr)
                return 1
            trailer_idx = trailer.target_rows.index(row)
            real_k_prior = trailer.k_codes[trailer_idx]
            real_v_prior = trailer.v_codes[trailer_idx]

            layer = artifact.layers[layer_index]

            def rec(name: str):
                return site_records[f"layer{layer_index}.{name}"]

            # --- q_proj.requant: real attn_norm codes/scale -> shadow ------
            attn_norm_rec = rec("attn_norm")
            q_status, q_shadow = S.project_and_funnel_shadow_codes(
                attn_norm_rec.codes, attn_norm_rec.m_out, attn_norm_rec.e_out, layer.q_weight,
                layer.q_fold[:, 0], layer.q_fold[:, 1], layer.q_fold[:, 2], layer.q_bias)
            q_real = rec("q_proj.requant")
            total_cells += 1
            if q_status != "Ok" or not np.array_equal(q_shadow, q_real.codes):
                return stop(label, role, row, layer_index, "q_proj.requant",
                            f"  status={q_status}\n"
                            f"  shadow={q_shadow.tolist() if q_shadow is not None else None}\n"
                            f"  real  ={q_real.codes.tolist()}")

            # --- attn_ctx: real q_proj.requant + real attn_norm + real ----
            # prior-position K/V (K/V-context trailer) -> shadow. See this
            # module's own docstring: the CURRENT position's own K/V landing
            # and BOTH RoPE rotations are this drive's own reproduction, not
            # traced from the real engine (D-SLM745) -- named here again at
            # the point of use.
            ctx_status, ctx_shadow = S.attn_ctx_shadow_codes(
                q_real.codes, q_real.m_out, q_real.e_out, attn_norm_rec.codes, attn_norm_rec.m_out,
                attn_norm_rec.e_out, position, width, layer, artifact.rope_cos, artifact.rope_sin,
                real_k_prior, real_v_prior, num_heads, num_kv_heads, head_dim)
            ctx_real = rec("attn_ctx")
            total_cells += 1
            if ctx_status != "Ok" or not np.array_equal(ctx_shadow, ctx_real.codes):
                return stop(label, role, row, layer_index, "attn_ctx",
                            f"  status={ctx_status}\n"
                            f"  shadow={ctx_shadow.tolist() if ctx_shadow is not None else None}\n"
                            f"  real  ={ctx_real.codes.tolist()}\n"
                            f"  NOTE: attn_ctx's own current-position K/V-landing and RoPE are this "
                            f"drive's own reproduction, not real-engine-traced (D-SLM745) -- a "
                            f"mismatch here does not by itself localize to the real engine.")

            # --- o_proj.requant: real attn_ctx codes/scale -> shadow ------
            o_status, o_shadow = S.project_and_funnel_shadow_codes(
                ctx_real.codes, ctx_real.m_out, ctx_real.e_out, layer.o_weight,
                layer.o_fold[:, 0], layer.o_fold[:, 1], layer.o_fold[:, 2], None)
            o_real = rec("o_proj.requant")
            total_cells += 1
            if o_status != "Ok" or not np.array_equal(o_shadow, o_real.codes):
                return stop(label, role, row, layer_index, "o_proj.requant",
                            f"  status={o_status}\n"
                            f"  shadow={o_shadow.tolist() if o_shadow is not None else None}\n"
                            f"  real  ={o_real.codes.tolist()}")

            # --- attn_residual: real o_proj.requant + real pre-attention --
            # hidden-state stream (base dump's own row, BEFORE this layer's
            # attention branch) -> shadow.
            stream_m, stream_e = base.scales[layer_index]
            res_status, res_shadow = S.residual_reconcile_shadow_codes(
                o_real.codes, o_real.m_out, o_real.e_out, base.codes[layer_index], int(stream_m),
                int(stream_e), hidden_size)
            res_real = rec("attn_residual")
            total_cells += 1
            if res_status != "Ok" or not np.array_equal(res_shadow, res_real.codes):
                return stop(label, role, row, layer_index, "attn_residual",
                            f"  status={res_status}\n"
                            f"  shadow={res_shadow.tolist() if res_shadow is not None else None}\n"
                            f"  real  ={res_real.codes.tolist()}")

        print(f"prompt={label!r} role={role}: q_proj/attn_ctx/o_proj/attn_residual exact-match at "
              f"rows {TARGET_ROWS} -- OK")

    print(f"RESULT: the attention interior agrees with the real engine EXACTLY at every checked "
          f"cell ({total_cells} cells: {len(KSR.POPULATION)} prompts x {len(TARGET_ROWS)} rows x 4 "
          f"sites [q_proj.requant, attn_ctx, o_proj.requant, attn_residual], "
          f"{total_mismatches} mismatch(es))")
    return 0 if total_mismatches == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
