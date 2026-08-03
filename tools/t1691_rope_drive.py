#!/usr/bin/env python3
"""T-1691 build-sequencing session (site kind 4/4, RoPE, D-SLM746/D-SLM749)
-- the real drive: drives `out/sslm_layer_trace.exe` with `--site-dump` and
`--dump` against the REAL Qwen2.5-1.5B-Instruct artifact, over the same
nine-prompt population `t1691_rmsnorm_drive.py`/`t1691_attention_drive.py`/
`t1691_mlp_drive.py` already established, at the divergence band (layers
26-28) plus the layer-5 control.

Per prompt and target row, checks four things by EXACT integer-code equality
-- stopping at the first disagreement, per this campaign's own standing rule:

  1. `kv_landing.k.hN` / `kv_landing.v.hN` (N in [0, num_key_value_heads)) --
     `kv_landing_shadow_codes` against the real K/V landing write's own
     pre-rotation captured codes, fed the real `attn_norm` codes/scale.
  2. `rope_apply.q.hN` (N in [0, num_attention_heads)) -- `rope_apply_site_
     reference` against the real per-query-head Q rotation, fed the real
     captured pre-rotation input (RopeApplySite's own x_int).
  3. `rope_apply.k.hN` (N in [0, num_attention_heads)) -- the same shadow
     function against the real per-query-head K rotation, fed the real
     `kv_landing.k.h{kv_head}` codes as input (kv_head = N // group) rather
     than RopeApplySite's own captured x_int, so this cell also exercises
     that the K/V-landing capture and RopeApplySite's own K-rotation input
     agree on what "the pre-rotation landed K value" is -- the SAME property
     `test_site_dump_kv_landing_captures_the_pre_rotation_value_not_the_
     post_rotation_one` proves at fixture scale, checked here on the real
     artifact's own real weights instead of a synthetic fixture.
  4. `attn_ctx` (one cell per row) -- D-SLM749's closure of the D-SLM745/751
     ambiguity: `attn_ctx_shadow_codes_from_real_landing_and_rotation`, fed
     the real captured rope_apply.q/k (post-rotation) and kv_landing.v
     (pre-rotation, V is never rotated) codes, never this module's own
     internal RoPE/landing reproduction. Agreement here is no longer
     consistent with "this port's own untraced reproduction happens to
     agree" (D-SLM751's own named caveat) -- every input is the real
     engine's own captured value, exactly like q_proj/o_proj/attn_residual
     already were.

Every input this drive feeds a shadow function is the REAL captured value at
that site -- never a value this drive's own prior comparison produced,
matching this campaign's own established per-site-independent methodology
(D-SLM744/747).

Usage: python tools\\t1691_rope_drive.py
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
OUT_DIR = os.path.join(REPO_ROOT, "out", "t1691_rope_drive")

# Dan's own stated scope for this build-sequencing pass: layers 26-28 plus
# the layer-5 control -- matches every prior site-kind drive.
TARGET_ROWS = [5, 26, 27, 28]
TARGET_LAYER_INDICES = [row - 1 for row in TARGET_ROWS]  # 0-indexed


def _saturation_trailer_end(dump_path: str, base) -> int:
    """Independently transcribed (this module's own "no shared apparatus"
    construction, matching every prior drive's own identically-named
    helper) rather than imported."""
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
    group = num_heads // num_kv_heads

    total_cells = 0
    total_mismatches = 0

    def stop(label, role, row, layer_index, site, extra: str) -> int:
        nonlocal total_mismatches
        total_mismatches += 1
        print(f"MISMATCH: prompt={label!r} role={role} row={row} layer_index={layer_index} "
              f"site={site}\n{extra}")
        print(f"RESULT: STOP -- RoPE (site kind 4/4) disagrees with the real engine at the first "
              f"cell checked ({total_cells} cells checked, {total_mismatches} mismatch(es))")
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
        chain_records, kv_records = S.load_site_dump_full(site_dump_path)
        chain_by_site = {r.site: r for r in chain_records}
        kv_by_site = {r.site: r for r in kv_records}
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

            def chain(name: str):
                return chain_by_site[f"layer{layer_index}.{name}"]

            def kv(name: str):
                return kv_by_site[f"layer{layer_index}.{name}"]

            attn_norm_rec = chain("attn_norm")

            # --- 1. kv_landing.k / kv_landing.v: real attn_norm -> shadow --
            k_shadow = S.kv_landing_shadow_codes(
                attn_norm_rec.codes, attn_norm_rec.m_out, attn_norm_rec.e_out, layer.k_weight,
                layer.k_fold[:, 0], layer.k_fold[:, 1], layer.k_fold[:, 2], layer.k_bias,
                layer.kv_landing_r_t_k, layer.kv_landing_e_t_k, num_kv_heads, head_dim)
            v_shadow = S.kv_landing_shadow_codes(
                attn_norm_rec.codes, attn_norm_rec.m_out, attn_norm_rec.e_out, layer.v_weight,
                layer.v_fold[:, 0], layer.v_fold[:, 1], layer.v_fold[:, 2], layer.v_bias,
                layer.kv_landing_r_t_v, layer.kv_landing_e_t_v, num_kv_heads, head_dim)
            for kvh in range(num_kv_heads):
                k_real = kv(f"kv_landing.k.h{kvh}")
                total_cells += 1
                if not np.array_equal(k_shadow[kvh], k_real.codes):
                    return stop(label, role, row, layer_index, f"kv_landing.k.h{kvh}",
                                f"  shadow={k_shadow[kvh].tolist()}\n  real  ={k_real.codes.tolist()}")
                v_real = kv(f"kv_landing.v.h{kvh}")
                total_cells += 1
                if not np.array_equal(v_shadow[kvh], v_real.codes):
                    return stop(label, role, row, layer_index, f"kv_landing.v.h{kvh}",
                                f"  shadow={v_shadow[kvh].tolist()}\n  real  ={v_real.codes.tolist()}")

            # --- 2. rope_apply.q.hN: real RopeApplySite input -> shadow ----
            q_rot_real = np.empty((num_heads, head_dim), dtype=np.int8)
            for h in range(num_heads):
                rope_q_real = chain(f"rope_apply.q.h{h}")
                shadow_out = S.rope_apply_site_reference(
                    rope_q_real.x_int.astype(np.int8), position, artifact.rope_cos, artifact.rope_sin)
                total_cells += 1
                if not np.array_equal(shadow_out, rope_q_real.codes):
                    return stop(label, role, row, layer_index, f"rope_apply.q.h{h}",
                                f"  shadow={shadow_out.tolist()}\n  real  ={rope_q_real.codes.tolist()}")
                q_rot_real[h] = rope_q_real.codes

            # --- 3. rope_apply.k.hN: real kv_landing.k -> shadow (also -----
            # cross-checks the landing capture and RopeApplySite's own
            # rotation input agree on the pre-rotation value).
            k_current_rot_real = np.empty((num_kv_heads, head_dim), dtype=np.int8)
            for h in range(num_heads):
                kvh = h // group
                real_landed_k = kv(f"kv_landing.k.h{kvh}").codes
                shadow_out = S.rope_apply_site_reference(
                    real_landed_k, position, artifact.rope_cos, artifact.rope_sin)
                rope_k_real = chain(f"rope_apply.k.h{h}")
                total_cells += 1
                if not np.array_equal(shadow_out, rope_k_real.codes):
                    return stop(label, role, row, layer_index, f"rope_apply.k.h{h}",
                                f"  shadow={shadow_out.tolist()}\n  real  ={rope_k_real.codes.tolist()}")
                if h == kvh * group:
                    k_current_rot_real[kvh] = rope_k_real.codes

            # --- 4. attn_ctx (closed): real q_rot/k_current_rot/v_current --
            v_current_real = np.stack([kv(f"kv_landing.v.h{kvh}").codes for kvh in range(num_kv_heads)])
            q_rec = chain("q_proj.requant")
            ctx_status, ctx_shadow = S.attn_ctx_shadow_codes_from_real_landing_and_rotation(
                q_rot_real, k_current_rot_real, v_current_real, q_rec.m_out, q_rec.e_out, position,
                width, layer, real_k_prior, real_v_prior, num_heads, num_kv_heads, head_dim)
            ctx_real = chain("attn_ctx")
            total_cells += 1
            if ctx_status != "Ok" or not np.array_equal(ctx_shadow, ctx_real.codes):
                return stop(label, role, row, layer_index, "attn_ctx (closed re-drive)",
                            f"  status={ctx_status}\n"
                            f"  shadow={ctx_shadow.tolist() if ctx_shadow is not None else None}\n"
                            f"  real  ={ctx_real.codes.tolist()}")

        print(f"prompt={label!r} role={role}: kv_landing.k/v, rope_apply.q/k, attn_ctx (closed) "
              f"exact-match at rows {TARGET_ROWS} -- OK")

    per_row = 2 * num_kv_heads + 2 * num_heads + 1
    print(f"RESULT: RoPE (site kind 4/4) agrees with the real engine EXACTLY at every checked cell "
          f"({total_cells} cells: {len(KSR.POPULATION)} prompts x {len(TARGET_ROWS)} rows x "
          f"({num_kv_heads} kv_landing.k + {num_kv_heads} kv_landing.v + {num_heads} rope_apply.q + "
          f"{num_heads} rope_apply.k + 1 attn_ctx-closed = {per_row} per-row cells), "
          f"{total_mismatches} mismatch(es))")
    return 0 if total_mismatches == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
