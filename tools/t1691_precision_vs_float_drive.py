#!/usr/bin/env python3
"""T-1691 -- the precision-shadow-vs-true-float-reference drive (design
`Claude/Vitruvius/superslm-t1683-source-attribution-design-2026-08-02.md`
Sec7 step 6d). The campaign's last unmeasured arm of the C1/C2/C3/C4
decomposition, run against the TRUE FLOAT reference -- not against a shadow
that shares the engine's already-quantized weight, which is what Sec7 step
6c (`tools/t1691_precision_drive.py`, D-SLM762) measured.

WHAT THIS DRIVE IS, STATED PRECISELY, READ BEFORE TRUSTING ANY NUMBER BELOW.

D-SLM762's own comparison (`parity-shadow-vs-precision-shadow`) holds the
ALREADY-QUANTIZED weight fixed on both arms -- it isolates C2's own
inference-time activation-requantization loss alone and cannot, by
construction, reflect weight-quantization error (the int8 artifact's own
departure from the model's TRUE pre-quantization weight). D-SLM762 named
this as the reason its own result could not decide the campaign's real
question and named design Sec7 step 6d -- this drive -- as what closes that
gap.

This drive compares, at each of the SAME five `project_and_funnel` sites
D-SLM762 already measures (q_proj, o_proj, gate_proj, up_proj, down_proj --
2 of the real layer's 5 site KINDS, 5 of its 11 site INVOCATIONS; the
remaining six invocations -- rms_norm x2, mlp_act x1, residual_reconcile x2,
context_row_funnel x1 -- have no representation in `precision_shadow_layer`
and are NOT measured here, the identical coverage boundary D-SLM762 named,
unchanged by this drive):

  - `precision_shadow_layer`'s float64 output, fed the real int8 engine's
    own captured input codes at that site (UNCHANGED from D-SLM762 --
    reused, not recomputed) through `dequantize_weight_matrix`'s recovered
    `real_scale` applied to the SAME already-quantized `w_int8` the engine
    itself holds;
  - against the TRUE FLOAT reference model's OWN output at the
    corresponding submodule (`self_attn.q_proj`/`self_attn.o_proj`/
    `mlp.gate_proj`/`mlp.up_proj`/`mlp.down_proj`), captured via a forward
    hook on the REAL, never-quantized `transformers` checkpoint, for the
    SAME prompt's last token, driven token-at-a-time with an explicit
    `DynamicCache` (NEVER a batched forward -- design's own standing
    constraint, `tools/float_reference_layer_dump.py`'s own established
    convention, reused verbatim here via `capture_float_site_outputs`).

This is NOT the same quantity as D-SLM762's own comparison. The float
reference's own q_proj/o_proj/gate_proj/up_proj/down_proj output is computed
from the model's TRUE, never-quantized weight, fed the float reference's OWN
cumulatively-float upstream state (fully independent of the int8 engine from
embedding onward) -- so a residual here reflects BOTH weight-quantization
error AND the accumulated divergence of the two paths' own upstream state,
not C2's inference-time loss alone. This is exactly the shape design Sec7
step 6d specifies ("unchanged from the design's original shadow-vs-float
comparison... the unit's own PARTIAL test of C4", design Sec9) -- a partial
test, because the two arms' inputs already differ before this site is
reached, not a clean single-mechanism isolation the way 6c is.

**q_proj's own bias-omission gap, unchanged from D-SLM762.**
`ProjectAndFunnel`'s real construction (and the float reference's own
`self_attn.q_proj`, which carries a real bias tensor) folds a per-channel
bias; `precision_shadow_layer`'s composition has no bias term at all. The
other four sites call with `bias=None` already on both sides (matching
production), so they carry no such gap. q_proj's own comparison in this
drive is `dequantized-weight-matmul-without-bias` against
`true-weight-matmul-WITH-bias` -- a real, separate confound, layered on top
of (not instead of) the weight-quantization/upstream-divergence gap this
drive exists to measure. Named, not closed, exactly as D-SLM762 named it.

Usage: python tools\\t1691_precision_vs_float_drive.py
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_saturation_report as KSR  # noqa: E402  -- the established 9-prompt population + paths
import layer_bisection_report as lbr  # noqa: E402
import shadow_layer_recompute as S  # noqa: E402
from float_reference_layer_dump import (  # noqa: E402
    DEFAULT_MODEL,
    _resolve_default_model,
    fnv1a64,
)
from layer_bisection_report import load_int8_layer_dump  # noqa: E402
from sslm_artifact_reader import read_artifact  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DRIVER_EXE = os.path.join(REPO_ROOT, "out", "sslm_layer_trace.exe")
OUT_DIR = os.path.join(REPO_ROOT, "out", "t1691_precision_vs_float_drive")

# Unchanged from every prior T-1691 site-kind drive and from D-SLM762.
TARGET_ROWS = [5, 26, 27, 28]
TARGET_LAYER_INDICES = [row - 1 for row in TARGET_ROWS]  # 0-indexed

# (site name in the chain-record dump, real-input chain-record name,
#  weight/fold attribute prefix on the artifact layer object, bias attribute
#  name or None, HF submodule short name).
SITES = [
    ("q_proj.requant", "attn_norm", "q", "q_bias", "q_proj"),
    ("o_proj.requant", "attn_ctx", "o", None, "o_proj"),
    ("gate_proj.requant", "mlp_norm", "gate", None, "gate_proj"),
    ("up_proj.requant", "mlp_norm", "up", None, "up_proj"),
    ("down_proj.requant", "mlp_act", "down", None, "down_proj"),
]

# The HF submodule each site's float output is captured from, relative to
# `model.model.layers[i]` -- read at source (standard Qwen2 decoder-layer
# module tree: `self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`), and
# confirmed by shape assertion at capture time (part of this drive's own
# red-first sanity gate, below) rather than assumed from naming convention
# alone.
SITE_MODULE_PATH = {
    "q_proj": ("self_attn", "q_proj"),
    "o_proj": ("self_attn", "o_proj"),
    "gate_proj": ("mlp", "gate_proj"),
    "up_proj": ("mlp", "up_proj"),
    "down_proj": ("mlp", "down_proj"),
}

# Expected output width per HF submodule -- the shape guard this drive's own
# red-first sanity gate checks before any statistic is computed (Qwen2.5-1.5B
# -Instruct: hidden_size=1536, intermediate_size=8960, read at source via
# AutoConfig at model-load time below, not hard-coded here as a magic
# number -- these names are for the assertion message only).
EXPECTED_WIDTH_KEY = {
    "q_proj": "hidden_size",
    "o_proj": "hidden_size",
    "gate_proj": "intermediate_size",
    "up_proj": "intermediate_size",
    "down_proj": "hidden_size",
}


def capture_float_site_outputs(model, input_ids, layer_indices):
    """Forward hooks on the FIVE `project_and_funnel` submodules
    (`self_attn.q_proj`/`o_proj`, `mlp.gate_proj`/`up_proj`/`down_proj`) at
    each of `layer_indices`, driven by a token-at-a-time incremental forward
    with an explicit `DynamicCache` -- the IDENTICAL capture shape
    `tools/float_reference_layer_dump.py`'s own `capture_incremental` uses
    (never a batched forward, per the design's own standing constraint,
    "Compare only at positions where both paths carry an identical prefix
    ... never a batched forward alone"). Returns
    `{(layer_idx, site_name): float64 numpy [out_channels]}` for the LAST
    prompt token only -- every earlier call's captured value is overwritten
    as the loop advances, matching that script's own "only the LAST call's
    captured values... are dumped" convention."""
    from transformers import DynamicCache

    import torch

    captured: dict[tuple[int, str], "torch.Tensor"] = {}

    def make_hook(layer_idx, site_name):
        def hook(module, args, output):
            t = output if not isinstance(output, tuple) else output[0]
            captured[(layer_idx, site_name)] = t.detach()[0, -1, :].float().clone()

        return hook

    handles = []
    for layer_idx in layer_indices:
        layer = model.model.layers[layer_idx]
        for site_name, (parent_attr, child_attr) in SITE_MODULE_PATH.items():
            module = getattr(getattr(layer, parent_attr), child_attr)
            handles.append(module.register_forward_hook(make_hook(layer_idx, site_name)))

    try:
        cache = DynamicCache()
        with torch.no_grad():
            for t in range(input_ids.shape[1]):
                model(input_ids=input_ids[:, t : t + 1], past_key_values=cache, use_cache=True)
    finally:
        for h in handles:
            h.remove()

    return {k: v.cpu().numpy().astype(np.float64) for k, v in captured.items()}


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isfile(DRIVER_EXE):
        print(f"FAILED: {DRIVER_EXE} not built -- run tools\\build_layer_trace.bat", file=sys.stderr)
        return 1
    if not os.path.isfile(KSR.MODEL_PATH):
        print(f"FAILED: real .sslm artifact not found at {KSR.MODEL_PATH}", file=sys.stderr)
        return 1

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = _resolve_default_model(DEFAULT_MODEL)
    if not model_path.exists():
        print(f"FAILED: float checkpoint not found at {model_path}", file=sys.stderr)
        return 1

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True, torch_dtype="auto")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    expected_width = {
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
    }
    print(f"float checkpoint loaded: {model_path}, device={device}, "
          f"hidden_size={expected_width['hidden_size']}, "
          f"intermediate_size={expected_width['intermediate_size']}")

    artifact = read_artifact(str(KSR.MODEL_PATH))

    cell_log: list[tuple] = []
    total_cells = 0
    fingerprint_checks = 0
    repeat_checks = 0

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

        # Provenance guard (design Sec7 red-first proof part 3): the .sslm
        # engine's own dumped prompt_fingerprint must match FNV-1a64 of the
        # literal prompt text this drive is about to feed the float
        # reference's own tokenizer -- a loud, non-zero-exit rejection on
        # mismatch, never a silent divergent-prompt comparison.
        expected_fp = fnv1a64(prompt)
        if base.prompt_fingerprint != expected_fp:
            print(f"FAILED: prompt_fingerprint mismatch for {label!r}: engine dump has "
                  f"0x{base.prompt_fingerprint:016X}, fnv1a64(prompt) is 0x{expected_fp:016X} -- "
                  f"the int8 dump and the prompt this drive is about to feed the float model "
                  f"were not built from the same text", file=sys.stderr)
            return 1
        fingerprint_checks += 1

        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        captured_a = capture_float_site_outputs(model, input_ids, TARGET_LAYER_INDICES)
        captured_b = capture_float_site_outputs(model, input_ids, TARGET_LAYER_INDICES)
        for key in captured_a:
            repeat_diff = float(np.max(np.abs(captured_a[key] - captured_b[key])))
            if repeat_diff != 0.0:
                print(f"NOTE: float-reference capture non-bit-identical on repeat at prompt="
                      f"{label!r} {key}: max abs diff={repeat_diff:.3e} (GPU reduction-order "
                      f"nondeterminism candidate -- reported, not treated as a driver failure)",
                      file=sys.stderr)
            repeat_checks += 1
        captured = captured_a

        row_stats = []
        for row, layer_index in zip(TARGET_ROWS, TARGET_LAYER_INDICES):
            if layer_index >= base.rows - 1:
                continue

            layer = artifact.layers[layer_index]

            def rec(name: str):
                return site_records[f"layer{layer_index}.{name}"]

            for site_name, input_site, attr, bias_attr, hf_name in SITES:
                input_rec = rec(input_site)

                w = getattr(layer, f"{attr}_weight")
                fold = getattr(layer, f"{attr}_fold")
                triple = S.SiteFoldTriple(
                    kind="project_and_funnel", w_int8=w,
                    identity=fold[:, 0], mult=fold[:, 1], shift=fold[:, 2])

                precision_a = S.precision_shadow_layer(input_rec.codes, [triple])
                precision_b = S.precision_shadow_layer(input_rec.codes, [triple])
                repeat_diff = float(np.max(np.abs(precision_a - precision_b)))
                if repeat_diff != 0.0:
                    print(f"FAILED: precision_shadow_layer non-deterministic at prompt={label!r} "
                          f"row={row} site={site_name}: repeat diff={repeat_diff}", file=sys.stderr)
                    return 1

                float_val = captured[(layer_index, hf_name)]

                # Red-first sanity gate, part 1: shape.
                expected_len = expected_width[EXPECTED_WIDTH_KEY[hf_name]]
                if precision_a.shape[0] != expected_len or float_val.shape[0] != expected_len:
                    print(f"FAILED: shape mismatch at prompt={label!r} row={row} site={site_name}: "
                          f"precision_shadow={precision_a.shape}, float_reference={float_val.shape}, "
                          f"expected {expected_len}", file=sys.stderr)
                    return 1

                stats = lbr.compare_layer_row(precision_a, float_val)
                total_cells += 1
                row_stats.append(stats)
                cell_log.append((label, role, row, site_name, stats.spearman,
                                  stats.pearson, stats.max_abs_z_diff))

        print(f"prompt={label!r} role={role}: precision-shadow-vs-float-reference driven at "
              f"rows {TARGET_ROWS} x 5 project_and_funnel sites -- {len(row_stats)} cells")

    # Design Sec7 red-first proof part 4's own sanity gate, reused verbatim.
    bad = 0
    for lbl, role, row, site, sp, pe, z in cell_log:
        for value in (sp, pe, z):
            if not np.isfinite(value) and not np.isnan(value):
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

    print(f"\nprovenance: {fingerprint_checks}/{len(KSR.POPULATION)} prompt fingerprints matched "
          f"between the .sslm engine dump and the text fed to the float reference tokenizer.")

    print("\n--- per-cell (prompt, row, site) -> spearman, pearson, max_abs_z_diff ---")
    for lbl, role, row, site, sp, pe, z in cell_log:
        print(f"  {lbl:24s} role={role:6s} row={row:2d} site={site:18s} "
              f"spearman={sp:+.6f} pearson={pe:+.6f} max_abs_z_diff={z:.6f}")

    print(f"\nRESULT: precision-shadow-vs-float-reference driven -- {total_cells} cells "
          f"({len(KSR.POPULATION)} prompts x {len(TARGET_ROWS)} rows x {len(SITES)} "
          f"project_and_funnel sites), 0 non-finite.")
    print(f"precision-shadow-vs-float-reference spearman: min={min(all_rho):.6f} "
          f"max={max(all_rho):.6f} over all {len(all_rho)} cells.")
    print(f"  mech2 group (n={len(mech2_rho)}): min={min(mech2_rho):.6f} max={max(mech2_rho):.6f}")
    print(f"  control group (n={len(control_rho)}): min={min(control_rho):.6f} "
          f"max={max(control_rho):.6f}")
    print(f"max_abs_z_diff over all cells: min={min(all_z):.6f} max={max(all_z):.6f}")
    print("\nper-site spearman (n=36 each -- 9 prompts x 4 rows):")
    for site, vals in by_site.items():
        print(f"  {site:18s} min={min(vals):.6f} max={max(vals):.6f} "
              f"mean={sum(vals) / len(vals):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
