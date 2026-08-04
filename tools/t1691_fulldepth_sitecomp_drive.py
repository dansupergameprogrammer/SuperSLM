#!/usr/bin/env python3
"""T-1691 onset/amplification drive (Brunel, 2026-08-04).

WHAT THIS IS. Popper's debunk (Claude/Popper/superslm-t1691-precision-vs-
float-quantization-attribution-debunk-2026-08-03.md, D-SLM764) killed the
"weight quantization explains layers 26-28's divergence" claim and showed
the effect is carried by upstream-activation divergence already present at
row 5 -- present, not new at 26-28. It left two questions genuinely open,
named in its own boundary section:

  (1) ONSET -- at which layer does the int8-vs-float activation divergence
      FIRST carry the mechanism-2 grouping (as opposed to being present but
      ungrouped)? Only rows {5, 26, 27, 28} have ever been measured with
      this method; this drive measures ALL 28 rows.
  (2) AMPLIFICATION -- the divergence is present early (row 5, gap 0.0257)
      and only becomes a RANK COLLAPSE at 26-28 (gap ~0.11-0.20). What
      changes at those rows? This drive reports the gap-magnitude curve
      across all 28 rows to locate where the growth actually happens.
  (3) THE SIX UNMEASURED SITE INVOCATIONS -- D-SLM763/764's own drive only
      reaches the 5 `project_and_funnel` sites (q/o/gate/up/down_proj),
      because only those have a dequantizable weight MATRIX. The other 6
      invocations per layer (attn_norm, mlp_norm -- RMSNorm x2; attn_ctx,
      i.e. the context-row funnel; mlp_act; attn_residual, mlp_residual --
      residual reconcile x2) have no weight matrix to swap, so the
      weight-vs-input decomposition (Q_weight_alone / Q_input_alone) does
      not apply to them by construction -- there is no "weight" axis to
      hold fixed. What IS measurable, and what this drive computes for all
      six: Q_full, the direct rank correlation between the int8 engine's
      own real site-output codes (already exact by C1 -- parity_shadow's
      1,440-cell zero-mismatch proof, D-SLM747/750-751/753-754/757-759) and
      the float reference's own true output at the SAME point in the
      computation, captured via a forward hook (or forward-PRE-hook, where
      the site's output is not itself a discrete HF submodule's return
      value but IS the next submodule's input). This is a single,
      non-decomposed quantity -- it measures WHETHER divergence appears/
      grows at that site, never WHETHER a weight or an input carries it,
      because for these six sites that separation is not a question with
      an answer: there is nothing to attribute to a weight.

METHOD, reused verbatim from the packet this pass is extending (design
Claude/Vitruvius/superslm-t1683-source-attribution-design-2026-08-02.md,
Claude/Popper/probes/t1691_weightquant_vs_inputdivergence_isolation_probe.py):
  - Never compare raw or dequantized magnitudes across paths -- Spearman
    rank correlation only.
  - Float reference driven token-at-a-time with an explicit DynamicCache,
    never batched (design S2.5/strike fracture).
  - Never output_hidden_states=True (S2.4 -- corrupts the last layer).
  - One variable varied at a time; every quantity states which.
  - C(9,4)=126 permutation test floors significance at p=0.0079 (rank
    1/126) -- reported alongside the gap-magnitude distribution, not as a
    verdict on its own.

WHAT THIS DRIVE ADDS THAT NO PRIOR T-1691 ARM DID: it runs at ALL 28 layer
rows (1..28), not only {5, 26, 27, 28}, for BOTH the 5 decomposable sites
AND the 6 non-decomposable ones -- widening tools/sslm_layer_trace.cpp's own
site-dump target-row band (a tools/ change, never src/ or include/; the
underlying RequantChainChecked/trace-hook seam this reads is unmodified).

Usage
-----
    python tools\\t1691_fulldepth_sitecomp_drive.py
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_saturation_report as KSR  # noqa: E402  -- the established 9-prompt population + paths
import shadow_layer_recompute as S  # noqa: E402
from float_reference_layer_dump import DEFAULT_MODEL, _resolve_default_model, fnv1a64  # noqa: E402
from layer_bisection_report import load_int8_layer_dump  # noqa: E402
from sslm_artifact_reader import read_artifact  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DRIVER_EXE = os.path.join(REPO_ROOT, "out", "sslm_layer_trace.exe")
OUT_DIR = os.path.join(REPO_ROOT, "out", "t1691_fulldepth_sitecomp")

ALL_ROWS = list(range(1, 29))  # rows 1..28, 0-indexed layer = row-1
ALL_LAYER_INDICES = [r - 1 for r in ALL_ROWS]

# The 5 decomposable (project_and_funnel) sites -- unchanged construction
# from D-SLM762/763's own drives.
PROJECT_SITES = [
    ("q_proj.requant", "attn_norm", "q", "self_attn", "q_proj"),
    ("o_proj.requant", "attn_ctx", "o", "self_attn", "o_proj"),
    ("gate_proj.requant", "mlp_norm", "gate", "mlp", "gate_proj"),
    ("up_proj.requant", "mlp_norm", "up", "mlp", "up_proj"),
    ("down_proj.requant", "mlp_act", "down", "mlp", "down_proj"),
]
PROJECT_WIDTH_KEY = {
    "q_proj": "hidden_size", "o_proj": "hidden_size",
    "gate_proj": "intermediate_size", "up_proj": "intermediate_size", "down_proj": "hidden_size",
}

# The 6 non-decomposable site invocations -- each entry: (sitedump chain-
# record name (without "layerN." prefix), a stable short quantity label,
# hook kind ("post" = forward_hook on the named module, capturing its
# OUTPUT; "pre" = forward_pre_hook on the named module, capturing its
# INPUT arg 0), module path relative to layers[i], expected-width config
# key). "layer" hook kind captures the decoder layer's own output
# (register_forward_hook on model.model.layers[i] itself -- design S2.4's
# own confirmed-safe mechanism: a plain tensor, not the tuple
# output_hidden_states corrupts).
NONPROJECT_SITES = [
    ("attn_norm", "attn_norm", "post", ("input_layernorm",), "hidden_size"),
    ("attn_ctx", "attn_ctx", "pre", ("self_attn", "o_proj"), "hidden_size"),
    ("attn_residual", "attn_residual", "pre", ("post_attention_layernorm",), "hidden_size"),
    ("mlp_norm", "mlp_norm", "post", ("post_attention_layernorm",), "hidden_size"),
    ("mlp_act", "mlp_act", "pre", ("mlp", "down_proj"), "intermediate_size"),
    ("mlp_residual", "mlp_residual", "layer", (), "hidden_size"),
]


def capture_float_site_outputs(model, input_ids, layer_indices):
    """One incremental (token-at-a-time, DynamicCache, never batched)
    forward pass capturing ALL 11 site invocations' float-side true output
    at every requested layer index, in one pass. Returns
    {(layer_idx, key): float64 numpy [width]} for the LAST prompt token
    only, `key` one of the 5 project-site hf_names or one of the 6
    NONPROJECT_SITES short labels."""
    from transformers import DynamicCache
    import torch

    captured: dict[tuple[int, str], "torch.Tensor"] = {}

    def make_post_hook(layer_idx, key):
        def hook(module, args, output):
            t = output if not isinstance(output, tuple) else output[0]
            captured[(layer_idx, key)] = t.detach()[0, -1, :].float().clone()
        return hook

    def make_pre_hook(layer_idx, key):
        def hook(module, args):
            t = args[0]
            captured[(layer_idx, key)] = t.detach()[0, -1, :].float().clone()
        return hook

    def make_layer_hook(layer_idx, key):
        def hook(module, args, output):
            t = output if not isinstance(output, tuple) else output[0]
            captured[(layer_idx, key)] = t.detach()[0, -1, :].float().clone()
        return hook

    handles = []
    for layer_idx in layer_indices:
        layer = model.model.layers[layer_idx]
        for site_name, input_site, attr, parent_attr, hf_name in PROJECT_SITES:
            module = getattr(getattr(layer, parent_attr), hf_name)
            handles.append(module.register_forward_hook(make_post_hook(layer_idx, hf_name)))
        for chain_name, key, hook_kind, module_path, _width_key in NONPROJECT_SITES:
            if hook_kind == "layer":
                handles.append(layer.register_forward_hook(make_layer_hook(layer_idx, key)))
                continue
            target = layer
            for attr in module_path:
                target = getattr(target, attr)
            if hook_kind == "post":
                handles.append(target.register_forward_hook(make_post_hook(layer_idx, key)))
            elif hook_kind == "pre":
                handles.append(target.register_forward_pre_hook(make_pre_hook(layer_idx, key)))
            else:
                raise ValueError(hook_kind)

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

    # rows: (label, role, row, site, quantity_name, value)
    rows_out: list[tuple] = []
    total_cells = 0
    fingerprint_checks = 0

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

        expected_fp = fnv1a64(prompt)
        if base.prompt_fingerprint != expected_fp:
            print(f"FAILED: prompt_fingerprint mismatch for {label!r}: engine dump has "
                  f"0x{base.prompt_fingerprint:016X}, fnv1a64(prompt) is 0x{expected_fp:016X}",
                  file=sys.stderr)
            return 1
        fingerprint_checks += 1

        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        captured = capture_float_site_outputs(model, input_ids, ALL_LAYER_INDICES)

        row_cells = 0
        for row, layer_index in zip(ALL_ROWS, ALL_LAYER_INDICES):
            if layer_index >= base.rows - 1:
                continue
            layer = artifact.layers[layer_index]

            def rec(name: str):
                return site_records[f"layer{layer_index}.{name}"]

            # 5 decomposable sites: Q_full, Q_weight_alone, Q_input_alone.
            for site_name, input_site, attr, parent_attr, hf_name in PROJECT_SITES:
                input_rec = rec(input_site)
                w_int8 = getattr(layer, f"{attr}_weight")
                fold = getattr(layer, f"{attr}_fold")
                triple = S.SiteFoldTriple(kind="project_and_funnel", w_int8=w_int8,
                                           identity=fold[:, 0], mult=fold[:, 1], shift=fold[:, 2])
                x_int8 = input_rec.codes.astype(np.float64)
                s_shadow = S.precision_shadow_layer(x_int8, [triple])

                w = getattr(getattr(model.model.layers[layer_index], parent_attr), hf_name).weight
                w_true = w.detach().float().cpu().numpy().astype(np.float64)
                s_trueweight_sameinput = w_true @ x_int8

                f_true = captured[(layer_index, hf_name)]
                expected_len = expected_width[PROJECT_WIDTH_KEY[hf_name]]
                if s_shadow.shape[0] != expected_len or f_true.shape[0] != expected_len:
                    print(f"FAILED: shape mismatch {label} row={row} site={site_name}: "
                          f"{s_shadow.shape} vs expected {expected_len}", file=sys.stderr)
                    return 1

                q_full, _ = spearmanr(s_shadow, f_true)
                q_weight, _ = spearmanr(s_shadow, s_trueweight_sameinput)
                q_input, _ = spearmanr(s_trueweight_sameinput, f_true)
                for qname, qval in (("q_full", q_full), ("q_weight_alone", q_weight),
                                     ("q_input_alone", q_input)):
                    rows_out.append((label, role, row, site_name, qname, float(qval)))
                    total_cells += 1
                row_cells += 3

            # 6 non-decomposable sites: Q_full only (no weight axis exists).
            for chain_name, key, hook_kind, module_path, width_key in NONPROJECT_SITES:
                input_codes = rec(chain_name).codes.astype(np.float64)
                f_true = captured[(layer_index, key)]
                expected_len = expected_width[width_key]
                if input_codes.shape[0] != expected_len or f_true.shape[0] != expected_len:
                    print(f"FAILED: shape mismatch {label} row={row} site={chain_name}: "
                          f"int8 codes {input_codes.shape} float {f_true.shape} "
                          f"expected {expected_len}", file=sys.stderr)
                    return 1
                q_full, _ = spearmanr(input_codes, f_true)
                rows_out.append((label, role, row, chain_name, "q_full", float(q_full)))
                total_cells += 1
                row_cells += 1

        print(f"prompt={label!r} role={role}: {row_cells} cells across {len(ALL_ROWS)} rows")

    bad = sum(1 for r in rows_out if not np.isfinite(r[5]))
    if bad:
        print(f"FAILED: {bad} non-finite cell(s) among {total_cells} -- sanity gate failed", file=sys.stderr)
        return 1

    out_path = os.path.join(OUT_DIR, "fulldepth_results.tsv")
    with open(out_path, "w") as f:
        f.write("label\trole\trow\tsite\tquantity\tvalue\n")
        for r in rows_out:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"\nwrote {len(rows_out)} cells to {out_path}")
    print(f"provenance: {fingerprint_checks}/{len(KSR.POPULATION)} prompt fingerprints matched.")
    print(f"RESULT: {total_cells} cells, 0 non-finite, {len(KSR.POPULATION)} prompts x {len(ALL_ROWS)} "
          f"rows x (5 sites x 3 quantities + 6 sites x 1 quantity = 21 cells/row).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
