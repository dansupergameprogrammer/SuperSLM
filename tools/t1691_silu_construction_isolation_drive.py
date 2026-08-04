#!/usr/bin/env python3
"""T-1691 -- SiLU/sigmoid construction isolation drive (Brunel, 2026-08-04).

WHAT THIS ANSWERS. `Claude/Linnaeus/int8-forward-vs-float-reference-semantic-
divergence-fact-sheet-2026-08-04.md` names the engine's SwiGLU sigmoid
construction (C10's fixed-point LUT, `SiluSigmoidQ15`, a bounded-domain
[-16,16] piecewise-linear approximation of the true logistic function) as
the strongest implementation-side lead against the layer-21-onward int8-
vs-float divergence signature this campaign has measured (D-SLM765/767):
onset at row 21-22 of 28, peak at rows 25-26, receding by row 28, present
for the four "mech2" arithmetic-flavoured prompts and largely absent for
the five factual-recall prompts, with a reversal band at rows 12-20. This
drive runs the natural experiment Dan's directive names: recompute the
`mlp_act` site two ways over the SAME real captured `gate_proj.requant`/
`up_proj.requant` codes and scale -- once with the engine's own C10 LUT
(`shadow_layer_recompute.silu_sigmoid_q15_reference`, already proven exact
against the real compiled engine at 180/180 cells, D-SLM753/754, extended
to all 28 rows here as a byproduct) and once with the exact mathematical
sigmoid the LUT approximates (`1/(1+exp(-x))`, the same target function
`Qwen2MLP`'s real `nn.SiLU` computes, per the fact sheet's own Sec7.4 --
this is "the more faithful construction the reference uses," not the
UNRELATED excluded i-exp construction the fact sheet's Divergence 1 names,
which lives only in `Tools/superslm_spike/pipeline.py`'s peripheral forward
entry points and is not part of this campaign's own float-reference
comparison path at all (that path is the real `transformers` Qwen2 model,
driven token-at-a-time, per every prior T-1691 arm).

ONE VARIABLE VARIED AT A TIME (StandardsDocument Sec5.4 / this campaign's
own standing method): `gate_proj.requant`/`up_proj.requant` codes and the
gate's own carried scale are the REAL engine's captured values, identical
on both arms. Only the sigmoid computation inside `mlp_act` differs -- the
domain check, the triple product, and the funnel (`RequantChainChecked`)
are the SAME code (`shadow_layer_recompute.py`'s own primitives) on both
arms. This isolates the sigmoid construction's own contribution from every
other site in the layer.

MAGNITUDE is reported in TWO units that never mix int8-vs-float raw
magnitudes across paths (the campaign's own standing prohibition):
  (a) the Q15 sigmoid VALUE delta itself (LUT vs exact), a fixed,
      construction-independent unit both arms share by definition --
      `sig_lut[i]` and `sig_exact[i]` are both Q15 fixed-point
      representations of the identical target quantity, so their
      difference is directly meaningful with no implicit per-row rescale;
  (b) the resulting mlp_act INT8 CODE-LEVEL difference between the two
      arms (`compare_exact`, a structural exact-equality count, the same
      convention the C1 tests in this campaign already use) -- NOT a
      dequantized-magnitude comparison, and NOT compared against the float
      reference's own raw values.
SIGNATURE is reported via the same C(9,4)=126 permutation-test machinery
(`t1691_fulldepth_permtest.py`'s own method, reimplemented here rather than
imported, to keep this module self-contained and off that file's own
dependency surface) applied to Spearman rank correlation against the real
float reference's true mlp_act activation (captured via a forward-pre-hook
on `mlp.down_proj`, the SAME site and hook `t1691_fulldepth_sitecomp_drive.
py` already established) -- once for the real engine's own codes (Q_full,
reproducing D-SLM765's own number), once for the alt-sigmoid arm's codes
(Q_full_alt) -- so the onset/peak/reversal shape can be compared directly
between the two constructions.

ALSO MEASURED (Divergence 2, cheap once this harness's float forward pass
exists, per the commission's own framing): the RMSNorm epsilon-vs-floor-at-1
divergence's magnitude, on REAL captured pre-norm hidden states, captured
via the same float forward pass (pre-hooks on `input_layernorm`/
`post_attention_layernorm`, i.e. the actual variance-units domain epsilon
lives in -- the int8 construction's own h codes carry no real-valued scale
at all, per `rmsnorm_shadow_codes`'s own docstring, "never dequantized," so
this magnitude can only be measured in the float reference's own terms, not
reconstructed from the int8 side).

Usage: python tools\\t1691_silu_construction_isolation_drive.py
"""

from __future__ import annotations

import itertools
import math
import os
import subprocess
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_saturation_report as KSR  # noqa: E402  -- established 9-prompt population + paths
import shadow_layer_recompute as S  # noqa: E402  -- read-only import of primitives, no engine code
from float_reference_layer_dump import DEFAULT_MODEL, _resolve_default_model  # noqa: E402
from layer_bisection_report import load_int8_layer_dump  # noqa: E402
from sslm_artifact_reader import read_artifact  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DRIVER_EXE = os.path.join(REPO_ROOT, "out", "sslm_layer_trace.exe")
OUT_DIR = os.path.join(REPO_ROOT, "out", "t1691_silu_isolation")

ALL_ROWS = list(range(1, 29))
ALL_LAYER_INDICES = [r - 1 for r in ALL_ROWS]

_SILU_Q15_SCALE = 1 << 15  # SIGMOID_FRAC_BITS = 15, S2.4 design Sec4


def exact_sigmoid_q15(code: int, m: int, e: int) -> int:
    """The 'more faithful construction the reference uses': the exact
    mathematical sigmoid the C10 LUT approximates, evaluated at the SAME
    real-valued dequantized gate `x = code * realscale`, `realscale = m *
    2**e` (S2.4 design Sec5's own dequant identity, `pos = (code*realscale
    + X)*K`), then rounded to the SAME Q15 fixed-point representation
    `silu_sigmoid_q15_reference` returns -- no domain clamp to [-16,16]
    (the exact function needs none; the LUT's clamp is itself part of what
    is being isolated as a construction difference, not reproduced here)."""
    x = float(code) * float(m) * (2.0 ** e)
    sig = 1.0 / (1.0 + math.exp(-x)) if x > -700.0 else 0.0  # exp overflow guard, x this negative -> sigmoid ~ 0
    q15 = int(round(sig * _SILU_Q15_SCALE))
    if q15 < 0:
        q15 = 0
    if q15 > _SILU_Q15_SCALE:
        q15 = _SILU_Q15_SCALE
    return q15


def mlp_act_two_ways(gate_codes, gate_m: int, gate_e: int, up_codes, sigmoid_lut_table):
    """Recomputes `MlpActSite` twice over the IDENTICAL real gate/up codes
    and gate scale -- once via the engine's own LUT (mirrors
    `S.mlp_act_shadow_codes` exactly, but also returns the per-element LUT
    sigmoid values so they can be compared to the exact arm's), once via
    `exact_sigmoid_q15`. Domain check, triple product, and funnel
    (`S.requant_chain_reference`) are the SAME code on both arms -- only
    the per-element sigmoid computation differs. Returns
    (domain_status, sig_lut, sig_exact, codes_lut, status_lut, codes_exact,
    status_exact)."""
    domain = S.check_silu_composition_scale_domain_reference(gate_m, gate_e)
    if domain != "Ok":
        return domain, None, None, None, domain, None, domain

    n = len(gate_codes)
    sig_lut = np.empty(n, dtype=np.int64)
    sig_exact = np.empty(n, dtype=np.int64)
    wide_lut = np.empty(n, dtype=object)
    wide_exact = np.empty(n, dtype=object)
    for i in range(n):
        gc, uc = int(gate_codes[i]), int(up_codes[i])
        sig_lut[i] = S.silu_sigmoid_q15_reference(sigmoid_lut_table, gc, gate_m, gate_e)
        sig_exact[i] = exact_sigmoid_q15(gc, gate_m, gate_e)
        wide_lut[i] = gc * int(sig_lut[i]) * uc
        wide_exact[i] = gc * int(sig_exact[i]) * uc

    status_lut, codes_lut = S.requant_chain_reference(wide_lut)
    status_exact, codes_exact = S.requant_chain_reference(wide_exact)
    return "Ok", sig_lut, sig_exact, codes_lut, status_lut, codes_exact, status_exact


def capture_float_hooks(model, input_ids, layer_indices):
    """One incremental (token-at-a-time, DynamicCache, never batched --
    the design's own standing constraint) float forward pass capturing,
    at every requested layer index: `mlp_act`'s true activation (pre-hook
    on `mlp.down_proj`, the SAME site/hook `t1691_fulldepth_sitecomp_
    drive.py` already established for this exact quantity) AND the two
    RMSNorm sites' pre-norm inputs (pre-hooks on `input_layernorm`/
    `post_attention_layernorm` -- the real-valued variance-units domain
    RMSNorm's epsilon term actually lives in, unlike the int8 side's own
    scale-free integer codes)."""
    from transformers import DynamicCache
    import torch

    captured: dict[tuple[int, str], "torch.Tensor"] = {}

    def make_pre_hook(layer_idx, key):
        def hook(module, args):
            captured[(layer_idx, key)] = args[0].detach()[0, -1, :].float().clone()
        return hook

    handles = []
    for layer_idx in layer_indices:
        layer = model.model.layers[layer_idx]
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_pre_hook(layer_idx, "mlp_act")))
        handles.append(layer.input_layernorm.register_forward_pre_hook(
            make_pre_hook(layer_idx, "attn_norm_input")))
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
            make_pre_hook(layer_idx, "mlp_norm_input")))

    try:
        cache = DynamicCache()
        with torch.no_grad():
            for t in range(input_ids.shape[1]):
                model(input_ids=input_ids[:, t : t + 1], past_key_values=cache, use_cache=True)
    finally:
        for h in handles:
            h.remove()

    return {k: v.cpu().numpy().astype(np.float64) for k, v in captured.items()}


def rmsnorm_epsilon_magnitude(x_float: np.ndarray, eps: float = 1e-6):
    """Divergence 2's own magnitude (fact sheet Sec6, flagged underived
    there): on a REAL captured pre-norm hidden-state row, the relative
    perturbation the float reference's `variance + eps` term introduces
    to the normalization root, vs. the epsilon-free root -- the quantity
    the int8 side's floor-at-1 clamp stands in place of. Returns
    (rel_delta, variance)."""
    variance = float(np.mean(x_float.astype(np.float64) ** 2))
    root_no_eps = 1.0 / math.sqrt(variance) if variance > 0.0 else float("inf")
    root_with_eps = 1.0 / math.sqrt(variance + eps)
    rel_delta = (root_with_eps - root_no_eps) / root_no_eps if math.isfinite(root_no_eps) and root_no_eps != 0 else float("nan")
    return rel_delta, variance


def perm_test(values_by_label: dict, mech2_labels: frozenset, all_labels: list):
    """C(9,4)=126 exhaustive permutation test, reimplemented here (self-
    contained, not imported from t1691_fulldepth_permtest.py, which reads
    a different results file this drive does not produce) -- rank of the
    actual mech2 grouping among all 126 possible 4-of-9 splits by
    control-mean-minus-mech2-mean gap. Returns (rank, actual_gap) or None
    if any label is missing a value."""
    if any(lbl not in values_by_label for lbl in all_labels):
        return None
    label_mean = {lbl: float(np.mean(values_by_label[lbl])) for lbl in all_labels}

    def gap_for(group4):
        g = [label_mean[l] for l in group4]
        rest = [label_mean[l] for l in all_labels if l not in group4]
        return float(np.mean(rest) - np.mean(g))

    splits = list(itertools.combinations(sorted(all_labels), 4))
    assert len(splits) == 126
    gaps = [(frozenset(split), gap_for(frozenset(split))) for split in splits]
    gaps_sorted = sorted(gaps, key=lambda x: -x[1])
    rank = next(i for i, (grp, _g) in enumerate(gaps_sorted, start=1) if grp == mech2_labels)
    actual_gap = dict(gaps)[mech2_labels]
    return rank, actual_gap


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isfile(DRIVER_EXE):
        print(f"FAILED: {DRIVER_EXE} not built -- run tools\\build_layer_trace.bat", file=sys.stderr)
        return 1
    if not os.path.isfile(KSR.MODEL_PATH):
        print(f"FAILED: real artifact not found at {KSR.MODEL_PATH}", file=sys.stderr)
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

    artifact = read_artifact(str(KSR.MODEL_PATH))
    intermediate_size = artifact.config["intermediate_size"]

    per_row_cells: list[tuple] = []  # (label, role, row, quantity, value)
    rmsnorm_cells: list[tuple] = []  # (label, role, row, site, quantity, value)
    total_mlp_act_cells = 0
    total_mlp_act_mismatches_lut_vs_real = 0
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

        from float_reference_layer_dump import fnv1a64
        expected_fp = fnv1a64(prompt)
        if base.prompt_fingerprint != expected_fp:
            print(f"FAILED: prompt_fingerprint mismatch for {label!r}", file=sys.stderr)
            return 1
        fingerprint_checks += 1

        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        captured = capture_float_hooks(model, input_ids, ALL_LAYER_INDICES)

        for row, layer_index in zip(ALL_ROWS, ALL_LAYER_INDICES):
            if layer_index >= base.rows - 1:
                continue

            def rec(name: str):
                return site_records[f"layer{layer_index}.{name}"]

            # --- mlp_act two ways ---
            gate_rec = rec("gate_proj.requant")
            up_rec = rec("up_proj.requant")
            act_real = rec("mlp_act")

            (domain, sig_lut, sig_exact, codes_lut, status_lut, codes_exact, status_exact
             ) = mlp_act_two_ways(gate_rec.codes, int(gate_rec.m_out), int(gate_rec.e_out),
                                   up_rec.codes, artifact.sigmoid_lut)
            total_mlp_act_cells += 1
            if domain != "Ok":
                print(f"UNEXPECTED: prompt={label!r} row={row}: gate scale domain rejected "
                      f"({domain}) though the real engine produced an mlp_act output -- skipping cell",
                      file=sys.stderr)
                continue
            if status_lut != "Ok" or not np.array_equal(codes_lut, act_real.codes):
                total_mlp_act_mismatches_lut_vs_real += 1
                print(f"MISMATCH (sanity, LUT arm vs real engine): prompt={label!r} row={row} "
                      f"status={status_lut}", file=sys.stderr)

            sig_delta = sig_exact.astype(np.float64) - sig_lut.astype(np.float64)
            n_diff_codes, _idx = (0, []) if status_exact != "Ok" else S.compare_exact(act_real.codes, codes_exact)

            f_true_act = captured[(layer_index, "mlp_act")]
            q_full_real, _ = spearmanr(act_real.codes.astype(np.float64), f_true_act)
            q_full_alt = float("nan")
            if status_exact == "Ok":
                q_full_alt, _ = spearmanr(codes_exact.astype(np.float64), f_true_act)

            for quantity, value in (
                ("sig_delta_q15_mean_abs", float(np.mean(np.abs(sig_delta)))),
                ("sig_delta_q15_max_abs", float(np.max(np.abs(sig_delta)))),
                ("frac_codes_diff", n_diff_codes / intermediate_size),
                ("q_full_real", float(q_full_real)),
                ("q_full_alt", q_full_alt),
            ):
                per_row_cells.append((label, role, row, quantity, value))

            # --- RMSNorm epsilon magnitude, both sites ---
            for site_key, hook_key in (("attn_norm", "attn_norm_input"), ("mlp_norm", "mlp_norm_input")):
                x_float = captured[(layer_index, hook_key)]
                rel_delta, variance = rmsnorm_epsilon_magnitude(x_float)
                rmsnorm_cells.append((label, role, row, site_key, "eps_rel_delta_root", rel_delta))
                rmsnorm_cells.append((label, role, row, site_key, "variance", variance))

        print(f"prompt={label!r} role={role}: mlp_act two-way + RMSNorm-eps computed for "
              f"{sum(1 for r in ALL_LAYER_INDICES if r < base.rows - 1)} rows")

    bad = sum(1 for r in per_row_cells if not np.isfinite(r[4]) and r[3] != "q_full_alt")
    if bad:
        print(f"FAILED: {bad} non-finite cell(s) in a quantity that must always be finite", file=sys.stderr)
        return 1

    tsv_path = os.path.join(OUT_DIR, "silu_isolation_results.tsv")
    with open(tsv_path, "w") as f:
        f.write("label\trole\trow\tquantity\tvalue\n")
        for r in per_row_cells:
            f.write("\t".join(str(x) for x in r) + "\n")
    rms_path = os.path.join(OUT_DIR, "rmsnorm_epsilon_results.tsv")
    with open(rms_path, "w") as f:
        f.write("label\trole\trow\tsite\tquantity\tvalue\n")
        for r in rmsnorm_cells:
            f.write("\t".join(str(x) for x in r) + "\n")

    print(f"\nwrote {len(per_row_cells)} mlp_act cells to {tsv_path}")
    print(f"wrote {len(rmsnorm_cells)} RMSNorm-epsilon cells to {rms_path}")
    print(f"provenance: {fingerprint_checks}/{len(KSR.POPULATION)} prompt fingerprints matched.")
    print(f"LUT-arm sanity vs real engine: {total_mlp_act_cells} cells, "
          f"{total_mlp_act_mismatches_lut_vs_real} mismatch(es).")

    # --- Permutation test on Q_full_real and Q_full_alt, per row ---
    all_labels = sorted(set(r[0] for r in per_row_cells))
    mech2_labels = frozenset(lbl for lbl, _q, role in KSR.POPULATION if role == "mech2")
    print(f"\nPopulation: {len(all_labels)} prompts, mech2={sorted(mech2_labels)}.\n")

    for quantity in ("q_full_real", "q_full_alt", "frac_codes_diff", "sig_delta_q15_mean_abs"):
        by_row = {}
        for lbl, _role, row, q, val in per_row_cells:
            if q != quantity:
                continue
            by_row.setdefault(row, {}).setdefault(lbl, []).append(val)
        first_weak = None
        first_ceiling = None
        curve = []
        for row in ALL_ROWS:
            if row not in by_row:
                continue
            result = perm_test(by_row[row], mech2_labels, all_labels)
            if result is None:
                continue
            rank, gap = result
            curve.append((row, rank, gap))
            if first_weak is None and rank <= 6:
                first_weak = row
            if first_ceiling is None and rank == 1:
                first_ceiling = row
        print(f"=== {quantity} ===")
        print(f"  first row rank<=6/126: {first_weak}   first row rank=1/126: {first_ceiling}")
        gap_str = " ".join(f"{g:+.4f}" for _r, _rk, g in curve)
        rank_str = " ".join(f"{rk:3d}" for _r, rk, _g in curve)
        print(f"  rows {ALL_ROWS[0]}..{ALL_ROWS[-1]}")
        print(f"  gap:  {gap_str}")
        print(f"  rank: {rank_str}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
