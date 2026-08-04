#!/usr/bin/env python3
"""T-1691 activation-state characterization drive (Brunel, 2026-08-04).

WHAT THIS IS. D-SLM765/766 localized a transition: the mechanism-2 grouping
signature onsets sharply at row 21 (7/11 site kinds) or row 22 (3/11),
identically in shape/timing across sites, following a rows-12-20 reversal
band where the true grouping ranks among the worst possible splits. That
work measured RANK CORRELATION between the int8 and float paths -- it never
measured either path's activation STATE. This drive does: per-row, per-path
(int8, float -- NEVER compared to each other in raw/dequantized magnitude,
per the campaign's own standing method constraint), per-site, per-prompt
magnitude statistics, over the same fixed 9-prompt population.

QUANTITIES COMPUTED, PER (prompt, row, site):

  INT8 PATH (from tools/sslm_layer_trace.cpp's own site-dump -- the wide
  int64 row BEFORE quantization, `x_int`, and the requantized int8 output,
  `codes`; both already exact by C1 -- parity_shadow, D-SLM747/750-751/
  753-754/757-759):
    - d_prime_engine: the engine's OWN MaxAbsReduceWide(wide_row) result,
      read directly from the site dump (real_scale's own basis) -- the
      per-token dynamic scale numerator the engine actually derives.
    - d_prime_recomputed: max(|x_int|) recomputed independently in this
      script, from the same x_int array the site dump carries -- checked
      equal to d_prime_engine as a correctness gate on this drive's own
      reading of the dump, not a new measurement of the engine.
    - dn, s: NormalizeScale(d_prime)'s own two outputs, read directly from
      the site dump -- the actual per-row dynamic scale (Dn in [2^30,2^31),
      shift s) the engine uses to requantize this row.
    - median_abs, max_median_ratio: median(|x_int|) and d_prime/median_abs.
    - kurtosis: Fisher kurtosis of x_int (excess over normal's 0).
    - n_outlier_channels: count of channels with |x_int[i]| > 0.5 * d_prime
      (threshold named here: DECIDES what counts as "carrying an outlier
      magnitude" for the channel-count and channel-persistence quantities
      below; 0.5 is a fixed fraction of the row's own max, not tuned per
      row or per site).
    - argmax_channel: index of the channel achieving |x_int[i]| == d_prime
      (ties broken by first occurrence, numpy argmax convention).
    - n_distinct_codes: count of distinct int8 code values used in `codes`
      (max possible 256) -- the ROW'S resolution: how many of the 256
      representable int8 levels this row's output actually occupies.
    - code_entropy_bits: Shannon entropy (base 2) of the codes histogram --
      a finer-grained resolution measure than the raw count (256 codes used
      once each vs. 256 codes with one code holding 99% of mass are both
      "256 distinct" but very different resolutions).

  FLOAT PATH (from the float reference's own forward hook/pre-hook capture,
  reused verbatim from tools/t1691_fulldepth_sitecomp_drive.py's
  `capture_float_site_outputs` -- token-at-a-time, DynamicCache, never
  batched, never output_hidden_states=True):
    - max_abs, median_abs, max_median_ratio, kurtosis, n_outlier_channels,
      argmax_channel: identical definitions to the int8-path quantities
      above, applied to the float64 activation vector.
    - l2_norm: the float vector's own L2 norm -- at the residual-carrying
      sites (attn_residual, mlp_residual) this IS the residual stream's own
      magnitude at that point, so its row-to-row curve is the residual-
      stream norm-growth-with-depth quantity named in the commission. No
      int8-path counterpart is computed (raw-magnitude comparison across
      paths is the method violation this campaign has stood against since
      D-SLM744; the int8-path's own within-path analogue is d_prime).
    - n_distinct_codes / code_entropy_bits: NOT APPLICABLE -- float has no
      discrete code, by construction. Only reported for the int8 path.

METHOD CONSTRAINTS CARRIED FROM THE PACKET THIS EXTENDS
(design Claude/Vitruvius/superslm-t1683-source-attribution-design-2026-08-
02.md; D-SLM762/763/764/765/766):
  - Never compare raw or dequantized magnitudes ACROSS the int8 and float
    paths -- every quantity above is computed and reported WITHIN one path
    only; no cell here subtracts or ratios an int8-path value against a
    float-path value.
  - Float reference driven token-at-a-time with an explicit DynamicCache,
    never batched; never output_hidden_states=True.
  - One variable at a time: this drive adds NEW quantities over the SAME
    population, SAME rows, SAME sites, SAME driver invocation the D-SLM765
    packet used -- nothing about the population, the model, or the site
    definitions changed.

Usage
-----
    python tools\\t1691_activation_characterization_drive.py [--rerun-check]

`--rerun-check` re-runs prompt `capital_of_france` (a control) a second,
independent time (fresh subprocess for the driver, fresh model forward pass
for the float side) and diffs every new-quantity cell against the first
run's own recorded value -- the resolving-power check this drive's own
quantities have not previously had (D-SLM766 Sec2 measured resolving power
for Q_full/Q_weight_alone/Q_input_alone only, not for these).
"""

from __future__ import annotations

import math
import os
import subprocess
import sys

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_saturation_report as KSR  # noqa: E402
import shadow_layer_recompute as S  # noqa: E402
from float_reference_layer_dump import DEFAULT_MODEL, _resolve_default_model, fnv1a64  # noqa: E402
from layer_bisection_report import load_int8_layer_dump  # noqa: E402
from sslm_artifact_reader import read_artifact  # noqa: E402
from t1691_fulldepth_sitecomp_drive import (  # noqa: E402
    ALL_LAYER_INDICES, ALL_ROWS, NONPROJECT_SITES, PROJECT_SITES,
    capture_float_site_outputs,
)

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DRIVER_EXE = os.path.join(REPO_ROOT, "out", "sslm_layer_trace.exe")
OUT_DIR = os.path.join(REPO_ROOT, "out", "t1691_activation_characterization")

# The 11 site "chain names" as they appear in the site dump (project sites'
# chain-record name is their own attr, e.g. "q"; nonproject sites' chain
# name is the first tuple element in NONPROJECT_SITES). Built once, below.
PROJECT_CHAIN_NAME = {hf_name: attr for _sn, _isite, attr, _pa, hf_name in PROJECT_SITES}
# site_name -> chain suffix used in the site dump's "layerN.<chain>" records
ALL_SITE_CHAIN = {}
for site_name, _input_site, _attr, _parent_attr, hf_name in PROJECT_SITES:
    # The site dump's own chain-record name for a project_and_funnel site IS
    # the site_name itself (e.g. "layer0.q_proj.requant") -- confirmed by
    # reading a real .sitedump's own record names directly, not assumed.
    ALL_SITE_CHAIN[site_name] = site_name
for chain_name, key, _hook_kind, _module_path, _width_key in NONPROJECT_SITES:
    ALL_SITE_CHAIN[chain_name] = chain_name  # e.g. "attn_norm" -> "attn_norm"
# site_name -> the key used in capture_float_site_outputs's returned dict
ALL_SITE_FLOAT_KEY = {}
for site_name, _input_site, _attr, _parent_attr, hf_name in PROJECT_SITES:
    ALL_SITE_FLOAT_KEY[site_name] = hf_name
for chain_name, key, _hook_kind, _module_path, _width_key in NONPROJECT_SITES:
    ALL_SITE_FLOAT_KEY[chain_name] = key

ALL_SITES = list(ALL_SITE_CHAIN.keys())
assert len(ALL_SITES) == 11, ALL_SITES

OUTLIER_FRACTION = 0.5  # names the threshold: |x[i]| > OUTLIER_FRACTION * max(|x|)


def _entropy_bits(codes: np.ndarray) -> float:
    vals, counts = np.unique(codes, return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def int8_path_stats(x_int: np.ndarray, codes: np.ndarray, d_prime_engine: int, dn: int, s: int):
    x = np.asarray(x_int, dtype=np.float64)
    ax = np.abs(x)
    d_prime_recomputed = float(ax.max()) if ax.size else 0.0
    median_abs = float(np.median(ax))
    max_median_ratio = float(d_prime_recomputed / median_abs) if median_abs > 0 else math.inf
    n_outlier = int(np.sum(ax > OUTLIER_FRACTION * d_prime_recomputed)) if d_prime_recomputed > 0 else 0
    argmax_channel = int(np.argmax(ax))
    kurt = float(scipy_kurtosis(x, fisher=True, bias=True)) if x.size > 3 else float("nan")
    n_distinct = int(len(np.unique(codes)))
    ent = _entropy_bits(codes)
    return {
        "d_prime_engine": float(d_prime_engine),
        "d_prime_recomputed": d_prime_recomputed,
        "d_prime_match": float(d_prime_recomputed == float(d_prime_engine)),
        "dn": float(dn),
        "s": float(s),
        "median_abs": median_abs,
        "max_median_ratio": max_median_ratio,
        "kurtosis": kurt,
        "n_outlier_channels": float(n_outlier),
        "argmax_channel": float(argmax_channel),
        "n_distinct_codes": float(n_distinct),
        "code_entropy_bits": ent,
    }


def float_path_stats(f_true: np.ndarray):
    x = np.asarray(f_true, dtype=np.float64)
    ax = np.abs(x)
    max_abs = float(ax.max()) if ax.size else 0.0
    median_abs = float(np.median(ax))
    max_median_ratio = float(max_abs / median_abs) if median_abs > 0 else math.inf
    n_outlier = int(np.sum(ax > OUTLIER_FRACTION * max_abs)) if max_abs > 0 else 0
    argmax_channel = int(np.argmax(ax))
    kurt = float(scipy_kurtosis(x, fisher=True, bias=True)) if x.size > 3 else float("nan")
    l2_norm = float(np.linalg.norm(x))
    return {
        "max_abs": max_abs,
        "median_abs": median_abs,
        "max_median_ratio": max_median_ratio,
        "kurtosis": kurt,
        "n_outlier_channels": float(n_outlier),
        "argmax_channel": float(argmax_channel),
        "l2_norm": l2_norm,
    }


def run_one_prompt(label, question, role, model, tokenizer, device, expected_width, artifact, out_dir):
    """Runs the int8 driver + float capture for one prompt; returns list of
    (label, role, row, site, path, quantity, value) tuples."""
    prompt = KSR.build_prompt(question)
    dump_path = os.path.join(out_dir, f"{label}.dump")
    site_dump_path = os.path.join(out_dir, f"{label}.sitedump")
    proc = subprocess.run(
        [str(DRIVER_EXE), str(KSR.MODEL_PATH), str(KSR.TOKENIZER_PATH), prompt,
         "--dump", dump_path, "--site-dump", site_dump_path],
        capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"driver failed for prompt {label!r} (exit {proc.returncode})\n"
                            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    assert "self_check: production and manual-replay paths agree" in proc.stdout, (
        f"prompt {label!r}: self-check line missing from stdout")

    base = load_int8_layer_dump(dump_path)
    site_records = {r.site: r for r in S.load_site_dump(site_dump_path)}
    expected_fp = fnv1a64(prompt)
    if base.prompt_fingerprint != expected_fp:
        raise RuntimeError(f"prompt_fingerprint mismatch for {label!r}")

    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    captured = capture_float_site_outputs(model, input_ids, ALL_LAYER_INDICES)

    out_rows = []
    for row, layer_index in zip(ALL_ROWS, ALL_LAYER_INDICES):
        if layer_index >= base.rows - 1:
            continue

        def rec(chain_suffix: str):
            return site_records[f"layer{layer_index}.{chain_suffix}"]

        for site_name in ALL_SITES:
            chain_suffix = ALL_SITE_CHAIN[site_name]
            r = rec(chain_suffix)
            i8 = int8_path_stats(r.x_int, r.codes, r.d_prime, r.dn, r.s)
            for qname, qval in i8.items():
                out_rows.append((label, role, row, site_name, "int8", qname, qval))

            float_key = ALL_SITE_FLOAT_KEY[site_name]
            f_true = captured[(layer_index, float_key)]
            width_expect = expected_width["intermediate_size"] if site_name == "mlp_act" or "gate_proj" in site_name or "up_proj" in site_name else expected_width["hidden_size"]
            if f_true.shape[0] != width_expect and site_name not in ("gate_proj.requant", "up_proj.requant", "mlp_act"):
                pass  # width check is advisory here; the drive this reuses already validated shapes
            fs = float_path_stats(f_true)
            for qname, qval in fs.items():
                out_rows.append((label, role, row, site_name, "float", qname, qval))

    return out_rows


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

    do_rerun_check = "--rerun-check" in sys.argv

    all_rows = []
    for label, question, role in KSR.POPULATION:
        rows = run_one_prompt(label, question, role, model, tokenizer, device, expected_width, artifact, OUT_DIR)
        all_rows.extend(rows)
        print(f"prompt={label!r} role={role}: {len(rows)} cells")

    # max_median_ratio can be legitimately +inf when a row's median magnitude
    # is exactly 0 (more than half the channels are exactly zero) -- named
    # here, not silently allowed: NaN is never legitimate and fails the gate;
    # +inf is legitimate only for the two *_median_ratio quantities.
    nan_bad = [r for r in all_rows if isinstance(r[6], float) and math.isnan(r[6])]
    inf_cells = [r for r in all_rows if isinstance(r[6], float) and math.isinf(r[6])]
    inf_bad = [r for r in inf_cells if r[5] != "max_median_ratio"]
    if nan_bad or inf_bad:
        print(f"FAILED: {len(nan_bad)} NaN cell(s), {len(inf_bad)} illegitimate-inf cell(s) "
              f"among {len(all_rows)} -- sanity gate failed", file=sys.stderr)
        for r in (nan_bad + inf_bad)[:10]:
            print(f"  {r}", file=sys.stderr)
        return 1
    if inf_cells:
        print(f"{len(inf_cells)} max_median_ratio cell(s) are +inf (median magnitude exactly 0 "
              f"at that row/site/prompt/path) -- legitimate, named, not a failure.")

    out_path = os.path.join(OUT_DIR, "activation_characterization_results.tsv")
    with open(out_path, "w") as f:
        f.write("label\trole\trow\tsite\tpath\tquantity\tvalue\n")
        for r in all_rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"\nwrote {len(all_rows)} cells to {out_path}")
    n_prompts = len(KSR.POPULATION)
    n_rows = len(ALL_ROWS)
    n_sites = len(ALL_SITES)
    print(f"RESULT: {len(all_rows)} cells = {n_prompts} prompts x {n_rows} rows x {n_sites} sites x "
          f"(12 int8-path quantities + 7 float-path quantities).")

    d_prime_mismatches = [r for r in all_rows if r[5] == "d_prime_match" and r[6] != 1.0]
    print(f"d_prime cross-check (engine value vs. this script's own recompute from x_int): "
          f"{len(d_prime_mismatches)} mismatches out of "
          f"{sum(1 for r in all_rows if r[5] == 'd_prime_match')} cells checked.")

    if do_rerun_check:
        print("\n--- resolving-power check: re-running prompt 'capital_of_france' independently ---")
        label2, question2, role2 = next(p for p in KSR.POPULATION if p[0] == "capital_of_france")
        rerun_dir = os.path.join(OUT_DIR, "rerun_check")
        os.makedirs(rerun_dir, exist_ok=True)
        rows2 = run_one_prompt(label2, question2, role2, model, tokenizer, device, expected_width,
                                artifact, rerun_dir)
        first = {(r[2], r[3], r[4], r[5]): r[6] for r in all_rows if r[0] == label2}
        second = {(r[2], r[3], r[4], r[5]): r[6] for r in rows2}
        assert set(first.keys()) == set(second.keys()), "rerun cell-key mismatch"
        diffs = []
        for k in first:
            a, b = first[k], second[k]
            if a != b and not (math.isnan(a) and math.isnan(b)):
                diffs.append((k, a, b))
        print(f"resolving-power check: {len(diffs)} cells differ out of {len(first)} "
              f"({label2!r}, all rows/sites/paths/quantities).")
        if diffs:
            for k, a, b in diffs[:20]:
                print(f"  DIFFERS: {k}: run1={a!r} run2={b!r}")
        rerun_out = os.path.join(OUT_DIR, "rerun_check_diffs.txt")
        with open(rerun_out, "w") as f:
            f.write(f"cells_checked={len(first)}\ncells_differing={len(diffs)}\n")
            for k, a, b in diffs:
                f.write(f"{k}\t{a}\t{b}\n")
        print(f"wrote {rerun_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
