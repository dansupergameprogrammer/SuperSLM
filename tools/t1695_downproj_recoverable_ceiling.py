#!/usr/bin/env python3
"""T-1695 -- the mitigation-ceiling measurement (T-1683/T-1691 campaign's own
owed question, `Claude/Zelda/RESUME-2026-08-04.md`): every campaign result to
date CLOSES a hypothesis (ruled out); nothing has ever measured how much of
the int8-vs-float divergence is reachable by an intervention shaped like a
real mitigation. This is that measurement, at the ONE site D-SLM767 pinned
the outlier-crushing mechanism to: `down_proj.requant`'s own dynamic per-row
max-abs funnel (`RequantChainChecked`, `src/forward/checked_chain_funnel.cpp`).

Three arms, all SITE-LOCAL (this drive does not chain across layers -- see
"scope" below), computed from the SAME real captured accumulator
(`down_proj.requant`'s own `x_int`, the wide int64 row `RequantChainChecked`
receives, read directly from the site dump -- never reconstructed, since
D-SLM772-774 already proved the parity shadow reproduces it bit-exact at all
28 rows) and compared against the SAME float64 reference:

  * **Arm A** (baseline, "full int8"): the real engine's own captured
    `down_proj.requant` output codes and `d_prime`, dequantized back to the
    wide-row domain: `A[i] = codes[i] * d_prime / 127`.
  * **Arm B** (reference, "full float64"): `shadow_layer_recompute.
    precision_shadow_layer` -- the campaign's own existing, already-validated
    instrument, unmodified -- run on the SAME real captured `mlp_act` input
    codes through the SAME `down_proj` weight/fold, in float64, with no
    intermediate int8 requantization anywhere.
  * **Arm C** (NEW -- the point of this task): the real captured accumulator
    `x_int` used DIRECTLY, with `down_proj.requant`'s own funnel held out
    entirely (no 127-level quantization at this site at all). Bit-identical
    int8 arithmetic everywhere else (GEMM, weight-scale fold) is untouched --
    this arm changes nothing upstream of the funnel and nothing downstream is
    computed at all.
  * **Arm D** (NEW, narrower): the SAME funnel, but the per-row max-abs scale
    (`MaxAbsReduceWide`) is derived EXCLUDING channel 609's own contribution
    (D-SLM767's own dominant locus-band channel, 68.1% of the locus band) --
    channel 609 itself still gets quantized and clips under the resulting
    narrower scale; every other channel gets the resolution channel 609 was
    stealing. This is the direct proxy for what an offline per-channel
    outlier migration targeting channel 609 alone would buy.

Recoverable fraction, per (row, prompt): `(err_A - err_X) / err_A`, where
`err_X = RMSE(X_recon, B)` over the row's hidden_size channels -- the
fraction of the int8 arm's own distance from the float reference that the
ablation closes. Both RMSE and max-abs-error are reported (5.4's "raw
numbers, not just ratios").

SCOPE -- what this measures and what it does not
--------------------------------------------------
This is a SITE-LOCAL measurement. `precision_shadow_layer` takes a captured
seed row and replays ONE site; it does not chain across the 28 layers. Arms
A/C/D all consume the SAME real captured `mlp_act` codes (Arm A via the real
engine's own already-computed `x_int`, Arm B/C/D via `precision_shadow_layer`
or the same `x_int`) -- so this measures how much of `down_proj.requant`'s
OWN quantization damage, AT THIS SITE, given a fixed real input, an ablation
of this shape could recover. It does NOT measure end-to-end output recovery:
recovering this site's damage does not by itself say what happens to the
NEXT 2-7 layers that would consume a less-damaged `down_proj.requant` output,
because nothing here re-runs the forward pass with the ablation live and
propagates it. Chaining that would cost: an ablated forward pass (either a
build-time engine flag or a much larger Python re-implementation of every
remaining site kind's forward composition, currently only assembled
site-by-site across five different real-population drives), re-run over the
9-prompt population at all 28 rows, plus a second exact-parity proof that the
propagated ablation composes correctly with the sites downstream of
`down_proj` -- not attempted here, out of this task's scope.

The known `q_proj` per-channel bias omission in `precision_shadow_layer`
(named in `Claude/Zelda/RESUME-2026-08-04.md`'s "Owed" table) does NOT touch
this measurement: `down_proj` has no bias (`project_and_funnel_shadow_codes`
callers for gate/up/down all pass `bias=None`; only the real engine's
`q_proj` call site supplies one), and this drive never constructs a
`q_proj` site at all.

Usage: python tools\\t1695_downproj_recoverable_ceiling.py
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_saturation_report as KSR  # noqa: E402 -- the established 9-prompt population + paths
import shadow_layer_recompute as S  # noqa: E402
from sslm_artifact_reader import read_artifact  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DRIVER_EXE = os.path.join(REPO_ROOT, "out", "sslm_layer_trace.exe")
OUT_DIR = os.path.join(REPO_ROOT, "out", "t1695_downproj_ceiling")

# All 28 rows -- the site-dump band this branch's sslm_layer_trace.cpp
# carries covers all 28 (kSiteDumpTargetRows, verified at build time, see the
# build log's "population band" section). Reversal band (12-22) first, per
# the campaign's own established priority convention (t1691_mlp_drive.py),
# so a run that stops partway still covers the region D-SLM767's own
# reversal-band question needs.
_PRIORITY_ROWS = list(range(12, 23))  # 12..22
TARGET_ROWS = _PRIORITY_ROWS + [r for r in range(1, 29) if r not in _PRIORITY_ROWS]
TARGET_LAYER_INDICES = [row - 1 for row in TARGET_ROWS]

OUTLIER_CHANNEL = 609  # D-SLM767 -- 0-indexed, the down_proj.requant output
# channel (of hidden_size) that takes 68.1% of the locus-band cells.


def funnel_with_holdout(wide_row: np.ndarray, exclude_idx: int | None) -> tuple[int, np.ndarray]:
    """The SAME `RequantChainChecked` composition
    (`normalize_scale_reference` -> `dynamic_scale_reciprocal_reference` ->
    `requant_token_code_wide_reference` per element) `requant_chain_reference`
    already implements, except `MaxAbsReduceWide`'s own row is computed with
    `exclude_idx` removed when supplied -- so the derived scale is set by
    every OTHER channel, and `exclude_idx` itself is still quantized (and
    clips) under that narrower scale, exactly like every other channel.
    `exclude_idx=None` reproduces the real engine's own d_prime exactly (used
    as this function's own fault-injection self-check, see
    `fault_injection_check`)."""
    row = [int(x) for x in np.asarray(wide_row).tolist()]
    if exclude_idx is None:
        reduce_row = row
    else:
        reduce_row = row[:exclude_idx] + row[exclude_idx + 1:]
    d_prime = S.max_abs_reduce_wide_reference(np.array(reduce_row, dtype=object))
    dn, s = S.normalize_scale_reference(d_prime)
    r = S.dynamic_scale_reciprocal_reference(dn)
    codes = np.array(
        [S.requant_token_code_wide_reference(x_i, r, s) for x_i in row], dtype=np.int64
    )
    return d_prime, codes


def dequant(codes: np.ndarray, d_prime: int) -> np.ndarray:
    """The funnel's own intended inverse: `RequantTokenCodeWide` maps
    `x_i -> round(127 * x_i / d_prime)` (implemented via the fixed-point
    `r`/`s` reciprocal, `requant_token_code_wide_reference`'s own docstring);
    this is that map's algebraic inverse, `code -> code * d_prime / 127`,
    applied elementwise. A lossy reconstruction by construction -- that
    lossiness at 127 levels is exactly what this drive measures."""
    return codes.astype(np.float64) * (float(d_prime) / 127.0)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def max_abs_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def compute_arms(wide_row: np.ndarray, d_prime_real: int, codes_real: np.ndarray,
                  act_codes_real: np.ndarray, down_site: S.SiteFoldTriple):
    """Computes B (float64 reference), A_recon (baseline int8, dequantized),
    C (funnel held out entirely), D_recon (funnel with channel 609 excluded
    from the scale derivation, dequantized). Returns
    (B, A_recon, C, D_recon, d_prime_D)."""
    b = S.precision_shadow_layer(act_codes_real, [down_site])
    a_recon = dequant(codes_real, d_prime_real)
    c = wide_row.astype(np.float64)
    d_prime_d, codes_d = funnel_with_holdout(wide_row, OUTLIER_CHANNEL)
    d_recon = dequant(codes_d, d_prime_d)
    return b, a_recon, c, d_recon, d_prime_d


def fault_injection_check(wide_row: np.ndarray, d_prime_real: int, codes_real: np.ndarray,
                           act_codes_real: np.ndarray, down_site: S.SiteFoldTriple) -> None:
    """Proves the two ablations are LIVE -- a silently no-op hold-out would
    report 0% recoverable and look like a real finding (the task's own named
    trap). Two checks, both hard asserts (loud, non-silent):

    1. A deliberately BROKEN Arm C -- one that claims to hold out the funnel
       but actually re-applies it (i.e. is secretly just Arm A again) --
       must show recoverable_C == 0 (to float rounding), while the REAL Arm C
       must show a materially different (here: strictly smaller, since C
       removes a lossy quantization step) error against B. If the two are
       equal, the ablation is not live and this function raises.
    2. A deliberately BROKEN Arm D -- one that claims to exclude channel 609
       but passes `exclude_idx=None` (i.e. is secretly Arm A's own funnel
       again) -- must reproduce the real engine's OWN d_prime and codes
       exactly (this doubles as a correctness check on `funnel_with_holdout`
       itself: with no exclusion, it must exactly reproduce
       `RequantChainChecked`, which the real engine's own captured
       `d_prime_real`/`codes_real` already are, by C1/D-SLM772-774). The
       REAL Arm D (excluding 609) must differ from this broken reproduction
       whenever channel 609 is not already zero in this row -- checked
       explicitly, not assumed.
    """
    b = S.precision_shadow_layer(act_codes_real, [down_site])

    # --- Check 1: broken Arm C (secretly re-applies the funnel) ---
    broken_c_recon = dequant(codes_real, d_prime_real)  # == Arm A_recon, i.e. a no-op hold-out
    real_c = wide_row.astype(np.float64)
    err_a = rmse(dequant(codes_real, d_prime_real), b)
    err_broken_c = rmse(broken_c_recon, b)
    err_real_c = rmse(real_c, b)
    recoverable_broken_c = (err_a - err_broken_c) / err_a if err_a > 0 else float("nan")
    if abs(recoverable_broken_c) > 1e-9:
        raise AssertionError(
            f"fault_injection_check: broken (no-op) Arm C should recover exactly 0%, "
            f"measured {recoverable_broken_c!r} -- the broken-detector itself is broken")
    if err_real_c == err_broken_c:
        raise AssertionError(
            "fault_injection_check: real Arm C (funnel held out) is numerically IDENTICAL to "
            "the broken (no-op) Arm C -- the hold-out is not live")

    # --- Check 2: broken Arm D (secretly passes exclude_idx=None) ---
    d_prime_broken, codes_broken = funnel_with_holdout(wide_row, None)
    if d_prime_broken != d_prime_real or not np.array_equal(codes_broken, codes_real):
        raise AssertionError(
            f"fault_injection_check: funnel_with_holdout(exclude_idx=None) does not reproduce "
            f"the real engine's own d_prime/codes exactly -- funnel_with_holdout itself is "
            f"wrong (d_prime_broken={d_prime_broken}, d_prime_real={d_prime_real}, "
            f"codes differ={not np.array_equal(codes_broken, codes_real)})")
    row_list = [int(x) for x in np.asarray(wide_row).tolist()]
    channel_609_magnitude = abs(row_list[OUTLIER_CHANNEL])
    d_prime_real_ablated, codes_real_ablated = funnel_with_holdout(wide_row, OUTLIER_CHANNEL)
    if channel_609_magnitude > 0 and d_prime_real_ablated == d_prime_broken:
        raise AssertionError(
            f"fault_injection_check: excluding channel {OUTLIER_CHANNEL} (magnitude "
            f"{channel_609_magnitude}) did not change d_prime -- the exclusion is not live "
            f"on this row")

    print(f"  fault injection check: PASS -- broken Arm C detected (0.0% recoverable, real "
          f"Arm C={100*((err_a - err_real_c) / err_a if err_a > 0 else float('nan')):.2f}%); "
          f"broken Arm D reproduces the real funnel exactly; excluding channel {OUTLIER_CHANNEL} "
          f"(|x|={channel_609_magnitude}) changes d_prime "
          f"({d_prime_broken} -> {d_prime_real_ablated})")


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isfile(DRIVER_EXE):
        print(f"FAILED: {DRIVER_EXE} not built -- run tools\\build_layer_trace.bat", file=sys.stderr)
        return 1
    if not os.path.isfile(KSR.MODEL_PATH):
        print(f"FAILED: real artifact not found at {KSR.MODEL_PATH}", file=sys.stderr)
        return 1

    artifact = read_artifact(str(KSR.MODEL_PATH))
    print(f"population band: rows {sorted(TARGET_ROWS)} (all 28) -- verified against "
          f"sslm_layer_trace.cpp's kSiteDumpTargetRows before this run (build log)")

    rows_out = []
    checked_fault_injection = False

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

        site_records = {r.site: r for r in S.load_site_dump(site_dump_path)}

        for row, layer_index in zip(TARGET_ROWS, TARGET_LAYER_INDICES):
            layer = artifact.layers[layer_index]

            def rec(name: str):
                return site_records[f"layer{layer_index}.{name}"]

            down_rec = rec("down_proj.requant")
            act_rec = rec("mlp_act")

            down_site = S.SiteFoldTriple(
                kind="down_proj", w_int8=layer.down_weight,
                identity=layer.down_fold[:, 0], mult=layer.down_fold[:, 1], shift=layer.down_fold[:, 2])

            # Sanity cross-check (load-bearing): the funnel with no exclusion
            # must reproduce the real engine's own d_prime exactly -- a
            # mismatch here means x_int/d_prime were parsed or indexed wrong.
            d_prime_check = S.max_abs_reduce_wide_reference(down_rec.x_int)
            if d_prime_check != down_rec.d_prime:
                print(f"FAILED: prompt={label!r} row={row}: recomputed d_prime {d_prime_check} "
                      f"!= real engine's own {down_rec.d_prime} -- SiteRecord parsing suspect",
                      file=sys.stderr)
                return 1

            # Run the fault-injection self-check once, at a cell where channel
            # 609 is actually known to dominate (D-SLM767: the locus band,
            # rows 25-28, mech2 prompts) -- picking an early reversal-band
            # row here would silently pass check 2 as a no-op (channel 609
            # is not yet the argmax that early), which is exactly the kind
            # of vacuously-passing check this campaign's own defect class
            # warns about. row 26 is inside D-SLM767's own stated locus band.
            if not checked_fault_injection and role == "mech2" and row == 26:
                fault_injection_check(down_rec.x_int, down_rec.d_prime, down_rec.codes,
                                       act_rec.codes, down_site)
                checked_fault_injection = True

            b, a_recon, c, d_recon, d_prime_d = compute_arms(
                down_rec.x_int, down_rec.d_prime, down_rec.codes, act_rec.codes, down_site)

            err_a_rmse, err_c_rmse, err_d_rmse = rmse(a_recon, b), rmse(c, b), rmse(d_recon, b)
            err_a_max, err_c_max, err_d_max = max_abs_err(a_recon, b), max_abs_err(c, b), max_abs_err(d_recon, b)

            rec_c_rmse = (err_a_rmse - err_c_rmse) / err_a_rmse if err_a_rmse > 0 else float("nan")
            rec_d_rmse = (err_a_rmse - err_d_rmse) / err_a_rmse if err_a_rmse > 0 else float("nan")
            rec_c_max = (err_a_max - err_c_max) / err_a_max if err_a_max > 0 else float("nan")
            rec_d_max = (err_a_max - err_d_max) / err_a_max if err_a_max > 0 else float("nan")

            rows_out.append(dict(
                label=label, role=role, row=row, layer_index=layer_index,
                err_a_rmse=err_a_rmse, err_c_rmse=err_c_rmse, err_d_rmse=err_d_rmse,
                err_a_max=err_a_max, err_c_max=err_c_max, err_d_max=err_d_max,
                recoverable_c_rmse=rec_c_rmse, recoverable_d_rmse=rec_d_rmse,
                recoverable_c_max=rec_c_max, recoverable_d_max=rec_d_max,
                d_prime_a=down_rec.d_prime, d_prime_d=d_prime_d,
            ))

        print(f"prompt={label!r} role={role}: {len(TARGET_ROWS)} rows computed -- OK")

    if not checked_fault_injection:
        print("FAILED: fault_injection_check never ran (no mech2 prompt reached row 26) -- "
              "the population or row band changed underneath this drive", file=sys.stderr)
        return 1

    tsv_path = os.path.join(OUT_DIR, "results.tsv")
    cols = ["label", "role", "row", "layer_index", "err_a_rmse", "err_c_rmse", "err_d_rmse",
            "err_a_max", "err_c_max", "err_d_max", "recoverable_c_rmse", "recoverable_d_rmse",
            "recoverable_c_max", "recoverable_d_max", "d_prime_a", "d_prime_d"]
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows_out:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    print(f"RESULT: {len(rows_out)} cells written to {tsv_path} "
          f"({len(KSR.POPULATION)} prompts x {len(TARGET_ROWS)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
