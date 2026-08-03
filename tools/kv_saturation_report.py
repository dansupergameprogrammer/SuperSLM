#!/usr/bin/env python3
"""T-1689 -- the K/V-landing saturation census (design Claude/Vitruvius/
superslm-t1683-source-attribution-design-2026-08-02.md S5): does K/V-landing
saturation (a candidate C3 cause) track the mechanism-2/control split T-1683's
layer-bisection instrument already found at layers 21-28?

Drives `tools/sslm_layer_trace.exe` (T-1685, extended by T-1689 to append a
per-layer saturation-delta trailer to its own dump -- see that file's own
header comment) over the SAME nine-prompt population layer_bisection_report.py
already established, and reports the per-layer, per-prompt delta of
`SequenceLayerState::kv_saturation_count` -- the ONE site this campaign can
currently observe without a `src/` change (design S2.1). It does NOT
instrument the RoPE clamp site (design S2.2, uninstrumented, a named and not
closed gap) or the projection sites (design S2.3, argued but not executed to
fail loudly on overflow rather than saturate silently).

Usage
-----
    python tools\\kv_saturation_report.py --out-dir out\\t1689

Threshold (design S9): "K/V saturation is elevated for mechanism-2 prompts"
is measured as the per-layer saturation-event delta, mechanism-2 group (n=4)
versus control group (n=5), at layers 21-28, against each group's OWN
within-group dispersion -- never a single pooled mean (the same discipline
Popper's Attack 2 used on the original locus). This tool reports that
distribution; it asserts no threshold of its own.
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import logit_margin_report as lmr  # T-1683 S3 precedent: one committed prompt-text source.

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVER_EXE = REPO_ROOT / "out" / "sslm_layer_trace.exe"
MODEL_PATH = Path(r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct.sslm")
TOKENIZER_PATH = REPO_ROOT / "tests" / "fixtures" / "qwen2.5-1.5b.tok.sslm"
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# Design S6 population, the SAME ROLE_BY_LABEL grouping layer_bisection_report.py
# already established -- reused verbatim, never re-typed, so a future population
# change to one tool cannot silently diverge from the other (T-1683 S3 precedent).
ROLE_BY_LABEL = {
    "digit_symbol_spaced": "mech2",
    "digit_symbol_nospace": "mech2",
    "plain_language_plus": "mech2",
    "digit_symbol_mult": "mech2",
    "capital_of_germany": "mech1",
    "capital_of_france": "control",
    "capital_of_japan": "control",
    "largest_planet": "control",
    "days_in_week": "control",
}

_PROMPT_TEXT_BY_LABEL = {case.label: case.question for case in lmr.PROMPT_SET}
_missing_labels = [label for label in ROLE_BY_LABEL if label not in _PROMPT_TEXT_BY_LABEL]
if _missing_labels:
    raise SystemExit(
        f"kv_saturation_report.py: {_missing_labels} not found in "
        f"logit_margin_report.PROMPT_SET -- the two tools' prompt sets have diverged")

POPULATION = [(label, _PROMPT_TEXT_BY_LABEL[label], role) for label, role in ROLE_BY_LABEL.items()]

# Design S9/S5: the locus's own band, at which the threshold is measured.
LOCUS_LAYERS = list(range(21, 29))  # 21..28 inclusive


class ProvenanceError(Exception):
    """A shape/fingerprint mismatch or a driver failure -- a loud, non-zero-
    exit failure, matching layer_bisection_report.py's own convention."""


@dataclass
class SaturationCensus:
    rows: int
    hidden_size: int
    prompt_fingerprint: int
    capture_mode: int
    num_layers: int
    deltas: list[int]  # one per layer, 1-indexed layer i is deltas[i-1]


def load_saturation_census(path) -> SaturationCensus:
    """Reads T-1685's existing 29-row body (skipped, not parsed -- this tool
    reports the saturation trailer only) then T-1689's own trailer: uint64
    num_layers, then num_layers uint64 deltas (this file's own extension to
    tools/sslm_layer_trace.cpp, see that file's header comment)."""
    with open(path, "rb") as f:
        rows, hidden_size, prompt_fingerprint, capture_mode = struct.unpack("<QQQQ", f.read(32))
        row_bytes = 8 + 8 + hidden_size
        f.seek(rows * row_bytes, 1)
        (num_layers,) = struct.unpack("<Q", f.read(8))
        deltas = list(struct.unpack(f"<{num_layers}Q", f.read(8 * num_layers)))
    return SaturationCensus(rows=rows, hidden_size=hidden_size, prompt_fingerprint=prompt_fingerprint,
                             capture_mode=capture_mode, num_layers=num_layers, deltas=deltas)


def build_prompt(question: str) -> str:
    """Identical shape to tools/layer_bisection_report.py's own build_prompt."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def run_int8_trace(label: str, question: str, out_dir: Path) -> Path:
    if not DRIVER_EXE.exists():
        raise SystemExit(f"driver not built: {DRIVER_EXE} (run tools\\build_layer_trace.bat)")
    dump_path = out_dir / f"{label}.int8.bin"
    cmd = [str(DRIVER_EXE), str(MODEL_PATH), str(TOKENIZER_PATH), build_prompt(question), "--dump",
           str(dump_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise ProvenanceError(
            f"[{label}] int8 layer-trace failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return dump_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "out" / "t1689"))
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"T-1689 K/V-landing saturation census -- cell: {len(POPULATION)} prompts, position 0 only, "
          f"Qwen2.5-1.5B-Instruct int8 engine, per-layer kv_saturation_count delta "
          f"(K and V landing writes combined, every head)")
    print("=" * 100)

    censuses: dict[str, SaturationCensus] = {}
    groups: dict[str, str] = {}

    for label, question, group in POPULATION:
        groups[label] = group
        print(f"\n--- {label} [{group}] ---")
        print(f"    Q: {question!r}")
        int8_path = run_int8_trace(label, question, out_dir)
        census = load_saturation_census(int8_path)
        if census.num_layers < max(LOCUS_LAYERS):
            raise ProvenanceError(
                f"[{label}] census carries {census.num_layers} layers, need at least "
                f"{max(LOCUS_LAYERS)} to report the locus band {LOCUS_LAYERS}")
        censuses[label] = census
        total = sum(census.deltas)
        locus_total = sum(census.deltas[i - 1] for i in LOCUS_LAYERS)
        print(f"    total saturation events (all {census.num_layers} layers): {total}")
        print(f"    locus-band total (layers {LOCUS_LAYERS[0]}-{LOCUS_LAYERS[-1]}): {locus_total}")
        nonzero_layers = [(i, d) for i, d in enumerate(census.deltas, start=1) if d > 0]
        if nonzero_layers:
            print(f"    nonzero layers: {nonzero_layers}")
        else:
            print(f"    nonzero layers: none")

    # --- execution-level sanity: every reported delta is a real, non-negative --
    # integer read from a successful run (StandardsDocument S5.4's own gate
    # convention, T-1687's own F8-shaped precedent) -- checked before any table
    # below treats these numbers as a result.
    for label, census in censuses.items():
        for i, d in enumerate(census.deltas, start=1):
            if d < 0:
                raise ProvenanceError(f"{label}: layer {i} delta {d} is negative -- impossible for a "
                                       f"monotonically-incremented counter's own non-negative difference")

    print("\n" + "=" * 100)
    print("SUMMARY -- design S9's own threshold cell: per-layer saturation-event delta, "
          f"mechanism-2 (n=4) vs. control (n=5), layers {LOCUS_LAYERS[0]}-{LOCUS_LAYERS[-1]}, "
          "against EACH GROUP'S OWN within-group dispersion (never a single pooled mean)")
    print("=" * 100)

    # Design S9's own threshold wording is "mechanism-2 group (n=4) versus
    # control group (n=5)" -- the SAME n=5 the original locus claim measured
    # (design S1: "the other five prompts hold [0.8257, 0.8867]"), which is
    # mech1 UNION control (4+1=5), not the narrower 4-prompt "control" role
    # layer_bisection_report.py's own ROLE_BY_LABEL reports as a separate
    # bucket for its own diagnostic purposes. Both groupings are reported
    # below: the design's own n=4-vs-n=5 threshold cell, and the finer
    # 3-way mech2/mech1/control breakdown for anyone re-deriving the n=5
    # bucket's own internal composition.
    def _is_mech2(lbl: str) -> bool:
        return groups[lbl] == "mech2"

    print(f"\n{'layer':<8}{'mech2 (n=4)':<28}{'mean':<10}{'std':<10}"
          f"{'control=non-mech2 (n=5)':<32}{'mean':<10}{'std':<10}")
    for layer in LOCUS_LAYERS:
        mech2_vals = [censuses[lbl].deltas[layer - 1] for lbl in censuses if _is_mech2(lbl)]
        control_vals = [censuses[lbl].deltas[layer - 1] for lbl in censuses if not _is_mech2(lbl)]
        m2 = np.array(mech2_vals, dtype=np.float64)
        ct = np.array(control_vals, dtype=np.float64)
        print(f"{layer:<8}{str(mech2_vals):<28}{m2.mean():<10.4f}{m2.std():<10.4f}"
              f"{str(control_vals):<32}{ct.mean():<10.4f}{ct.std():<10.4f}")

    print("\nper-prompt locus-band (layers "
          f"{LOCUS_LAYERS[0]}-{LOCUS_LAYERS[-1]}) total, finer 3-way breakdown "
          "(mech2 / mech1 / control, layer_bisection_report.py's own ROLE_BY_LABEL):")
    for g in ("control", "mech1", "mech2"):
        vals = [sum(censuses[lbl].deltas[i - 1] for i in LOCUS_LAYERS) for lbl in censuses
                if groups[lbl] == g]
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        print(f"  {g:<9} n={len(vals):<3} values={vals} mean={arr.mean():.4f} std={arr.std():.4f}")

    mech2_totals = [sum(censuses[lbl].deltas[i - 1] for i in LOCUS_LAYERS) for lbl in censuses
                     if _is_mech2(lbl)]
    control_totals = [sum(censuses[lbl].deltas[i - 1] for i in LOCUS_LAYERS) for lbl in censuses
                       if not _is_mech2(lbl)]
    print(f"\ndesign S9's own threshold cell -- mech2 n={len(mech2_totals)}, "
          f"control(=non-mech2) n={len(control_totals)}:")
    print("\nPLAIN READING (not a statistical test -- n=4 vs n=5, StandardsDocument S5.4's own "
          "resolving-power discipline: an effect this population cannot resolve is not a result):")
    if all(v == 0 for v in mech2_totals) and all(v == 0 for v in control_totals):
        print("  Zero saturation events in the locus band on EVERY prompt in this population, both "
              "groups. K/V-landing saturation (this one instrumented site) does NOT track the "
              "mechanism-2 group on this population -- narrows C3 to \"not the K/V-landing site, on "
              "this population\" (design S5's own stated boundary). Does not clear C3 generally: the "
              "RoPE clamp site (design S2.2) remains unmeasured.")
    elif max(mech2_totals, default=0) == 0 and max(control_totals, default=0) == 0:
        print("  Zero saturation in both groups -- see above.")
    else:
        m2_arr = np.array(mech2_totals, dtype=np.float64)
        ct_arr = np.array(control_totals, dtype=np.float64)
        print(f"  mech2 locus-band totals: mean={m2_arr.mean():.4f} std={m2_arr.std():.4f}")
        print(f"  control locus-band totals: mean={ct_arr.mean():.4f} std={ct_arr.std():.4f}")
        if m2_arr.mean() > ct_arr.mean() and m2_arr.min() > ct_arr.max():
            print("  mech2 group's saturation is elevated over control with no overlap on this "
                  "population -- consistent with saturation tracking the mechanism-2 group. n=4/n=5: "
                  "this is a candidate, not a certified finding (design S12's own population-size "
                  "boundary; no permutation test is run by this tool).")
        else:
            print("  No clean separation between groups on this population (some overlap, or control "
                  "not below mech2). Inconclusive at this population size -- design S12's own boundary "
                  "applies: n=4 vs n=5 is not enough to certify a null or a positive finding beyond "
                  "\"not separated on THIS population\".")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
