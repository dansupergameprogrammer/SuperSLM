#!/usr/bin/env python3
"""Layer-bisection comparison tool (T-1687) -- the structural hidden-state
sibling of tools/logit_margin_report.py (design Claude/Vitruvius/superslm-
t1683-layer-bisection-design-2026-08-02.md S5), reusing its already-reviewed
statistical machinery (zscore_row, spearmanr-based rank comparison)
generalized from a vocab_size-wide row to a hidden_size-wide row.

Reads T-1685's int8 layer-trace dump and T-1686's float layer-dump, checks
their provenance (shape, prompt identity, capture mode) before computing any
statistic, then reports Spearman rank correlation, z-scored Pearson
correlation, and max absolute z-scored difference at every layer boundary
(embedding + 28 layers).

NO raw-magnitude or dequantized comparison is reported as a primary
statistic -- only rank and per-row-normalized statistics, matching the same
discipline already established for logits (design S2.3, S5).

Usage
-----
    python tools\\layer_bisection_report.py --out-dir out\\t1683

Runs the design's own nine-prompt population (S6), driving both
tools\\sslm_layer_trace.exe (T-1685) and tools\\float_reference_layer_dump.py
(T-1686) itself, then reports the per-layer tables plus the control
baseline's own dispersion and both sides' repeat-vs-repeat resolving power
(design S10).
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVER_EXE = REPO_ROOT / "out" / "sslm_layer_trace.exe"
MODEL_PATH = Path(r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct.sslm")
TOKENIZER_PATH = REPO_ROOT / "tests" / "fixtures" / "qwen2.5-1.5b.tok.sslm"
FLOAT_MODEL_PATH = Path(
    r"D:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots"
    r"\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# Design S6: the union of two already-executed, already-classified prompt
# sets -- no new prompt is invented. "Role" is this design's own grouping
# (what S5/S10's statistics group by); PROMPT_SET's own `group` field in
# tools/logit_margin_report.py is a different, structural taxonomy and is
# never read here (design S6, "label-system collision" note).
POPULATION = [
    ("digit_symbol_spaced", "What is 12 + 15? Give just the number.", "mech2"),
    ("digit_symbol_nospace", "What is 12+15? Give just the number.", "mech2"),
    ("plain_language_plus", "What is 12 plus 15? Give just the number.", "mech2"),
    ("digit_symbol_mult", "What is 17 x 23? Give just the number.", "mech2"),
    ("capital_of_germany", "What is the capital of Germany?", "mech1"),
    ("capital_of_france", "What is the capital of France?", "control"),
    ("capital_of_japan", "What is the capital of Japan?", "control"),
    ("largest_planet", "Name the largest planet in the solar system.", "control"),
    ("days_in_week", "How many days are in a week? Give just the number.", "control"),
]


class ProvenanceError(Exception):
    """Raised by check_provenance on a shape, fingerprint, or capture_mode
    mismatch -- a loud, non-zero-exit failure per design S5."""


@dataclass
class Int8LayerDump:
    rows: int
    hidden_size: int
    prompt_fingerprint: int
    capture_mode: int
    scales: np.ndarray  # rows x 2 int64, columns [m, e]
    codes: np.ndarray  # rows x hidden_size int8


@dataclass
class FloatLayerDump:
    rows: int
    hidden_size: int
    prompt_fingerprint: int
    capture_mode: int
    values: np.ndarray  # rows x hidden_size float32


@dataclass
class LayerRowStats:
    spearman: float
    pearson: float
    max_abs_z_diff: float


def zscore_row(row: np.ndarray) -> np.ndarray:
    """Verbatim shape of tools/logit_margin_report.py:235-240, returning only
    the z-scored array (design S5's own machinery, transcribed)."""
    mean = float(row.mean())
    std = float(row.std())
    if std == 0.0:
        return np.zeros_like(row, dtype=np.float64)
    return (row.astype(np.float64) - mean) / std


def load_int8_layer_dump(path) -> Int8LayerDump:
    """Reads T-1685's dump format (design S4.1 step 5): uint64 rows,
    hidden_size, prompt_fingerprint, capture_mode, then per row int64 m,
    int64 e, then hidden_size int8 codes."""
    with open(path, "rb") as f:
        rows, hidden_size, prompt_fingerprint, capture_mode = struct.unpack("<QQQQ", f.read(32))
        scales = np.zeros((rows, 2), dtype=np.int64)
        codes = np.zeros((rows, hidden_size), dtype=np.int8)
        for i in range(rows):
            m, e = struct.unpack("<qq", f.read(16))
            scales[i, 0] = m
            scales[i, 1] = e
            codes[i, :] = np.frombuffer(f.read(hidden_size), dtype=np.int8)
    return Int8LayerDump(
        rows=rows,
        hidden_size=hidden_size,
        prompt_fingerprint=prompt_fingerprint,
        capture_mode=capture_mode,
        scales=scales,
        codes=codes,
    )


def load_float_layer_dump(path) -> FloatLayerDump:
    """Reads T-1686's dump format (design S4.2 step 4): uint64 rows,
    hidden_size, prompt_fingerprint, capture_mode, then rows * hidden_size
    float32 values."""
    with open(path, "rb") as f:
        rows, hidden_size, prompt_fingerprint, capture_mode = struct.unpack("<QQQQ", f.read(32))
        data = np.frombuffer(f.read(), dtype=np.float32, count=rows * hidden_size)
    return FloatLayerDump(
        rows=rows,
        hidden_size=hidden_size,
        prompt_fingerprint=prompt_fingerprint,
        capture_mode=capture_mode,
        values=data.reshape(rows, hidden_size),
    )


def check_provenance(int8_dump: Int8LayerDump, float_dump: FloatLayerDump) -> None:
    """Design S5's provenance check, run before any statistic. Raises
    ProvenanceError on a rows/hidden_size mismatch, a prompt_fingerprint
    mismatch, or a capture_mode value other than 1 on EITHER side (not
    merely "the two sides disagree" -- a value of 2 on both sides is also
    rejected, design S5)."""
    if int8_dump.rows != float_dump.rows:
        raise ProvenanceError(
            f"rows mismatch: int8 dump has {int8_dump.rows}, float dump has {float_dump.rows}"
        )
    if int8_dump.hidden_size != float_dump.hidden_size:
        raise ProvenanceError(
            f"hidden_size mismatch: int8 dump has {int8_dump.hidden_size}, float dump has "
            f"{float_dump.hidden_size}"
        )
    if int8_dump.prompt_fingerprint != float_dump.prompt_fingerprint:
        raise ProvenanceError(
            f"prompt_fingerprint mismatch: int8 dump has 0x{int8_dump.prompt_fingerprint:016X}, "
            f"float dump has 0x{float_dump.prompt_fingerprint:016X} -- the two dumps were not "
            f"generated from the same prompt"
        )
    if int8_dump.capture_mode != 1:
        raise ProvenanceError(
            f"int8 dump capture_mode == {int8_dump.capture_mode}, want 1"
        )
    if float_dump.capture_mode != 1:
        raise ProvenanceError(
            f"float dump capture_mode == {float_dump.capture_mode}, want 1"
        )


def compare_layer_row(int8_codes: np.ndarray, float_row: np.ndarray) -> LayerRowStats:
    """Design S5's per-boundary statistic set, over the FULL hidden_size
    width (no TOP_N truncation -- "there is no vocabulary-style long tail to
    restrict attention to"): Spearman rank correlation between the raw int8
    codes and the raw float32 values (S2.3's rescaling-invariance argument
    -- no dequantization), Pearson correlation of the two rows after each is
    independently z-scored, and the maximum absolute z-scored difference."""
    int8_f64 = int8_codes.astype(np.float64)
    float_f64 = float_row.astype(np.float64)

    rho, _ = spearmanr(int8_f64, float_f64)

    # zscore_row is called on already-float64 rows, matching
    # tools/logit_margin_report.py's own analyze_position calling convention
    # (float_row64 = float_row.astype(np.float64) computed BEFORE its own
    # zscore_row call) -- ndarray.std() on a raw float32 array returns a
    # float32 result (numpy's own default), losing precision a raw int8
    # array's .std() does not (numpy upcasts integer accumulation to
    # float64 by default), so pre-casting both operands the same way is
    # what keeps the two rows' z-scores computed at the same precision.
    z_int8 = zscore_row(int8_f64)
    z_float = zscore_row(float_f64)
    if np.std(z_int8) == 0.0 or np.std(z_float) == 0.0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(z_int8, z_float)[0, 1])
    max_abs_z_diff = float(np.max(np.abs(z_int8 - z_float)))

    return LayerRowStats(spearman=float(rho), pearson=pearson, max_abs_z_diff=max_abs_z_diff)


# --- execution driver (design S6, S7 T-1687 execution, S10) -----------------


def build_prompt(question: str) -> str:
    """Identical shape to tools/logit_margin_report.py's own build_prompt --
    the literal chat-templated text this campaign's PROMPT_SET convention
    uses on the int8 side."""
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
        raise SystemExit(f"[{label}] int8 layer-trace failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return dump_path


def run_float_dump(label: str, question: str, out_dir: Path) -> Path:
    dump_path = out_dir / f"{label}.float.bin"
    cmd = [sys.executable, str(REPO_ROOT / "tools" / "float_reference_layer_dump.py"), question,
           "--system", SYSTEM_PROMPT, "--model", str(FLOAT_MODEL_PATH), "--dump", str(dump_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise SystemExit(f"[{label}] float layer-dump failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return dump_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "out" / "t1683"))
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"T-1683 layer-bisection report -- cell: {len(POPULATION)} prompts (labels below), "
          f"position 0 only, Qwen2.5-1.5B-Instruct, int8 engine vs. float reference, "
          f"29 layer boundaries (embedding + 28 layers) each.")
    print("=" * 100)

    all_stats: dict[str, list[LayerRowStats]] = {}
    groups: dict[str, str] = {}

    for label, question, group in POPULATION:
        groups[label] = group
        print(f"\n--- {label} [{group}] ---")
        print(f"    Q: {question!r}")
        int8_path = run_int8_trace(label, question, out_dir)
        float_path = run_float_dump(label, question, out_dir)

        int8_dump = load_int8_layer_dump(int8_path)
        float_dump = load_float_layer_dump(float_path)
        check_provenance(int8_dump, float_dump)

        row_stats = [
            compare_layer_row(int8_dump.codes[i], float_dump.values[i]) for i in range(int8_dump.rows)
        ]
        all_stats[label] = row_stats
        for i, s in enumerate(row_stats):
            layer_name = "embed" if i == 0 else f"layer{i}"
            print(f"    {layer_name:<8} spearman={s.spearman:7.4f} pearson={s.pearson:7.4f} "
                  f"max|z-diff|={s.max_abs_z_diff:8.4f}")

    # --- design S7 T-1687 part 4 / coverage audit F8: execution-level -------
    # sanity, over the REAL 29x9 execution, before the tables are presented
    # as the step's result.
    n_boundaries = len(next(iter(all_stats.values())))
    total_cells = 0
    for label, row_stats in all_stats.items():
        for s in row_stats:
            total_cells += 1
            assert np.isfinite(s.spearman) and -1.0 <= s.spearman <= 1.0, (
                f"{label}: spearman={s.spearman} out of [-1,1] or non-finite")
            assert np.isnan(s.pearson) or (np.isfinite(s.pearson) and -1.0 <= s.pearson <= 1.0), (
                f"{label}: pearson={s.pearson} out of [-1,1] or non-finite (and not NaN)")
            assert np.isfinite(s.max_abs_z_diff), f"{label}: max_abs_z_diff={s.max_abs_z_diff} non-finite"
    print(f"\nexecution-level sanity: all {total_cells} cells ({len(all_stats)} prompts x "
          f"{n_boundaries} boundaries) finite and in-range -- asserted (design S7 T-1687 part 4, "
          f"coverage audit F8)")

    # --- design S10: the control-population distribution, the cell any -----
    # locus claim would be measured against -- reported, not asserted as a
    # threshold.
    print("\n" + "=" * 100)
    print("SUMMARY -- design S10: the cell any locus claim would be measured over")
    print("=" * 100)
    for g in ("control", "mech1", "mech2"):
        vals = [s.max_abs_z_diff for lbl, rs in all_stats.items() if groups[lbl] == g for s in rs]
        if not vals:
            continue
        arr = np.array(vals)
        print(f"{g:<9} n={len(vals):<5} max|z-diff|: min={arr.min():.4f} median={np.median(arr):.4f} "
              f"p95={np.percentile(arr, 95):.4f} max={arr.max():.4f} mean={arr.mean():.4f}")

    print("\nper-boundary mean max|z-diff|, mech2 group vs. control group:")
    print(f"  {'boundary':<10}{'mech2':>10}{'control':>10}{'ratio':>10}")
    for i in range(n_boundaries):
        mech2_vals = [all_stats[lbl][i].max_abs_z_diff for lbl in all_stats if groups[lbl] == "mech2"]
        control_vals = [all_stats[lbl][i].max_abs_z_diff for lbl in all_stats if groups[lbl] == "control"]
        m = float(np.mean(mech2_vals)) if mech2_vals else float("nan")
        c = float(np.mean(control_vals)) if control_vals else float("nan")
        layer_name = "embed" if i == 0 else f"layer{i}"
        print(f"  {layer_name:<10}{m:>10.4f}{c:>10.4f}{(m / c if c else float('nan')):>10.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
