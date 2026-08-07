#!/usr/bin/env python3
"""T-1690 witness report -- drives the third-witness composition
(`tools/independent_layer_reference.py`) across the T-1683 campaign's own
nine-prompt population, validates it against the existing library-based
float reference (T-1686) at an early, uncontested layer boundary, then
compares it against the int8 engine's own captured hidden states (T-1685)
at layers 21-28 -- the same shape T-1687's `layer_bisection_report.py`
already established for the int8-vs-float comparison, reused here for its
comparison MACHINERY (Spearman rank correlation, a generic statistic) only
-- never for its arithmetic composition, which is what this witness exists
to be independent of.

Comparison rule (design S10, S6's own threshold spec, S9): Spearman rank
correlation ONLY, never raw or dequantized magnitude. int8 codes are
compared directly (a positive per-row scale is rank-invariant, the same
"no dequantization needed for rank statistics" argument
`layer_bisection_report.py` already uses).

Population (design S9's own threshold; the exact split D-SLM708 measured,
Claude/Decisions/DecisionLog.md):
  mechanism-2 (n=4): digit_symbol_spaced, digit_symbol_nospace,
    plain_language_plus, digit_symbol_mult
  the other five  (n=5): capital_of_germany, capital_of_france,
    capital_of_japan, largest_planet, days_in_week
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
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
LAYER_TRACE_EXE = REPO_ROOT / "out" / "sslm_layer_trace.exe"
MODEL_PATH = Path(r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct.sslm")
TOKENIZER_PATH = REPO_ROOT / "tests" / "fixtures" / "qwen2.5-1.5b.tok.sslm"
FLOAT_MODEL_PATH = Path(
    r"D:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots"
    r"\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

PROMPTS = {
    "digit_symbol_spaced": "What is 12 + 15? Give just the number.",
    "digit_symbol_nospace": "What is 12+15? Give just the number.",
    "plain_language_plus": "What is 12 plus 15? Give just the number.",
    "digit_symbol_mult": "What is 17 x 23? Give just the number.",
    "capital_of_germany": "What is the capital of Germany?",
    "capital_of_france": "What is the capital of France?",
    "capital_of_japan": "What is the capital of Japan?",
    "largest_planet": "Name the largest planet in the solar system.",
    "days_in_week": "How many days are in a week? Give just the number.",
}
MECH2 = ["digit_symbol_spaced", "digit_symbol_nospace", "plain_language_plus", "digit_symbol_mult"]
CONTROL5 = ["capital_of_germany", "capital_of_france", "capital_of_japan", "largest_planet", "days_in_week"]

VALIDATION_LAYERS = [1, 2, 3, 4, 5]
LOCUS_LAYERS = list(range(21, 29))


def build_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


@dataclass
class Int8Dump:
    rows: int
    hidden_size: int
    fingerprint: int
    capture_mode: int
    codes: np.ndarray  # rows x hidden_size int8


@dataclass
class FloatDump:
    rows: int
    hidden_size: int
    fingerprint: int
    capture_mode: int
    values: np.ndarray  # rows x hidden_size float32


def load_int8_dump(path: Path) -> Int8Dump:
    with open(path, "rb") as f:
        rows, hidden_size, fingerprint, capture_mode = struct.unpack("<QQQQ", f.read(32))
        codes = np.zeros((rows, hidden_size), dtype=np.int8)
        for i in range(rows):
            m, e = struct.unpack("<qq", f.read(16))
            codes[i, :] = np.frombuffer(f.read(hidden_size), dtype=np.int8)
    return Int8Dump(rows, hidden_size, fingerprint, capture_mode, codes)


def load_float_dump(path: Path) -> FloatDump:
    with open(path, "rb") as f:
        rows, hidden_size, fingerprint, capture_mode = struct.unpack("<QQQQ", f.read(32))
        data = np.frombuffer(f.read(), dtype=np.float32, count=rows * hidden_size)
    return FloatDump(rows, hidden_size, fingerprint, capture_mode, data.reshape(rows, hidden_size))


def check_provenance(a, b, label: str):
    if a.rows != b.rows or a.hidden_size != b.hidden_size:
        raise SystemExit(f"[{label}] shape mismatch: {a.rows}x{a.hidden_size} vs {b.rows}x{b.hidden_size}")
    if a.fingerprint != b.fingerprint:
        raise SystemExit(f"[{label}] prompt fingerprint mismatch: 0x{a.fingerprint:016X} vs 0x{b.fingerprint:016X}")
    if a.capture_mode != 1 or b.capture_mode != 1:
        raise SystemExit(f"[{label}] capture_mode != 1 (incremental) on at least one side")


def run_int8_trace(label: str, question: str, out_dir: Path) -> Path:
    dump_path = out_dir / f"{label}.int8.bin"
    if not LAYER_TRACE_EXE.exists():
        raise SystemExit(f"driver not built: {LAYER_TRACE_EXE} (run tools\\build_layer_trace.bat)")
    prompt = build_prompt(question)
    cmd = [str(LAYER_TRACE_EXE), str(MODEL_PATH), str(TOKENIZER_PATH), prompt, "--dump", str(dump_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise SystemExit(f"[{label}] sslm_layer_trace failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return dump_path


def run_float_dump(label: str, question: str, out_dir: Path) -> Path:
    dump_path = out_dir / f"{label}.float.bin"
    cmd = [
        sys.executable, str(REPO_ROOT / "tools" / "float_reference_layer_dump.py"), question,
        "--system", SYSTEM_PROMPT, "--model", str(FLOAT_MODEL_PATH), "--dump", str(dump_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise SystemExit(f"[{label}] float_reference_layer_dump failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return dump_path


def run_witness_dump(label: str, question: str, out_dir: Path, dtype: str) -> Path:
    dump_path = out_dir / f"{label}.witness_{dtype}.bin"
    cmd = [
        sys.executable, str(REPO_ROOT / "tools" / "independent_layer_reference.py"), question,
        "--system", SYSTEM_PROMPT, "--model", str(FLOAT_MODEL_PATH), "--dtype", dtype, "--dump", str(dump_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise SystemExit(f"[{label}] independent_layer_reference failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return dump_path


def spearman_int8_vs_float(int8_row: np.ndarray, float_row: np.ndarray) -> float:
    rho, _ = spearmanr(int8_row.astype(np.float64), float_row.astype(np.float64))
    return float(rho)


def spearman_float_vs_float(row_a: np.ndarray, row_b: np.ndarray) -> float:
    rho, _ = spearmanr(row_a.astype(np.float64), row_b.astype(np.float64))
    return float(rho)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "out" / "t1690"))
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Validation gate (design S6): layers 1-5 vs the library-based
    # float reference, on one representative prompt, both witness dtypes.
    if not args.skip_validation:
        print("=== VALIDATION GATE: witness vs. library-based float reference, layers 1-5 ===")
        label = "capital_of_france"
        question = PROMPTS[label]
        float_path = run_float_dump(label, question, out_dir)
        float_dump = load_float_dump(float_path)
        sanity_fail = False
        for dtype in ("bfloat16", "float32"):
            witness_path = run_witness_dump(label, question, out_dir, dtype)
            witness_dump = load_float_dump(witness_path)
            check_provenance(float_dump, witness_dump, f"validation/{dtype}")
            print(f"  dtype={dtype}:")
            for layer in VALIDATION_LAYERS:
                rho = spearman_float_vs_float(witness_dump.values[layer], float_dump.values[layer])
                flag = "" if rho > 0.9 else "  <-- LOW"
                if not np.isfinite(rho):
                    sanity_fail = True
                print(f"    layer {layer:2d}: spearman(witness, float_reference) = {rho:.6f}{flag}")
        if sanity_fail:
            raise SystemExit("VALIDATION SANITY FAILED: non-finite Spearman value")
        print()

    # ---- Headline: witness vs. int8 engine, layers 21-28, all nine prompts,
    # both witness dtypes.
    print("=== HEADLINE: witness vs. int8 engine, layers 21-28, nine-prompt population ===")
    per_prompt_per_layer: dict[str, dict[int, dict[str, float]]] = {}
    for label, question in PROMPTS.items():
        int8_path = run_int8_trace(label, question, out_dir)
        int8_dump = load_int8_dump(int8_path)
        per_prompt_per_layer[label] = {}
        for dtype in ("bfloat16", "float32"):
            witness_path = run_witness_dump(label, question, out_dir, dtype)
            witness_dump = load_float_dump(witness_path)
            check_provenance(int8_dump, witness_dump, f"{label}/{dtype}")
            for layer in LOCUS_LAYERS + [5]:
                rho = spearman_int8_vs_float(int8_dump.codes[layer], witness_dump.values[layer])
                per_prompt_per_layer[label].setdefault(layer, {})[dtype] = rho

    for dtype in ("bfloat16", "float32"):
        print(f"\n--- dtype={dtype} ---")
        print(f"  {'prompt':<22}{'group':<8}" + "".join(f"L{l:<5}" for l in LOCUS_LAYERS) + "  L5(ctrl-band)")
        for label in list(PROMPTS.keys()):
            group = "mech2" if label in MECH2 else "other5"
            row = per_prompt_per_layer[label]
            vals = "".join(f"{row[l][dtype]:6.3f}" for l in LOCUS_LAYERS)
            l5 = row[5][dtype]
            print(f"  {label:<22}{group:<8}{vals}  {l5:6.3f}")

        for layer in LOCUS_LAYERS:
            mech2_vals = [per_prompt_per_layer[l][layer][dtype] for l in MECH2]
            other_vals = [per_prompt_per_layer[l][layer][dtype] for l in CONTROL5]
            mech2_range = (min(mech2_vals), max(mech2_vals))
            other_range = (min(other_vals), max(other_vals))
            gap = other_range[0] - mech2_range[1]
            print(
                f"  layer {layer}: mech2 [{mech2_range[0]:.4f}, {mech2_range[1]:.4f}]  "
                f"other5 [{other_range[0]:.4f}, {other_range[1]:.4f}]  gap={gap:+.4f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
