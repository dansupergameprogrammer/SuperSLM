#!/usr/bin/env python3
"""T-1740 -- pooled-fidelity report: reads a tools/t1740_pooled_trace.cpp int8
dump and a tools/t1740_pooled_float_dump.py float dump of the SAME prompt,
checks their provenance, dequantizes the int8 side, mean-pools each side's
residual over positions 1..n (excluding position 0, the sequence's FIRST
token -- D-SLM447's ratified encoder scheme), and reports the pooled cosine
defect at every layer boundary plus position 0's own share of pooled
magnitude.

Dequantization: this codebase's own CarriedScale contract
(include/superslm/checked_chain_funnel.h: "value == m * 2^e") is a single
scalar (m, e) pair per captured row, shared across every element of that
row -- so real_value[i] = code[i] * m * 2^e, computed here in float64.

Cosine defect := 1 - cosine_similarity(pooled_int8, pooled_float), so 0 is
perfect agreement (matching D-SLM447's own reporting convention, "0.53%").

Usage
-----
    python tools\\t1740_pooled_fidelity_report.py \\
        --int8 out\\t1740\\capital_of_france.int8.bin \\
        --float out\\t1740\\capital_of_france.float.bin \\
        --label capital_of_france
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


class ProvenanceError(Exception):
    """Raised on a shape or prompt-fingerprint mismatch between the two dumps
    -- a loud, non-zero-exit failure, before any statistic is computed."""


@dataclass
class Int8PooledDump:
    num_positions: int
    num_rows: int
    hidden_size: int
    prompt_fingerprint: int
    # [position, row, 2] int64: columns [m, e]
    scales: np.ndarray
    # [position, row, hidden_size] int8
    codes: np.ndarray


@dataclass
class FloatPooledDump:
    num_positions: int
    num_rows: int
    hidden_size: int
    prompt_fingerprint: int
    # [position, row, hidden_size] float32
    values: np.ndarray


def load_int8_pooled_dump(path) -> Int8PooledDump:
    with open(path, "rb") as f:
        num_positions, num_rows, hidden_size, fingerprint = struct.unpack("<QQQQ", f.read(32))
        scales = np.zeros((num_positions, num_rows, 2), dtype=np.int64)
        codes = np.zeros((num_positions, num_rows, hidden_size), dtype=np.int8)
        for pos in range(num_positions):
            for row in range(num_rows):
                m, e = struct.unpack("<qq", f.read(16))
                scales[pos, row, 0] = m
                scales[pos, row, 1] = e
                codes[pos, row, :] = np.frombuffer(f.read(hidden_size), dtype=np.int8)
    return Int8PooledDump(num_positions, num_rows, hidden_size, fingerprint, scales, codes)


def load_float_pooled_dump(path) -> FloatPooledDump:
    with open(path, "rb") as f:
        num_positions, num_rows, hidden_size, fingerprint = struct.unpack("<QQQQ", f.read(32))
        data = np.frombuffer(f.read(), dtype=np.float32, count=num_positions * num_rows * hidden_size)
    return FloatPooledDump(
        num_positions, num_rows, hidden_size, fingerprint,
        data.reshape(num_positions, num_rows, hidden_size),
    )


def check_provenance(a: Int8PooledDump, b: FloatPooledDump) -> None:
    if a.num_positions != b.num_positions:
        raise ProvenanceError(f"num_positions mismatch: int8={a.num_positions} float={b.num_positions}")
    if a.num_rows != b.num_rows:
        raise ProvenanceError(f"num_rows mismatch: int8={a.num_rows} float={b.num_rows}")
    if a.hidden_size != b.hidden_size:
        raise ProvenanceError(f"hidden_size mismatch: int8={a.hidden_size} float={b.hidden_size}")
    if a.prompt_fingerprint != b.prompt_fingerprint:
        raise ProvenanceError(
            f"prompt_fingerprint mismatch: int8=0x{a.prompt_fingerprint:016X} "
            f"float=0x{b.prompt_fingerprint:016X} -- the two dumps were not generated from the same prompt"
        )


def dequantize(dump: Int8PooledDump) -> np.ndarray:
    """real_value[pos, row, i] = code[pos, row, i] * m[pos, row] * 2^e[pos, row],
    float64. m/e broadcast across hidden_size."""
    codes = dump.codes.astype(np.float64)
    m = dump.scales[:, :, 0].astype(np.float64)[:, :, None]
    e = dump.scales[:, :, 1].astype(np.float64)
    scale = m * np.exp2(e)[:, :, None]
    return codes * scale


def cosine_defect(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    cos = float(np.dot(a, b) / (na * nb))
    return 1.0 - cos


def pooled_vector(values: np.ndarray, row: int, exclude_position_0: bool) -> np.ndarray:
    """values: [num_positions, num_rows, hidden_size]. Mean over positions,
    at the given row (layer boundary)."""
    if exclude_position_0:
        return values[1:, row, :].mean(axis=0)
    return values[:, row, :].mean(axis=0)


def position0_magnitude_share(values: np.ndarray, row: int) -> float:
    """||row for position 0|| / sum_i ||row for position i||, at the given
    row (layer boundary) -- this report's own definition (D-SLM447's own
    record states the resulting percentages but not its internal formula;
    this is the natural reading and is not claimed to be bit-identical to
    that arm's own computation)."""
    norms = np.linalg.norm(values[:, row, :], axis=1)
    total = norms.sum()
    if total == 0.0:
        return float("nan")
    return float(norms[0] / total)


def analyze_one(int8_dump: Int8PooledDump, float_dump: FloatPooledDump, label: str) -> dict:
    check_provenance(int8_dump, float_dump)
    int8_values = dequantize(int8_dump)
    float_values = float_dump.values.astype(np.float64)

    num_rows = int8_dump.num_rows
    per_layer_mean_excl0 = []
    per_layer_mean_incl0 = []
    per_layer_pos0_share_float = []
    per_layer_pos0_share_int8 = []

    for row in range(num_rows):
        pooled_int8_excl0 = pooled_vector(int8_values, row, exclude_position_0=True)
        pooled_float_excl0 = pooled_vector(float_values, row, exclude_position_0=True)
        per_layer_mean_excl0.append(cosine_defect(pooled_int8_excl0, pooled_float_excl0))

        pooled_int8_incl0 = pooled_vector(int8_values, row, exclude_position_0=False)
        pooled_float_incl0 = pooled_vector(float_values, row, exclude_position_0=False)
        per_layer_mean_incl0.append(cosine_defect(pooled_int8_incl0, pooled_float_incl0))

        per_layer_pos0_share_float.append(position0_magnitude_share(float_values, row))
        per_layer_pos0_share_int8.append(position0_magnitude_share(int8_values, row))

    return {
        "label": label,
        "num_positions": int8_dump.num_positions,
        "num_rows": num_rows,
        "mean_excl0": per_layer_mean_excl0,
        "mean_incl0": per_layer_mean_incl0,
        "pos0_share_float": per_layer_pos0_share_float,
        "pos0_share_int8": per_layer_pos0_share_int8,
    }


def print_report(result: dict) -> None:
    n = result["num_positions"]
    final_row = result["num_rows"] - 1
    print(f"\n=== {result['label']} -- {n} positions ===")
    print(f"{'row':<10}{'mean excl. pos0':>18}{'mean incl. pos0':>18}{'pos0 share (float)':>20}"
          f"{'pos0 share (int8)':>20}")
    for row in range(result["num_rows"]):
        layer_name = "embed" if row == 0 else f"layer{row - 1}"
        print(f"{layer_name:<10}{result['mean_excl0'][row]:>18.6f}{result['mean_incl0'][row]:>18.6f}"
              f"{result['pos0_share_float'][row]:>20.6f}{result['pos0_share_int8'][row]:>20.6f}")
    print(f"\nfinal-layer (layer{final_row - 1}) pooled cosine defect, mean excl. pos0: "
          f"{result['mean_excl0'][final_row]:.6%}")
    print(f"final-layer (layer{final_row - 1}) position-0 share of pooled magnitude (float): "
          f"{result['pos0_share_float'][final_row]:.4%}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--int8", required=True, help="path to a tools/t1740_pooled_trace.cpp dump")
    parser.add_argument("--float", required=True, dest="float_path",
                         help="path to a tools/t1740_pooled_float_dump.py dump")
    parser.add_argument("--label", default="prompt")
    args = parser.parse_args(argv)

    int8_dump = load_int8_pooled_dump(args.int8)
    float_dump = load_float_pooled_dump(args.float_path)
    result = analyze_one(int8_dump, float_dump, args.label)
    print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
