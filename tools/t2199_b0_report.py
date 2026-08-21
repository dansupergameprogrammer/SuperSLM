"""Aggregate commissioned T-2199 B0 probe cells into the ruled 63-row reporting table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


ALPHAS = (0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0)
ORDERS = (1, 2, 3)
WIDTHS = (3, 6, 10)
MODEL_GOVERNS_M = 2.0
STABILITY_BOUND = 10.0


@dataclass
class Cell:
    label: str
    probe: Path
    baseline: Path
    length: int


def median(values):
    return statistics.median(values)


def parse_cell_spec(value: str) -> Cell:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("cell must be label=probe.tsv,baseline.jsonl,length")
    fields = parts[1].split(",")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("cell must be label=probe.tsv,baseline.jsonl,length")
    return Cell(parts[0], Path(fields[0]), Path(fields[1]), int(fields[2]))


def load_baseline(cell: Cell):
    result = {}
    for line in cell.baseline.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["max_new_tokens"] != cell.length:
            continue
        result[row["prompt_id"] + ".logits"] = row
    if len(result) != 48:
        raise RuntimeError(f"{cell.label}: expected 48 baseline rows, got {len(result)}")
    if any(row.get("error") for row in result.values()):
        raise RuntimeError(f"{cell.label}: baseline contains an error row")
    return result


def load_probe(path: Path):
    a, stability, pom, cost = {}, {}, {}, {}
    with path.open(encoding="utf-8") as fh:
        for fields in csv.reader(fh, delimiter="\t"):
            if not fields:
                continue
            if fields[0] == "A" and fields[1] != "file":
                a[(fields[1], float(fields[2]), int(fields[3]), int(fields[4]))] = {
                    "steps": int(fields[5]), "ndiv": int(fields[6]), "first_div": int(fields[7])}
            elif fields[0] == "S" and fields[1] != "file":
                stability[(fields[1], int(fields[2]))] = {
                    "steps": int(fields[3]), "pmin": float(fields[4]), "pmax": float(fields[5]),
                    "dispersion": float(fields[6]), "qspread": int(fields[7])}
            elif fields[0] == "SN" and fields[1] != "file":
                pom[(fields[1], int(fields[2]), int(fields[3]))] = int(fields[5])
            elif fields[0] == "C" and fields[1] != "file":
                cost[(fields[1], int(fields[2]))] = float(fields[3])
    return a, stability, pom, cost


def commission(label, a, files):
    alpha_zero_div = sum(a[(name, 0.0, n, k)]["ndiv"]
                         for name in files for n in ORDERS for k in WIDTHS)
    alpha_thirty_div = sum(a[(name, 30.0, n, k)]["ndiv"]
                           for name in files for n in ORDERS for k in WIDTHS)
    if alpha_zero_div != 0:
        raise RuntimeError(f"{label}: alpha=0 identity control diverged {alpha_zero_div} times")
    if alpha_thirty_div == 0:
        raise RuntimeError(f"{label}: ABI-legal alpha=30 vitality control never diverged")


def model_governance(qspread: float, pomspread: float, alpha: float):
    component_b = qspread >= pomspread
    denominator = MODEL_GOVERNS_M * alpha * pomspread
    margin_a = math.inf if denominator == 0 else qspread / denominator
    component_a = qspread >= denominator
    return component_b, component_a, margin_a


def evaluate_cell(cell: Cell):
    baseline = load_baseline(cell)
    a, stability, pom, cost = load_probe(cell.probe)
    files = sorted(baseline)
    commission(cell.label, a, files)
    baseline_rep3 = statistics.mean(row["rep_3"] for row in baseline.values())
    rep3_se = statistics.stdev(row["rep_3"] for row in baseline.values()) / math.sqrt(48)
    baseline_locks = sum(bool(row["loop_lock"]) for row in baseline.values())
    unlocked_files = [name for name in files if not baseline[name]["loop_lock"]]
    baseline_rep3_without_locks = statistics.mean(
        baseline[name]["rep_3"] for name in unlocked_files)
    rep3_se_without_locks = (statistics.stdev(baseline[name]["rep_3"] for name in unlocked_files)
                             / math.sqrt(len(unlocked_files)))
    rows = {}
    for alpha in ALPHAS:
        for order in ORDERS:
            for width in WIDTHS:
                divergent = {name for name in files if a[(name, alpha, order, width)]["ndiv"] > 0}
                unlocked = sum(bool(baseline[name]["loop_lock"]) for name in divergent)
                max_drop = sum(baseline[name]["rep_3"] for name in divergent) / 48
                max_drop_without_locks = (sum(baseline[name]["rep_3"] for name in divergent
                                              if name in unlocked_files) / len(unlocked_files))
                dispersions = [stability[(name, width)]["dispersion"] for name in files]
                qspread = median([stability[(name, width)]["qspread"] for name in files])
                pomspread = median([pom[(name, order, width)] for name in files])
                comp_b, comp_a, margin_a = model_governance(qspread, pomspread, alpha)
                rows[(alpha, order, width)] = {
                    "reach": len(divergent),
                    "locked_moved": unlocked,
                    "baseline_locks": baseline_locks,
                    "rep3_baseline": baseline_rep3,
                    "rep3_drop_ceiling": max_drop,
                    "rep3_ceiling": baseline_rep3 - max_drop,
                    "rep3_drop_over_se": max_drop / rep3_se if rep3_se else math.inf,
                    "rep3_se": rep3_se,
                    "rep3_baseline_without_locks": baseline_rep3_without_locks,
                    "rep3_drop_ceiling_without_locks": max_drop_without_locks,
                    "rep3_ceiling_without_locks": baseline_rep3_without_locks - max_drop_without_locks,
                    "rep3_drop_over_se_without_locks": (
                        max_drop_without_locks / rep3_se_without_locks
                        if rep3_se_without_locks else math.inf),
                    "rep3_se_without_locks": rep3_se_without_locks,
                    "stability_median": median(dispersions),
                    "stability_max": max(dispersions),
                    "stability_over_10x": sum(v > STABILITY_BOUND for v in dispersions),
                    "qspread": qspread,
                    "pomspread": pomspread,
                    "component_b": comp_b,
                    "component_a": comp_a,
                    "margin_a": margin_a,
                    "cost_ns": median([cost[(name, width)] for name in files]),
                }
    return rows


def fmt(value, digits=2):
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", action="append", required=True, type=parse_cell_spec)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    args = parser.parse_args()
    if len(args.cell) != 4:
        raise RuntimeError(f"B0 requires four checkpoint/length cells, got {len(args.cell)}")

    evaluated = {cell.label: evaluate_cell(cell) for cell in args.cell}
    # Commission model-governance itself with discriminating synthetic pairs.
    degenerate_b, _, _ = model_governance(0, 1, 1.0)
    if degenerate_b:
        raise RuntimeError("component-B degenerate commissioning failed")
    _, known_a, known_margin = model_governance(100, 10, 1.0)
    if known_margin != 5.0 or not known_a:
        raise RuntimeError("component-A nonzero-alpha commissioning failed")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    labels = [cell.label for cell in args.cell]
    payload = {
        "method": "counterfactual replay of frozen greedy prefixes through production Phase A/C",
        "scale": {"q_ln2": 493, "q_b": 964, "q_c": 487361},
        "controls": {"alpha_0_identity": "pass", "alpha_30_vitality": "pass",
                     "component_b_degenerate": "pass", "component_a_nonzero": "pass"},
        "cells": labels,
        "rows": [],
    }
    fieldnames = ["alpha", "n", "k", "all_component_b", "min_margin_a",
                  "worst_stability_median", "worst_stability_max", "max_generations_over_10x",
                  "median_cost_ns"]
    for label in labels:
        fieldnames += [f"{label}_reach", f"{label}_locked_moved", f"{label}_rep3_baseline",
                       f"{label}_rep3_ceiling", f"{label}_rep3_se",
                       f"{label}_rep3_drop_over_se", f"{label}_rep3_baseline_without_locks",
                       f"{label}_rep3_ceiling_without_locks", f"{label}_rep3_se_without_locks",
                       f"{label}_rep3_drop_over_se_without_locks"]

    with args.out_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for alpha in ALPHAS:
            for order in ORDERS:
                for width in WIDTHS:
                    details = {label: evaluated[label][(alpha, order, width)] for label in labels}
                    row = {
                        "alpha": alpha, "n": order, "k": width,
                        "all_component_b": all(v["component_b"] for v in details.values()),
                        "min_margin_a": min(v["margin_a"] for v in details.values()),
                        "worst_stability_median": max(v["stability_median"] for v in details.values()),
                        "worst_stability_max": max(v["stability_max"] for v in details.values()),
                        "max_generations_over_10x": max(v["stability_over_10x"] for v in details.values()),
                        "median_cost_ns": median([v["cost_ns"] for v in details.values()]),
                    }
                    for label, value in details.items():
                        row[f"{label}_reach"] = value["reach"]
                        row[f"{label}_locked_moved"] = value["locked_moved"]
                        row[f"{label}_rep3_baseline"] = value["rep3_baseline"]
                        row[f"{label}_rep3_ceiling"] = value["rep3_ceiling"]
                        row[f"{label}_rep3_se"] = value["rep3_se"]
                        row[f"{label}_rep3_drop_over_se"] = value["rep3_drop_over_se"]
                        row[f"{label}_rep3_baseline_without_locks"] = value["rep3_baseline_without_locks"]
                        row[f"{label}_rep3_ceiling_without_locks"] = value["rep3_ceiling_without_locks"]
                        row[f"{label}_rep3_se_without_locks"] = value["rep3_se_without_locks"]
                        row[f"{label}_rep3_drop_over_se_without_locks"] = value[
                            "rep3_drop_over_se_without_locks"]
                    writer.writerow(row)
                    payload["rows"].append({"coordinate": {"alpha": alpha, "n": order, "k": width},
                                            "summary": row, "cells": details})

    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    header = ["α", "n", "k", "reach 05/100", "locks",
              "rep3 05/100 ceiling ± SE; Δ/SE (all / no-lock)", "reach 05/300",
              "reach 15/100", "reach 15/300", "stability med/max/>10", "MG B/min A", "cost ns"]
    lines = ["# T-2199 Phase B0 primary calibration table", "",
             "Counterfactual reach/effectiveness ceiling; candidate rows require genuine autoregressive confirmation.", "",
             "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    short = labels
    for item in payload["rows"]:
        coordinate, row, cells = item["coordinate"], item["summary"], item["cells"]
        values = [fmt(coordinate["alpha"], 1), str(coordinate["n"]), str(coordinate["k"]),
                  str(cells[short[0]]["reach"]),
                  f"{cells[short[0]]['locked_moved']}/{cells[short[0]]['baseline_locks']}",
                  (f"{fmt(cells[short[0]]['rep3_ceiling'], 4)} ± "
                   f"{fmt(cells[short[0]]['rep3_se'], 4)}; "
                   f"{fmt(cells[short[0]]['rep3_drop_over_se'], 2)}× / "
                   f"{fmt(cells[short[0]]['rep3_ceiling_without_locks'], 4)} ± "
                   f"{fmt(cells[short[0]]['rep3_se_without_locks'], 4)}; "
                   f"{fmt(cells[short[0]]['rep3_drop_over_se_without_locks'], 2)}×"),
                  str(cells[short[1]]["reach"]),
                  str(cells[short[2]]["reach"]), str(cells[short[3]]["reach"]),
                  f"{fmt(row['worst_stability_median'])}/{fmt(row['worst_stability_max'])}/"
                  f"{row['max_generations_over_10x']}",
                  f"{'pass' if row['all_component_b'] else 'fail'}/{fmt(row['min_margin_a'])}",
                  fmt(row["median_cost_ns"], 0)]
        lines.append("| " + " | ".join(values) + " |")
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['rows'])} rows: {args.out_csv}, {args.out_json}, {args.out_md}")


if __name__ == "__main__":
    main()
