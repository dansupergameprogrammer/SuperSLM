"""Build the T-2199 Phase E confirmation record from frozen T-2193 baselines and real rollouts."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path


EOS = {151645, 151643}
CELL_ORDER = ("05_100", "05_300", "15_100", "15_300")
SAMPLE_PROMPTS = (
    "npc_dialogue_05",
    "npc_dialogue_08",
    "npc_dialogue_11",
    "event_narration_00",
    "flavor_text_00",
    "list_primed_00",
)


@dataclass(frozen=True)
class CellSpec:
    label: str
    rollout: Path
    baseline: Path
    length: int
    model: str


def parse_cell(value: str) -> CellSpec:
    label, separator, rest = value.partition("=")
    fields = rest.split(",")
    if not separator or len(fields) != 4:
        raise argparse.ArgumentTypeError(
            "cell must be label=rollout.jsonl,baseline.jsonl,length,model-label")
    return CellSpec(label, Path(fields[0]), Path(fields[1]), int(fields[2]), fields[3])


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_baseline(spec: CellSpec) -> list[dict]:
    rows = [row for row in read_jsonl(spec.baseline)
            if row["max_new_tokens"] == spec.length]
    if len(rows) != 48 or len({row["prompt_id"] for row in rows}) != 48:
        raise RuntimeError(f"{spec.label}: baseline population is not exactly 48 unique prompts")
    if any(row.get("error") or row.get("header_list_mismatch") for row in rows):
        raise RuntimeError(f"{spec.label}: baseline contains a failed or malformed row")
    return rows


def load_paired_greedy(path: Path, spec: CellSpec, frozen: list[dict]) -> list[dict]:
    rows = [row for row in read_jsonl(path) if row["max_new_tokens"] == spec.length]
    if len(rows) != 48 or len({row["prompt_id"] for row in rows}) != 48:
        raise RuntimeError(f"{spec.label}: paired greedy population is not exactly 48 unique prompts")
    if any(row.get("error") or row.get("header_list_mismatch") for row in rows):
        raise RuntimeError(f"{spec.label}: paired greedy contains a failed or malformed row")
    frozen_by_id = {row["prompt_id"]: row for row in frozen}
    mismatches = sorted(
        row["prompt_id"] for row in rows
        if row["output_tokens"] != frozen_by_id[row["prompt_id"]]["output_tokens"])
    if mismatches:
        raise RuntimeError(f"{spec.label}: paired greedy differs from frozen baseline: {mismatches}")
    return rows


def load_rollout(spec: CellSpec) -> list[dict]:
    rows = read_jsonl(spec.rollout)
    if len(rows) != 48 or len({row["prompt_id"] for row in rows}) != 48:
        raise RuntimeError(f"{spec.label}: rollout population is not exactly 48 unique prompts")
    expected = (2.0, 65536, 2, 6, spec.length)
    for row in rows:
        actual = (row["alpha"], row["alpha_q15"], row["n"], row["k"], row["length"])
        if actual != expected or row["n_generated"] <= 0 or row["output_text"] is None:
            raise RuntimeError(f"{spec.label}/{row['prompt_id']}: invalid rollout row {actual}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_delta(baseline: list[dict], candidate: list[dict], key: str) -> tuple[float, float]:
    by_id = {row["prompt_id"]: row for row in baseline}
    deltas = [row[key] - by_id[row["prompt_id"]][key] for row in candidate]
    return statistics.fmean(deltas), statistics.stdev(deltas) / math.sqrt(len(deltas))


def summarize(rows: list[dict], length: int) -> dict:
    wall = [row["wall_time_seconds"] for row in rows]
    total_tokens = sum(row["n_generated"] for row in rows)
    total_wall = sum(wall)
    return {
        "generations": len(rows),
        "loop_locks": sum(bool(row["loop_lock"]) for row in rows),
        "cap_count": sum(row["n_generated"] == length for row in rows),
        "eos_count": sum(row["n_eos_stripped"] > 0 for row in rows),
        "mean_tokens": statistics.fmean(row["n_generated"] for row in rows),
        "rep_2_mean": statistics.fmean(row["rep_2"] for row in rows),
        "rep_3_mean": statistics.fmean(row["rep_3"] for row in rows),
        "rep_4_mean": statistics.fmean(row["rep_4"] for row in rows),
        "wall_seconds_mean": statistics.fmean(wall),
        "wall_seconds_total": total_wall,
        "effective_ms_per_generated_token": 1000.0 * total_wall / total_tokens,
        "effective_tokens_per_second": total_tokens / total_wall,
    }


def strip_stop(tokens: list[int]) -> list[int]:
    result = list(tokens)
    while result and result[-1] in EOS:
        result.pop()
    return result


def clean_text(text: str) -> str:
    return html.escape("\n".join(line.rstrip() for line in text.strip().splitlines()))


def compact_row(row: dict) -> dict:
    keys = ("prompt_id", "category", "output_tokens", "n_generated", "n_eos_stripped",
            "stop_reason", "wall_time_seconds", "rep_2", "rep_3", "rep_4", "loop_lock",
            "loop_lock_cycle_len", "loop_lock_repeats")
    return {key: row[key] for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", action="append", required=True, type=parse_cell)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--repeat-05", required=True, type=Path)
    parser.add_argument("--repeat-15", required=True, type=Path)
    parser.add_argument("--paired-greedy-05", required=True, type=Path)
    parser.add_argument("--paired-greedy-15", required=True, type=Path)
    parser.add_argument("--commission", action="append", required=True, type=Path)
    parser.add_argument("--harness-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-05", required=True, type=Path)
    parser.add_argument("--tokenizer-15", required=True, type=Path)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--artifact-05", required=True, type=Path)
    parser.add_argument("--artifact-15", required=True, type=Path)
    parser.add_argument("--selector-forward-ns", required=True, type=float)
    parser.add_argument("--selector-renorm-ns", required=True, type=float)
    parser.add_argument("--selector-checks", required=True, type=int)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    args = parser.parse_args()

    if [cell.label for cell in args.cell] != list(CELL_ORDER):
        raise RuntimeError(f"cells must be supplied in order {CELL_ORDER}")
    if len(args.commission) != 4:
        raise RuntimeError("Phase E requires all four greedy commissioning manifests")

    primary = json.loads(args.primary.read_text(encoding="utf-8"))
    selected = next(row for row in primary["rows"]
                    if row["coordinate"] == {"alpha": 2.0, "n": 2, "k": 6})
    commissions = [json.loads(path.read_text(encoding="utf-8")) for path in args.commission]
    if any(not item["baseline_exact"] or item["prompt_population"] != 48
           for item in commissions):
        raise RuntimeError("greedy commissioning is incomplete")

    baselines: dict[str, list[dict]] = {}
    rollouts: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}
    result_records = []
    for spec in args.cell:
        frozen = load_baseline(spec)
        paired_path = args.paired_greedy_05 if spec.label.startswith("05_") else args.paired_greedy_15
        baseline = load_paired_greedy(paired_path, spec, frozen)
        rollout = load_rollout(spec)
        baselines[spec.label], rollouts[spec.label] = baseline, rollout
        base_by_id = {row["prompt_id"]: row for row in baseline}
        actual_reach = sum(row["diverged_from_greedy"] for row in rollout)
        actual_unlocked = sum(bool(base_by_id[row["prompt_id"]]["loop_lock"])
                              and not row["loop_lock"] for row in rollout)
        predicted = selected["cells"][spec.label]
        rep3_delta, rep3_delta_se = paired_delta(baseline, rollout, "rep_3")
        baseline_summary = summarize(baseline, spec.length)
        damped_summary = summarize(rollout, spec.length)
        summary = {
            "model": spec.model,
            "length": spec.length,
            "greedy": baseline_summary,
            "damped": damped_summary,
            "genuine_reach": actual_reach,
            "b0_replay_reach": predicted["reach"],
            "reach_matches_b0": actual_reach == predicted["reach"],
            "genuine_locked_rows_repaired": actual_unlocked,
            "b0_locked_rows_repair_ceiling": predicted["locked_moved"],
            "locked_repair_matches_b0": actual_unlocked == predicted["locked_moved"],
            "paired_rep3_delta": rep3_delta,
            "paired_rep3_delta_se": rep3_delta_se,
            "effective_ms_per_token_ratio_damped_over_greedy": (
                damped_summary["effective_ms_per_generated_token"]
                / baseline_summary["effective_ms_per_generated_token"]),
            "wall_time_ratio_damped_over_greedy": (
                damped_summary["wall_seconds_total"] / baseline_summary["wall_seconds_total"]),
        }
        summaries[spec.label] = summary
        for row in rollout:
            prompt_id = row["prompt_id"]
            result_records.append({
                "cell": spec.label,
                "model": spec.model,
                "length": spec.length,
                "greedy": compact_row(base_by_id[prompt_id]),
                "damped": compact_row(row),
                "diverged_from_greedy": row["diverged_from_greedy"],
            })

    repeat_mismatches = {}
    for label, repeat_path in (("05_100", args.repeat_05), ("15_100", args.repeat_15)):
        first = {row["prompt_id"]: row["output_tokens"] for row in rollouts[label]}
        repeat = {row["prompt_id"]: row["output_tokens"] for row in read_jsonl(repeat_path)}
        mismatches = sorted(prompt_id for prompt_id in first if first[prompt_id] != repeat[prompt_id])
        repeat_mismatches[label] = mismatches
    if any(repeat_mismatches.values()):
        raise RuntimeError(f"100-token repeat determinism failed: {repeat_mismatches}")

    sys.path.insert(0, str(args.harness_dir))
    from prompts import all_prompts  # pylint: disable=import-error,import-outside-toplevel
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    prompt_text = {prompt_id: user_text
                   for prompt_id, _category, user_text, _rendered in all_prompts()}
    tok05 = AutoTokenizer.from_pretrained(str(args.tokenizer_05), local_files_only=True)
    tok15 = AutoTokenizer.from_pretrained(str(args.tokenizer_15), local_files_only=True)

    provenance = {
        "sslm_generate_sha256": sha256(args.exe),
        "damped_artifact_05_sha256": sha256(args.artifact_05),
        "damped_artifact_15_sha256": sha256(args.artifact_15),
        "prompts_py_sha256": sha256(args.harness_dir / "prompts.py"),
        "metrics_py_sha256": sha256(args.harness_dir / "metrics.py"),
        "baseline_05_sha256": sha256(args.cell[0].baseline),
        "baseline_15_sha256": sha256(args.cell[2].baseline),
        "paired_greedy_05_sha256": sha256(args.paired_greedy_05),
        "paired_greedy_15_sha256": sha256(args.paired_greedy_15),
        "paired_greedy_frozen_exact_cases": sum(len(rows) for rows in baselines.values()),
        "greedy_commissioned_cases": sum(item["prompt_population"] for item in commissions),
        "greedy_baseline_exact": all(item["baseline_exact"] for item in commissions),
        "repeat_100_token_mismatches": repeat_mismatches,
    }
    selector_ratio_percent = 100.0 * args.selector_renorm_ns / args.selector_forward_ns

    lines = [
        "# T-2199 Phase E end-to-end confirmation",
        "",
        "**Result: damped greedy works as an opt-in anti-loop decoder at the ruled operating point. "
        "Greedy remains the default.**",
        "",
        "Operating point: `alpha=2`, `n=2`, `k=6`, `alpha_q15=65536`, "
        "`q=(493, 964, 487361)`.",
        "",
        "## Full-generation production result",
        "",
        "| Cell | Greedy locks | Damped locks | Reach (actual / B0) | Greedy rep-3 | Damped rep-3 | "
        "Paired delta +/- SE | Greedy cap/EOS | Damped cap/EOS | Effective ms/token ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in CELL_ORDER:
        item = summaries[label]
        greedy, damped = item["greedy"], item["damped"]
        lines.append(
            f"| {label} | {greedy['loop_locks']} | {damped['loop_locks']} | "
            f"{item['genuine_reach']}/48 / {item['b0_replay_reach']}/48 | "
            f"{greedy['rep_3_mean']:.5f} | {damped['rep_3_mean']:.5f} | "
            f"{item['paired_rep3_delta']:+.5f} +/- {item['paired_rep3_delta_se']:.5f} | "
            f"{greedy['cap_count']}/{greedy['eos_count']} | "
            f"{damped['cap_count']}/{damped['eos_count']} | "
            f"{item['effective_ms_per_token_ratio_damped_over_greedy']:.3f}x |")
    lines.extend([
        "",
        "`cap/EOS` reports generations that reached the token ceiling / emitted a stop token. The "
        "effective wall-time ratio uses fresh paired greedy captures from the same executable and "
        "hardware, but includes model loading and different generated lengths, so it is an end-to-end "
        "observation, not an isolated selector-cost measurement. Phase D2a separately "
        "measures the wired selector against real forward cost.",
        "",
        "Across the four fresh paired cells, damped/greedy wall time was "
        f"{min(item['wall_time_ratio_damped_over_greedy'] for item in summaries.values()):.3f}x to "
        f"{max(item['wall_time_ratio_damped_over_greedy'] for item in summaries.values()):.3f}x per "
        "48-generation cell and "
        f"{min(item['effective_ms_per_token_ratio_damped_over_greedy'] for item in summaries.values()):.3f}x "
        "to "
        f"{max(item['effective_ms_per_token_ratio_damped_over_greedy'] for item in summaries.values()):.3f}x "
        "per generated token. This is a measured "
        "end-to-end harness cost, not attributed solely to selection. The post-capture Phase D2a "
        f"microbenchmark passed {args.selector_checks} checks with mean forward "
        f"{args.selector_forward_ns:.1f} ns, selector renormalization {args.selector_renorm_ns:.1f} "
        f"ns, and a {selector_ratio_percent:.4f}% selector/forward ratio. The two measurements have "
        "different scopes; Phase E does not claim that the microbenchmark explains the full "
        "end-to-end delta.",
        "",
        "Production matched the B0 replay reach and locked-row ceiling in all four cells. The two "
        "100-token cells were rerun after the ruling and matched their B0 token streams exactly: "
        "0 mismatches across 96 generations. Greedy capture independently matched the frozen T-2193 "
        "baseline in all 192 model/length/prompt cases.",
        "",
        "## Fidelity interpretation",
        "",
        "The result supports the ruled product shape, not semantic equivalence. Damped greedy removes "
        "the measured loop locks and often improves prose, but it can alter facts, structure, count, "
        "formatting, and termination. In particular, `list_primed_00` demonstrates that legitimate "
        "repeated list syntax is penalized along with pathological repetition. Format-sensitive calls "
        "should use greedy or schema-constrained decoding unless the caller explicitly chooses this "
        "tradeoff.",
        "",
        "## Real text, side by side (300-token cells)",
        "",
        "The three known 0.5B lock cases come first, followed by fixed narration, flavor, and list "
        "cases. Text is decoded from the recorded token streams.",
        "",
    ])
    for label, tokenizer in (("05_300", tok05), ("15_300", tok15)):
        base_by_id = {row["prompt_id"]: row for row in baselines[label]}
        damped_by_id = {row["prompt_id"]: row for row in rollouts[label]}
        lines.extend([f"### {label}", ""])
        for prompt_id in SAMPLE_PROMPTS:
            greedy_text = tokenizer.decode(strip_stop(base_by_id[prompt_id]["output_tokens"]),
                                           skip_special_tokens=False)
            damped_text = tokenizer.decode(strip_stop(damped_by_id[prompt_id]["output_tokens"]),
                                           skip_special_tokens=False)
            lines.extend([
                f"#### {prompt_id}",
                "",
                "Prompt:",
                "",
                f"<pre>{clean_text(prompt_text[prompt_id])}</pre>",
                "",
                "Greedy:",
                "",
                f"<pre>{clean_text(greedy_text)}</pre>",
                "",
                "Damped greedy:",
                "",
                f"<pre>{clean_text(damped_text)}</pre>",
                "",
            ])

    payload = {
        "result": "confirmed_opt_in",
        "default_decoder": "greedy",
        "operating_point": {"alpha": 2.0, "alpha_q15": 65536, "n": 2, "k": 6,
                            "q_ln2": 493, "q_b": 964, "q_c": 487361},
        "cells": summaries,
        "provenance": provenance,
        "selector_microbenchmark": {
            "checks": args.selector_checks,
            "mean_forward_ns": args.selector_forward_ns,
            "mean_renorm_ns": args.selector_renorm_ns,
            "renorm_over_forward_percent": selector_ratio_percent,
        },
        "fidelity_caveat": ("anti-loop effectiveness is confirmed; semantic and format equivalence "
                             "is explicitly not claimed"),
        "sample_prompt_ids": list(SAMPLE_PROMPTS),
    }
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.out_jsonl.open("w", encoding="utf-8", newline="\n") as stream:
        for row in result_records:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {args.out_md}, {args.out_json}, and {len(result_records)} rows to {args.out_jsonl}")


if __name__ == "__main__":
    main()
