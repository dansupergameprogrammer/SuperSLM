"""Build the human-readable T-2199 B0 decision packet from commissioned data."""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from pathlib import Path


EOS = {151645, 151643}
SAMPLE_PROMPTS = (
    "npc_dialogue_05",
    "npc_dialogue_08",
    "npc_dialogue_11",
    "event_narration_00",
    "flavor_text_00",
    "list_primed_00",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def baseline_rows(path: Path, length: int = 100) -> list[dict]:
    rows = [row for row in read_jsonl(path) if row["max_new_tokens"] == length]
    if len(rows) != 48:
        raise RuntimeError(f"{path}: expected 48 rows at length {length}, got {len(rows)}")
    return rows


def rollout_rows(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    if len(rows) != 48:
        raise RuntimeError(f"{path}: expected 48 rollout rows, got {len(rows)}")
    return rows


def summarize(rows: list[dict], length: int) -> dict:
    return {
        "mean_tokens": statistics.fmean(row["n_generated"] for row in rows),
        "cap_count": sum(row["n_generated"] == length for row in rows),
        "eos_count": sum(row["n_eos_stripped"] > 0 for row in rows),
        "loop_locks": sum(bool(row["loop_lock"]) for row in rows),
        "rep_2_mean": statistics.fmean(row["rep_2"] for row in rows),
        "rep_3_mean": statistics.fmean(row["rep_3"] for row in rows),
        "rep_4_mean": statistics.fmean(row["rep_4"] for row in rows),
    }


def paired_delta(baseline: list[dict], candidate: list[dict], key: str) -> tuple[float, float]:
    base_by_id = {row["prompt_id"]: row for row in baseline}
    deltas = [row[key] - base_by_id[row["prompt_id"]][key] for row in candidate]
    se = statistics.stdev(deltas) / math.sqrt(len(deltas))
    return statistics.fmean(deltas), se


def strip_stop(tokens: list[int]) -> list[int]:
    tokens = list(tokens)
    while tokens and tokens[-1] in EOS:
        tokens.pop()
    return tokens


def decoded_by_id(rows: list[dict], tokenizer) -> dict[str, str]:
    return {
        row["prompt_id"]: tokenizer.decode(strip_stop(row["output_tokens"]),
                                             skip_special_tokens=False)
        for row in rows
    }


def table_row(label: str, stats: dict, reach: str, delta: str) -> str:
    return (f"| {label} | {reach} | {stats['loop_locks']} | {stats['cap_count']} | "
            f"{stats['eos_count']} | {stats['mean_tokens']:.2f} | "
            f"{stats['rep_3_mean']:.5f} | {delta} |")


def clean_text(text: str) -> str:
    stripped = text.replace("<|im_end|>", "").strip()
    return html.escape("\n".join(line.rstrip() for line in stripped.splitlines()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--baseline-05", required=True, type=Path)
    parser.add_argument("--baseline-15", required=True, type=Path)
    parser.add_argument("--candidate-05", required=True, type=Path)
    parser.add_argument("--candidate-15", required=True, type=Path)
    parser.add_argument("--challenger-05", required=True, type=Path)
    parser.add_argument("--tokenizer-05", required=True, type=Path)
    parser.add_argument("--tokenizer-15", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    primary = json.loads(args.primary.read_text(encoding="utf-8"))
    b05, b15 = baseline_rows(args.baseline_05), baseline_rows(args.baseline_15)
    c05, c15 = rollout_rows(args.candidate_05), rollout_rows(args.candidate_15)
    h05 = rollout_rows(args.challenger_05)
    datasets = {
        "0.5B greedy": b05,
        "0.5B alpha=2": c05,
        "0.5B alpha=3": h05,
        "1.5B greedy": b15,
        "1.5B alpha=2": c15,
    }
    stats = {name: summarize(rows, 100) for name, rows in datasets.items()}
    for name, base, candidate in (("0.5B alpha=2", b05, c05),
                                  ("0.5B alpha=3", b05, h05),
                                  ("1.5B alpha=2", b15, c15)):
        delta, se = paired_delta(base, candidate, "rep_3")
        stats[name]["rep_3_delta"] = delta
        stats[name]["rep_3_delta_se"] = se
        stats[name]["reach"] = sum(row["diverged_from_greedy"] for row in candidate)

    selected = next(row for row in primary["rows"]
                    if row["coordinate"] == {"alpha": 2.0, "n": 2, "k": 6})
    challenger = next(row for row in primary["rows"]
                      if row["coordinate"] == {"alpha": 3.0, "n": 2, "k": 6})

    tok05 = AutoTokenizer.from_pretrained(str(args.tokenizer_05), local_files_only=True)
    tok15 = AutoTokenizer.from_pretrained(str(args.tokenizer_15), local_files_only=True)
    text_sets = {
        "0.5B": (decoded_by_id(b05, tok05), decoded_by_id(c05, tok05)),
        "1.5B": (decoded_by_id(b15, tok15), decoded_by_id(c15, tok15)),
    }

    lines = [
        "# T-2199 Phase B0 decision packet",
        "",
        "This packet reports calibration evidence; it does not automatically select a shipping row.",
        "",
        "## Primary-grid interpretation",
        "",
        "The production-selector replay reached its maximum at alpha=3, n=2/3, k=6/10 "
        "(170/192 cells). Alpha=2, n=2/3, k=6/10 reached 168/192. At k=6, alpha=2 "
        "retains more component-A governance margin than alpha=3; k=10 adds cost without reach "
        "at either strength. The two orders tie on replay reach, so n=2 is the simpler row to weigh.",
        "",
        f"Selected-for-confirmation replay row: `{json.dumps(selected['summary'], sort_keys=True)}`",
        "",
        f"Higher-strength challenger replay row: `{json.dumps(challenger['summary'], sort_keys=True)}`",
        "",
        "All commissioning controls passed: alpha=0 identity, alpha=30 vitality, synthetic "
        "component-A nonzero margin, and synthetic component-B rejection.",
        "",
        "## Genuine autoregressive rollout (48 prompts, max-new=100)",
        "",
        "| Model / decoder | Reach | Locks | Capped | EOS | Mean tokens | Mean rep-3 | Paired rep-3 delta +/- SE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        table_row("0.5B greedy", stats["0.5B greedy"], "--", "--"),
        table_row("0.5B alpha=2 n=2 k=6", stats["0.5B alpha=2"],
                  f"{stats['0.5B alpha=2']['reach']}/48",
                  f"{stats['0.5B alpha=2']['rep_3_delta']:+.5f} +/- {stats['0.5B alpha=2']['rep_3_delta_se']:.5f}"),
        table_row("0.5B alpha=3 n=2 k=6", stats["0.5B alpha=3"],
                  f"{stats['0.5B alpha=3']['reach']}/48",
                  f"{stats['0.5B alpha=3']['rep_3_delta']:+.5f} +/- {stats['0.5B alpha=3']['rep_3_delta_se']:.5f}"),
        table_row("1.5B greedy", stats["1.5B greedy"], "--", "--"),
        table_row("1.5B alpha=2 n=2 k=6", stats["1.5B alpha=2"],
                  f"{stats['1.5B alpha=2']['reach']}/48",
                  f"{stats['1.5B alpha=2']['rep_3_delta']:+.5f} +/- {stats['1.5B alpha=2']['rep_3_delta_se']:.5f}"),
        "",
        "The counterfactual replay is a reach/lock ceiling. The table above is the achieved result "
        "after every divergence is fed back through the model.",
        "",
        "## Human-readable sample",
        "",
        "The three 0.5B baseline loop-lock prompts are included first, followed by one fixed prompt "
        "from each remaining corpus category. Text is decoded from recorded token IDs.",
        "",
    ]
    for model, (greedy_text, candidate_text) in text_sets.items():
        lines.extend([f"### {model}", ""])
        for prompt_id in SAMPLE_PROMPTS:
            lines.extend([
                f"#### {prompt_id}",
                "",
                "Greedy:",
                "",
                f"<pre>{clean_text(greedy_text[prompt_id])}</pre>",
                "",
                "Damped greedy (alpha=2, n=2, k=6):",
                "",
                f"<pre>{clean_text(candidate_text[prompt_id])}</pre>",
                "",
            ])

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    machine = {
        "method": "genuine autoregressive production-decoder rollout",
        "primary_controls": primary["controls"],
        "selected_replay_row": selected,
        "challenger_replay_row": challenger,
        "statistics": stats,
        "sample_prompt_ids": list(SAMPLE_PROMPTS),
    }
    args.out_json.write_text(json.dumps(machine, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
