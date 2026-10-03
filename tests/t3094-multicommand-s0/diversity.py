"""Read-only diversity census and predeclared v3 freeze gates.

This module never creates utterances or labels. All counts come from the
committed corpus, with entity substitutions removed before shape counting.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from math import ceil
import math
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
BANNED_TRACE_FIELDS = frozenset({
    "expected_action_items", "wait_scheduled", "wait_has_completion",
    "gold", "gold_status", "segmentation", "label", "status", "route",
})
BANNED_LABEL_TEXT = (
    "do not schedule a separate wait", "schedule and track a wait",
    "at cap four", "above cap four", "training transcript",
)


def rows(path=HERE / "corpus.jsonl"):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def stratum(row):
    identifier = row["id"]
    if identifier.startswith("P-"):
        return "pc2/" + identifier.split("-")[1] + "/" + identifier[-1]
    if identifier.startswith("D-"):
        return "development"
    return identifier[0] + "/" + row["source_class"]


def shape(row):
    text = row["utterance"].casefold()
    trace = row["game_action_trace"]
    entities = [trace.get(key) for key in ("actor", "target")]
    # The fixture's recurrent actors and places are masked even when pronouns
    # or shared arguments leave them outside this row's primary trace fields.
    entities += [
        "alpha wing", "bravo wing", "charlie team", "delta scout", "echo battery",
        "foxtrot patrol", "golf squad", "hotel drone", "india unit",
        "juliet group", "reserve wing", "north ridge", "east bridge",
        "south gate", "west tower", "upper deck", "lower tunnel",
        "river crossing", "supply depot", "radar mast", "landing zone",
    ]
    for entity in sorted({e.casefold() for e in entities if isinstance(e, str)}, key=len, reverse=True):
        text = re.sub(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", " ENTITY ", text)
    text = re.sub(r"\b\d+\b", " NUMBER ", text)
    return " ".join(re.findall(r"[a-z]+(?:'[a-z]+)?", text))


def ngrams(sentence, size=4):
    words = sentence.split()
    return set(zip(*(words[i:] for i in range(size))))


def lexical_probe(pairs):
    """Leave-one-pair-out set-of-words baseline, with no trace access."""
    pc2 = [r for members in pairs.values() for r in members]
    def words(row):
        return set(re.findall(r"[a-z]+", row["utterance"].casefold()))
    right = 0
    for pair, members in pairs.items():
        train = [r for r in pc2 if r["pair"] != pair]
        counts = {arm: Counter(word for r in train if r["id"].endswith("-" + arm)
                               for word in words(r)) for arm in "TW"}
        vocabulary = set(counts["T"]) | set(counts["W"])
        totals = {arm: sum(counts[arm].values()) for arm in "TW"}
        for row in members:
            scores = {arm: sum(math.log((counts[arm][word] + 1) /
                                        (totals[arm] + len(vocabulary))) for word in words(row))
                      for arm in "TW"}
            right += max(scores, key=scores.get) == row["id"][-1]
    return round(right / max(1, len(pc2)), 6)


def trace_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from trace_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from trace_keys(child)


def census(population):
    strata = defaultdict(list)
    for row in population:
        strata[stratum(row)].append(shape(row))
    per_stratum = {}
    for name, sentences in sorted(strata.items()):
        counts = Counter(sentences)
        per_stratum[name] = {
            "rows": len(sentences), "distinct_shapes": len(counts),
            "max_shape_share": round(max(counts.values()) / len(sentences), 6),
        }
    held = set().union(*(ngrams(shape(r)) for r in population if r["split"] == "held_out"))
    dev = set().union(*(ngrams(shape(r)) for r in population if r["split"] == "development"))
    pairs = defaultdict(list)
    for row in population:
        if row.get("pair"):
            pairs[row["pair"]].append(row)
    pair_metrics = {}
    for condition in ("simple_trigger", "dependency", "ordering"):
        selected = [members for key, members in pairs.items() if key.startswith("P-" + condition + "-")]
        signatures = [tuple(sorted(shape(row) for row in members)) for members in selected]
        similarities = []
        for members in selected:
            if len(members) == 2:
                a, b = (set(re.findall(r"[a-z]+", r["utterance"].casefold())) for r in members)
                similarities.append(len(a & b) / len(a | b))
        pair_metrics[condition] = {
            "pairs": len(selected),
            "distinct_pair_shapes": len(set(signatures)),
            "complete_pairs": sum(len(members) == 2 for members in selected),
            "minimum_pair_token_jaccard": round(min(similarities, default=0), 6),
            "mean_pair_token_jaccard": round(sum(similarities) / max(1, len(similarities)), 6),
        }
    trace_leaks = sorted({key for r in population for key in trace_keys(r["game_action_trace"])
                          if key in BANNED_TRACE_FIELDS})
    label_text_ids = [r["id"] for r in population if any(label in r["utterance"].casefold() for label in BANNED_LABEL_TEXT)]
    return {
        "per_stratum": per_stratum,
        "cross_split_4gram_jaccard": round(len(held & dev) / max(1, len(held | dev)), 6),
        "pc2": pair_metrics,
        "pc2_lexical_leave_pair_out_accuracy": lexical_probe(pairs),
        "trace_label_fields": trace_leaks,
        "literal_label_text_count": len(label_text_ids),
        "literal_label_text_examples": label_text_ids[:8],
    }


def gate(report, limits):
    failures = []
    for name, result in report["per_stratum"].items():
        minimum = ceil(result["rows"] * limits["minimum_distinct_shape_fraction_per_stratum"])
        if result["distinct_shapes"] < minimum:
            failures.append(f"{name}: {result['distinct_shapes']} shapes < {minimum}")
        if result["max_shape_share"] > limits["maximum_single_shape_share"]:
            failures.append(f"{name}: shape share {result['max_shape_share']}")
    if report["cross_split_4gram_jaccard"] > limits["maximum_cross_split_4gram_jaccard"]:
        failures.append("held-out/development 4-gram overlap")
    for name, result in report["pc2"].items():
        if result["pairs"] != 40 or result["complete_pairs"] != 40:
            failures.append(f"{name}: incomplete 40-pair challenge")
        if result["distinct_pair_shapes"] < limits["minimum_distinct_pc2_pair_shapes_per_condition"]:
            failures.append(f"{name}: pair-shape diversity")
        if result["minimum_pair_token_jaccard"] < limits["minimum_pc2_pair_token_jaccard"]:
            failures.append(f"{name}: pair words insufficiently matched")
    if report["pc2_lexical_leave_pair_out_accuracy"] > limits["maximum_pc2_lexical_baseline_accuracy"]:
        failures.append("PC-2 word-only arm baseline")
    if report["trace_label_fields"]:
        failures.append("answer-bearing trace fields: " + ",".join(report["trace_label_fields"]))
    if report["literal_label_text_count"]:
        failures.append("literal label wording in utterances")
    return failures


if __name__ == "__main__":
    report = census(rows())
    limits = json.loads((HERE / "freeze.json").read_text())["diversity_gates"]
    report["gate_failures"] = gate(report, limits)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(bool(report["gate_failures"]))
