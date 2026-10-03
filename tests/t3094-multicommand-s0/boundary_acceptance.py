"""Rev-4 PC-2/cap corpus acceptance, independent of authored wording.

The two raw-only readings are external evidence, pinned by SHA-256. This code
checks their action and arm decisions against the text-free spec, then checks
the packaged model input and old shortcut controls. It never writes utterances.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import argparse
import json
from pathlib import Path
import re
import subprocess
import unicodedata


HERE = Path(__file__).resolve().parent
ARTIFACTS = Path("D:/_artifacts/SuperSLM")
ARTIFACT_SHA256 = {
    "t3100/boundary-spec.jsonl": "1005b9c80f812d1686b4285fc3ff8d35fcc4c01564da4bca286845fb42ced78b",
    "t3100/utterances.jsonl": "1765a6727efaae4807b52df4d3a5e38c5bc92fc64e1d6efbeba2f85fe102e4ec",
    "t3101/arm-A/labels.jsonl": "3f09ad9556af571b791878a84dc6550c9b943dbd294c46ebaf5387f21aa06b1e",
    "t3101/arm-B/labels.jsonl": "8271ff99a7c9eac1a19cbca238816144b15f17d0ca11d3e1147f1092ac6d3370",
}
V4 = "7a5cddd1a3a2924efe94600256c052498b72c4ab"
V5 = "d298c6842f74f8198f33f5e4b9904b08d80486a7"
FIXED_CONDITION = "there is no separate task completion"
FIXED_ACTION = "there is no fire until separate task completion"
TIMING_NOUN = re.compile(
    r"\b(?:window|interval|phase|pulse|period|signal|span|cycle|message|stage|light)"
    r"\s+(?:task|order|job|entry)\b", re.I)
SUPPORT = 20
ALERT_PURITY = 0.95
SUBSTRING_LENGTHS = (8, 12, 16, 24, 32, 48, 64)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_artifact(relative: str) -> list[dict]:
    path = ARTIFACTS / relative
    actual = digest(path)
    if actual != ARTIFACT_SHA256[relative]:
        raise ValueError(f"independent artifact changed: {relative}: {actual}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def unique(rows: list[dict], key: str, expected: int) -> dict[str, dict]:
    result = {r[key]: r for r in rows}
    if len(rows) != len(result) or len(result) != expected:
        raise ValueError(f"{key} population: {len(rows)} rows, {len(result)} unique")
    return result


def inputs() -> tuple[dict, dict, dict]:
    spec = unique(load_artifact("t3100/boundary-spec.jsonl"), "row_id", 320)
    writer = unique(load_artifact("t3100/utterances.jsonl"), "row_id", 320)
    readers = {arm: unique(load_artifact(f"t3101/arm-{arm}/labels.jsonl"), "row_id", 320)
               for arm in "AB"}
    if set(spec) != set(writer) or any(set(reader) != set(spec) for reader in readers.values()):
        raise ValueError("independent artifact IDs do not cover the same 320 rows")
    return spec, writer, readers


def tokens(text: str) -> tuple[str, ...]:
    """Rev-4 tokenizer: casefold, maximal Unicode L/N runs, internal apostrophes."""
    text = text.casefold()
    result = []
    current = []
    for i, char in enumerate(text):
        letter_or_number = unicodedata.category(char)[0] in ("L", "N")
        apostrophe = (char in "'’" and current and i + 1 < len(text)
                      and unicodedata.category(text[i + 1])[0] in ("L", "N"))
        if letter_or_number or apostrophe:
            current.append(char)
        elif current:
            result.append("".join(current))
            current = []
    if current:
        result.append("".join(current))
    return tuple(result)


def punctuation(text: str) -> Counter:
    return Counter(char for char in text if unicodedata.category(char).startswith("P"))


def trace_arm(row: dict) -> str:
    events = row["game_action_trace"]["events"]
    queued = any(e.get("event") == "task_queued" and e.get("task") == "signal_watch" for e in events)
    callback = any(e.get("event") == "completion_callback_registered" for e in events)
    listener = any(e.get("event") == "signal_listener_registered" for e in events)
    if queued and callback and not listener:
        return "W"
    if listener and not queued and not callback:
        return "T"
    raise ValueError("boundary trace has no unique wait interpretation")


def boundary(rows: list[dict]) -> dict[str, dict]:
    selected = [r for r in rows if r["source_class"] in ("pc2", "at_cap", "cap_plus_one")]
    return unique(selected, "id", 320)


def reader_scores(rows: list[dict], spec: dict, writer: dict, readers: dict) -> dict:
    selected = boundary(rows)
    if set(selected) != set(spec):
        raise ValueError("corpus boundary IDs differ from the text-free spec")
    writer_failures = []
    for row_id, row in selected.items():
        draft = writer[row_id]
        if draft["flag"] or draft["reason"] or row["utterance"] != draft["utterance"]:
            writer_failures.append({"row_id": row_id, "flag": draft["flag"],
                                    "reason": draft["reason"],
                                    "verbatim": row["utterance"] == draft["utterance"]})
        if trace_arm(row) != spec[row_id]["arm"]:
            raise ValueError(f"text-free spec and private trace disagree: {row_id}")
        trace_actions = ["wait" if e["task"] == "signal_watch" else e["task"]
                         for e in row["game_action_trace"]["events"] if e["event"] == "task_queued"]
        expected_actions = [cmd["action"] for cmd in spec[row_id]["intended_commands"]["commands"]]
        if trace_actions != expected_actions:
            raise ValueError(f"text-free spec and private action trace disagree: {row_id}")
    results = {}
    for arm, labels in readers.items():
        failures = []
        for row_id, label in labels.items():
            gold = spec[row_id]
            actions = [command["action"] for command in label["commands"]]
            expected_actions = [command["action"] for command in gold["intended_commands"]["commands"]]
            got_arm = label.get("wait_class", label.get("wait_type"))
            flagged = bool(label.get("ambiguity", label.get("ambiguous", False)))
            status = label.get("route_status", label.get("status"))
            if got_arm != gold["arm"] or actions != expected_actions or flagged or status != "Executable":
                failures.append({"row_id": row_id, "expected_arm": gold["arm"], "reader_arm": got_arm,
                                 "expected_actions": expected_actions, "reader_actions": actions,
                                 "flagged": flagged, "status": status,
                                 "reason": label.get("reason", "")})
        results[arm] = {"rows": len(labels), "correct": len(labels) - len(failures),
                        "flagged": sum(bool(r.get("ambiguity", r.get("ambiguous", False)))
                                       for r in labels.values()),
                        "disagreements": sorted(failures, key=lambda item: item["row_id"])}
    return {"scored_fields": ["ordered action names", "T/W arm", "Executable status", "ambiguity flag"],
            "writer_failures": writer_failures, "readers": results,
            "passed": not writer_failures and all(r["correct"] == 320 and r["flagged"] == 0
                                                  for r in results.values())}


def pairs(spec: dict) -> dict[str, dict[str, str]]:
    result = defaultdict(dict)
    for row in spec.values():
        result[row["pair_id"]][row["arm"]] = row["row_id"]
    if len(result) != 160 or any(set(members) != {"T", "W"} for members in result.values()):
        raise ValueError("text-free spec does not define 160 complete pairs")
    return dict(result)


def pair_metrics(rows: list[dict], spec: dict) -> dict:
    selected = boundary(rows)
    classes = pairs(spec)
    report = {}
    for family in ("simple_trigger", "dependency", "ordering", "cap_boundary"):
        group = {pair: members for pair, members in classes.items()
                 if (pair.startswith("C-") if family == "cap_boundary"
                     else pair.startswith("P-" + family + "-"))}
        mismatched_words = []
        mismatched_punctuation = []
        mismatched_arguments = []
        for pair, members in group.items():
            a, b = (selected[members[arm]]["utterance"] for arm in "TW")
            if Counter(tokens(a)) != Counter(tokens(b)):
                mismatched_words.append(pair)
            if punctuation(a) != punctuation(b):
                mismatched_punctuation.append(pair)
            actors = {selected[members[arm]]["game_action_trace"]["actor"] for arm in "TW"}
            targets = {selected[members[arm]]["game_action_trace"]["target"] for arm in "TW"}
            if len(actors) != 1 or len(targets) != 1:
                mismatched_arguments.append(pair)
        report[family] = {"pairs": len(group), "exact_word_multisets": len(group) - len(mismatched_words),
                          "punctuation_inventory_matches": len(group) - len(mismatched_punctuation),
                          "same_actor_target": len(group) - len(mismatched_arguments),
                          "word_mismatches": mismatched_words,
                          "punctuation_mismatches": mismatched_punctuation,
                          "argument_mismatches": mismatched_arguments,
                          "word_presence_count_ceiling": 0.5 if not mismatched_words else None}
    return report


def copied_bodies(selected: dict[str, dict]) -> list[dict]:
    copied = []
    for index in range(40):
        for arm in "TW":
            simple = selected[f"P-simple_trigger-{index:03}-{arm}"]["utterance"]
            # The historical defect copied the entire simple request after its
            # actor prefix into the dependency and ordering requests.
            body = simple.split(", ", 1)[-1]
            for condition in ("dependency", "ordering"):
                row_id = f"P-{condition}-{index:03}-{arm}"
                if len(body) >= 24 and body in selected[row_id]["utterance"]:
                    copied.append({"simple": f"P-simple_trigger-{index:03}-{arm}", "copy": row_id})
    return copied


def cue_alerts(selected: dict[str, dict]) -> dict:
    counts = defaultdict(lambda: [0, 0])
    substrings = defaultdict(lambda: [0, 0])
    for row_id, row in selected.items():
        arm = int(trace_arm(row) == "W")
        word_list = tokens(row["utterance"])
        grams = {tuple(word_list[i:i + width]) for width in range(1, 8)
                 for i in range(len(word_list) - width + 1)}
        for gram in grams:
            counts[gram][arm] += 1
        normalized = row["utterance"].casefold()
        chunks = {normalized[i:i + width] for width in SUBSTRING_LENGTHS
                  for i in range(len(normalized) - width + 1)}
        for chunk in chunks:
            substrings[chunk][arm] += 1
    def alerts(population, stringify):
        found = [{"cue": stringify(cue), "T": count[0], "W": count[1],
                  "support": sum(count), "purity": round(max(count) / sum(count), 6)}
                 for cue, count in population.items()
                 if sum(count) >= SUPPORT and max(count) / sum(count) >= ALERT_PURITY]
        return sorted(found, key=lambda item: (-item["support"], -item["purity"], item["cue"]))
    return {"minimum_support": SUPPORT, "alert_purity": ALERT_PURITY,
            "ordered_1_to_7_grams": alerts(counts, lambda cue: " ".join(cue)),
            "character_substrings": alerts(substrings, lambda cue: cue)}


def surface(rows: list[dict], spec: dict, *, scan_cues: bool = True) -> dict:
    selected = boundary(rows)
    texts = {row_id: row["utterance"] for row_id, row in selected.items()}
    fixed_condition = [row_id for row_id, text in texts.items() if FIXED_CONDITION in text.casefold()]
    fixed_action = [row_id for row_id, text in texts.items() if FIXED_ACTION in text.casefold()]
    timing = Counter({"T": 0, "W": 0})
    for row_id, text in texts.items():
        if TIMING_NOUN.search(text):
            timing[trace_arm(selected[row_id])] += 1
    result = {"pair_metrics": pair_metrics(rows, spec),
              "fixed_condition_clause_hits": len(fixed_condition),
              "fixed_action_clause_hits": len(fixed_action),
              "timing_noun_pattern_hits": dict(timing),
              "copied_simple_bodies": copied_bodies(selected)}
    if scan_cues:
        result["cue_sweep"] = cue_alerts(selected)
    return result


def historical(commit: str) -> list[dict]:
    raw = subprocess.check_output(
        ["git", "show", f"{commit}:tests/t3094-multicommand-s0/corpus.jsonl"],
        cwd=HERE, encoding="utf-8", text=True)
    return [json.loads(line) for line in raw.splitlines()]


def model_export(rows: list[dict], model_path: Path = HERE / "model-input.jsonl",
                 order_path: Path = HERE / "model-input-order.jsonl") -> dict:
    if not model_path.is_file() or not order_path.is_file():
        return {"passed": False, "reason": "model input or private order map is absent"}
    held = {r["id"]: r for r in rows if r["split"] == "held_out"}
    model = [json.loads(line) for line in model_path.read_text(encoding="utf-8").splitlines()]
    order = [json.loads(line) for line in order_path.read_text(encoding="utf-8").splitlines()]
    keys_only = all(set(item) == {"utterance"} for item in model)
    mapping = (len(model) == len(order) == len(held) == 1760 and
               all(set(item) == {"id", "index"} for item in order) and
               {item["id"] for item in order} == set(held) and
               all(item["index"] == i and model[i]["utterance"] == held[item["id"]]["utterance"]
                   for i, item in enumerate(order)))
    boundary_order = [item["id"] for item in order if item["id"].startswith(("P-", "U-at_cap-", "U-cap_plus_one-"))]
    adjacent_pair_count = 0
    spec, _, _ = inputs()
    pair_by_id = {row_id: item["pair_id"] for row_id, item in spec.items()}
    for first, second in zip(boundary_order, boundary_order[1:]):
        adjacent_pair_count += pair_by_id[first] == pair_by_id[second]
    source_order = [r["id"] for r in rows if r["split"] == "held_out"]
    return {"rows": len(model), "utterance_only": keys_only, "private_order_mapping": mapping,
            "shuffled_from_corpus_order": [item["id"] for item in order] != source_order,
            "adjacent_boundary_pairs": adjacent_pair_count,
            "passed": keys_only and mapping and adjacent_pair_count == 0
                      and [item["id"] for item in order] != source_order}


def audit(rows: list[dict], *, include_historical: bool = True,
          model_path: Path = HERE / "model-input.jsonl",
          order_path: Path = HERE / "model-input-order.jsonl") -> dict:
    spec, writer, readers = inputs()
    raw = reader_scores(rows, spec, writer, readers)
    current = surface(rows, spec)
    pair_ok = all(item["pairs"] == item["exact_word_multisets"] ==
                  item["punctuation_inventory_matches"] == item["same_actor_target"] == 40
                  for item in current["pair_metrics"].values())
    timing = current["timing_noun_pattern_hits"]
    timing_ok = sum(timing.values()) < SUPPORT or max(timing.values()) / sum(timing.values()) < ALERT_PURITY
    cue_sweep = current["cue_sweep"]
    cue_ok = not cue_sweep["ordered_1_to_7_grams"] and not cue_sweep["character_substrings"]
    export = model_export(rows, model_path, order_path)
    report = {"artifact_sha256": ARTIFACT_SHA256, "reader_scores": raw,
              "v6_surface": current, "model_export": export,
              "v6_passed": raw["passed"] and pair_ok and timing_ok and cue_ok and export["passed"]
                           and current["fixed_condition_clause_hits"] == 0
                           and current["fixed_action_clause_hits"] == 0
                           and not current["copied_simple_bodies"]}
    if include_historical:
        old4 = surface(historical(V4), spec, scan_cues=False)
        old5 = surface(historical(V5), spec, scan_cues=False)
        v4_fixed_red = old4["fixed_condition_clause_hits"] == 160 and old4["fixed_action_clause_hits"] == 160
        v5_word_red = sum(x["exact_word_multisets"] for x in old5["pair_metrics"].values()) == 0
        v5_noun_red = old5["timing_noun_pattern_hits"] == {"T": 0, "W": 160}
        v5_copy_red = len(old5["copied_simple_bodies"]) == 160
        report["legacy_controls"] = {
            "v4_fixed_clause_red": v4_fixed_red,
            "v4_fixed_condition_hits": old4["fixed_condition_clause_hits"],
            "v4_fixed_action_hits": old4["fixed_action_clause_hits"],
            "v5_word_multiset_red": v5_word_red,
            "v5_exact_word_pairs": sum(x["exact_word_multisets"] for x in old5["pair_metrics"].values()),
            "v5_timing_noun_red": v5_noun_red,
            "v5_timing_noun_hits": old5["timing_noun_pattern_hits"],
            "v5_copied_body_red": v5_copy_red,
            "v5_copied_bodies": len(old5["copied_simple_bodies"]),
        }
        report["passed"] = report["v6_passed"] and all(
            report["legacy_controls"][key] for key in
            ("v4_fixed_clause_red", "v5_word_multiset_red", "v5_timing_noun_red", "v5_copied_body_red"))
    else:
        report["passed"] = report["v6_passed"]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=HERE / "corpus.jsonl")
    parser.add_argument("--model-input", type=Path, default=HERE / "model-input.jsonl")
    parser.add_argument("--model-order", type=Path, default=HERE / "model-input-order.jsonl")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines()]
    result = audit(rows, model_path=args.model_input, order_path=args.model_order)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(not result["passed"])
