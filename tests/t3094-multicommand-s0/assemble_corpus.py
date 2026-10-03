"""Package individually authored T-3094 utterances into immutable JSONL.

Every utterance is a complete literal line in a tracked source text file. This
module attaches IDs, split provenance and raw game event context; it performs
no utterance generation, phrase substitution or lexical variation.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from diversity import census, gate
from raw_identifiability import audit as raw_audit
from shortcut_audit import audit as shortcut_audit

HERE = Path(__file__).resolve().parent
ACTORS = ("alpha wing", "bravo wing", "charlie team", "delta scout", "echo battery",
          "foxtrot patrol", "golf squad", "hotel drone", "india unit", "juliet group")
TARGETS = ("north ridge", "east bridge", "south gate", "west tower", "upper deck",
           "lower tunnel", "river crossing", "supply depot", "radar mast", "landing zone")
GENERAL = ("conjunction", "punctuation_free", "correction", "shared_argument",
           "pronoun", "nested_condition", "parallel", "cancellation",
           "zero_action", "above_cap")
UNIQUE = ("correction", "shared_argument", "pronoun", "nested_condition",
          "parallel", "at_cap", "cap_plus_one")


def authored(name, expected):
    file = HERE / name
    lines = file.read_text(encoding="utf-8").splitlines()
    if len(lines) != expected or any(not line.strip() for line in lines):
        raise ValueError(f"{name}: expected {expected} complete utterances, got {len(lines)}")
    return lines


def event_trace(cls, actor, target, *, old_target=None, condition=None, arm=None):
    """Provide observations and queue events, never an expected segmentation."""
    trace = {"actor": actor, "target": target, "events": []}
    events = trace["events"]
    def queued(task, place=target):
        events.append({"event": "task_queued", "task": task, "place": place})
    if cls in ("conjunction", "pronoun"):
        queued("move")
        queued("scan")
    elif cls == "punctuation_free":
        queued("move")
        queued("hold")
    elif cls == "correction":
        events.append({"event": "utterance_revision", "discarded_place": old_target})
        queued("move")
        if condition == "unique":
            queued("scan")
    elif cls == "shared_argument":
        queued("scan")
        queued("hold")
    elif cls == "nested_condition":
        events.extend(({"event": "condition_source_registered", "name": "area_clear", "place": target},
                       {"event": "condition_source_registered", "name": "scout_confirmation", "place": target}))
        events.append({"event": "conditional_request_received", "task": "move", "place": target})
    elif cls == "parallel":
        queued("scan")
        events.append({"event": "task_queued", "task": "hold", "actor": "reserve wing", "place": "west tower"})
    elif cls == "cancellation":
        events.append({"event": "pending_movement_retracted", "place": target})
    elif cls == "zero_action":
        events.append({"event": "information_request_received", "topic_place": target})
    elif cls == "above_cap":
        start = TARGETS.index(target)
        for offset in range(5):
            queued("move", TARGETS[(start + offset) % len(TARGETS)])
    elif cls in ("at_cap", "cap_plus_one"):
        for task in ("move", "scan", "hold"):
            queued(task)
        if cls == "cap_plus_one":
            events.extend(({"event": "task_queued", "task": "signal_watch", "place": target, "task_id": "w1"},
                           {"event": "completion_callback_registered", "task_id": "w1"}))
        else:
            events.append({"event": "signal_listener_registered", "signal": "fire_mark"})
        queued("fire")
    elif cls == "ambiguous_executable":
        events.append({"event": "pending_movement_retracted", "place": target})
        queued("scan")
    elif cls == "development":
        queued("scan")
    elif cls == "pc2":
        if condition == "dependency":
            events.append({"event": "preexisting_scan_active", "place": target})
        elif condition == "ordering":
            events.append({"event": "preexisting_move_active", "place": target})
        if arm == "W":
            events.extend(({"event": "task_queued", "task": "signal_watch", "place": target, "task_id": "w1"},
                           {"event": "completion_callback_registered", "task_id": "w1"}))
            events.append({"event": "task_queued", "task": "fire", "place": target, "after_task_id": "w1"})
        else:
            events.append({"event": "signal_listener_registered", "signal": "fire_mark"})
            queued("fire")
    else:
        raise ValueError(cls)
    return trace


def row(identifier, utterance, source_class, trace, split, session, pair=None):
    case_id = "C-" + hashlib.sha256(("T3094-v5/" + identifier).encode()).hexdigest()[:16]
    return {"id": identifier, "annotation_case_id": case_id,
            "utterance": utterance, "source_class": source_class,
            "game_action_trace": trace, "split": split, "source_session": session,
            "speaker_id": "synthetic-player-" + session, "provenance": "individually_authored_synthetic",
            "pair": pair}


def build():
    result = []
    for cls in GENERAL:
        for i, utterance in enumerate(authored(f"general-{cls}.txt", 100)):
            actor, target = ACTORS[i % 10], TARGETS[i // 10]
            old = "west tower" if target == "south gate" else "south gate"
            result.append(row(f"G-{cls}-{i:03}", utterance, cls,
                              event_trace(cls, actor, target, old_target=old),
                              "held_out", f"H-G-{cls}-{i // 10:02}"))
    for cls in UNIQUE:
        for i, utterance in enumerate(authored(f"unique-{cls}.txt", 40)):
            actor, target = ACTORS[i % 10], TARGETS[i // 10]
            result.append(row(f"U-{cls}-{i:03}", utterance, cls,
                              event_trace(cls, actor, target, old_target="old checkpoint", condition="unique"),
                              "held_out", f"H-U-{cls}-{i // 10:02}"))
    pair_lines = authored("pc2-authored.tsv", 120)
    pair_counts = Counter()
    for parts in (line.split("\t") for line in pair_lines):
        if len(parts) != 5:
            raise ValueError("PC-2 source line must carry condition, actor, target and two full utterances")
        condition, actor, target, trigger_text, watch_text = parts
        if condition not in ("simple_trigger", "dependency", "ordering"):
            raise ValueError(f"unregistered PC-2 condition: {condition}")
        i = pair_counts[condition]
        pair_counts[condition] += 1
        pair = f"P-{condition}-{i:03}"
        for arm, utterance in (("T", trigger_text), ("W", watch_text)):
            result.append(row(f"{pair}-{arm}", utterance, "pc2",
                              event_trace("pc2", actor, target, condition=condition, arm=arm),
                              "held_out", f"H-P-{condition}-{i // 10:02}", pair=pair))
    if set(pair_counts.values()) != {40} or len(pair_counts) != 3:
        raise ValueError(f"PC-2 pair quotas: {pair_counts}")
    for cls in ("cancellation", "zero_action"):
        for i, utterance in enumerate(authored(f"dedicated-{cls}.txt", 100)):
            actor, target = ACTORS[i % 10], TARGETS[i // 10]
            result.append(row(f"N-{cls}-{i:03}", utterance, cls,
                              event_trace(cls, actor, target), "held_out", f"H-N-{cls}-{i // 10:02}"))
    for i, utterance in enumerate(authored("dedicated-ambiguous_executable.txt", 40)):
        actor, target = ACTORS[i % 10], TARGETS[i // 10]
        result.append(row(f"N-ambiguous_executable-{i:03}", utterance,
                          "ambiguous_executable", event_trace("ambiguous_executable", actor, target),
                          "held_out", f"H-N-ambiguous-{i // 10:02}"))
    for i, utterance in enumerate(authored("development.txt", 200)):
        actor, target = ACTORS[i % 10], TARGETS[(i // 10) % 10]
        result.append(row(f"D-{i:03}", utterance, "development", event_trace("development", actor, target),
                          "development", f"D-{i // 10:02}"))
    if len(result) != 1960 or len({r["id"] for r in result}) != 1960:
        raise ValueError("corpus quota or ID collision")
    if len({r["utterance"].casefold() for r in result}) != 1960:
        raise ValueError("duplicate authored utterance")
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = build()
    corpus = HERE / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in result),
                      encoding="utf-8", newline="\n")
    annotation_input = HERE / "annotation-input.jsonl"
    annotation_input.write_text("".join(json.dumps({
        "case_id": r["annotation_case_id"], "utterance": r["utterance"],
        "game_action_trace": r["game_action_trace"]}, ensure_ascii=False, sort_keys=True) + "\n"
        for r in result if r["split"] == "held_out"), encoding="utf-8", newline="\n")
    blind_input = HERE / "blind-annotation-input.jsonl"
    blind_input.write_text("".join(json.dumps({
        "case_id": r["annotation_case_id"], "utterance": r["utterance"]},
        ensure_ascii=False, sort_keys=True) + "\n"
        for r in result if r["split"] == "held_out"), encoding="utf-8", newline="\n")
    # This is the only candidate-facing payload. Line order maps to the
    # evaluator's private corpus order; no ID, class, arm, trace or label ships.
    model_input = HERE / "model-input.jsonl"
    model_input.write_text("".join(json.dumps({"utterance": r["utterance"]}, ensure_ascii=False, sort_keys=True) + "\n"
        for r in result if r["split"] == "held_out"), encoding="utf-8", newline="\n")
    identifiability = raw_audit(result)
    identifiability_file = HERE / "identifiability-report.json"
    identifiability_file.write_text(json.dumps(identifiability, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8", newline="\n")
    if not identifiability["passed"]:
        raise ValueError(f"v5 raw-only identifiability freeze failed: {identifiability['failure_count']} rows")
    shortcut = shortcut_audit(result)
    shortcut_file = HERE / "shortcut-report.json"
    shortcut_file.write_text(json.dumps(shortcut, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8", newline="\n")
    if not shortcut["passed"]:
        raise ValueError("v5 n-gram shortcut freeze failed")
    diversity = census(result)
    diversity["gate_failures"] = gate(diversity, json.loads((HERE / "freeze.json").read_text())["diversity_gates"])
    diversity_file = HERE / "diversity-report.json"
    diversity_file.write_text(json.dumps(diversity, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8", newline="\n")
    if diversity["gate_failures"]:
        raise ValueError("v5 diversity freeze failed: " + "; ".join(diversity["gate_failures"]))
    counts = Counter(r["source_class"] for r in result if r["split"] == "held_out")
    source_files = sorted([*HERE.glob("general-*.txt"), *HERE.glob("unique-*.txt"),
                           *HERE.glob("dedicated-*.txt"), HERE / "development.txt", HERE / "pc2-authored.tsv"])
    manifest = {"rows": len(result), "held_out": 1760, "development": 200,
                "counts": dict(sorted(counts.items())), "sha256": sha(corpus),
                "annotation_input_sha256": sha(annotation_input),
                "blind_annotation_input_sha256": sha(blind_input),
                "model_input_sha256": sha(model_input),
                "identifiability_report_sha256": sha(identifiability_file),
                "shortcut_report_sha256": sha(shortcut_file),
                "shortcut_v4_negative_report_sha256": sha(HERE / "shortcut-v4-negative-report.json"),
                "diversity_report_sha256": sha(diversity_file),
                "authored_source_sha256": {p.name: sha(p) for p in source_files},
                "guide_sha256": sha(HERE / "annotation-guide.md"),
                "baseline_prompt_sha256": sha(HERE / "baseline-prompt.txt"),
                "schema_sha256": sha(HERE / "command.schema.json"),
                "freeze_sha256": sha(HERE / "freeze.json"),
                "resource_counts_sha256": sha(HERE / "resource-model-counts.json"),
                "assembler_sha256": sha(HERE / "assemble_corpus.py"),
                "diversity_script_sha256": sha(HERE / "diversity.py"),
                "identifiability_script_sha256": sha(HERE / "raw_identifiability.py"),
                "shortcut_script_sha256": sha(HERE / "shortcut_audit.py"),
                "attributes_sha256": sha(HERE / ".gitattributes"),
                "keyword_catalogue_sha256": sha(HERE / "keyword_catalogue.py")}
    (HERE / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                                encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
