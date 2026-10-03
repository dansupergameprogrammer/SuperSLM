"""Deterministically freeze T-3094 raw utterances before any model output.

No model or grader is imported. Annotation labels are deliberately absent: two
independent annotators provide them, followed by adjudication.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACTORS = ("alpha wing", "bravo wing", "charlie team", "delta scout", "echo battery",
          "foxtrot patrol", "golf squad", "hotel drone", "india unit", "juliet group")
TARGETS = ("north ridge", "east bridge", "south gate", "west tower", "upper deck",
           "lower tunnel", "river crossing", "supply depot", "radar mast", "landing zone")


def row(identifier, text, source_class, trace, split="held_out", pair=None):
    return {"id": identifier, "utterance": text, "source_class": source_class,
            "game_action_trace": trace, "split": split,
            "source_session": f"{split}-{source_class}-{identifier.split('-')[-1]}", "pair": pair}


def build():
    rows = []
    classes = (
        "conjunction", "punctuation_free", "correction", "shared_argument", "pronoun",
        "nested_condition", "parallel", "cancellation", "zero_action", "above_cap")
    for cls in classes:
        for i in range(100):
            actor, target = ACTORS[i % 10], TARGETS[(i // 10) % 10]
            variant = ((i % 10) + (i // 10)) % 4
            old_target = "south gate" if target != "south gate" else "west tower"
            if cls == "conjunction":
                text = (
                    f"Move {actor} to {target}, then scan {target}.",
                    f"Have {actor} move to {target}; afterward scan that location.",
                    f"First send {actor} to {target} and then scan the area.",
                    f"{actor} should move to {target} and scan once there.",
                )[variant]
            elif cls == "punctuation_free":
                text = (
                    f"move {actor} to {target} then hold there",
                    f"uh send {actor} to {target} and hold position",
                    f"{actor} move to {target} then hold it",
                    f"please move {actor} to {target} hold on arrival",
                )[variant]
            elif cls == "correction":
                text = (
                    f"Send {actor} to {old_target}—correction, move them to {target}.",
                    f"Move {actor} to {old_target}; no, change the destination to {target}.",
                    f"For {actor}, strike the {old_target} destination and use {target} instead.",
                    f"{actor} to {old_target}—actually make that {target}.",
                )[variant]
            elif cls == "shared_argument":
                text = (
                    f"Have {actor} scan and hold {target}.",
                    f"Tell {actor} to scan {target} and then hold the same place.",
                    f"{actor}: scan {target}, then hold it.",
                    f"At {target}, have {actor} scan before holding position there.",
                )[variant]
            elif cls == "pronoun":
                text = (
                    f"Move {actor} to {target}; have them scan it.",
                    f"Send {actor} toward {target}; when they arrive, scan it.",
                    f"{actor} should move to {target}. Then they should scan that location.",
                    f"Get {actor} to {target}, and have them scan it afterward.",
                )[variant]
            elif cls == "nested_condition":
                text = (
                    f"If {target} is clear, move {actor} there after the scout reports in.",
                    f"Move {actor} to {target} only if it is clear and the scout confirms.",
                    f"If the scout reports {target} clear, then and only then send {actor} there.",
                    f"{actor} moves to {target} when both the clearance and scout report are in.",
                )[variant]
            elif cls == "parallel":
                text = (
                    f"While {actor} scans {target}, tell reserve wing to hold the west tower.",
                    f"At the same time, {actor} scans {target} and reserve wing holds the west tower.",
                    f"Run two parallel orders: {actor} scans {target}; reserve wing holds west tower.",
                    f"Have reserve wing hold west tower as {actor} scans {target}.",
                )[variant]
            elif cls == "cancellation":
                text = (
                    f"Cancel the move order for {actor} at {target}; do nothing now.",
                    f"Never mind the {actor} order for {target}. No replacement order.",
                    f"Stand down {actor}'s pending move to {target}; issue nothing else.",
                    f"Withdraw the instruction for {actor} at {target}, with no new action.",
                )[variant]
            elif cls == "zero_action":
                text = (
                    f"Did {actor} report anything about {target}? This is only a question.",
                    f"What was {actor}'s last observation at {target}? No order here.",
                    f"I wonder whether {actor} has reached {target}; I am not issuing a command.",
                    f"Any news from {actor} near {target}? Please just answer.",
                )[variant]
            else:
                text = (
                    f"From the staging bunker near {target}, send {actor} in order to north ridge, east bridge, south gate, west tower and upper deck; record each move separately.",
                    f"Starting near {target}, order {actor} to move to north ridge, then east bridge, then south gate, then west tower, then upper deck.",
                    f"Give {actor} five moves from the bunker by {target}: north ridge, east bridge, south gate, west tower, upper deck, in that order.",
                    f"{actor} departs the staging area by {target}; send them to north ridge, east bridge, south gate, west tower and upper deck in sequence.",
                )[variant]
            trace = {"kind": cls, "actor": actor, "target": target}
            if cls == "above_cap":
                trace.update(item_cap=4, expected_action_items=5)
            elif cls == "correction":
                trace.update(retracted_target=old_target, effective_target=target)
            elif cls == "shared_argument":
                trace.update(shared_target=target, ordered_actions=["scan", "hold"])
            elif cls == "pronoun":
                trace.update(they_refers_to=actor, it_refers_to=target)
            elif cls == "nested_condition":
                trace.update(target_clear=True, scout_confirmed=True)
            elif cls == "parallel":
                trace.update(parallel_actors=[actor, "reserve wing"])
            rows.append(row(f"G-{cls}-{i:03}", text, cls, trace))
    unique = ("correction", "shared_argument", "pronoun", "nested_condition", "parallel", "at_cap", "cap_plus_one")
    for cls in unique:
        for i in range(40):
            actor, target = ACTORS[i % 10], TARGETS[(i // 10) % 10]
            if cls == "at_cap":
                text = f"At cap four: {actor} move to {target}, scan, hold, then fire on signal."
            elif cls == "cap_plus_one":
                text = f"Above cap four: {actor} move to {target}, scan, hold, wait for signal, then fire."
            elif cls == "correction":
                text = f"Send {actor} to the old checkpoint; no, change that: move to {target} and scan there."
            elif cls == "shared_argument":
                text = f"Have {actor} scan and then hold {target}; both orders concern that same location."
            elif cls == "pronoun":
                text = f"Move {actor} to {target}; once they arrive, have them scan it."
            elif cls == "nested_condition":
                text = f"If {target} is clear, and only if {actor}'s scout confirms it, move there; otherwise hold."
            elif cls == "parallel":
                text = f"While {actor} scans {target}, order reserve wing to hold the west tower at the same time."
            else:
                raise AssertionError(cls)
            trace = {"kind": cls, "actor": actor, "target": target}
            if cls in ("at_cap", "cap_plus_one"):
                trace.update(item_cap=4, expected_action_items=4 if cls == "at_cap" else 5,
                             wait_scheduled=cls == "cap_plus_one",
                             wait_has_completion=cls == "cap_plus_one")
            elif cls == "correction":
                trace.update(retracted_target="old checkpoint", effective_target=target)
            elif cls == "shared_argument":
                trace.update(shared_target=target, ordered_actions=["scan", "hold"])
            elif cls == "pronoun":
                trace.update(they_refers_to=actor, it_refers_to=target)
            elif cls == "nested_condition":
                trace.update(target_clear=(i % 2 == 0), scout_confirmed=(i % 3 != 0))
            elif cls == "parallel":
                trace.update(parallel_actors=[actor, "reserve wing"])
            rows.append(row(f"U-{cls}-{i:03}", text, cls, trace))
    for condition in ("simple_trigger", "dependency", "ordering"):
        for i in range(40):
            actor, target = ACTORS[i % 10], TARGETS[(i // 10) % 10]
            pair = f"P-{condition}-{i:03}"
            if condition == "simple_trigger":
                base = f"{actor}, fire at {target}"
            elif condition == "dependency":
                base = f"When the already-scheduled scan finishes, {actor}, fire at {target}"
            else:
                base = f"After the existing move order completes, {actor}, fire at {target}"
            rows.append(row(f"{pair}-T", base + " when my signal arrives; do not schedule a separate wait.",
                            "pc2_trigger", {"wait_scheduled": False, "action": "fire", "actor": actor,
                                            "target": target, "condition": condition,
                                            "expected_action_items": 1,
                                            "external_dependency_already_scheduled": condition != "simple_trigger"}, pair=pair))
            rows.append(row(f"{pair}-W", base + "; schedule and track a wait for my signal, then fire after that wait completes.",
                            "pc2_tracked_wait", {"wait_scheduled": True, "wait_has_completion": True,
                                                 "action": "fire", "actor": actor, "target": target,
                                                 "condition": condition,
                                                 "expected_action_items": 2,
                                                 "external_dependency_already_scheduled": condition != "simple_trigger"}, pair=pair))
    for cls in ("cancellation", "zero_action"):
        for i in range(100):
            actor, target = ACTORS[i % 10], TARGETS[(i // 10) % 10]
            text = (f"Cancel every pending instruction for {actor} at {target}; take no new action."
                    if cls == "cancellation" else
                    f"What was {actor}'s last report about {target}? I am asking, not issuing an order.")
            rows.append(row(f"N-{cls}-{i:03}", text, cls,
                            {"kind": cls, "authorized_actions": 0}))
    for i in range(40):
        actor, target = ACTORS[i % 10], TARGETS[(i // 10) % 10]
        rows.append(row(f"N-ambiguous_executable-{i:03}",
                        f"Cancel the old move for {actor}, but issue a new scan of {target} now.",
                        "ambiguous_executable", {"old_action_cancelled": True, "new_action": "scan",
                                                 "actor": actor, "target": target}))
    # Development is disjoint by source session and phrase, and is never a test denominator.
    for i in range(200):
        actor, target = ACTORS[i % 10], TARGETS[(i // 10) % 10]
        rows.append(row(f"D-{i:03}", f"Training transcript {i}: order {actor} to scan {target}.",
                        "development", {"action": "scan", "actor": actor, "target": target}, split="development"))
    ids = [r["id"] for r in rows]
    texts = [r["utterance"] for r in rows]
    assert len(ids) == len(set(ids)) and len(texts) == len(set(texts))
    return rows


def main():
    rows = build()
    output = HERE / "corpus.jsonl"
    output.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8", newline="\n")
    counts = {c: sum(r["source_class"] == c and r["split"] == "held_out" for r in rows)
              for c in sorted({r["source_class"] for r in rows if r["split"] == "held_out"})}
    manifest = {"rows": len(rows), "held_out": sum(r["split"] == "held_out" for r in rows),
                "development": sum(r["split"] == "development" for r in rows),
                "counts": counts,
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "guide_sha256": hashlib.sha256((HERE / "annotation-guide.md").read_bytes()).hexdigest(),
                "baseline_prompt_sha256": hashlib.sha256((HERE / "baseline-prompt.txt").read_bytes()).hexdigest(),
                "schema_sha256": hashlib.sha256((HERE / "command.schema.json").read_bytes()).hexdigest(),
                "freeze_sha256": hashlib.sha256((HERE / "freeze.json").read_bytes()).hexdigest(),
                "resource_counts_sha256": hashlib.sha256((HERE / "resource-model-counts.json").read_bytes()).hexdigest()}
    (HERE / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
