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
            if cls == "conjunction":
                text = f"Move {actor} to {target}, then scan {target}."
            elif cls == "punctuation_free":
                text = f"move {actor} to {target} then hold there"
            elif cls == "correction":
                text = f"Send {actor} to the south gate—correction, move them to {target}."
            elif cls == "shared_argument":
                text = f"Have {actor} scan and hold {target}."
            elif cls == "pronoun":
                text = f"Move {actor} to {target}; have them scan it."
            elif cls == "nested_condition":
                text = f"If {target} is clear, move {actor} there after the scout reports in."
            elif cls == "parallel":
                text = f"While {actor} scans {target}, tell bravo wing to hold the west tower."
            elif cls == "cancellation":
                text = f"Cancel the move order for {actor} at {target}; do nothing now."
            elif cls == "zero_action":
                text = f"Did {actor} report anything about {target}? This is only a question."
            else:
                text = f"Send {actor} through {target}, north ridge, east bridge, and south gate; one move per location."
            rows.append(row(f"G-{cls}-{i:03}", text, cls, {"kind": cls, "actor": actor, "target": target}))
    unique = ("correction", "shared_argument", "pronoun", "nested_condition", "parallel", "at_cap", "cap_plus_one")
    for cls in unique:
        for i in range(40):
            actor, target = ACTORS[i % 10], TARGETS[(i // 10) % 10]
            if cls == "at_cap":
                text = f"At cap four: {actor} move to {target}, scan, hold, then fire on signal."
            elif cls == "cap_plus_one":
                text = f"Above cap four: {actor} move to {target}, scan, hold, wait for signal, then fire."
            else:
                text = f"{cls.replace('_', ' ')} case: {actor} to {target}; scan it, then hold position."
            rows.append(row(f"U-{cls}-{i:03}", text, cls, {"kind": cls, "actor": actor, "target": target}))
    for condition in ("simple_trigger", "dependency", "ordering"):
        for i in range(40):
            actor, target = ACTORS[i % 10], TARGETS[(i // 10) % 10]
            pair = f"P-{condition}-{i:03}"
            if condition == "simple_trigger":
                base = f"{actor}, fire at {target}"
            elif condition == "dependency":
                base = f"After the scan, {actor} fire at {target}"
            else:
                base = f"Once the move completes, {actor} fire at {target}"
            rows.append(row(f"{pair}-T", base + " when my signal arrives; no wait action is scheduled.",
                            "pc2_trigger", {"wait_scheduled": False, "action": "fire", "actor": actor,
                                            "target": target, "condition": condition}, pair=pair))
            rows.append(row(f"{pair}-W", base + "; schedule and track a wait for my signal, then fire after it completes.",
                            "pc2_tracked_wait", {"wait_scheduled": True, "wait_has_completion": True,
                                                 "action": "fire", "actor": actor, "target": target,
                                                 "condition": condition}, pair=pair))
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
                "schema_sha256": hashlib.sha256((HERE / "command.schema.json").read_bytes()).hexdigest(),
                "freeze_sha256": hashlib.sha256((HERE / "freeze.json").read_bytes()).hexdigest(),
                "resource_counts_sha256": hashlib.sha256((HERE / "resource-model-counts.json").read_bytes()).hexdigest()}
    (HERE / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
