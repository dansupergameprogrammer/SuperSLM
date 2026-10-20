"""Export a meaning-only brief for the independent T-3100 boundary writer.

This program never uses an old utterance to form a spec record. The frozen
corpus supplies row identity and scheduler events; semantic notes below state
the action distinction, leaving all player language to the independent writer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess


HERE = Path(__file__).resolve().parent
DESTINATION = Path("D:/_artifacts/SuperSLM/t3100")
SOURCE_COMMITS = (
    "d872be238f9cf7c1c90ef5c2fc745e2db73178a7",  # v3
    "7a5cddd1a3a2924efe94600256c052498b72c4ab",  # v4
)

# These describe the scheduler semantics. They are never candidate utterances.
MEANINGS = {
    ("pc2/simple_trigger", "T"): (
        "One newly requested operation attacks its assigned location. An external cue "
        "governs its execution; the player does not request independent monitoring "
        "producing a completion event."
    ),
    ("pc2/simple_trigger", "W"): (
        "Begin an independently managed monitoring operation for an external cue. "
        "Its completion authorizes another attacking operation."
    ),
    ("pc2/dependency", "T"): (
        "A survey is currently underway and is contextual history, not a newly "
        "issued command. Its completion and an external cue govern the sole new "
        "attack; no independent monitoring operation is added."
    ),
    ("pc2/dependency", "W"): (
        "A current survey is contextual history. Following its finish, "
        "initiate a new, independently managed monitoring operation for a cue; "
        "attack upon that operation's completion."
    ),
    ("pc2/ordering", "T"): (
        "Relocation is currently underway and is contextual history. The sole "
        "new operation is an attack gated by relocation completion and an "
        "external cue; no independent monitoring is requested."
    ),
    ("pc2/ordering", "W"): (
        "Relocation currently underway must finish before a newly requested "
        "monitoring operation begins. That operation is managed independently; "
        "its completion authorizes an attack."
    ),
    ("cap_boundary/at_cap", "T"): (
        "Four new operations are requested consecutively: relocation, reconnaissance, "
        "position maintenance, and attack. An external cue governs operation four "
        "operation without creating an independently managed monitor."
    ),
    ("cap_boundary/cap_plus_one", "W"): (
        "Five new operations are requested consecutively: relocation, reconnaissance, "
        "position maintenance, independent monitoring of an external cue, "
        "and attack following monitor completion."
    ),
}


def command(action: str, subject: str, target: str, trigger: str,
            depends_on_previous: bool) -> dict:
    return {"action": action, "subject": subject, "target": target,
            "trigger": trigger, "depends_on_previous": depends_on_previous}


def expected_events(stratum: str, arm: str) -> list[str]:
    if stratum.startswith("pc2/"):
        prefix = {"simple_trigger": [], "dependency": ["preexisting_scan_active"],
                  "ordering": ["preexisting_move_active"]}[stratum.split("/")[1]]
        return prefix + (["signal_listener_registered", "task_queued"] if arm == "T"
                         else ["task_queued", "completion_callback_registered", "task_queued"])
    return (["task_queued"] * 3 +
            (["signal_listener_registered", "task_queued"] if arm == "T"
             else ["task_queued", "completion_callback_registered", "task_queued"]))


def intended_commands(stratum: str, arm: str, subject: str, target: str) -> dict:
    if stratum.startswith("pc2/"):
        variant = stratum.split("/")[1]
        prerequisite = {"simple_trigger": "immediate", "dependency": "prior_scan_complete",
                        "ordering": "prior_move_complete"}[variant]
        if arm == "T":
            fire_trigger = ("external_cue" if variant == "simple_trigger" else
                            prerequisite + "_and_external_cue")
            items = [command("fire", subject, target, fire_trigger, False)]
        else:
            items = [command("wait", subject, target, prerequisite, False),
                     command("fire", subject, target, "tracked_wait_complete", True)]
    else:
        items = [command("move", subject, target, "immediate", False),
                 command("scan", subject, target, "previous_complete", True),
                 command("hold", subject, target, "previous_complete", True)]
        if arm == "W":
            items.append(command("wait", subject, target, "previous_complete", True))
        items.append(command("fire", subject, target,
                             "tracked_wait_complete" if arm == "W" else "external_cue",
                             True))
    return {"commands": items}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def old_utterances() -> list[str]:
    source = "tests/t3094-multicommand-s0/corpus.jsonl"
    utterances = []
    for commit in SOURCE_COMMITS:
        raw = subprocess.check_output(["git", "show", f"{commit}:{source}"],
                                      cwd=HERE, text=True, encoding="utf-8")
        utterances.extend(json.loads(line)["utterance"] for line in raw.splitlines())
    utterances.extend(json.loads(line)["utterance"] for line in
                      (HERE / "corpus.jsonl").read_text(encoding="utf-8").splitlines())
    return utterances


def main() -> None:
    manifest = json.loads((HERE / "corpus-manifest.json").read_text(encoding="utf-8"))
    raw = (HERE / "corpus.jsonl").read_bytes()
    assert sha256(raw).hexdigest() == manifest["sha256"], "source corpus differs from v5 freeze"
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    boundary = [row for row in rows if row["source_class"] in ("pc2", "at_cap", "cap_plus_one")]
    assert len(boundary) == 320
    output = []
    pairs: dict[str, list[dict]] = defaultdict(list)
    for row in boundary:
        source_class = row["source_class"]
        if source_class == "pc2":
            assert re.fullmatch(r"P-(simple_trigger|dependency|ordering)-\d{3}-[TW]", row["id"])
            stratum = "pc2/" + row["id"].split("-")[1]
            pair_id = row["pair"]
            arm = row["id"][-1]
        else:
            assert re.fullmatch(r"U-(at_cap|cap_plus_one)-\d{3}", row["id"])
            index = row["id"].split("-")[-1]
            stratum = "cap_boundary/" + source_class
            pair_id = f"C-boundary-{index}"
            arm = "T" if source_class == "at_cap" else "W"
        trace = row["game_action_trace"]
        events = trace["events"]
        assert [event["event"] for event in events] == expected_events(stratum, arm), row["id"]
        tasks = [event["task"] for event in events if event["event"] == "task_queued"]
        assert tasks == (["fire"] if stratum.startswith("pc2/") and arm == "T" else
                         ["signal_watch", "fire"] if stratum.startswith("pc2/") else
                         ["move", "scan", "hold", "fire"] if arm == "T" else
                         ["move", "scan", "hold", "signal_watch", "fire"]), row["id"]
        if arm == "W":
            assert sum(event["event"] == "completion_callback_registered" for event in events) == 1
        else:
            assert sum(event["event"] == "signal_listener_registered" for event in events) == 1
        assert trace["actor"] and trace["target"]
        spec = {"row_id": row["id"], "stratum": stratum, "pair_id": pair_id,
                "arm": arm,
                "intended_commands": intended_commands(stratum, arm, trace["actor"], trace["target"]),
                "meaning": MEANINGS[(stratum, arm)]}
        output.append(spec)
        pairs[pair_id].append(spec)
    assert len(pairs) == 160
    assert all(len(pair) == 2 and {record["arm"] for record in pair} == {"T", "W"}
               and len({(cmd["subject"], cmd["target"]) for record in pair
                        for cmd in record["intended_commands"]["commands"]}) == 1
               for pair in pairs.values())
    assert Counter(record["stratum"] for record in output) == {
        "pc2/simple_trigger": 80, "pc2/dependency": 80, "pc2/ordering": 80,
        "cap_boundary/at_cap": 40, "cap_boundary/cap_plus_one": 40}
    schema = json.loads((HERE / "command.schema.json").read_text(encoding="utf-8"))
    item_schema = schema["properties"]["commands"]["items"]
    for record in output:
        commands = record["intended_commands"]["commands"]
        assert len(commands) == (1 if record["stratum"].startswith("pc2/") and record["arm"] == "T"
                                 else 2 if record["stratum"].startswith("pc2/") else
                                 4 if record["arm"] == "T" else 5)
        for item in commands:
            assert set(item) == set(item_schema["required"])
            assert item["action"] in item_schema["properties"]["action"]["enum"]
            assert all(isinstance(item[key], str) and item[key] for key in
                       ("action", "subject", "target", "trigger"))
            assert isinstance(item["depends_on_previous"], bool)
    # Protect the independent writer from copied old sentence fragments. This
    # scans meaning prose only: actor and target names must remain in commands.
    prior = set()
    for utterance in old_utterances():
        tokens = words(utterance)
        prior.update(tuple(tokens[i:i + 2]) for i in range(len(tokens) - 1))
    overlap = sorted({" ".join(ngram) for text in MEANINGS.values()
                      for tokens in [words(text)]
                      for ngram in (tuple(tokens[i:i + 2]) for i in range(len(tokens) - 1))
                      if ngram in prior})
    assert not overlap, f"meaning prose copied prior two-word fragments: {overlap}"
    output.sort(key=lambda record: (record["pair_id"], record["arm"]))
    DESTINATION.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                      for record in output).encode("utf-8")
    destination = DESTINATION / "boundary-spec.jsonl"
    destination.write_bytes(content)
    digest = sha256(content).hexdigest()
    (DESTINATION / "boundary-spec.sha256").write_text(
        f"{digest}  boundary-spec.jsonl\n", encoding="ascii", newline="\n")
    print(f"rows={len(output)} pairs={len(pairs)} sha256={digest}")


if __name__ == "__main__":
    main()
