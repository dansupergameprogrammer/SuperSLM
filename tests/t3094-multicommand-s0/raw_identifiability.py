"""A raw-utterance-only necessary check for the signal/wait distinction.

The classifier reads only the text the model receives. Expected behavior is
consulted afterwards, solely to score that independent reading. A bare
"watch", sequence word, or trace event cannot satisfy the evidence rule.
"""
from __future__ import annotations

import json
import argparse
import hashlib
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ACTION = re.compile(
    r"\b(queue|open|start|log|assign|register|schedule|create|add|activate)\s+"
    r"(?:a\s+|an\s+|the\s+)?(?:separate|independent|standalone|own)\s+"
    r"(?:signal\s+)?(?:watch|wait|hold)\s+(?:task|order|job|entry)\b",
    re.I,
)
WAIT_COMPLETION = re.compile(r"\b(?:task|order|job|entry)\s+(?:complet(?:e|es|ion)|finish(?:es)?|close(?:s)?)\b", re.I)
NEGATION = re.compile(r"\b(?:do\s+not|don['’]t|never)\s+$", re.I)
NO_TASK = re.compile(
    r"\b(?:no|without)\s+(?:a\s+|an\s+|the\s+)?"
    r"(?:separate|independent|standalone|own)\s+(?:signal\s+)?"
    r"(?:watch|wait|hold)\s+(?:task|order|job|entry)\b", re.I,
)


def reading(utterance: str) -> str:
    """Return action, condition, or ambiguous using utterance text alone."""
    affirmative = negative = False
    for match in ACTION.finditer(utterance):
        prefix = utterance[max(0, match.start() - 20):match.start()]
        if NEGATION.search(prefix):
            negative = True
        else:
            affirmative = True
    negative |= bool(NO_TASK.search(utterance))
    if affirmative and not negative and WAIT_COMPLETION.search(utterance):
        return "action"
    if negative and not affirmative:
        return "condition"
    return "ambiguous"


def audit(rows: list[dict]) -> dict:
    tested = [r for r in rows if r["source_class"] in {"pc2", "cap_plus_one", "at_cap"}]
    failures = []
    counts = {"action": 0, "condition": 0, "ambiguous": 0}
    for row in tested:
        got = reading(row["utterance"])
        counts[got] += 1
        expected = "action" if (row["source_class"] == "cap_plus_one" or
                                row["source_class"] == "pc2" and row["id"].endswith("-W")) else "condition"
        if got != expected:
            failures.append({"id": row["id"], "expected": expected, "raw_reading": got})
    return {"rows": len(tested), "readings": counts, "failures": failures,
            "failure_count": len(failures), "passed": not failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=HERE / "corpus.jsonl")
    args = parser.parse_args()
    raw = args.corpus.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    result = audit(rows)
    result["corpus_sha256"] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(result, indent=2))
    raise SystemExit(not result["passed"])


if __name__ == "__main__":
    main()
