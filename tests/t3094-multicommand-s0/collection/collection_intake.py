"""T-3103 structural intake for natural-utterance submissions (collection-spec.md §5).

Exit 0 + "INTAKE OK" when every submission is structurally valid (and, with --quotas, every
held-out quota is met). Exit 1 + "REFUSED:" lines otherwise. Intake is not a naturalness,
identifiability or shortcut verdict, and it labels nothing.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

KEYS = {"row_id", "utterance", "writer_id", "session_id", "speaker_id", "source_kind", "commission",
        "writer_flag", "writer_note"}
SOURCE_KINDS = {"synthetic_writer", "recorded_player"}
QUOTAS = {"conjunction": 100, "punctuation_free": 100, "correction": 140, "shared_argument": 140,
          "pronoun": 140, "nested_condition": 140, "parallel": 140, "cancellation": 100, "zero_action": 100,
          "above_cap": 100, "at_cap": 40, "cap_plus_one": 40, "ambiguous_executable": 40}
DEVELOPMENT_MINIMUM = 200
COMMISSIONS = set(QUOTAS) | {"development"}
LABEL_WORDS = re.compile(r"(intent|command|gold|label|trigger|condition|cue|watch|arm|status|route|"
                         r"expected|answer|count|action|split|stratum|class)", re.I)


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def intake(paths: list[Path], quotas: bool) -> list[str]:
    errors: list[str] = []
    seen: dict[str, str] = {}
    sessions: dict[tuple[str, str], set] = defaultdict(set)
    speakers: dict[tuple[str, str], set] = defaultdict(set)
    counts: Counter = Counter()
    ids: set = set()
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            where = f"{path.name}:{number}"
            if not line.strip():
                continue
            row = json.loads(line)
            keys = set(row)
            if keys != KEYS:
                extra = sorted(keys - KEYS)
                labelled = [k for k in extra if LABEL_WORDS.search(k)]
                errors.append(f"{where}: keys {sorted(keys)}" + (f" carry label-bearing {labelled}" if labelled else ""))
                continue
            if not isinstance(row["utterance"], str) or not row["utterance"].strip():
                errors.append(f"{where}: empty utterance")
                continue
            if row["source_kind"] not in SOURCE_KINDS or row["commission"] not in COMMISSIONS \
                    or not isinstance(row["writer_flag"], bool):
                errors.append(f"{where}: bad source_kind/commission/writer_flag")
                continue
            for field in ("row_id", "writer_id", "session_id", "speaker_id"):
                if not isinstance(row[field], str) or not row[field].strip():
                    errors.append(f"{where}: empty {field}")
            key = (row["writer_id"], row["row_id"])
            if key in ids:
                errors.append(f"{where}: duplicate row_id {row['row_id']} for writer {row['writer_id']}")
            ids.add(key)
            norm = normalize(row["utterance"])
            if norm in seen:
                errors.append(f"{where}: duplicate utterance of {seen[norm]}")
            seen[norm] = where
            session = (row["writer_id"], row["session_id"])
            sessions[session].add("development" if row["commission"] == "development" else "held_out")
            speakers[session].add(row["speaker_id"])
            if row["writer_flag"]:
                errors.append(f"{where}: writer-flagged row must be rewritten before intake")
            counts[row["commission"]] += 1
    for session, splits in sessions.items():
        if len(splits) > 1:
            errors.append(f"session {session} spans development and held-out")
        if len(speakers[session]) > 1:
            errors.append(f"session {session} has {len(speakers[session])} speakers")
    print("COUNTS " + json.dumps(dict(sorted(counts.items()))))
    if quotas:
        for name, minimum in QUOTAS.items():
            if counts[name] < minimum:
                errors.append(f"quota {name}: {counts[name]} < {minimum}")
        if counts["development"] < DEVELOPMENT_MINIMUM:
            errors.append(f"quota development: {counts['development']} < {DEVELOPMENT_MINIMUM}")
    return errors


def main(argv: list[str]) -> int:
    quotas = "--quotas" in argv
    paths = [Path(a) for a in argv if a != "--quotas"]
    if not paths:
        print("usage: collection_intake.py <submission.jsonl>... [--quotas]")
        return 2
    errors = intake(paths, quotas)
    for error in errors:
        print("REFUSED: " + error)
    if errors:
        return 1
    print("INTAKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
