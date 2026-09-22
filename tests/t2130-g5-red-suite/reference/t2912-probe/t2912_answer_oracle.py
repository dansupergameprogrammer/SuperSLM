"""Commissioned T-2912 answer-value oracle.

The oracle classifies the decoded Prompt_Result value, not its first token.  It is deliberately
independent of the grammar compiler.  A value is a usable answer when it contains at least one
alphanumeric code point, is not punctuation/whitespace only, is not a same-digit repetition of
length four or greater, and selected no tokenizer special ID.

Commissioning mode reads a JSON array of {value, special_selected} rows and exits zero only when
every row is an answer.  Thus the same path supplies a must-accept (exit 0) and a must-reject
(nonzero) construction.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import unicodedata
from typing import Any


def display_value(value: Any) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    prefix = "<unparsed> "
    return text[len(prefix):] if text.startswith(prefix) else text


def is_punctuation_or_whitespace(text: str) -> bool:
    return bool(text) and all(c.isspace() or unicodedata.category(c).startswith("P") for c in text)


def is_degenerate_repetition(text: str) -> bool:
    compact = "".join(c for c in text if not c.isspace())
    return len(compact) >= 4 and compact.isdigit() and len(set(compact)) == 1


def is_answer(value: Any, special_selected: bool = False) -> bool:
    text = display_value(value)
    if special_selected or not text or is_punctuation_or_whitespace(text):
        return False
    if is_degenerate_repetition(text):
        return False
    return any(c.isalnum() for c in text)


def commission_fixture(path: pathlib.Path) -> int:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("fixture must be a non-empty JSON array")
    accepted = 0
    for index, row in enumerate(rows):
        answer = is_answer(row["value"], bool(row.get("special_selected", False)))
        accepted += int(answer)
        print(f"{index:02d} answer={str(answer).lower()} value={row['value']!r}")
    print(f"accepted={accepted}/{len(rows)}")
    return 0 if accepted == len(rows) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=pathlib.Path)
    args = parser.parse_args()
    return commission_fixture(args.fixture)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"oracle input error: {exc}", file=sys.stderr)
        raise SystemExit(2)
