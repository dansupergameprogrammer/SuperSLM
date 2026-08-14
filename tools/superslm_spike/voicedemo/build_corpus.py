#!/usr/bin/env python3
"""P1 build entry point: materialize the corpus, Loki's two red fixtures, and their provenance
record under `corpus/` (this directory). Run with `python -m superslm_spike.voicedemo.build_corpus`
from `Tools/`, or `python build_corpus.py` from this directory.

Regenerates deterministically from `event_bank.py` and `validation_reactions.py` -- no randomness
anywhere in this pipeline, so re-running produces byte-identical output and the provenance hashes are
stable across machines (once `.gitattributes`' `eol=lf` on `*.json` is honored, which this repo's
root `.gitattributes` already sets).
"""

from __future__ import annotations

import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    # Allow running as a bare script (`python build_corpus.py`) as well as `python -m ...`. Running
    # as a bare script puts `_THIS_DIR` itself on sys.path automatically (Python's own behavior),
    # which is *not* what makes `import voicedemo` resolve -- it is `_PARENT_DIR` (the directory
    # containing the `voicedemo` package) that needs to be on the path. An earlier version checked
    # `_THIS_DIR not in sys.path`, which is always false for a bare script run (Python already put
    # it there), so the needed insertion was silently skipped -- caught by execution
    # (StandardsDocument.md Sec.5.4): `python build_corpus.py` raised ModuleNotFoundError.
    sys.path.insert(0, _PARENT_DIR)

from voicedemo.event_bank import build_entries  # noqa: E402
from voicedemo.events import CorpusEntry, validate_corpus  # noqa: E402
from voicedemo.fixtures import cipher_fixture, tell_fixture  # noqa: E402
from voicedemo.provenance import write_provenance  # noqa: E402
from voicedemo.validation_reactions import all_validation_reactions, attach_validation_reactions  # noqa: E402

CORPUS_VERSION = "voicedemo-corpus-2026-08-13-v1"
CORPUS_DIR = os.path.join(_THIS_DIR, "corpus")
FIXTURES_DIR = os.path.join(CORPUS_DIR, "fixtures")


def _write_json(path: str, obj: object) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def build_fixture_json(entries: list[CorpusEntry], module) -> dict:
    held_out = [e for e in entries if e.split == "held_out"]
    lines = []
    for e in held_out:
        for race, text in module.render_all_races(e).items():
            lines.append({"event_id": e.event.id, "race": race, "text": text})
    return {
        "corpus_version": CORPUS_VERSION,
        "n_events": len(held_out),
        "n_lines": len(lines),
        "lines": lines,
    }


def main() -> int:
    entries = build_entries()
    entries = attach_validation_reactions(entries)
    validate_corpus(entries)

    held_out = [e for e in entries if e.split == "held_out"]
    train = [e for e in entries if e.split == "train"]
    if len(held_out) != 50:
        raise AssertionError(f"expected 50 held-out events (plan Sec.5 item 1), got {len(held_out)}")

    for e in held_out:
        tell_fixture.verify_content_identical_across_races(e)
        cipher_fixture.check_no_output_token_collision(e)
        cipher_fixture.verify_reversible_across_races(e)

    vrs = all_validation_reactions(entries)
    if len(vrs) < 100:
        raise AssertionError(f"expected >=100 validation reactions (plan Sec.8 P2), got {len(vrs)}")
    for field in ("info_present", "attitude_present"):
        n_true = sum(1 for v in vrs if getattr(v, field))
        n_false = len(vrs) - n_true
        if n_true < 30 or n_false < 30:
            raise AssertionError(
                f"{field}: need >=30 per direction (plan Sec.8 P2), got {n_true} True / {n_false} False"
            )

    os.makedirs(CORPUS_DIR, exist_ok=True)
    os.makedirs(FIXTURES_DIR, exist_ok=True)

    corpus_path = os.path.join(CORPUS_DIR, "corpus.json")
    _write_json(
        corpus_path,
        {
            "corpus_version": CORPUS_VERSION,
            "n_train": len(train),
            "n_held_out": len(held_out),
            "n_validation_reactions": len(vrs),
            "entries": [e.to_json() for e in entries],
        },
    )

    tell_path = os.path.join(FIXTURES_DIR, "tell_cast.json")
    _write_json(tell_path, build_fixture_json(entries, tell_fixture))

    cipher_path = os.path.join(FIXTURES_DIR, "cipher_cast.json")
    _write_json(cipher_path, build_fixture_json(entries, cipher_fixture))

    provenance_path = os.path.join(CORPUS_DIR, "PROVENANCE.md")
    rel_paths = [
        os.path.relpath(corpus_path, CORPUS_DIR),
        os.path.relpath(tell_path, CORPUS_DIR),
        os.path.relpath(cipher_path, CORPUS_DIR),
    ]
    write_provenance(provenance_path, CORPUS_DIR, rel_paths, CORPUS_VERSION)

    print(f"corpus.json: {len(train)} train + {len(held_out)} held_out = {len(entries)} entries")
    print(f"validation_reactions: {len(vrs)} total")
    print(f"fixtures: tell_cast.json ({len(held_out) * 3} lines), cipher_cast.json ({len(held_out) * 3} lines)")
    print(f"provenance written: {provenance_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
