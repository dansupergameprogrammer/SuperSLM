"""Version-pin and hash the built corpus artifacts (plan Sec.8 P1: "Version-pinned, hashed,
`.gitattributes eol=lf` pinned on the hashed files, mirroring Sec.11's hashed-corpus discipline in
`SuperSLM_Plan.md`"). Mirrors this repo's existing `tests/reference/check_provenance.py` pattern: a
PROVENANCE.md table of relative-path -> SHA-256, and a checker that recomputes the hash from disk and
asserts equality rather than trusting a hand-maintained value.

`*.json` already carries `text eol=lf` in this repo's root `.gitattributes` (see that file's header
comment), so the corpus JSON files' line endings are already pinned repo-wide -- this module does not
need to add a new `.gitattributes` rule, only to record and check the resulting hashes.
"""

from __future__ import annotations

import hashlib
import os
import re

_ROW_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64}|PENDING)`\s*\|\s*$", re.MULTILINE)


def sha256_of(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_provenance(provenance_path: str, checked_root: str, relative_paths: list[str], corpus_version: str) -> None:
    lines = [
        "# T-1980 voice-demo corpus provenance",
        "",
        f"Corpus version: `{corpus_version}`. Regenerate with `build_corpus.py`; do not hand-edit.",
        "Checked by `provenance.py:check_provenance` (mirrors `tests/reference/check_provenance.py`'s pattern).",
        "",
        "| file | sha256 |",
        "|---|---|",
    ]
    for rel in sorted(relative_paths):
        digest = sha256_of(os.path.join(checked_root, rel))
        # Always record forward slashes: os.path.relpath (used by callers to build `rel`) returns
        # OS-native separators, and a provenance record with backslashes on Windows would not match
        # the same record regenerated on Linux/Mac -- exactly the cross-platform hash-identity
        # problem this repo's own `.gitattributes` header comment exists to prevent (SuperSLM_Plan.md
        # Sec.11). `check_provenance` below reads whatever separator is on disk when it re-derives
        # the path, so normalizing only on write is sufficient.
        rel_posix = rel.replace(os.sep, "/")
        lines.append(f"| `{rel_posix}` | `{digest}` |")
    with open(provenance_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def _parse_provenance(text: str) -> dict[str, str]:
    return dict(_ROW_RE.findall(text))


def check_provenance(provenance_path: str, checked_root: str) -> list[str]:
    """Returns a list of mismatch descriptions (empty if everything matches)."""

    with open(provenance_path, "r", encoding="utf-8") as f:
        recorded = _parse_provenance(f.read())

    mismatches: list[str] = []
    if not recorded:
        return [f"{provenance_path}: no provenance rows parsed"]
    for rel_path, recorded_digest in recorded.items():
        if recorded_digest == "PENDING":
            mismatches.append(f"{rel_path}: PROVENANCE.md still records PENDING")
            continue
        abs_path = os.path.join(checked_root, rel_path)
        if not os.path.isfile(abs_path):
            mismatches.append(f"{rel_path}: file not found on disk at {abs_path}")
            continue
        actual = sha256_of(abs_path)
        if actual != recorded_digest:
            mismatches.append(f"{rel_path}: recorded {recorded_digest}, disk hashes to {actual}")
    return mismatches
