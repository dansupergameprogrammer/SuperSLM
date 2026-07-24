#!/usr/bin/env python3
"""Machine-checked provenance for the vendored reference (S-HARDEN-5, F3, S3.1).

PROVENANCE.md's recorded SHA-256 values are, without this script, a value a human
reads when re-vendoring later -- nothing computes them against the committed files
and asserts equality. This script recomputes SHA-256 of each vendored file from
disk and compares it against PROVENANCE.md's recorded values, exiting non-zero and
naming whichever file mismatched if any disagree.

Wired into the `generators` CI job before either fixture-regeneration step runs,
so a stale or hand-edited provenance record -- or a vendored-file edit that
happens not to move any fixture byte -- fails the job immediately rather than
passing silently through a gate that only indirectly covers it.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROVENANCE_PATH = os.path.join(_THIS_DIR, "PROVENANCE.md")

# Files whose content this script certifies, relative to this directory.
_CHECKED_FILES = (
    "superslm_spike/intmath.py",
    "superslm_spike/rope.py",
    "superslm_spike/rope_tables_pinned.json",
)

_ROW_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64}|PENDING)`\s*\|\s*$", re.MULTILINE)


def _parse_provenance(text: str) -> dict[str, str]:
    recorded: dict[str, str] = {}
    for name, digest in _ROW_RE.findall(text):
        recorded[name] = digest
    return recorded


def _sha256_of(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main() -> int:
    with open(PROVENANCE_PATH, "r", encoding="utf-8") as f:
        recorded = _parse_provenance(f.read())

    mismatches: list[str] = []
    for rel_path in _CHECKED_FILES:
        # PROVENANCE.md's table keys files as "superslm_spike/<name>".
        recorded_digest = recorded.get(rel_path)
        if recorded_digest is None:
            mismatches.append(f"{rel_path}: no entry recorded in PROVENANCE.md")
            continue
        if recorded_digest == "PENDING":
            mismatches.append(f"{rel_path}: PROVENANCE.md still records PENDING, never updated with a real hash")
            continue
        abs_path = os.path.join(_THIS_DIR, rel_path)
        if not os.path.isfile(abs_path):
            mismatches.append(f"{rel_path}: file not found on disk at {abs_path}")
            continue
        actual_digest = _sha256_of(abs_path)
        if actual_digest != recorded_digest:
            mismatches.append(
                f"{rel_path}: SHA-256 mismatch -- PROVENANCE.md records {recorded_digest}, "
                f"disk content hashes to {actual_digest}"
            )

    if mismatches:
        print("check_provenance.py: FAILED", file=sys.stderr)
        for m in mismatches:
            print(f"  - {m}", file=sys.stderr)
        return 1

    print(f"check_provenance.py: OK -- {len(_CHECKED_FILES)} vendored files match PROVENANCE.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
