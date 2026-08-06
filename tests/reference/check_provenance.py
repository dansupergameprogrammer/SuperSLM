#!/usr/bin/env python3
"""Machine-checked provenance for the vendored reference (S-HARDEN-5, F3, S3.1;
extended for the group-commit-identity property, T-1529, §11 S3.1c item 2a).

PROVENANCE.md's recorded SHA-256 values are, without this script, a value a human
reads when re-vendoring later -- nothing computes them against the committed files
and asserts equality. This script recomputes SHA-256 of each vendored file from
disk and compares it against PROVENANCE.md's recorded values, exiting non-zero and
naming whichever file mismatched if any disagree.

PROVENANCE.md's table also carries a `Source commit` and a `Group` column per
row. Rows sharing a `Group` value must record an identical `Source commit` --
without this second pass, a hand-edit recording two files of one closure at two
individually-correct-but-different commits would pass the per-file hash check
silently, defeating the joint-pin property the closure's own bit-equality claim
depends on (dynamic_engine.py's bit-equality-to-pipeline.forward_dynamic claim
is proven by test_dynamic_engine.py against one specific state of both files
together; pinning them at different commits could combine a dynamic_engine.py
from one point in time with a pipeline.py from another where that equivalence
was never proven to hold jointly). This script groups the parsed rows by
`Group` and asserts every row within a group records an identical
`Source commit`, exiting non-zero and naming the group and the disagreeing
files/commits if not.

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
    "superslm_spike/pipeline_prob_width_ceiling.py",
    "superslm_spike/dynamic_engine.py",
    "superslm_spike/pipeline.py",
    "superslm_spike/silu_lut.py",
    "superslm_spike/constrain.py",
)

# Matches a four-column `| File | SHA-256 | Source commit | Group |` row. The
# SHA-256 and Source-commit fields each accept PENDING as a not-yet-recorded
# placeholder, reported as its own mismatch below rather than a silent match.
_ROW_RE = re.compile(
    r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64}|PENDING)`\s*\|\s*`([0-9a-f]{40}|PENDING)`\s*\|\s*`([^`]+)`\s*\|\s*$",
    re.MULTILINE,
)


class _Row:
    __slots__ = ("digest", "source_commit", "group")

    def __init__(self, digest: str, source_commit: str, group: str) -> None:
        self.digest = digest
        self.source_commit = source_commit
        self.group = group


def _parse_provenance(text: str) -> dict[str, _Row]:
    recorded: dict[str, _Row] = {}
    for name, digest, source_commit, group in _ROW_RE.findall(text):
        recorded[name] = _Row(digest, source_commit, group)
    return recorded


def _sha256_of(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _check_group_commit_identity(recorded: dict[str, _Row], checked_files: tuple[str, ...]) -> list[str]:
    """Groups `checked_files`' recorded rows by `Group` and asserts every row
    within a group carries an identical `Source commit`. Files with no
    recorded row are skipped here -- that is already reported as its own
    mismatch by the per-file hash pass above, and is not this pass's job to
    repeat."""
    groups: dict[str, dict[str, str]] = {}
    for rel_path in checked_files:
        row = recorded.get(rel_path)
        if row is None or row.group == "" or row.source_commit == "PENDING":
            continue
        groups.setdefault(row.group, {})[rel_path] = row.source_commit

    mismatches: list[str] = []
    for group, commits_by_file in groups.items():
        distinct_commits = set(commits_by_file.values())
        if len(distinct_commits) > 1:
            detail = ", ".join(f"{f}={c}" for f, c in sorted(commits_by_file.items()))
            mismatches.append(
                f"group '{group}': disagreeing Source commit across its rows -- {detail}"
            )
    return mismatches


def main() -> int:
    with open(PROVENANCE_PATH, "r", encoding="utf-8") as f:
        recorded = _parse_provenance(f.read())

    mismatches: list[str] = []
    for rel_path in _CHECKED_FILES:
        # PROVENANCE.md's table keys files as "superslm_spike/<name>".
        row = recorded.get(rel_path)
        if row is None:
            mismatches.append(f"{rel_path}: no entry recorded in PROVENANCE.md")
            continue
        if row.digest == "PENDING":
            mismatches.append(f"{rel_path}: PROVENANCE.md still records PENDING, never updated with a real hash")
            continue
        abs_path = os.path.join(_THIS_DIR, rel_path)
        if not os.path.isfile(abs_path):
            mismatches.append(f"{rel_path}: file not found on disk at {abs_path}")
            continue
        actual_digest = _sha256_of(abs_path)
        if actual_digest != row.digest:
            mismatches.append(
                f"{rel_path}: SHA-256 mismatch -- PROVENANCE.md records {row.digest}, "
                f"disk content hashes to {actual_digest}"
            )

    mismatches += _check_group_commit_identity(recorded, _CHECKED_FILES)

    if mismatches:
        print("check_provenance.py: FAILED", file=sys.stderr)
        for m in mismatches:
            print(f"  - {m}", file=sys.stderr)
        return 1

    print(f"check_provenance.py: OK -- {len(_CHECKED_FILES)} vendored files match PROVENANCE.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
