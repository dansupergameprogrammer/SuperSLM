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
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
PROVENANCE_PATH = os.path.join(_THIS_DIR, "PROVENANCE.md")

# Files whose content this script certifies, relative to this directory.
_CHECKED_FILES = (
    "superslm_spike/intmath.py",
    "superslm_spike/rope.py",
    "superslm_spike/rope_tables_pinned.json",
    "superslm_spike/pipeline_prob_width_ceiling.py",
)

# B7 (T-2123/T-2137, design SS3.5 ruling item 1): the live, product copy of intmath.py/rope.py
# under tools/reference_pipeline/ and this directory's frozen golden-fixture copy are SUPPOSED
# to be able to diverge -- a live copy gets a bug fix, the frozen copy stays frozen until
# someone deliberately re-vendors it. What must not happen is a divergence nobody decided and
# nobody can see: each pair below passes on byte-identity, or on an explicit, dated
# PROVENANCE.md entry (in the "Cross-repo reach" section) recording the two as intentionally
# decoupled -- never a silent, undocumented pass either way.
_CROSS_REPO_PAIRS = (
    ("intmath.py", os.path.join(_REPO_ROOT, "tools", "reference_pipeline", "intmath.py"),
     "superslm_spike/intmath.py"),
    ("rope.py", os.path.join(_REPO_ROOT, "tools", "reference_pipeline", "rope.py"),
     "superslm_spike/rope.py"),
)

_ROW_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64}|PENDING)`\s*\|\s*$", re.MULTILINE)

# T-2137 fix round (Poirot casebook f83afe0-t2137-vendoring-review.md, S1): a decoupling row
# names the ONE divergence a stated commit produced -- `| File | Date | Commit | Live SHA-256
# | Reason |` -- not a standing exemption for the file. Scoped between the section's own start
# marker and the next `##` heading (M1: a bare re.MULTILINE scan over the whole document would
# also match any other five-column table that happened to start with a backticked filename).
_CROSS_REPO_SECTION_RE = re.compile(
    r"^## Cross-repo reach.*?(?=^## |\Z)", re.MULTILINE | re.DOTALL
)
_DECOUPLING_RE = re.compile(
    r"^\|\s*`([^`]+)`\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*`([0-9a-f]{7,40})`\s*\|\s*`([0-9a-f]{64})`\s*\|\s*(.+?)\s*\|\s*$",
    re.MULTILINE,
)


def _parse_provenance(text: str) -> dict[str, str]:
    recorded: dict[str, str] = {}
    for name, digest in _ROW_RE.findall(text):
        recorded[name] = digest
    return recorded


def _sha256_of(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _parse_decoupling_entries(text: str) -> dict[str, tuple[str, str, str]]:
    """Rows of the "Cross-repo reach" section's own table:
    `| file.py | YYYY-MM-DD | commit | live-SHA-256 | free-text reason |`.

    Scoped to the section (not the whole document, M1) so a future unrelated table that
    happens to start with a backticked filename and a date cannot become a silent exemption
    anywhere else in this file.
    """
    section_match = _CROSS_REPO_SECTION_RE.search(text)
    section_text = section_match.group(0) if section_match else ""
    entries: dict[str, tuple[str, str, str]] = {}
    for name, date, commit, live_sha256, reason in _DECOUPLING_RE.findall(section_text):
        entries[name] = (date, commit, live_sha256)
    return entries


def _check_cross_repo_reach(provenance_text: str) -> list[str]:
    decoupled = _parse_decoupling_entries(provenance_text)
    problems: list[str] = []
    for name, live_path, frozen_rel in _CROSS_REPO_PAIRS:
        frozen_path = os.path.join(_THIS_DIR, frozen_rel)
        if not os.path.isfile(live_path):
            problems.append(f"cross-repo reach: {name}: live copy not found at {live_path}")
            continue
        if not os.path.isfile(frozen_path):
            problems.append(f"cross-repo reach: {name}: frozen copy not found at {frozen_path}")
            continue
        # Normalized to LF before comparing: .gitattributes pins `*.py text eol=lf` for both
        # locations, but an uncommitted working-tree file can still carry the checkout
        # platform's native line endings before that normalization applies at commit time --
        # comparing raw bytes here would report a false divergence on Windows.
        live_bytes = open(live_path, "rb").read().replace(b"\r\n", b"\n")
        frozen_bytes = open(frozen_path, "rb").read().replace(b"\r\n", b"\n")
        if live_bytes == frozen_bytes:
            continue  # no divergence at all -- nothing to record or check against a pin
        entry = decoupled.get(name)
        if entry is None:
            problems.append(
                f"cross-repo reach: {name}: tools/reference_pipeline/{name} diverges from "
                f"tests/reference/{frozen_rel} with no PROVENANCE.md 'Cross-repo reach' entry "
                f"recording the divergence as intentional"
            )
            continue
        _date, commit, pinned_live_sha256 = entry
        actual_live_sha256 = hashlib.sha256(live_bytes).hexdigest()
        if actual_live_sha256 != pinned_live_sha256:
            problems.append(
                f"cross-repo reach: {name}: PROVENANCE.md's decoupling entry (commit {commit}) "
                f"pins the live copy's SHA-256 to {pinned_live_sha256}, but the live copy on "
                f"disk hashes to {actual_live_sha256} -- the recorded row names ONE approved "
                f"divergence, not blanket permission for this file to drift further (or to "
                f"revert to some other, unrecorded state); update the pinned hash with a new, "
                f"dated row if this change was intentional"
            )
        # else: byte-identical to the pinned, approved divergence -- passes.
    return problems


def main() -> int:
    with open(PROVENANCE_PATH, "r", encoding="utf-8") as f:
        provenance_text = f.read()
    recorded = _parse_provenance(provenance_text)

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

    mismatches.extend(_check_cross_repo_reach(provenance_text))

    if mismatches:
        print("check_provenance.py: FAILED", file=sys.stderr)
        for m in mismatches:
            print(f"  - {m}", file=sys.stderr)
        return 1

    print(f"check_provenance.py: OK -- {len(_CHECKED_FILES)} vendored files match PROVENANCE.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
