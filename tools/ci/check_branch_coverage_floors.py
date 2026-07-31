#!/usr/bin/env python3
"""S-HARDEN-8 (design Sec4.1, Sec4.3; F12): the per-file branch-coverage
floor gate. Reads `llvm-cov export --format=text`'s JSON (source-only --
`src/` and `include/superslm/`, per the plan's own "a source-only LLVM
coverage leg" framing) and fails if any file's measured branch-coverage
percentage drops below its pinned floor in
tools/ci/branch_coverage_floors.json.

Per-file floors, not one aggregate threshold (F12's own finding: "a single
aggregate line threshold hides exactly the parser gaps that matter" --
artifact.cpp and model.cpp were the two files a 97%/92%/75.7% aggregate
concealed real gaps inside). Floors are measured, never invented: the first
run of this gate records the actual per-file percentages
(`--record-floors`); every subsequent run compares against the committed
file.

Usage:
    llvm-cov export --format=text -instr-profile=default.profdata \\
        out/cov/superslm_tests_cov -- <object args> \\
        src/*.cpp include/superslm/*.h > coverage.json
    python tools/ci/check_branch_coverage_floors.py coverage.json
"""
from __future__ import annotations

import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_FLOORS_PATH = os.path.join(_THIS_DIR, "branch_coverage_floors.json")
_ALLOWLIST_PATH = os.path.join(_THIS_DIR, "branch_coverage_allowlist.txt")

# Source-only: the measured set is restricted to these two directories
# (design Sec4.1) -- tests/ and tools/ are excluded from the measured set.
_MEASURED_PREFIXES = ("src/", "include/superslm/")


def _relpath(abs_path: str) -> str:
    rel = os.path.relpath(abs_path, _REPO_ROOT).replace("\\", "/")
    return rel


def _load_allowlist() -> dict[str, str]:
    """file:line -> reason. An allowlist entry with no reason is a defect in
    the allowlist, not a pass (design Sec4.1) -- enforced by the parse
    itself: a malformed line raises rather than silently admitting an
    unreasoned entry."""
    entries: dict[str, str] = {}
    if not os.path.isfile(_ALLOWLIST_PATH):
        return entries
    with open(_ALLOWLIST_PATH, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "#" not in line:
                raise ValueError(
                    f"{_ALLOWLIST_PATH}:{lineno}: allowlist entry has no reason "
                    f"(format is 'file:line  # reason'): {raw!r}"
                )
            key, reason = line.split("#", 1)
            key = key.strip()
            reason = reason.strip()
            if not reason:
                raise ValueError(f"{_ALLOWLIST_PATH}:{lineno}: empty reason for {key!r}")
            entries[key] = reason
    return entries


def per_file_branch_percentages(export_json: dict) -> dict[str, float]:
    """Extract {relative_file_path: branch_coverage_percent} from
    `llvm-cov export --format=text`'s JSON, restricted to the measured
    source set."""
    out: dict[str, float] = {}
    for export in export_json.get("data", []):
        for file_entry in export.get("files", []):
            filename = file_entry.get("filename", "")
            rel = _relpath(filename)
            if not rel.startswith(_MEASURED_PREFIXES):
                continue
            branches = file_entry.get("summary", {}).get("branches", {})
            count = branches.get("count", 0)
            if count == 0:
                # A file with zero branches (e.g. a pure-declaration header)
                # has no branch-coverage percentage to gate -- not a floor
                # violation, not a floor to pin.
                continue
            out[rel] = float(branches.get("percent", 0.0))
    return out


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_json", help="Path to llvm-cov export --format=text output")
    parser.add_argument(
        "--record-floors",
        action="store_true",
        help="Write the measured percentages to branch_coverage_floors.json "
        "instead of checking them (the design's own step 1: run unmodified, "
        "record the current state, before any floor is pinned).",
    )
    args = parser.parse_args()

    with open(args.export_json, "r", encoding="utf-8") as f:
        export_json = json.load(f)

    measured = per_file_branch_percentages(export_json)

    if args.record_floors:
        with open(_FLOORS_PATH, "w", encoding="utf-8") as f:
            json.dump(dict(sorted(measured.items())), f, indent=2)
            f.write("\n")
        print(f"Recorded {len(measured)} file floor(s) to {_FLOORS_PATH}")
        return 0

    if not os.path.isfile(_FLOORS_PATH):
        print(
            f"check_branch_coverage_floors.py: {_FLOORS_PATH} does not exist -- "
            "run with --record-floors first (design Sec4.3 step 1: measure "
            "before pinning).",
            file=sys.stderr,
        )
        return 1

    with open(_FLOORS_PATH, "r", encoding="utf-8") as f:
        floors: dict[str, float] = json.load(f)

    allowlist = _load_allowlist()

    failures: list[str] = []
    for rel, floor in sorted(floors.items()):
        got = measured.get(rel)
        if got is None:
            failures.append(f"{rel}: no measured branch coverage in this run (file missing or unmeasured)")
            continue
        if got < floor - 1e-9:
            failures.append(f"{rel}: branch coverage {got:.2f}% < floor {floor:.2f}%")

    # A file the pinned floors don't yet know about is not itself a
    # failure (new source files start unpinned) -- but it is reported so a
    # human notices and pins it deliberately in the same change, matching
    # the pinned-oracle discipline this gate otherwise enforces.
    unpinned = sorted(set(measured) - set(floors))
    if unpinned:
        print(
            "check_branch_coverage_floors.py: NOTE -- measured but unpinned file(s), "
            "not gated: " + ", ".join(unpinned),
            file=sys.stderr,
        )

    if allowlist:
        print(
            f"check_branch_coverage_floors.py: {len(allowlist)} allowlisted branch(es) "
            "loaded (T-1456: this gate version does not cross-check the allowlist "
            "against llvm-cov's per-branch report, so no failure is actually excluded "
            "by it -- these entries are documented for a human reviewer only).",
        )

    if failures:
        for line in failures:
            print(line, file=sys.stderr)
        print(
            f"check_branch_coverage_floors.py: FAILED -- {len(failures)} file(s) below "
            "their pinned branch-coverage floor",
            file=sys.stderr,
        )
        return 1

    print(f"check_branch_coverage_floors.py: OK -- {len(floors)} file(s) at or above their pinned floor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
