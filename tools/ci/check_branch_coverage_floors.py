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
concealed real gaps inside). Floors are measured, never invented: a run of
this gate with `--record-floors-to <path>` records the actual per-file
percentages to a build artifact, never to the committed floors file itself
(fold round 6, D-SLM3603 -- the record step and the check step never share
a file); a human commits the recorded value(s) into
tools/ci/branch_coverage_floors.json deliberately, and every subsequent run
compares against that committed file.

Usage (T-2149 fold round 4, D-SLM3569: the branch-coverage CI job merges profiles
across the non-forced binary and the forced-tier binaries so each SIMD leaf reaches
its own coverage in its own forced binary -- see .github/workflows/tests.yml,
branch-coverage job, for the full multi-binary build/run/merge sequence):
    llvm-profdata merge -sparse *.profraw -o merged.profdata
    llvm-cov export --format=text -instr-profile=merged.profdata \\
        out/cov/superslm_tests -object out/cov/superslm_tests_sse2_forced ... \\
        -- src/*.cpp include/superslm/*.h > coverage.json
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
        "--record-floors-to",
        metavar="PATH",
        default=None,
        help="Write the measured percentages to PATH (a build artifact) instead "
        "of checking them, and never to the committed floors file (fold round 6, "
        "D-SLM3603: --record-floors used to write _FLOORS_PATH directly, which is "
        "the same file the check step below reads -- at HEAD, with no committed "
        "src/matmul.cpp entry, every run took this branch and the check that "
        "followed was checking this step's own output, reporting OK "
        "unconditionally. The record step and the check step never share a file "
        "now: a human reads this artifact and commits the value into "
        "tools/ci/branch_coverage_floors.json deliberately.).",
    )
    args = parser.parse_args()

    with open(args.export_json, "r", encoding="utf-8") as f:
        export_json = json.load(f)

    measured = per_file_branch_percentages(export_json)

    if args.record_floors_to:
        # T-2149 design §10 dimension 7 item (h), fold round 5 (D-SLM3584): a
        # `_`-prefixed key (e.g. "_measured_cell") is a comment, not a per-file
        # floor -- JSON has no native comment syntax, so this is the file's own
        # convention for stating which cell (runner, toolchain, binary set) every
        # numeric floor in it is pinned from. Preserved across a re-record rather
        # than dropped, so recording a fresh set of numbers never silently erases
        # the statement of where they came from. Fold round 6 (D-SLM3603): the
        # comment carried forward is read from the COMMITTED floors file
        # (`_FLOORS_PATH`), never from `--record-floors-to`'s own output path,
        # since this step no longer writes `_FLOORS_PATH` at all.
        existing_comments: dict[str, str] = {}
        if os.path.isfile(_FLOORS_PATH):
            with open(_FLOORS_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_comments = {k: v for k, v in existing.items() if k.startswith("_")}
        recorded = dict(existing_comments)
        recorded.update(sorted(measured.items()))
        out_path = args.record_floors_to
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dict(sorted(recorded.items())), f, indent=2)
            f.write("\n")
        print(f"Recorded {len(measured)} file floor(s) to {out_path} (a build "
              f"artifact -- commit the value(s) you need into {_FLOORS_PATH} "
              "deliberately; this step never writes the committed file).")
        return 0

    if not os.path.isfile(_FLOORS_PATH):
        print(
            f"check_branch_coverage_floors.py: {_FLOORS_PATH} does not exist -- "
            "run with --record-floors-to <path> first, then commit the recorded "
            "value(s) into this file (design Sec4.3 step 1: measure before "
            "pinning).",
            file=sys.stderr,
        )
        return 1

    with open(_FLOORS_PATH, "r", encoding="utf-8") as f:
        floors: dict[str, float] = json.load(f)

    allowlist = _load_allowlist()

    # `_`-prefixed keys are comments (fold round 5, D-SLM3584), not per-file floors
    # -- e.g. "_measured_cell", stating which runner/toolchain/binary-set cell every
    # real floor below is pinned from. Skipped here so a comment is never compared
    # against a measured percentage as though it were one.
    failures: list[str] = []
    for rel, floor in sorted(floors.items()):
        if rel.startswith("_"):
            continue
        got = measured.get(rel)
        if got is None:
            failures.append(f"{rel}: no measured branch coverage in this run (file missing or unmeasured)")
            continue
        if got < floor - 1e-9:
            failures.append(f"{rel}: branch coverage {got:.2f}% < floor {floor:.2f}%")

    # Fold round 6 (D-SLM3603): a measured, gated (`_MEASURED_PREFIXES`-scoped)
    # file with a nonzero branch count and no entry in the committed floors file
    # is a hard FAILURE, replacing the prior advisory-only NOTE. The committed
    # JSON alone cannot distinguish "never measured before" from "measured
    # before, entry since lost" -- both are the same shape ("this gate's own
    # docstring claim, floors are measured never invented, does not hold for
    # this file right now"), so both are closed by the same general rule rather
    # than a file-specific special case that would silently reopen for the next
    # file this happens to.
    # T-2177 Finding 7: `unpinned` entries are a distinct failure class from
    # `failures` above -- a file with no floor is not below one -- but both
    # were appended to the same `failures` list and summarised by one line
    # naming only "below their pinned branch-coverage floor". Counted
    # separately here so the summary names what actually happened.
    below_floor_count = len(failures)
    unpinned = sorted(set(measured) - set(floors))
    for rel in unpinned:
        failures.append(
            f"floor unrecorded for {rel} -- commit the artifact value from "
            "this run's measured-branch-coverage-floors artifact into "
            "tools/ci/branch_coverage_floors.json"
        )
    unpinned_count = len(unpinned)

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
        summary_parts = []
        if below_floor_count:
            summary_parts.append(
                f"{below_floor_count} file(s) below their pinned branch-coverage floor"
            )
        if unpinned_count:
            summary_parts.append(f"{unpinned_count} file(s) with no pinned floor")
        print(
            "check_branch_coverage_floors.py: FAILED -- " + "; ".join(summary_parts),
            file=sys.stderr,
        )
        return 1

    # Fold round 6 (D-SLM3606): `len(floors)` counts every key, including
    # `_`-prefixed comment keys the check loop above already skips (today,
    # "_measured_cell") -- the same exclusion is applied here so the success
    # line's file count matches what was actually gated, not the dict's raw key
    # count.
    pinned_count = len([k for k in floors if not k.startswith("_")])
    print(f"check_branch_coverage_floors.py: OK -- {pinned_count} file(s) at or above their pinned floor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
