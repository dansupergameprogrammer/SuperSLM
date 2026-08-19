"""Curie's red suite for tools/ci/check_branch_coverage_floors.py (T-2179 Finding 4).

T-2177 Finding 7 and its confirmation-round successor, T-2179 Finding 3, both found the
same defect in the same eleven-line loop: a pinned file with no measured branch coverage
in this run (the file dropped out of the coverage export entirely -- renamed, excluded
from the merged profile, a build variant that stopped emitting a profraw) was folded into
the same `below_floor_count` as a file that measured LOW but was measured, and the summary
line named only "below their pinned branch-coverage floor" for both. The reviewer's own
stated root cause for why the defect survived T-2177's remedy for its sibling class
(unpinned-but-measured) is that `check_branch_coverage_floors.py` is the only checker
under tools/ci/ with no `tests/ci/test_check_*` sibling -- this file closes that gap.

Four classes, each asserted on its own detail line AND its own summary phrase so a future
change to the wording is caught by the same cell that catches a future change to the
classification:

- **below-floor** -- a pinned file, measured, below its pinned floor.
- **absent-from-run** -- a pinned file with NO measured branch coverage in this run
  (this is the class T-2177 Finding 7 closed for "no floor" and left open for "no
  measurement" -- T-2179 Finding 3).
- **unrecorded** -- a measured file with no pinned floor at all.
- **comment-key** -- a `_`-prefixed key (e.g. `_measured_cell`) is a comment, never a
  per-file floor: it is skipped by the gate loop and excluded from the OK line's count.

Mirrors test_matmul_avx_isolation_population.py's own convention: the module under test is
imported by inserting tools/ci onto sys.path once, never by editing the real committed
tools/ci/branch_coverage_floors.json -- every constructed floors file and coverage export
here lives under pytest's own tmp_path, and _FLOORS_PATH / _ALLOWLIST_PATH are monkeypatched
per test rather than read from the real repository, so this suite is independent of
whatever the real committed floors file currently pins.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_CI_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools", "ci")

if _CI_TOOLS_DIR not in sys.path:
    sys.path.insert(0, _CI_TOOLS_DIR)

import check_branch_coverage_floors as chk  # noqa: E402


def _rel_to_abs(rel: str) -> str:
    """The checker's own `_relpath` computes `os.path.relpath(abs, _REPO_ROOT)` --
    building the export's `filename` field as an absolute path under the real
    _REPO_ROOT (the file need not actually exist; only the export JSON's own
    string is read) means the fixture never has to monkeypatch _REPO_ROOT
    itself, only _FLOORS_PATH and _ALLOWLIST_PATH."""
    return os.path.join(_REPO_ROOT, *rel.split("/"))


def _export_json(measured: dict[str, float]) -> dict:
    """Builds a minimal `llvm-cov export --format=text`-shaped dict: one `data`
    entry, one `files` entry per (relative path, branch percent), each with a
    nonzero branch count (a zero-branch file is a different, untested code path
    -- `per_file_branch_percentages` skips it deliberately, per its own comment,
    and is out of this fixture's scope)."""
    return {
        "data": [
            {
                "files": [
                    {
                        "filename": _rel_to_abs(rel),
                        "summary": {"branches": {"count": 10, "percent": pct}},
                    }
                    for rel, pct in measured.items()
                ]
            }
        ]
    }


def _write_export(tmp_path, measured: dict[str, float]) -> str:
    path = os.path.join(str(tmp_path), "coverage.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_export_json(measured), f)
    return path


def _run_capturing(tmp_path, monkeypatch, capsys, floors: dict, measured: dict[str, float]):
    export_path = _write_export(tmp_path, measured)
    floors_path = os.path.join(str(tmp_path), "branch_coverage_floors.json")
    with open(floors_path, "w", encoding="utf-8") as f:
        json.dump(floors, f)
    monkeypatch.setattr(chk, "_FLOORS_PATH", floors_path)
    monkeypatch.setattr(
        chk, "_ALLOWLIST_PATH", os.path.join(str(tmp_path), "no_such_allowlist.txt")
    )
    monkeypatch.setattr(sys, "argv", ["check_branch_coverage_floors.py", export_path])
    exit_code = chk.main()
    captured = capsys.readouterr()
    return exit_code, captured.out, captured.err


def test_below_floor_class(tmp_path, monkeypatch, capsys):
    """A pinned file, measured, below its floor -- MUST fail, name the shortfall on its
    own detail line, and summarise as "below their pinned branch-coverage floor",
    never as any other class."""
    exit_code, out, err = _run_capturing(
        tmp_path,
        monkeypatch,
        capsys,
        floors={"src/a.cpp": 90.0, "src/b.cpp": 80.0},
        measured={"src/a.cpp": 70.0, "src/b.cpp": 80.0},
    )
    assert exit_code == 1
    assert "src/a.cpp: branch coverage 70.00% < floor 90.00%" in err
    assert "src/b.cpp" not in err
    assert "1 file(s) below their pinned branch-coverage floor" in err
    assert "no pinned floor" not in err
    assert "no measured branch coverage" not in err


def test_absent_from_run_class(tmp_path, monkeypatch, capsys):
    """T-2177 Finding 7 / T-2179 Finding 3: a pinned file with NO measured branch
    coverage in this run is not below a floor -- it was never measured. MUST fail,
    name the file on its own detail line, and summarise as its own class, distinct
    from "below their pinned branch-coverage floor"."""
    exit_code, out, err = _run_capturing(
        tmp_path,
        monkeypatch,
        capsys,
        floors={"src/a.cpp": 90.0},
        measured={},  # src/a.cpp drops out of the export entirely
    )
    assert exit_code == 1
    assert "src/a.cpp: no measured branch coverage in this run (file missing or unmeasured)" in err
    assert "1 file(s) with no measured branch coverage" in err
    assert "below their pinned branch-coverage floor" not in err
    assert "no pinned floor" not in err


def test_unrecorded_class(tmp_path, monkeypatch, capsys):
    """A measured file with no pinned floor at all -- MUST fail, name the file on its
    own detail line ("floor unrecorded for ..."), and summarise as "no pinned floor"."""
    exit_code, out, err = _run_capturing(
        tmp_path,
        monkeypatch,
        capsys,
        floors={},
        measured={"src/c.cpp": 50.0},
    )
    assert exit_code == 1
    assert "floor unrecorded for src/c.cpp" in err
    assert "1 file(s) with no pinned floor" in err
    assert "below their pinned branch-coverage floor" not in err
    assert "no measured branch coverage" not in err


def test_comment_key_excluded_from_gate_and_count(tmp_path, monkeypatch, capsys):
    """A `_`-prefixed key is a comment, never a per-file floor: it is skipped by the
    gate loop (never appears as a failure) and excluded from the OK line's pinned
    count."""
    exit_code, out, err = _run_capturing(
        tmp_path,
        monkeypatch,
        capsys,
        floors={"_measured_cell": "clang-18 / ubuntu-latest / RelWithDebInfo", "src/a.cpp": 50.0},
        measured={"src/a.cpp": 60.0},
    )
    assert exit_code == 0
    assert "_measured_cell" not in out
    assert "_measured_cell" not in err
    assert "OK -- 1 file(s) at or above their pinned floor" in out


def test_all_three_failure_classes_in_one_run_are_all_named(tmp_path, monkeypatch, capsys):
    """The state that exposed the original defect (T-2179 Finding 3's own executed
    table): a below-floor file, an absent-from-run file, and an unrecorded file all in
    the same run. All three summary phrases MUST appear, each with its own count, and
    the below-floor / absent-from-run detail lines must not be conflated."""
    exit_code, out, err = _run_capturing(
        tmp_path,
        monkeypatch,
        capsys,
        floors={"src/a.cpp": 90.0, "src/b.cpp": 90.0},
        measured={"src/a.cpp": 70.0, "src/c.cpp": 50.0},  # b.cpp absent; c.cpp unpinned
    )
    assert exit_code == 1
    assert "src/a.cpp: branch coverage 70.00% < floor 90.00%" in err
    assert "src/b.cpp: no measured branch coverage in this run" in err
    assert "floor unrecorded for src/c.cpp" in err
    assert "1 file(s) below their pinned branch-coverage floor" in err
    assert "1 file(s) with no measured branch coverage" in err
    assert "1 file(s) with no pinned floor" in err


def test_all_pinned_files_at_floor_is_ok(tmp_path, monkeypatch, capsys):
    """Control: every pinned file measured at or above its floor, nothing unpinned --
    exit 0, no failure class named."""
    exit_code, out, err = _run_capturing(
        tmp_path,
        monkeypatch,
        capsys,
        floors={"src/a.cpp": 90.0},
        measured={"src/a.cpp": 90.0},
    )
    assert exit_code == 0
    assert err == ""
    assert "OK -- 1 file(s) at or above their pinned floor" in out
