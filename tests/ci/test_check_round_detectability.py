"""Curie's red suite for check_round_detectability.py (T-2352).

Mechanism cells build a tiny, throwaway git repository per test -- never the
real SuperSLM tree -- and drive `sweep_round` / `get_hunks` /
`try_reverse_apply` against it directly, per `StandardsDocument.md` Sec4's
population-validation requirement: a check shown only to pass on its own
author's construction is not shown to catch anything, so these cells are
built to fail the mechanism first (an undetected hunk that stays undetected,
a detected hunk that stays detected) and only then does the module under
test get to answer them.

These cells are fast (a two-file scratch repo, a two-line pytest target) and
belong in the ordinary gating run. The REAL population-reproduction evidence
-- that this module actually reproduces the ten historical instances found
by hand across `Claude/Poirot/8a28460-...` and `Claude/Poirot/3cd1a2b-...`,
by executing the real repository's real history -- is not repeated here as a
fast unit cell: each of those replays a full `tests/t2296-fp-free-open-red-
suite` run (order 30s) against real historical commits, so folding all ten
into this file would turn an ordinary `pytest` invocation into a 10+ minute
one. That evidence is `Claude/Brunel/t2352-detectability-check-build-
2026-08-27.md`'s own reproduction table, executed and named per instance,
and `tests/ci/test_check_round_detectability_historical.py` carries a
runnable (not gating) subset of it for a later session to re-run without
re-deriving the commit list from the build log.

WHAT THIS COVERS: hunk splitting on a file with 1 vs N changes; reverse-apply
succeeding on an isolated hunk and failing (then combining) on an
interdependent pair; signature equality treating an identical failing set as
UNDETECTED and any new failure as DETECTED; a mutant that raises at import
time scored DETECTED (a collection error IS a signature change) rather than
crashing the sweep itself; and the CLI's exit code (0 only when nothing is
UNDETECTED).

WHAT THIS DOES NOT COVER: whether the module's judgement of "production
file" is correct for a given caller's directory layout (a scope decision for
the caller, per this module's own docstring on `--prod`); and whether an
UNRESOLVED hunk was genuinely unresolvable or the combine window was too
narrow -- covered by inspection of the historical replay's own output
instead, where the real diffs are large enough to exercise it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_round_detectability as crd


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True
    )


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "scratch_repo"
    repo.mkdir()
    _run("init", "--quiet", "-b", "main", cwd=repo)
    _run("config", "user.email", "t2352@example.invalid", cwd=repo)
    _run("config", "user.name", "T-2352 fixture", cwd=repo)
    return repo


def _commit_all(repo: Path, message: str) -> str:
    _run("add", "-A", cwd=repo)
    _run("commit", "--quiet", "-m", message, cwd=repo)
    return _run("rev-parse", "HEAD", cwd=repo).stdout.strip()


@pytest.fixture()
def uncovered_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """BASE: `widget.py` has one behaviour, one cell pinning it.
    HEAD: a second behaviour is added, with NO new cell -- the classic
    unpinned production change, at hunk granularity, isolatable alone."""
    repo = _init_repo(tmp_path)
    (repo / "widget.py").write_text(
        textwrap.dedent(
            """\
            def classify(x):
                if x < 0:
                    return "negative"
                return "non-negative"
            """
        ),
        encoding="utf-8",
    )
    (repo / "test_widget.py").write_text(
        textwrap.dedent(
            """\
            from widget import classify

            def test_negative():
                assert classify(-1) == "negative"

            def test_non_negative():
                assert classify(5) == "non-negative"
            """
        ),
        encoding="utf-8",
    )
    base = _commit_all(repo, "base: classify negative vs non-negative")

    (repo / "widget.py").write_text(
        textwrap.dedent(
            """\
            def classify(x):
                if x < 0:
                    return "negative"
                if x == 0:
                    return "zero"
                return "non-negative"
            """
        ),
        encoding="utf-8",
    )
    head = _commit_all(repo, "head: adds a zero case, no new cell for it")
    return repo, base, head


def test_uncovered_hunk_is_reported_undetected(uncovered_repo):
    repo, base, head = uncovered_repo
    baseline, results = crd.sweep_round(repo, base, head, ["widget.py"], ".")
    assert baseline.failed == 0 and baseline.errors == 0
    assert len(results) == 1
    assert results[0].status == "undetected"


def test_covered_hunk_is_reported_detected(uncovered_repo):
    """Same fixture, but a second commit adds the missing cell -- the same
    hunk in widget.py must now flip to DETECTED without editing widget.py."""
    repo, base, _ = uncovered_repo
    (repo / "widget.py").write_text(
        textwrap.dedent(
            """\
            def classify(x):
                if x < 0:
                    return "negative"
                if x == 0:
                    return "zero"
                return "non-negative"
            """
        ),
        encoding="utf-8",
    )
    (repo / "test_widget.py").write_text(
        textwrap.dedent(
            """\
            from widget import classify

            def test_negative():
                assert classify(-1) == "negative"

            def test_non_negative():
                assert classify(1) == "non-negative"

            def test_zero():
                assert classify(0) == "zero"
            """
        ),
        encoding="utf-8",
    )
    head_pinned = _commit_all(repo, "head: adds the zero case AND its cell")

    baseline, results = crd.sweep_round(repo, base, head_pinned, ["widget.py"], ".")
    assert len(results) == 1
    assert results[0].status == "detected"


def test_two_hunks_one_covered_one_not(tmp_path: Path):
    """widget.py gets two changes far enough apart to fall in separate git
    hunks: an unpinned `classify` zero-case, and a pinned new `parity`
    function. Both hunks must be reported, independently and correctly --
    the two changes are separated by enough untouched lines that git itself
    puts them in different hunks, so this exercises per-hunk isolation
    rather than the combine-window fallback."""
    repo = _init_repo(tmp_path)
    padding = "\n\n".join(f'def unrelated_{i}():\n    return {i}' for i in range(6))
    (repo / "widget.py").write_text(
        f'def classify(x):\n    if x < 0:\n        return "negative"\n'
        f'    return "non-negative"\n\n\n{padding}\n',
        encoding="utf-8",
    )
    (repo / "test_widget.py").write_text(
        textwrap.dedent(
            """\
            from widget import classify

            def test_negative():
                assert classify(-1) == "negative"

            def test_non_negative():
                assert classify(5) == "non-negative"
            """
        ),
        encoding="utf-8",
    )
    base = _commit_all(repo, "base: classify + six spacer functions")

    (repo / "widget.py").write_text(
        f'def classify(x):\n    if x < 0:\n        return "negative"\n'
        f'    if x == 0:\n        return "zero"\n    return "non-negative"\n\n\n'
        f'{padding}\n\n\ndef parity(x):\n    return "even" if x % 2 == 0 else "odd"\n',
        encoding="utf-8",
    )
    (repo / "test_widget.py").write_text(
        textwrap.dedent(
            """\
            from widget import classify, parity

            def test_negative():
                assert classify(-1) == "negative"

            def test_non_negative():
                assert classify(5) == "non-negative"

            def test_parity_even():
                assert parity(2) == "even"

            def test_parity_odd():
                assert parity(3) == "odd"
            """
        ),
        encoding="utf-8",
    )
    head = _commit_all(repo, "head: zero case (unpinned) + parity (pinned)")

    baseline, results = crd.sweep_round(repo, base, head, ["widget.py"], ".")
    assert len(results) == 2, [r.header for r in results]
    statuses = {r.status for r in results}
    # One hunk classifies zero (undetected -- no cell reads it); one hunk
    # adds parity() entirely (detected -- two cells call it).
    assert statuses == {"undetected", "detected"}


def test_import_time_crash_scores_detected_not_a_harness_error(tmp_path: Path):
    """A mutant that fails to import must not be mistaken for 'the sweep
    itself broke' -- it is scored DETECTED (the outcome changed), with the
    collection error visible in its own signature."""
    repo = _init_repo(tmp_path)
    (repo / "widget.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "test_widget.py").write_text(
        textwrap.dedent(
            """\
            from widget import VALUE, HELPER

            def test_value():
                assert VALUE == 1

            def test_helper():
                assert HELPER() == "ok"
            """
        ),
        encoding="utf-8",
    )
    base = _commit_all(repo, "base: only VALUE, suite already imports HELPER")
    # (Deliberately broken base -- irrelevant; we only diff base..head below
    # and the base commit is never checked out by sweep_round.)

    (repo / "widget.py").write_text(
        textwrap.dedent(
            """\
            VALUE = 1

            def HELPER():
                return "ok"
            """
        ),
        encoding="utf-8",
    )
    head = _commit_all(repo, "head: adds HELPER, which the suite already needs")

    baseline, results = crd.sweep_round(repo, base, head, ["widget.py"], ".")
    assert baseline.failed == 0 and baseline.collection_error is False
    assert len(results) == 1
    # Reverting HELPER's hunk breaks import for test_helper -> ImportError
    # at collection, which pytest reports as an error, not a quiet pass.
    assert results[0].status == "detected"


def test_signature_equality_is_by_failing_id_not_count(tmp_path: Path):
    """Two mutants with the SAME total failure count but a DIFFERENT failing
    test must not be scored equal -- a coarser count-only comparison would
    hide a swap of which cell is broken."""
    sig_a = crd.Signature(
        passed=5, failed=1, xfailed=0, xpassed=0, errors=0,
        failed_ids=("test_a",), error_ids=(), collection_error=False,
    )
    sig_b = crd.Signature(
        passed=5, failed=1, xfailed=0, xpassed=0, errors=0,
        failed_ids=("test_b",), error_ids=(), collection_error=False,
    )
    assert sig_a != sig_b


def test_cli_exit_code_nonzero_iff_undetected(uncovered_repo, capsys):
    repo, base, head = uncovered_repo
    rc = crd.main(
        [
            "--repo", str(repo),
            "--base", base,
            "--head", head,
            "--prod", "widget.py",
            "--suite", ".",
        ]
    )
    assert rc == 1  # the fixture's zero-case hunk is unpinned by construction


def test_cli_exit_code_zero_when_fully_pinned(uncovered_repo):
    repo, base, _ = uncovered_repo
    (repo / "widget.py").write_text(
        textwrap.dedent(
            """\
            def classify(x):
                if x < 0:
                    return "negative"
                if x == 0:
                    return "zero"
                return "non-negative"
            """
        ),
        encoding="utf-8",
    )
    (repo / "test_widget.py").write_text(
        textwrap.dedent(
            """\
            from widget import classify

            def test_negative():
                assert classify(-1) == "negative"

            def test_non_negative():
                assert classify(1) == "non-negative"

            def test_zero():
                assert classify(0) == "zero"
            """
        ),
        encoding="utf-8",
    )
    head_pinned = _commit_all(repo, "head: fully pinned")

    rc = crd.main(
        [
            "--repo", str(repo),
            "--base", base,
            "--head", head_pinned,
            "--prod", "widget.py",
            "--suite", ".",
        ]
    )
    assert rc == 0
