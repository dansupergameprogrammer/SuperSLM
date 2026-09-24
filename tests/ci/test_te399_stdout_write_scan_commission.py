"""TE-399 (Curie), TE-402 S3 fix round -- commissioning constructions for the hosted-CI source
scan Poirot's remedy #4 asks the builder for (Claude/Poirot/fcbc6a7-slm171-stdout-fix.md): "add a
hosted-CI source scan that fails on any stdout write in src/ and include/. Per Sec4, validate it
against the 1.7.0 base tree before trusting it: it must flag the old harness print site there."

This module is the test side only, authored independently of the scanner's own implementation
(conductor's dispatch, 2026-09-23: "authored by you, blind to any controls the builder writes") --
the two population-validation cases below, must-reject and must-accept, are each materialized
fresh from a real git ref this module resolves itself (never a fixture file the scanner's own
author supplied), the same population-validation discipline StandardsDocument Sec4 requires of any
new check and this exact directory's own siblings already carry (check_no_pow_operator.py's
"**"-ban commission, check_no_forward_leaf_calls.py's leaf-call-ban commission). A scanner whose
own self-test passes only against fixtures its own author chose is not evidence its check catches
what it claims to catch on a tree neither side hand-picked.

INTERFACE CONTRACT THIS MODULE COMMISSIONS (not yet implemented anywhere in this repo as of this
commit -- red by ModuleNotFoundError until the builder adds it, the Python-import equivalent of
T-2019's own LINK-RED convention for a C++ symbol declared but not yet defined): a module at
tests/ci/check_no_library_stdout_write.py, matching this directory's own established shape
(check_no_pow_operator.py, check_no_forward_leaf_calls.py), exposing:

    def scan_for_stdout_writes(root_dirs: list[str]) -> list[str]:
        '''Every source line under any of `root_dirs` (walked recursively; .c/.cpp/.h/.hpp/.hlsl/
        .hlsli by extension, matching this repo's own shipped-library source set) that writes to
        stdout -- printf/wprintf/puts/std::cout/fwrite(..., stdout)/fprintf(stdout, ...) and
        equivalents, the same enumeration CHANGELOG.md's own 1.7.1 entry and
        docs/releases/1.7.1.md's "Sweep" paragraph describe as already performed by hand for this
        release. Returns "path:line: reason" strings, empty if clean.'''

    def main(argv: list[str]) -> int:
        '''CLI entry point: argv names one or more root directories (this repo's own src/ and
        include/ in normal CI use); prints every hit from scan_for_stdout_writes and returns 1 if
        any exist, 0 if the scan is clean -- the same contract check_no_pow_operator.main() and
        check_no_forward_leaf_calls.main() already use in this directory, so this scan wires into
        CI (.github/workflows/tests.yml) and a local `pytest tests/ci/` run the same way every
        sibling check here does.'''

Both fixtures below are the REAL src/+include/ trees of this repo at two real commits -- not a
synthetic one-file snippet -- so the must-accept leg also proves the scan does not false-positive
anywhere else in the real tree it is committed to run against, and the must-reject leg proves the
scan actually walks a multi-file tree to find the one real site, not merely a hand-fed path.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

# f43ab15: SuperSLM v1.7.0's exact tip (TE-393's own reviewed pin, D-SLM7752) -- the tree that
# still carries the one known stdout-write site (src/gpu/d3d12_harness.h, the `std::wprintf` T-393
# found). A fixed, stable historical ref in this same repo: always resolvable via `git archive`
# regardless of which branch or worktree runs this file, and never a ref the scanner's own author
# chose.
_MUST_REJECT_REF = "f43ab15"

sys.path.insert(0, _THIS_DIR)
try:
    import check_no_library_stdout_write as scanner  # type: ignore[import-not-found]

    _IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:  # pragma: no cover -- exercised by the collection itself
    scanner = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


def _git_archive_tree(ref: str, dest_dir: str) -> None:
    """Materializes `ref`'s real src/ and include/ trees (this repo's own shipped-library source,
    the same two directories Poirot's remedy names) fresh into `dest_dir`, via a real `git
    archive` of a real commit -- never a copy of the currently-checked-out working tree (which
    would silently pick up whichever branch this module happens to run from), and never a fixture
    file authored for this test alone."""
    os.makedirs(dest_dir, exist_ok=True)
    archive_path = os.path.join(dest_dir, "archive.tar")
    subprocess.run(
        ["git", "archive", ref, "-o", archive_path, "--", "src", "include"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
    )
    # No --force-local: that flag is bsdtar/GNU-tar's (Git Bash's own tar), and this module runs
    # under python.exe -- whose resolved `tar` on this machine is Windows' native bsdtar
    # (System32\tar.exe), which does not recognize it and refuses outright. Plain `-xf` extracts
    # correctly under both; the flag is unnecessary here since tar's own cwd is already a real
    # Windows path, never a POSIX-style "D:" a shell could misparse as a remote host.
    subprocess.run(
        ["tar", "-xf", archive_path],
        cwd=dest_dir,
        check=True,
        capture_output=True,
    )
    os.remove(archive_path)


def _resolve_head() -> str:
    """The commit this module itself is running from -- the must-accept case's ref. Never
    hardcoded: a fixture pinned to today's fix commit would stop proving anything the moment a
    later, unrelated commit lands on top of it, and "the tree this test suite is currently
    checked out against" is exactly the tree a CI run of this same scan is meant to accept."""
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


def test_scanner_module_exists():
    """The commission's own precondition, stated as a named, readable failure rather than every
    test below failing on a bare ModuleNotFoundError with no context. Red until the builder adds
    tests/ci/check_no_library_stdout_write.py per this module's own interface contract (see the
    module docstring)."""
    assert scanner is not None, (
        "tests/ci/check_no_library_stdout_write.py does not exist yet (or fails to import: "
        f"{_IMPORT_ERROR!r}) -- TE-402 S3's remedy #4, commissioned by this module's own "
        "docstring. Every other test in this file is meaningless until it does."
    )


def test_must_reject_the_v1_7_0_tree_and_names_the_site():
    """Poirot's remedy #4, verbatim: "validate it against the 1.7.0 base tree before trusting it:
    it must flag the old harness print site there." Materializes f43ab15's real src/+include/
    fresh (never a hand-picked snippet) and requires the scan to both fail AND name the exact
    known site -- a scanner that merely reports SOME failure without pointing at the real cause
    is not the diagnostic Poirot's own S3 finding asked for ("a future printf in src/ ships
    silently" is the failure mode this exists to prevent, and a mute failure reproduces half of
    it)."""
    if scanner is None:
        import pytest

        pytest.skip(f"scanner module not yet implemented: {_IMPORT_ERROR!r}")
    with tempfile.TemporaryDirectory(prefix="te399_ci_scan_reject_") as tmp:
        _git_archive_tree(_MUST_REJECT_REF, tmp)
        hits = scanner.scan_for_stdout_writes(
            [os.path.join(tmp, "src"), os.path.join(tmp, "include")]
        )
        assert hits, (
            f"the scan reported ZERO hits against SuperSLM {_MUST_REJECT_REF} (v1.7.0), which "
            "still contains the known stdout-write site (src/gpu/d3d12_harness.h) -- the scan "
            "must-reject this tree, per Poirot's own remedy #4"
        )
        assert any("d3d12_harness.h" in h for h in hits), (
            f"the scan flagged {len(hits)} hit(s) against {_MUST_REJECT_REF} but none names "
            f"d3d12_harness.h, the one real site TE-393/D-SLM7752 found -- hits: {hits!r}"
        )
        code = scanner.main([os.path.join(tmp, "src"), os.path.join(tmp, "include")])
        assert code != 0, (
            f"scanner.main() returned {code} (success) against {_MUST_REJECT_REF}, which must "
            "be refused"
        )


def test_must_accept_the_fixed_tree():
    """The positive control: the real src/+include/ tree of the commit this test suite is
    currently checked out against (never a pinned SHA this module hardcodes) must scan clean.
    Proves the scanner does not merely fail loudly on any input -- it also has to genuinely pass
    the real, current, already-fixed tree, over every file the scan walks, not only the one file
    the must-reject leg exercises."""
    if scanner is None:
        import pytest

        pytest.skip(f"scanner module not yet implemented: {_IMPORT_ERROR!r}")
    fixed_ref = _resolve_head()
    with tempfile.TemporaryDirectory(prefix="te399_ci_scan_accept_") as tmp:
        _git_archive_tree(fixed_ref, tmp)
        hits = scanner.scan_for_stdout_writes(
            [os.path.join(tmp, "src"), os.path.join(tmp, "include")]
        )
        assert hits == [], (
            f"the scan reported {len(hits)} hit(s) against the current, already-fixed tree "
            f"({fixed_ref}), which must scan clean (D-SLM7753: the engine library writes "
            f"nothing to stdout) -- hits: {hits!r}"
        )
        code = scanner.main([os.path.join(tmp, "src"), os.path.join(tmp, "include")])
        assert code == 0, f"scanner.main() returned {code} (failure) against the fixed tree {fixed_ref}"


def test_must_reject_a_synthetic_new_stdout_write_the_known_site_alone_would_miss():
    """S3's own headline finding: "a future printf in src/ ships silently." Proves the scan is a
    general property check, not a regex for the one line TE-393 happened to find -- introduces a
    brand-new stdout write, in a file and a form the known site never used (plain std::printf in
    a different translation unit), into a scratch copy of the ALREADY-FIXED tree, and requires
    the scan to catch it anyway. Never writes into the real checkout (StandardsDocument's mutant-
    proof discipline: a temp copy only)."""
    if scanner is None:
        import pytest

        pytest.skip(f"scanner module not yet implemented: {_IMPORT_ERROR!r}")
    fixed_ref = _resolve_head()
    with tempfile.TemporaryDirectory(prefix="te399_ci_scan_mutant_") as tmp:
        _git_archive_tree(fixed_ref, tmp)
        planted = os.path.join(tmp, "src", "te399_synthetic_new_stdout_write.cpp")
        with open(planted, "w", encoding="utf-8") as f:
            f.write('#include <cstdio>\nvoid TE399SyntheticNoise() { std::printf("noise\\n"); }\n')
        hits = scanner.scan_for_stdout_writes(
            [os.path.join(tmp, "src"), os.path.join(tmp, "include")]
        )
        assert any("te399_synthetic_new_stdout_write" in h for h in hits), (
            "planting a brand-new std::printf in a new file under the already-fixed tree was not "
            f"caught -- the scan is not a fixed-line pin, it must generalize. hits: {hits!r}"
        )
