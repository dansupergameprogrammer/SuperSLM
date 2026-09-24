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

import pytest

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



# ---------------------------------------------------------------------------------------------
# TE-407 (Poirot, Claude/Poirot/7f7436d-slm171-stdout-reconfirm.md, S1) re-commission: the prior
# round's population was the two forms the scanner is surest to catch (a plain `fprintf(stdout,
# ...)` and `std::cout`), so "neither the pattern set nor the input set was validated." Poirot ran
# an independent 15-form population (`Claude/Poirot/7f7436d-slm171-stdout-reconfirm-probe/
# scan_mutants.py`, executed against the fixed tree, one planted form per run) and found 13 of 15
# escape today's scanner. That exact population, reproduced verbatim here (not re-derived) per the
# conductor's dispatch, 2026-09-24 ("use the reviewer's executed 15-form population as the
# independent population"): each form plants ONE realistic stdout-write shape into a fresh
# git-archive copy of the fixed tree (never the known site's own text, and never a copy of a live
# worktree that could pick up an unrelated branch), and the scan must flag the planted file by
# name. The must-accept leg above already proves the real tree has none of these; this section
# proves the scan would catch each one if it appeared.
# ---------------------------------------------------------------------------------------------

# Verbatim from Claude/Poirot/7f7436d-slm171-stdout-reconfirm-probe/scan_mutants.py -- the
# reviewer's own executed population, not re-derived by this seat. Each maps a planted file's
# name to its content; every file is planted alone, one temp tree per case.
_TE407_MUTANT_FORMS = {
    "ctrl_fprintf_stdout.cpp": 'void f(){ std::fprintf(stdout, "x"); }\n',
    "ctrl_cout.cpp": "void f(){ std::cout << 1; }\n",
    "multiline_fprintf.cpp": 'void f(){ std::fprintf(\n    stdout, "x %d", 1); }\n',
    "multiline_fwrite.cpp": (
        "void f(const char* b, size_t n){ std::fwrite(b, 1, n,\n    stdout); }\n"
    ),
    "vprintf.cpp": (
        "#include <cstdarg>\nvoid f(const char* fmt, va_list ap){ std::vprintf(fmt, ap); }\n"
    ),
    "vfprintf_stdout.cpp": (
        "#include <cstdarg>\nvoid f(const char* fmt, va_list ap){ "
        "std::vfprintf(stdout, fmt, ap); }\n"
    ),
    "printf_s.cpp": 'void f(){ printf_s("x"); }\n',
    "putchar.cpp": "void f(){ std::putchar(120); }\n",
    "fputws_stdout.cpp": 'void f(){ std::fputws(L"x", stdout); }\n',
    "fputc_stdout.cpp": "void f(){ std::fputc(120, stdout); }\n",
    "using_cout.cpp": "using namespace std;\nvoid f(){ cout << 1; }\n",
    "posix_write_fd1.cpp": "void f(const char* b, unsigned n){ _write(1, b, n); }\n",
    "stderr_then_stdout_same_line.cpp": (
        'void f(){ std::fprintf(stderr, "a"); std::fprintf(stdout, "b"); }\n'
    ),
    "stdout_via_variable.cpp": 'void f(){ FILE* o = stdout; std::fprintf(o, "x"); }\n',
}
# The 15th form: a write appended to a shipped, public .inc header (sslm_abi.h:324 pulls it in),
# not a scanned extension at 7f7436d and not a new .cpp translation unit -- the escape family
# every other planted form in this population cannot exercise.
_TE407_INC_MUTANT_PATH = "include/superslm/sslm_abi_functions.inc"
_TE407_INC_MUTANT_BODY = '\ninline void TE407Noise(){ std::printf("x"); }\n'


def _plant_and_scan(rel_path: str, body: str, prefix: str):
    """Materializes the current fixed tree fresh, plants one file (or appends to one, for the
    .inc case) at `rel_path` relative to the tree root, and returns the scan's hits restricted to
    that path -- so a case's own assertion reads directly as "was the planted form caught",
    independent of whether some unrelated file also happens to hit (it should not, per
    test_must_accept_the_fixed_tree, but this keeps each case's failure message unambiguous about
    which form it is about)."""
    fixed_ref = _resolve_head()
    tmp = tempfile.mkdtemp(prefix=prefix)
    try:
        _git_archive_tree(fixed_ref, tmp)
        full_path = os.path.join(tmp, *rel_path.split("/"))
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        mode = "a" if os.path.exists(full_path) else "w"
        with open(full_path, mode, encoding="utf-8") as f:
            if mode == "w":
                f.write("#include <cstdio>\n#include <iostream>\n" + body)
            else:
                f.write(body)
        hits = scanner.scan_for_stdout_writes(
            [os.path.join(tmp, "src"), os.path.join(tmp, "include")]
        )
        needle = os.path.basename(rel_path)
        return [h for h in hits if needle in h]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.parametrize("name,body", sorted(_TE407_MUTANT_FORMS.items()))
def test_must_reject_each_te407_population_form(name, body):
    """One case per form in Poirot's own executed 15-form population (minus the .inc form, its
    own case immediately below). RED today (7f7436d): 13 of these 14 plus the .inc form (15 of 15
    minus the 2 controls `ctrl_fprintf_stdout.cpp`/`ctrl_cout.cpp`) are reported ESCAPED by
    `scan_mutants.py`'s own executed run -- this reproduces that population as real, executable
    pytest cases rather than a report, so "N of 15 escaping" is a live count (`pytest ... -v`'s own
    pass/fail tally), not a transcribed table. Checked against a scratch reproduction of the
    reviewer's own probe (`Claude/Poirot/7f7436d-slm171-stdout-reconfirm-probe/tokenban.py`'s
    exact token pattern, never committed here): 18 of these 19 cases (all 14 parametrized forms
    but `posix_write_fd1.cpp`, plus the `.inc` case and the pre-existing generalization case) go
    green under that token list. `posix_write_fd1.cpp` (`_write(1, b, n)`) does NOT -- the token
    list (`stdout|STD_OUTPUT_HANDLE|cout|wcout|v?w?printf(_s)?|puts|_putws|putw?char`) names no
    POSIX write call, so this one case is a genuine residual the token ban as probed does not
    close; left in this population rather than removed, since surfacing that gap is this
    commission's job, not smoothing it over. Named in
    Claude/Curie/te399-slm171-stdout-red-2026-09-23.md Sec.11 for the conductor to route."""
    if scanner is None:
        pytest.skip(f"scanner module not yet implemented: {_IMPORT_ERROR!r}")
    hits = _plant_and_scan(f"src/{name}", body, prefix=f"te407_form_{name}_")
    assert hits, (
        f"planting {name!r} (Poirot's own executed population, "
        "Claude/Poirot/7f7436d-slm171-stdout-reconfirm-probe/scan_mutants.py) into the fixed tree "
        f"was NOT caught -- content: {body!r}"
    )


def test_must_reject_a_stdout_write_in_a_shipped_public_inc_header():
    """The 15th form, kept as its own named case rather than folded into the parametrize above:
    unlike every other form, this one is not a new .cpp translation unit -- it appends to a real,
    shipped, public header (`include/superslm/sslm_abi_functions.inc`, pulled in by
    `sslm_abi.h:324`) whose extension (`.inc`) is not in `_SCANNED_EXTENSIONS` at 7f7436d at all,
    so this case also stands as this population's own test of extension coverage, not only pattern
    coverage. RED today; must go green once `.inc`/`.def`/`.inl`/`.ipp` are scanned (Poirot's
    remedy #1)."""
    if scanner is None:
        pytest.skip(f"scanner module not yet implemented: {_IMPORT_ERROR!r}")
    hits = _plant_and_scan(_TE407_INC_MUTANT_PATH, _TE407_INC_MUTANT_BODY, prefix="te407_inc_")
    assert hits, (
        "appending a std::printf to the shipped, public "
        f"{_TE407_INC_MUTANT_PATH} was not caught -- content appended: {_TE407_INC_MUTANT_BODY!r}"
    )


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
