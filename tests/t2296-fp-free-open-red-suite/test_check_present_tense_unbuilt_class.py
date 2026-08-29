"""T-2385 (Brunel, fold round 43) -- the test pin for
`tests/ci/check_present_tense_defect_comments.py`'s second, independent
defect check (`find_stale_unbuilt_claim`/`find_stale_unbuilt_claims`/
`scan_unbuilt_claims`, and their wiring into `main()`).

WHY THIS FILE LIVES HERE, NOT BESIDE THE MODULE IT TESTS. The natural home
for a check module's own red suite is `tests/ci/test_check_<name>.py`
(`tests/ci/test_check_present_tense_defect_comments.py`, already present) --
but this ticket's own writable scope names `tools/ci/
check_present_tense_defect_comments.py` (this repo's actual path:
`tests/ci/check_present_tense_defect_comments.py`) as the one file to edit,
not its existing test file, and per Brunel's own charter a builder does not
edit another persona's test suite. `tests/t2296-fp-free-open-red-suite/` is
writable in full, so this pin lives there instead, importing the production
module the same way `test_archive_gate.py` and its siblings already do
(`sys.path.insert(0, _CI_DIR)`).

WHAT IS PINNED. Per `StandardsDocument.md` Sec4's population-validation
requirement -- a structural check earns trust by reproducing an
independently-found population BEFORE any fix lands, not by a fault its own
author injects -- this file:

  1. Confirms the new check's OWN rule coverage: a constructed fixture for
     each alternative in `_UNBUILT_CLAIM_PATTERN` is flagged when the file
     carries no live `xfail` marker, and NOT flagged when it does (the file-
     granularity marker rule this module's own docstring documents under
     "THE MARKER, AND WHY FILE-GRANULARITY").
  2. Replays the REAL historical population: `test_archive_composition.py`
     and `test_archive_gate.py` as they stood at this ticket's own starting
     commit (`7a77a07543ce159617e075a4fa72cc972c1255bd`, recovered via
     `git show`, never a hand-transcribed stand-in) -- both must flag. Both
     files' CURRENT content (this round's own fix) must be clean. This is
     the exact population T-2382 finding S3 measured (nine present-tense
     sites across these two files) and D-SLM5018 S1/S3 measured one round
     earlier on the same suite.
  3. Confirms the deliberately narrower input coverage
     (`_UNBUILT_CLAIM_GLOBS`, scoped to this suite alone): the two
     `tests/ci/` files this module's own docstring names as a measured,
     accurate false positive under the wider `tests/**/*.py` net are not
     scanned by the narrower one.
  4. An end-to-end run of the real `main()` against this actual repository
     confirms it exits 0 today -- StandardsDocument Sec4's "a test exercises
     the real workload" clause, applied here rather than only to synthetic
     fixtures.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS_ROOT = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_TESTS_ROOT)
_CI_DIR = os.path.join(_TESTS_ROOT, "ci")

sys.path.insert(0, _CI_DIR)
import check_present_tense_defect_comments as cptdc  # noqa: E402

# This ticket's own starting commit (T-2385's brief: "branch@7a77a07") --
# the state of the archive-gate red suite BEFORE this round's fix, and the
# real, independently-found population this check is validated against.
_HISTORICAL_COMMIT = "7a77a07543ce159617e075a4fa72cc972c1255bd"

_HISTORICAL_SITES = (
    "tests/t2296-fp-free-open-red-suite/test_archive_composition.py",
    "tests/t2296-fp-free-open-red-suite/test_archive_gate.py",
)


def _write(tmpdir: str, rel_path: str, content: str) -> str:
    abs_path = os.path.join(tmpdir, rel_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)
    return abs_path


def _git_show(rev: str, path: str) -> str:
    r = subprocess.run(
        ["git", "show", "{}:{}".format(rev, path)],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    if r.returncode != 0:
        pytest.skip(
            "cannot recover {}:{} via git show (not a git checkout, or the "
            "commit is unreachable in this environment): {}".format(
                rev, path, r.stderr)
        )
    return r.stdout


# ===========================================================================
# Rule coverage -- each alternative in _UNBUILT_CLAIM_PATTERN, individually.
# ===========================================================================

_TRIGGER_PHRASES = (
    ("not yet built", "check_fp_free_scan.py is not yet built at this path.\n"),
    ("NONE OF THIS IS BUILT", "NONE OF THIS IS BUILT YET, confirmed by execution.\n"),
    ("still globs", "find_target_objects still globs a directory for object files.\n"),
    ("genuinely red-unimplemented", "every cell here is genuinely red-unimplemented today.\n"),
    ("unbuilt", "this is the still-unbuilt half of the contract.\n"),
    ("does not read archives", "scan_build_output.py does not read archives at all yet.\n"),
    ("coincidentally the correct exit code",
     "it exits 2 today -- coincidentally the correct exit code, for the wrong reason.\n"),
    ("coincidentally exits", "the driver coincidentally exits 2 for an unrelated reason.\n"),
    ("future-tense will/once/lands",
     "the design specifies will keep reading it once the archive path lands.\n"),
)


@pytest.mark.parametrize("label,line", _TRIGGER_PHRASES)
def test_each_trigger_phrase_with_no_marker_is_flagged(label, line):
    result = cptdc.find_stale_unbuilt_claim('"""{}"""\n'.format(line))
    assert result is not None, "expected {!r} to be flagged, file text: {!r}".format(label, line)


@pytest.mark.parametrize("label,line", _TRIGGER_PHRASES)
def test_each_trigger_phrase_with_a_live_marker_is_not_flagged(label, line):
    """The file-granularity marker rule: a live xfail marker ANYWHERE in the
    file clears every trigger phrase in it, by design -- this is what keeps
    test_check_fp_free_scan.py's own genuinely-open D-SLM5009 cells (three
    live xfail markers, one stale "NOT YET BUILT" import-guard comment
    predating the archive round) from being flagged."""
    text = '"""{}"""\n@pytest.mark.xfail(strict=True, reason="open")\ndef test_x():\n    pass\n'.format(line)
    result = cptdc.find_stale_unbuilt_claim(text)
    assert result is None, "expected {!r} to clear once a live marker is present, got {!r}".format(label, result)


def test_a_file_with_neither_a_trigger_nor_a_marker_is_not_flagged():
    result = cptdc.find_stale_unbuilt_claim('"""a perfectly ordinary docstring."""\n')
    assert result is None


def test_scan_unbuilt_claims_reports_a_missing_file_distinctly():
    failures = cptdc.scan_unbuilt_claims(["does/not/exist.py"], repo_root=_REPO_ROOT)
    assert len(failures) == 1
    assert "file not found" in failures[0]


def test_scan_unbuilt_claims_end_to_end_over_a_synthetic_tree():
    with tempfile.TemporaryDirectory() as tmp:
        clean = _write(tmp, "site_clean.py", '"""nothing unusual here."""\n')
        dirty = _write(tmp, "site_dirty.py", '"""NONE OF THIS IS BUILT YET."""\n')
        failures = cptdc.scan_unbuilt_claims([clean, dirty], repo_root=tmp)
        assert len(failures) == 1
        assert "site_dirty.py" in failures[0]


# ===========================================================================
# The real historical population (T-2382 finding S3 / D-SLM5018 S1/S3):
# both files, as they stood before this round's fix, must flag; both files'
# current content must be clean. Fixing first would destroy this population
# (StandardsDocument.md Sec4) -- these cells replay recovered text, they do
# not rely on the working tree ever having been in the pre-fix state.
# ===========================================================================

@pytest.mark.parametrize("rel_path", _HISTORICAL_SITES)
def test_fires_on_the_historical_pre_fix_site(rel_path):
    text = _git_show(_HISTORICAL_COMMIT, rel_path)
    result = cptdc.find_stale_unbuilt_claim(text)
    assert result is not None, (
        "{} at {} is the real, measured T-2382 finding S3 population -- "
        "expected this check to flag it".format(rel_path, _HISTORICAL_COMMIT)
    )


@pytest.mark.parametrize("rel_path", _HISTORICAL_SITES)
def test_does_not_fire_on_the_current_post_fix_site(rel_path):
    abs_path = os.path.join(_REPO_ROOT, rel_path)
    result = cptdc.find_stale_unbuilt_claims(abs_path)
    assert result is None, (
        "{} still flags after this round's own fix: {!r} -- the fix did not "
        "close every trigger phrase, or a marker this check relies on was "
        "removed".format(rel_path, result)
    )


# ===========================================================================
# Input coverage, deliberately narrower than the severity-label check: the
# two tests/ci/ files this module's own docstring names as a measured,
# accurate false positive under tests/**/*.py must not be reached by
# _UNBUILT_CLAIM_GLOBS's own narrower scope.
# ===========================================================================

_KNOWN_FALSE_POSITIVE_FILES = (
    "tests/ci/check_checked_chain_funnel_position_cap_not_a_stub.py",
    "tests/ci/test_check_checked_chain_funnel_position_cap_not_a_stub.py",
)


@pytest.mark.parametrize("rel_path", _KNOWN_FALSE_POSITIVE_FILES)
def test_known_false_positive_files_would_trigger_under_the_bare_pattern(rel_path):
    """Confirms these two files are a live false positive under the trigger
    pattern alone (accurate historical narration about a DIFFERENT function's
    own past defect, not a current unbuilt-status claim) -- the reason
    _UNBUILT_CLAIM_GLOBS is scoped to this suite rather than the whole tree,
    per this module's own docstring ("INPUT COVERAGE, DELIBERATELY
    NARROWER"). If this stops matching, the false-positive rationale in that
    docstring section is stale and the input coverage may be safe to widen."""
    abs_path = os.path.join(_REPO_ROOT, rel_path)
    with open(abs_path, "r", encoding="utf-8") as f:
        text = f.read()
    assert cptdc._UNBUILT_CLAIM_PATTERN.search(text) is not None, (
        "{} was expected to contain an 'unbuilt'-shaped phrase (the known, "
        "accurate false positive this module's own docstring names)".format(rel_path)
    )


def test_unbuilt_claim_globs_do_not_reach_the_known_false_positive_files():
    scanned = set(cptdc._glob_files(cptdc._UNBUILT_CLAIM_GLOBS, _REPO_ROOT))
    for rel_path in _KNOWN_FALSE_POSITIVE_FILES:
        abs_path = os.path.join(_REPO_ROOT, rel_path)
        assert abs_path not in scanned, (
            "{} is reached by _UNBUILT_CLAIM_GLOBS -- the narrower input "
            "coverage this module's own docstring documents has widened; "
            "confirm the known false positive is still correctly excluded "
            "or has been resolved".format(rel_path)
        )


# ===========================================================================
# End-to-end: the real production entry point, against the real repository.
# ===========================================================================

def test_main_is_clean_against_the_real_repository_today():
    """StandardsDocument.md Sec4's "a test exercises the real workload"
    clause: at least one cell runs the actual check against the actual
    repository, not only synthetic fixtures. Must be 0 after this round's
    own fix -- if this goes red, either a new stale-unbuilt claim was
    introduced somewhere in tests/t2296-fp-free-open-red-suite/, or an
    unrelated severity-label citation regressed."""
    code = cptdc.main()
    assert code == 0
