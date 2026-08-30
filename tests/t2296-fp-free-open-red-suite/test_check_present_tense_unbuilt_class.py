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
     carries no covering marker, and NOT flagged when a live
     `@pytest.mark.xfail` decorator's own `reason=` text also matches the
     same alternative (the marker-reason rule this module's own docstring
     documents under "WHAT COUNTS AS THE MARKER") -- one rule-coverage cell
     per alternative on a single physical line, and (T-2387) one more per
     multi-word alternative wrapped across a line break the way this
     suite's own prose actually wraps.
  2. Replays the REAL historical population: `test_archive_composition.py`
     and `test_archive_gate.py` as they stood at this ticket's own starting
     commit (`7a77a07543ce159617e075a4fa72cc972c1255bd`), vendored verbatim
     in `present_tense_unbuilt_class_historical_fixtures/` (T-2387; see
     "POPULATION RECOVERY IS VENDORED" below) -- both must flag. Both
     files' CURRENT content (this round's own fix) must be clean. This is
     the exact population T-2382 finding S3 measured (nine present-tense
     sites across these two files) and D-SLM5018 S1/S3 measured one round
     earlier on the same suite. A dedicated cell also reproduces the
     reviewer's own whitespace-tolerance measurement (12 literal-space
     matches, 14 once the pattern is `\\s+`-tolerant) directly against the
     vendored blobs.
  3. Confirms the deliberately narrower input coverage
     (`_UNBUILT_CLAIM_GLOBS`, scoped to this suite alone): the two
     `tests/ci/` files this module's own docstring names as a measured,
     accurate false positive under the wider `tests/**/*.py` net are not
     scanned by the narrower one; and (T-2387) that this pin's OWN file is
     excluded by name from the same glob's scan surface, and would flag
     under the bare pattern if it were not.
  4. Two regression cells for the two Significant findings closed by
     T-2387 (Poirot 8788b01-t2386-archive-gate-confirmation.md N3/N4):
     an unrelated live marker must not launder a stale claim it says
     nothing about, and reintroducing the exact stale comment this round
     removed from `test_check_fp_free_scan.py:245`, alongside that file's
     own three genuinely unrelated markers, must still flag.
  5. An end-to-end run of the real `main()` against this actual repository
     confirms it exits 0 today -- StandardsDocument Sec4's "a test exercises
     the real workload" clause, applied here rather than only to synthetic
     fixtures.

POPULATION RECOVERY IS VENDORED, NOT LIVE `git show` (T-2387, Poirot
8788b01-t2386-archive-gate-confirmation.md N1). This pin used to recover
`7a77a07`'s pre-fix text of both historical files with a live
`git show <sha>` subprocess call at test time, skipping (not failing) when
the command could not resolve the commit. Measured: a real
`git clone --depth 1` of this repository -- the checkout every job in
`.github/workflows/tests.yml` performs, none of them setting
`fetch-depth` -- resolves that `git show` with exit 128, so both
population cells skipped silently and this whole pin reported green in the
exact environment that gates it (`fp-free-scan-gate`, which runs
`pytest tests/t2296-fp-free-open-red-suite -q`). The two files' pre-fix
text is now vendored once, verbatim, as
`present_tense_unbuilt_class_historical_fixtures/test_archive_composition_
pre.txt` and `..._test_archive_gate_pre.txt` (byte-identical to
`git show 7a77a07543ce159617e075a4fa72cc972c1255bd:<path>`, diffed at
authoring time) -- recovery no longer depends on git history being present
at all, so the population cells below run, and can fail, in any checkout
depth.
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS_ROOT = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_TESTS_ROOT)
_CI_DIR = os.path.join(_TESTS_ROOT, "ci")
_HISTORICAL_FIXTURES_DIR = os.path.join(_HERE, "present_tense_unbuilt_class_historical_fixtures")

sys.path.insert(0, _CI_DIR)
import check_present_tense_defect_comments as cptdc  # noqa: E402

# This ticket's own starting commit (T-2385's brief: "branch@7a77a07") --
# the state of the archive-gate red suite BEFORE this round's fix, and the
# real, independently-found population this check is validated against.
# Named here for provenance only; recovery no longer touches git (T-2387,
# see the module docstring's own "POPULATION RECOVERY IS VENDORED" section).
_HISTORICAL_COMMIT = "7a77a07543ce159617e075a4fa72cc972c1255bd"

# (repo-relative path, vendored fixture basename) -- one pair per historical
# site. The vendored file holds that path's exact content at
# _HISTORICAL_COMMIT, recovered once via `git show` at authoring time and
# diffed byte-for-byte against the live blob before being committed.
_HISTORICAL_SITES = (
    ("tests/t2296-fp-free-open-red-suite/test_archive_composition.py", "test_archive_composition_pre.txt"),
    ("tests/t2296-fp-free-open-red-suite/test_archive_gate.py", "test_archive_gate_pre.txt"),
)


def _write(tmpdir: str, rel_path: str, content: str) -> str:
    abs_path = os.path.join(tmpdir, rel_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)
    return abs_path


def _read_vendored_fixture(fixture_basename: str) -> str:
    """The vendored pre-fix text for one historical site (T-2387) -- a
    plain file read, with no git dependency and no skip path: a missing or
    unreadable fixture file is this module's own test infrastructure being
    broken and must fail loud, the same posture
    `check_present_tense_defect_comments.py` itself takes toward a file it
    cannot vouch for."""
    path = os.path.join(_HISTORICAL_FIXTURES_DIR, fixture_basename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ===========================================================================
# Rule coverage -- each alternative in _UNBUILT_CLAIM_PATTERN, individually,
# on a single physical line and (T-2387) wrapped across a line break the way
# this suite's own hard-wrapped prose actually wraps.
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
def test_each_trigger_phrase_with_a_covering_marker_is_not_flagged(label, line):
    """T-2387 (N3/N4): the marker must be a real `@pytest.mark.xfail(...)`
    decorator (verified via `ast`, not a string that merely spells one)
    whose OWN `reason=` text also matches the same trigger phrase -- not
    merely any marker's presence anywhere in the file. Embedding the exact
    trigger `line` in the decorator's own reason is what makes it
    "covering" rather than "present but unrelated"; see
    test_an_unrelated_live_marker_does_not_clear_a_stale_unbuilt_claim for
    the negative case this positive case is paired against."""
    text = '"""{}"""\n@pytest.mark.xfail(strict=True, reason={!r})\ndef test_x():\n    pass\n'.format(
        line, line.strip()
    )
    result = cptdc.find_stale_unbuilt_claim(text)
    assert result is None, "expected {!r} to clear once a covering marker is present, got {!r}".format(
        label, result
    )


# T-2387 (N2): each multi-word alternative, wrapped across a line break at
# the point a real occurrence actually wrapped (`does not read` / `archives`
# at test_archive_gate.py:743-744, `coincidentally` / `the correct exit
# code` at :406-407) or a representative equivalent point for the
# alternatives that were not themselves found wrapped in the historical
# population. Single-word alternatives (`unbuilt`) have no internal space to
# wrap and are omitted.
_WRAPPED_TRIGGER_PHRASES = (
    ("not yet built", "check_fp_free_scan.py is not yet\n    built at this path.\n"),
    ("NONE OF THIS IS BUILT", "NONE OF THIS IS\n    BUILT YET, confirmed by execution.\n"),
    ("still globs", "find_target_objects still\n    globs a directory for object files.\n"),
    ("genuinely red-unimplemented", "every cell here is genuinely\n    red-unimplemented today.\n"),
    ("does not read archives", "scan_build_output.py does not read\n    archives at all yet.\n"),
    ("coincidentally the correct exit code",
     "it exits 2 today -- coincidentally\n    the correct exit code, for the wrong reason.\n"),
    ("coincidentally exits", "the driver coincidentally\n    exits 2 for an unrelated reason.\n"),
)


@pytest.mark.parametrize("label,line", _WRAPPED_TRIGGER_PHRASES)
def test_each_wrapped_trigger_phrase_with_no_marker_is_flagged(label, line):
    """T-2387 (N2): a trigger phrase split across a line break -- this
    suite's own real, hard-wrapped prose shape -- must still be found. The
    literal-space pattern this check shipped with could not see this; see
    the module docstring's own "WHITESPACE-TOLERANT MATCHING" section."""
    result = cptdc.find_stale_unbuilt_claim('"""{}"""\n'.format(line))
    assert result is not None, "expected wrapped {!r} to be flagged, file text: {!r}".format(label, line)


# ===========================================================================
# T-2387 (N3/N4): the marker-reason mechanism, both directions.
# ===========================================================================

def test_an_unrelated_live_marker_does_not_clear_a_stale_unbuilt_claim():
    """N3 (Poirot 8788b01-t2386-archive-gate-confirmation.md): a live xfail
    marker whose OWN reason has nothing to do with the trigger phrase must
    not launder an unrelated stale claim -- the exact shape that let
    test_check_fp_free_scan.py:245's stale "NOT YET BUILT" import-guard
    comment hide behind three markers about T-2347's and D-SLM5009's own,
    genuinely unrelated, open questions."""
    text = (
        '"""NONE OF THIS IS BUILT YET."""\n'
        '@pytest.mark.xfail(strict=True, reason="an unrelated open question")\n'
        "def test_x():\n    pass\n"
    )
    result = cptdc.find_stale_unbuilt_claim(text)
    assert result is not None, "an unrelated marker must not clear the stale claim, got {!r}".format(result)


def test_a_fixture_string_spelling_a_marker_is_not_treated_as_a_real_one():
    """N4 (Poirot 8788b01-t2386-archive-gate-confirmation.md): a Python
    string literal that merely SPELLS `@pytest.mark.xfail(` -- this file's
    own fixture data one section up -- is not an AST decorator node and
    must not be credited as a live marker. Without the ast-based check,
    this text would have cleared, the same accidental protection N4 found
    saving this pin's own file."""
    text = (
        '"""NONE OF THIS IS BUILT YET."""\n'
        "FIXTURE = '@pytest.mark.xfail(strict=True, reason=\"NONE OF THIS IS BUILT YET\")'\n"
    )
    result = cptdc.find_stale_unbuilt_claim(text)
    assert result is not None, (
        "a string literal spelling marker syntax must not be read as a real "
        "decorator, got {!r}".format(result)
    )


def test_the_real_line_245_shape_is_caught_when_reintroduced():
    """N3, mutation test in the direction that proves the mechanism (not
    just the one comment) is fixed: read test_check_fp_free_scan.py's own
    CURRENT text -- which carries three real, unrelated xfail markers
    (T-2347's O4/S5, D-SLM5009b) -- and reintroduce, in memory only, the
    exact stale comment this round removed from line 245. The file must
    still flag: if this regresses to clean, the marker-reason check has
    regressed to file-wide presence and N3 is open again."""
    abs_path = os.path.join(
        _REPO_ROOT, "tests", "t2296-fp-free-open-red-suite", "test_check_fp_free_scan.py"
    )
    with open(abs_path, "r", encoding="utf-8") as f:
        current_text = f.read()
    fixed_line = "    import check_fp_free_scan as scan  # noqa: E402"
    assert fixed_line in current_text, "the fixed import line moved or was reworded; update this mutation"
    mutated_text = current_text.replace(
        fixed_line,
        "    import check_fp_free_scan as scan  # noqa: E402  -- NOT YET BUILT, design Sec4.1",
        1,
    )
    result = cptdc.find_stale_unbuilt_claim(mutated_text)
    assert result is not None, (
        "reintroducing the real line-245 stale comment alongside this file's own three "
        "unrelated markers must still flag, got {!r}".format(result)
    )


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
# (StandardsDocument.md Sec4) -- these cells replay vendored text (T-2387),
# they do not rely on the working tree ever having been in the pre-fix
# state, and they do not depend on git history being present (see the
# module docstring's own "POPULATION RECOVERY IS VENDORED" section).
# ===========================================================================

@pytest.mark.parametrize("rel_path,fixture_basename", _HISTORICAL_SITES)
def test_fires_on_the_historical_pre_fix_site(rel_path, fixture_basename):
    text = _read_vendored_fixture(fixture_basename)
    result = cptdc.find_stale_unbuilt_claim(text)
    assert result is not None, (
        "{} at {} is the real, measured T-2382 finding S3 population -- "
        "expected this check to flag it".format(rel_path, _HISTORICAL_COMMIT)
    )


@pytest.mark.parametrize("rel_path,fixture_basename", _HISTORICAL_SITES)
def test_does_not_fire_on_the_current_post_fix_site(rel_path, fixture_basename):
    abs_path = os.path.join(_REPO_ROOT, rel_path)
    result = cptdc.find_stale_unbuilt_claims(abs_path)
    assert result is None, (
        "{} still flags after this round's own fix: {!r} -- the fix did not "
        "close every trigger phrase, or a marker this check relies on was "
        "removed".format(rel_path, result)
    )


def test_whitespace_tolerant_pattern_match_count_against_the_vendored_blobs():
    """Reproduces Poirot 8788b01-t2386-archive-gate-confirmation.md's own
    E7/E8 measurement directly against the vendored historical population
    (T-2387): the literal-space pattern matches 12 real occurrences across
    both files combined; the shipped, `\\s+`-tolerant pattern matches 14.
    Pins the exact reviewer-measured counts, not just "more than before"."""
    literal_pattern_source = cptdc._UNBUILT_CLAIM_PATTERN.pattern.replace(r"\s+", " ")
    import re
    literal_pattern = re.compile(literal_pattern_source, re.IGNORECASE)
    literal_total = 0
    shipped_total = 0
    for _rel_path, fixture_basename in _HISTORICAL_SITES:
        text = _read_vendored_fixture(fixture_basename)
        literal_total += len(literal_pattern.findall(text))
        shipped_total += len(cptdc._UNBUILT_CLAIM_PATTERN.findall(text))
    assert literal_total == 12, "literal-space match count drifted from the reviewer's measured 12: got {}".format(
        literal_total
    )
    assert shipped_total == 14, "whitespace-tolerant match count drifted from the reviewer's measured 14: got {}".format(
        shipped_total
    )


# ===========================================================================
# Input coverage, deliberately narrower than the severity-label check: the
# two tests/ci/ files this module's own docstring names as a measured,
# accurate false positive under tests/**/*.py must not be reached by
# _UNBUILT_CLAIM_GLOBS's own narrower scope; neither must this pin's own
# file (T-2387, N4).
# ===========================================================================

_KNOWN_FALSE_POSITIVE_FILES = (
    "tests/ci/check_checked_chain_funnel_position_cap_not_a_stub.py",
    "tests/ci/test_check_checked_chain_funnel_position_cap_not_a_stub.py",
)

_THIS_PIN_FILE = "tests/t2296-fp-free-open-red-suite/test_check_present_tense_unbuilt_class.py"


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


def test_this_pin_file_would_trigger_under_the_bare_pattern_and_marker_check():
    """N4: proves the exclusion in `main()` is load-bearing, not decorative.
    Without `_UNBUILT_CLAIM_EXCLUDE_BASENAMES`, this pin's own file would be
    scanned (it lives inside `_UNBUILT_CLAIM_GLOBS`'s own scope) and would
    flag: it carries every trigger phrase verbatim as fixture data and has
    no real `@pytest.mark.xfail` decorator of its own to cover any of them."""
    abs_path = os.path.join(_REPO_ROOT, _THIS_PIN_FILE)
    result = cptdc.find_stale_unbuilt_claims(abs_path)
    assert result is not None, (
        "this pin file was expected to trigger the bare check (proving the "
        "named exclusion in main() is load-bearing); got None"
    )


def test_unbuilt_claim_globs_scan_surface_excludes_this_pin_file():
    this_basename = os.path.basename(_THIS_PIN_FILE)
    scanned_basenames = {os.path.basename(p) for p in cptdc._glob_files(cptdc._UNBUILT_CLAIM_GLOBS, _REPO_ROOT)}
    assert this_basename in scanned_basenames, "this pin file is expected to be inside the glob's raw scan surface"
    assert this_basename in cptdc._UNBUILT_CLAIM_EXCLUDE_BASENAMES, (
        "this pin file's basename is expected to be in _UNBUILT_CLAIM_EXCLUDE_BASENAMES"
    )


# ===========================================================================
# T-2403 (Curie), R9 -- the RENDERED-value-only population this check's
# raw-text-only scan cannot see (`Claude/Vitruvius/t2265-fold46-delta-
# manifest.md` Sec4; D-SLM5209/D-SLM5211/D-SLM5215). test_check_fp_free_
# scan.py's own prose wraps its "is not yet built" claim across two adjacent
# Python string literals (a quote/newline/indent/quote gap between "yet" and
# "built" that `\s+` does not bridge); the RENDERED string Python produces by
# concatenating the literals at parse time matches _UNBUILT_CLAIM_PATTERN
# cleanly. `find_stale_unbuilt_claim` today searches only raw source text and
# cannot see this shape.
#
# This population was independently derived three times before any repair
# (the adversary, the planner, and the conductor's own AST walk) and is
# pinned here a fourth time, permanently, against a VENDORED snapshot of
# test_check_fp_free_scan.py (commit b2325597a906 -- the same discipline
# "POPULATION RECOVERY IS VENDORED" above already applies: a pin against the
# live, mutable file would stop discriminating the day R9's own text fix
# lands and the rendered claims stop existing to be found). Per
# `StandardsDocument.md` Sec4, the repaired checker's new scan surface must
# reproduce this exact population -- 2 rendered-value matches, 0 at
# raw-source level, at the file's own lines 295 and 913 -- BEFORE either
# claim is fixed; fixing first would destroy the only population the
# repaired checker could be shown to catch.
# ===========================================================================

_R9_POPULATION_FIXTURE = "test_check_fp_free_scan_pre_r9.txt"
_R9_EXPECTED_LINES = (295, 913)


def test_r9_population_ast_walk_independently_finds_two_rendered_matches_zero_raw():
    """Independent verification, NOT the (absent) checker surface's own
    logic: walks the vendored file's own AST directly (this test's own
    code, not `find_stale_unbuilt_claim`) and checks each string-constant
    node's RENDERED value against `_UNBUILT_CLAIM_PATTERN` (the checker's
    shared vocabulary, not its detection mechanism) versus that same node's
    own raw source segment. Confirms the exact population this fold's own
    manifest derived: two rendered-only matches, at lines 295 and 913, and
    a whole-file raw-text search finds neither."""
    text = _read_vendored_fixture(_R9_POPULATION_FIXTURE)
    tree = ast.parse(text)
    rendered_hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if cptdc._UNBUILT_CLAIM_PATTERN.search(node.value):
                segment = ast.get_source_segment(text, node) or ""
                raw_visible = bool(cptdc._UNBUILT_CLAIM_PATTERN.search(segment))
                rendered_hits.append((node.lineno, raw_visible))
    lines_found = sorted(lineno for lineno, _raw_visible in rendered_hits)
    assert lines_found == list(_R9_EXPECTED_LINES), (
        "expected exactly the rendered-value matches at lines {}, found at "
        "{}".format(list(_R9_EXPECTED_LINES), lines_found)
    )
    assert all(not raw_visible for _lineno, raw_visible in rendered_hits), (
        "expected every rendered-value match to be INVISIBLE at raw-source "
        "level (the whole point of this population); got {}".format(rendered_hits)
    )
    assert cptdc._UNBUILT_CLAIM_PATTERN.search(text) is None, (
        "a whole-file raw-text search was expected to find nothing on this "
        "vendored population -- if it now matches, the population has "
        "changed and this fixture needs re-vendoring, not this assertion "
        "loosened"
    )


def test_r9_checker_flags_the_pinned_population_once_the_rendered_value_surface_exists():
    """THE genuinely red half, today: `find_stale_unbuilt_claim`'s current
    raw-text-only scan returns None on this vendored population (confirmed
    by the independent AST walk above to carry two real rendered-value
    claims) -- a false "clean." Per R9's own repair (`Claude/Vitruvius/
    t2265-fold46-delta-manifest.md` Sec4, item 2), the checker gains a
    second scan surface: `_UNBUILT_CLAIM_PATTERN` also runs against the
    RENDERED value of every `ast.Constant` string node, unioned with the
    existing raw-text result. Against a VENDORED, frozen population (not
    the live file, which R5/R9's own text fixes will make legitimately
    clean), this assertion stays the permanent regression guard the ticket
    asks for: it fails today because the surface does not exist, passes
    once it is added, and fails again if that surface is ever removed."""
    text = _read_vendored_fixture(_R9_POPULATION_FIXTURE)
    result = cptdc.find_stale_unbuilt_claim(text)
    assert result is not None, (
        "expected the repaired checker's rendered-value scan surface to "
        "flag this vendored population (two claims split across adjacent "
        "string literals, invisible to a raw-text-only search); got None -- "
        "the checker's rendered-value scan surface is absent or was removed"
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
