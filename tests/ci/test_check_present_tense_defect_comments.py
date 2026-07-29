"""Curie's red suite for check_present_tense_defect_comments.py
(SuperSLM_S3a_WalkingSkeletonPlan.md Sec11 intro, Sec14.12; D-SLM411, D-SLM414,
D-SLM418; board T-1337).

Mirrors tests/ci/test_check_no_forward_leaf_calls.py's own convention: mechanism
cells (rule coverage, marker recognition, block boundaries, input coverage) drive
the module against constructed scratch files, never the real production tree,
per StandardsDocument Sec4's population-validation requirement -- a check shown
only to pass on unchanged input is not shown to catch anything. The historical-
population cells at the bottom of this file are the OTHER half of Sec4: they
replay the seven real sites T-1331/T-1334 found, recovered from git history,
rather than a fault the module's own author injected.
"""

import os
import tempfile

import check_present_tense_defect_comments as cptdc

_FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "present_tense_defect_historical_fixtures")


def _write(tmpdir: str, rel_path: str, content: str) -> str:
    abs_path = os.path.join(tmpdir, rel_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)
    return abs_path


def _read_lines(rel_fixture_name: str) -> list[str]:
    path = os.path.join(_FIXTURES_DIR, rel_fixture_name)
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


# --- Rule coverage: each severity-label family is individually detected. ---


def test_each_severity_label_family_with_no_marker_is_flagged():
    for label in ("Critical 1", "Significant 5", "Weak 2", "Structural 3"):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(
                tmp,
                "tests/site.cpp",
                f"// {label} (Poirot deadbeef-some-review.md): the guard does not exist.\n",
            )
            hits = cptdc.find_uncited_defect_citations(path)
            assert len(hits) == 1, f"expected exactly one hit for {label!r}, got {hits}"
            assert hits[0][2] == [label], f"expected label {label!r} reported, got {hits[0][2]}"


def test_a_label_without_a_digit_is_not_matched():
    """"Significant B" is this codebase's OWN, separate convention for a coverage-
    dimension tag (plan Sec13's own "Significant B" callouts), not a severity
    finding -- it must not be mistaken for one."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "tests/site.cpp", "// Significant B: the exact status, not merely not-Ok.\n")
        assert cptdc.find_uncited_defect_citations(path) == []


# --- Resolution markers: either suppresses a flag. ---


def test_the_word_closed_case_insensitive_suppresses_the_flag():
    for spelling in ("closed", "CLOSED", "Closed"):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "tests/site.cpp", f"// Critical 1 ({spelling}; Poirot deadbeef.md): now guards it.\n")
            assert cptdc.find_uncited_defect_citations(path) == [], f"{spelling!r} must suppress the flag"


def test_a_cited_ticket_id_suppresses_the_flag():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "tests/site.cpp", "// Critical 2 (T-1322): the guard now exists.\n")
        assert cptdc.find_uncited_defect_citations(path) == []


def test_a_marker_in_a_different_block_does_not_suppress_an_unrelated_citation():
    """Negative control for block boundaries: a blank line (code or whitespace)
    between two comment runs must stop a resolution marker in the SECOND block
    from silently satisfying a citation in the FIRST -- otherwise any 'closed'
    anywhere later in the file would clear every uncited citation before it."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "tests/site.cpp",
            "// Critical 1 (Poirot deadbeef.md): the guard does not exist.\n"
            "\n"
            "// Critical 2 (closed; Poirot deadbeef.md): this one is fixed.\n",
        )
        hits = cptdc.find_uncited_defect_citations(path)
        assert len(hits) == 1
        assert hits[0][2] == ["Critical 1"]


def test_a_citation_split_across_a_line_break_is_still_matched_against_the_merged_block():
    """This codebase's own fixed text splits a label across a line break
    ("...(Significant " / "5, closed by...)") -- the merged-block join must still
    find both the label and the marker."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "tests/site.cpp",
            "// ParseConfigImpl now enforces the check (Significant\n// 5, closed by the review).\n",
        )
        assert cptdc.find_uncited_defect_citations(path) == []


def test_a_check_msg_style_multiline_string_literal_is_scanned_as_one_block():
    """The one historical site (D-SLM411) that sat inside a live CHECK_MSG
    failure-message string, rather than a `//` comment -- this codebase writes a
    multi-line string as one quoted literal per physical line."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "tests/site.cpp",
            'CHECK_MSG(status == kOk,\n'
            '          "the parity check == %s, want Ok -- ParseConfigImpl performs no "\n'
            '          "parity check today (Significant 5)",\n'
            "          StatusName(status));\n",
        )
        hits = cptdc.find_uncited_defect_citations(path)
        assert len(hits) == 1
        assert hits[0][2] == ["Significant 5"]


def test_a_clean_check_msg_string_with_the_marker_inline_is_not_flagged():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "tests/site.cpp",
            'CHECK_MSG(status == kOk,\n'
            '          "the parity check == %s, want Ok (Significant 5, closed by the review)",\n'
            "          StatusName(status));\n",
        )
        assert cptdc.find_uncited_defect_citations(path) == []


def test_a_file_with_no_severity_citation_at_all_is_never_flagged():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "tests/site.cpp", "// A perfectly ordinary comment about a passing test.\n")
        assert cptdc.find_uncited_defect_citations(path) == []


# --- main()/scan_files() end-to-end and input coverage. ---


def test_main_fails_when_a_scanned_file_has_an_uncited_citation():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "tests/site.cpp", "// Critical 1: the guard does not exist.\n")
        code = cptdc.main(globs=("tests/*.cpp",), repo_root=tmp)
        assert code == 1


def test_main_passes_when_every_citation_carries_a_marker():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "tests/site.cpp", "// Critical 1 (closed): the guard now exists.\n")
        code = cptdc.main(globs=("tests/*.cpp",), repo_root=tmp)
        assert code == 0


def test_input_coverage_a_file_in_a_freshly_added_nested_directory_is_still_scanned():
    """Input coverage: the file set is glob-derived (`tests/**/*.cpp`), not
    enumerated by hand, so a file under a subdirectory that did not exist when
    this check was authored is still reached."""
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "tests/newly/added/subdir/site.cpp", "// Critical 1: the guard does not exist.\n")
        code = cptdc.main(globs=cptdc._DEFAULT_TEST_GLOBS, repo_root=tmp)
        assert code == 1, "a citation under a freshly added tests/ subdirectory must still be caught"


def test_input_coverage_a_python_test_file_is_scanned_too():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "tests/tool_test.py", "# Critical 1: the guard does not exist.\n")
        # Python's own line-comment token is '#', not '//' -- confirm this
        # module's rule (keyed to '//') does NOT falsely match a '#' comment,
        # which is a real, honest input-coverage gap for .py files rather than a
        # bug: no historical site lives in a Python comment, and none of the
        # seven recovered sites are Python, so this gap does not touch the
        # validated population. Documented here so it is a known limitation,
        # not a silent one.
        code = cptdc.main(globs=cptdc._DEFAULT_TEST_GLOBS, repo_root=tmp)
        assert code == 0, (
            "a '#'-style Python comment citing a severity label is NOT matched by this "
            "module's '//'-keyed comment-block rule -- known gap, see comment above"
        )


def test_missing_scanned_file_is_reported_distinctly_from_a_dirty_one():
    failures = cptdc.scan_files(["/does/not/exist/site.cpp"], repo_root="/does/not/exist")
    assert len(failures) == 1
    assert "not found" in failures[0]


# --- Historical population validation (StandardsDocument Sec4). ---
#
# tests/ci/present_tense_defect_historical_fixtures/site{1..7}_{pre,post}.txt hold
# the exact recovered text of all seven sites T-1331/T-1334 found (D-SLM411,
# D-SLM414): site1-6 from `D:\SuperSLM`@fa5113d^ (before the six-site rewrite) and
# @fa5113d (after); site7 from @ee76dbd^ and @ee76dbd (the seventh, found by the
# scrub for the sixth). Fixing nothing here -- these are read-only replays of
# text that was already fixed at those two real commits, satisfying Sec4's "found
# independently of the check, verified, and reproduced before anything is fixed."

_SITE_NAMES = ("site1", "site2", "site3", "site4", "site5", "site6", "site7")


def test_fires_on_every_historical_pre_fix_site():
    misses = []
    for name in _SITE_NAMES:
        lines = _read_lines(f"{name}_pre.txt")
        hits = cptdc.scan_text(lines)
        if not hits:
            misses.append(name)
    assert not misses, (
        f"rule coverage gap: the following historical PRE-fix sites were NOT flagged: {misses} "
        f"-- this check does not reproduce the full T-1331/T-1334 population"
    )


def test_does_not_fire_on_any_historical_post_fix_site():
    false_positives = []
    for name in _SITE_NAMES:
        lines = _read_lines(f"{name}_post.txt")
        hits = cptdc.scan_text(lines)
        if hits:
            false_positives.append((name, hits))
    assert not false_positives, (
        f"the check fires on the CURRENT, already-fixed text of these sites: {false_positives} "
        f"-- it would never stop reporting the correct, current comment"
    )


def test_historical_population_is_the_full_named_set_not_a_subset():
    """Guards the fixture set itself: seven pre and seven post files, one pair
    per site named in D-SLM411/D-SLM414, so a fixture accidentally left out does
    not silently shrink the population this suite claims to validate against."""
    for name in _SITE_NAMES:
        for suffix in ("pre", "post"):
            path = os.path.join(_FIXTURES_DIR, f"{name}_{suffix}.txt")
            assert os.path.isfile(path), f"missing historical fixture: {path}"
