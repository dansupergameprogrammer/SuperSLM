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
            # One single physical Python line, not one quoted literal per
            # target line: a quoted literal that opens a physical line with
            # nothing else on it is itself indistinguishable from a genuine
            # generated `//`-comment line (see check_present_tense_defect_
            # comments.py's own _PY_QUOTED_CPP_COMMENT_PATTERN) -- so split
            # across separate physical lines, THIS fixture's own first
            # citation reads, to this module scanning its own source under
            # its default globs, as an uncited "Critical 1" with the "closed"
            # marker on an unrelated later line (T-1485's real-tree run
            # caught this). Folding both citations onto one physical source
            # line keeps them one block for this module's self-scan while the
            # OUTPUT file `_write` produces is byte-for-byte identical -- two
            # `//` lines separated by a blank line -- so the test still
            # exercises exactly the two-block shape it asserts.
            "// Critical 1 (Poirot deadbeef.md): the guard does not exist.\n\n"
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


# --- Python triple-quoted docstrings: joined as one block, not fragmented at
# the opening line (T-1485). ---


def test_a_multiline_python_docstring_is_joined_into_one_block():
    """T-1485's real-tree run: a Python docstring's OPENING line happens to
    start with `"` and so passes the plain-STRING check, but every line after
    it does not start with a quote character at all -- without explicit
    triple-quote tracking, only the opening line is scanned, silently hiding
    a resolution marker that in fact appears later in the same docstring."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "tests/site_test.py",
            'def test_something():\n'
            '    """Pins a mutation (Poirot deadbeef review, Critical 1,\n'
            '    Execution evidence): the property T-9001 restored, and the\n'
            '    guard now holds for every case in the pin."""\n'
            "    assert True\n",
        )
        assert cptdc.find_uncited_defect_citations(path) == [], (
            "the resolution marker T-9001 appears two physical lines below the citation, "
            "inside the same docstring -- the merged block must still find it"
        )


def test_a_multiline_python_docstring_with_no_marker_anywhere_is_still_flagged():
    # The target docstring's `"""` delimiters are built from `_TQ` rather than
    # written as a literal three-quote run: a literal run here would itself be
    # an (uncited, marker-free) instance of this exact block shape in THIS
    # module's own source, which this module's own default globs scan (a
    # self-application collision T-1485's real-tree run hit once already; see
    # test_a_marker_in_a_different_block_does_not_suppress_an_unrelated_
    # citation's comment above).
    _tq = '"' * 3
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "tests/site_test.py",
            'def test_something():\n'
            f"    {_tq}The mutation this guard exists to catch (Critical 1's own\n"
            '    text): a regression must fail loudly rather than pass\n'
            f"    silently.{_tq}\n"
            "    assert True\n",
        )
        hits = cptdc.find_uncited_defect_citations(path)
        assert len(hits) == 1
        assert hits[0][2] == ["Critical 1"]


def test_a_single_line_python_docstring_is_unaffected():
    """A one-line docstring (opens and closes its triple-quote on the same
    physical line) is not swept into an erroneous multi-line continuation --
    the very next, unrelated line must not be pulled into its block."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "tests/site_test.py",
            'def test_something():\n'
            '    """Critical 1 (closed): a one-line docstring."""\n'
            "    assert True\n",
        )
        assert cptdc.find_uncited_defect_citations(path) == []


# --- Tokenize-derived Python string spans replace triple-quote parity
# tracking (T-1538): the parity tracker desynchronised on any ordinary line
# outside a string carrying an odd triple-quote count, inverting docstring/
# code polarity for the rest of the file, and had no closing marker at all
# in `.cpp`/`.h`, where it ran to end of file. ---


def test_a_preceding_comment_naming_the_triple_quote_delimiter_does_not_hide_a_later_citation():
    """T-1538 Significant 1 (Poirot 884ee74-t1485-present-tense-gate-blocking-
    review-2026-07-31.md): the per-line triple-quote-parity tracker read this
    file's own leading `#` comment -- which names the delimiter and so carries
    an odd triple-quote count -- as OPENING a phantom string, inverting every
    later docstring boundary's polarity for the rest of the file and hiding
    the uncited citation below entirely. Tokenize-derived spans are immune:
    a `#` comment is never part of any STRING token."""
    # Interpolated, not written literally: a bare "Critical 1" on a line
    # starting with a quote character would itself be an uncited instance of
    # this exact defect class in THIS module's own source, which this
    # module's own default globs scan (the Minor 4 self-scan collision
    # T-1485's real-tree run hit once already; see
    # test_a_marker_in_a_different_block_does_not_suppress_an_unrelated_
    # citation's comment above).
    _label = "Critical" + " 1"
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "tests/site_test.py",
            '# The fixture generators write """-delimited blocks.\n'
            "\n"
            "def test_something():\n"
            '    """Pins the mutation this guard exists to catch.\n'
            "\n"
            f"    {_label} (Poirot deadbeef.md): the guard does not exist and the\n"
            '    kernel reads unmapped heap memory on every call."""\n'
            "    assert True\n",
        )
        hits = cptdc.find_uncited_defect_citations(path)
        assert len(hits) == 1, (
            f"the preceding comment must not suppress the uncited {_label} citation, got {hits}"
        )
        assert hits[0][2] == [_label]


def test_a_stray_triple_quote_length_run_in_a_cpp_comment_does_not_collapse_the_file():
    """T-1538 Significant 2 (same casebook): no `.cpp`/`.h` line ever closes a
    phantom Python string, so the parity tracker's collapse ran to end of
    file -- an uncited citation followed, anywhere later in the file, by an
    unrelated block that happens to contain the word "closed" was silenced,
    because the whole remainder of the file became one block. `.cpp`/`.h`
    inputs now get no triple-quote handling at all, so the stray `'''` below
    (an apostrophe-heavy aside, not a string delimiter in C++) must not merge
    anything across the blank/code lines that separate real comment blocks."""
    # Interpolated for the same self-scan-collision reason as the test above.
    _label = "Critical" + " 1"
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "tests/site.cpp",
            "// possessives run together: foos''' bar\n"
            "int x = 0;\n"
            "int y = 0;\n"
            "int z = 0;\n"
            f"// {_label}: the guard does not exist.\n"
            "int w = 0;\n"
            "int v = 0;\n"
            "// unrelated, closed elsewhere\n",
        )
        hits = cptdc.find_uncited_defect_citations(path)
        assert hits == [(5, 5, [_label])], (
            f"the later, unrelated 'closed' comment must not silence the uncited {_label} "
            f"citation by way of a file-wide collapse, got {hits}"
        )


def test_a_multiline_fstring_citation_is_caught_not_silently_admitted():
    """T-1545 (Poirot 28aa351-t1544-present-tense-confirmation-2026-07-31.md
    Significant 1): on Python 3.12+ a multi-line f-string is not a
    tokenize.STRING token at all -- it is FSTRING_START/MIDDLE/END -- so the
    prior filter (STRING only) found no span for it, every physical line fell
    through to `_line_kind`, the opening line starts with `f` rather than a
    quote character so it (and every following line) classified OTHER, and no
    block formed: an uncited citation inside a multi-line f-string was
    silently admitted. Matched against a plain triple-quoted control that
    differs by exactly the leading `f`, which was already caught before this
    fix and must remain caught after it."""
    _label = "Critical" + " 1"
    with tempfile.TemporaryDirectory() as tmp:
        fstring_path = _write(
            tmp,
            "tests/fstring_site_test.py",
            "def test_something():\n"
            f'    x = f"""line1\n'
            f"    {_label} unresolved\n"
            '    line3"""\n'
            "    assert x\n",
        )
        hits = cptdc.find_uncited_defect_citations(fstring_path)
        assert len(hits) == 1, (
            f"a citation inside a multi-line f-string must be caught, got {hits}"
        )
        assert hits[0][2] == [_label]

    with tempfile.TemporaryDirectory() as tmp:
        control_path = _write(
            tmp,
            "tests/plain_site_test.py",
            "def test_something():\n"
            '    x = """line1\n'
            f"    {_label} unresolved\n"
            '    line3"""\n'
            "    assert x\n",
        )
        hits = cptdc.find_uncited_defect_citations(control_path)
        assert len(hits) == 1, (
            f"the plain triple-quoted control (no leading f) must still be caught, got {hits}"
        )
        assert hits[0][2] == [_label]


def test_a_resolution_marker_in_one_paragraph_of_a_multiline_string_does_not_clear_an_unrelated_paragraph():
    """T-1545 (Poirot 28aa351-t1544-present-tense-confirmation-2026-07-31.md
    Significant 2, the prior review's Significant 3 reopened): every physical
    line a `.py` file's tokenizer places inside one multi-line STRING span
    was forced to kind STRING regardless of blank lines within it, so the
    WHOLE span merged into one block -- a resolution marker anywhere inside
    it silently satisfied a citation anywhere else inside it, however many
    blank lines separated them, which is exactly what `_line_kind`'s own
    docstring says must not happen ("a resolution marker written for one
    comment can never silently satisfy a citation in an unrelated, later
    block"). A blank line inside the string span now breaks the block the
    same way a blank line already breaks a `//` comment run, so a genuine
    uncited citation thirty lines below an unrelated resolved paragraph, in
    the same docstring, is still flagged."""
    _label = "Critical" + " 1"
    filler = "\n".join(f"    filler line {n} of the docstring." for n in range(30))
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "tests/site_test.py",
            'def test_something():\n'
            '    """Module docstring.\n'
            "\n"
            "    Significant 5 (closed; fixed 2026-07-01).\n"
            f"{filler}\n"
            "\n"
            f"    {_label}: the guard does not exist and the kernel reads\n"
            '    unmapped heap memory on every call.\n'
            '    """\n'
            "    assert True\n",
        )
        hits = cptdc.find_uncited_defect_citations(path)
        assert hits, (
            "an unrelated 'closed' resolution thirty lines above, in the same "
            f"docstring, must not silently clear the genuine uncited {_label} "
            f"citation below the blank-line paragraph break, got {hits}"
        )
        labels = {label for _, _, block_labels in hits for label in block_labels}
        assert _label in labels, f"expected {_label} among the flagged labels, got {labels}"
        significant5_hit = any("Significant 5" in block_labels for _, _, block_labels in hits)
        assert not significant5_hit, (
            f"'Significant 5 (closed; ...)' carries its own resolution marker in its own "
            f"paragraph and must not be flagged, got {hits}"
        )


def test_a_py_file_tokenize_cannot_parse_is_reported_as_a_failure_not_treated_as_clean():
    """T-1538: a `.py` file this check cannot tokenize must not be silently
    scanned as clean (StandardsDocument Sec4) -- scan_files reports it as its
    own failure line, the same posture as a missing file."""
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "tests/broken.py", 'x = """unterminated triple-quoted string\n')
        failures = cptdc.scan_files(["tests/broken.py"], repo_root=tmp)
        assert len(failures) == 1
        assert "tokeniz" in failures[0].lower(), f"expected a tokenize-failure message, got {failures}"


# --- Python-quoted `//` comment lines: a fixture generator that builds
# multi-line C++ output as one Python string literal per physical line
# (T-1485). ---


def test_a_run_of_python_quoted_cpp_comment_lines_forms_one_block():
    """tests/gen_*.py's own convention: a Python list of string literals, one
    per generated-file physical line, where several consecutive elements are
    themselves quoted `//` comment lines destined for the generated .h/.cpp
    file. These must merge into one block the same way the equivalent real
    `//` lines would if read directly from the generated output."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "tests/gen_site_fixtures.py",
            "lines = [\n"
            '    "// Witness 1 (Poirot deadbeef review, Critical 1): the same",\n'
            '    "// off-ratio mechanism as witness 0.",\n'
            "]\n",
        )
        hits = cptdc.find_uncited_defect_citations(path)
        assert len(hits) == 1
        assert hits[0][2] == ["Critical 1"]


def test_a_python_quoted_cpp_comment_run_does_not_swallow_neighboring_code_lines():
    """T-1485's real-tree run: gen_c32_softmax_row_width_gate_fixtures.py's
    five-line `//` comment was merged, under the plain bare-quote STRING rule,
    into an eighteen-line block that also swallowed a struct definition and
    blank-line placeholders -- because every one of those lines is ALSO,
    coincidentally, its own complete Python string literal starting with `"`.
    A quoted `//` line must merge only with adjacent quoted `//` lines, never
    with a neighboring quoted line that is not itself a `//` line."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "tests/gen_site_fixtures.py",
            "lines = [\n"
            '    "};",\n'
            '    "",\n'
            '    "// Witness 4 (Poirot deadbeef review, Critical 1): the same",\n'
            '    "// off-ratio mechanism as witness 1.",\n'
            '    "",\n'
            '    "struct Witness4 { int64_t x; };",\n'
            '    "// Critical 1 (closed): restated for a later, unrelated witness.",\n'
            "]\n",
        )
        hits = cptdc.find_uncited_defect_citations(path)
        assert len(hits) == 1, f"expected exactly the uncited Witness 4 block, got {hits}"
        assert hits[0][0] == 4 and hits[0][1] == 5, (
            f"expected the flagged block to span exactly lines 4-5 (the two `// Witness 4` "
            f"lines), not the surrounding struct/blank-line/later-citation lines: got {hits[0]}"
        )


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
