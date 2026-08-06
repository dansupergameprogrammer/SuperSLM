"""Red suite for check_no_criterion2_closure_imports.py (T-1523, SuperSLM_S3a_
WalkingSkeleton_Plan.md Sec11 S3.1c item 4).

Mirrors tests/ci/test_check_no_forward_leaf_calls.py's own convention:
constructed scratch files, never the real production tree, per
StandardsDocument Sec4's population-validation requirement -- a check must be
shown able to FAIL on a fault it exists to catch, not only shown to pass on
unchanged input. The four cells the plan's own item 4 red-cell list names are
realized here against constructed scratch directories; the real tree's own
wiring (the default globs/allowlist agreeing on the real files that exist
today) is realized separately by test_main_end_to_end_against_the_real_tree.

FIXTURE CONTENT IS BUILT FROM PARTS, NEVER AS A LITERAL CONTIGUOUS SUBSTRING,
throughout this file. This module itself lives under tests/ci/, which the
checker under test scans (clause ii) and which is not on its allowlist -- a
literal `"from superslm_spike import pipeline"` string constant in THIS
file's own source would trip the checker's own text scan the moment it
scanned itself (the same accepted-false-positive shape the module docstring
under test names for a comment or string literal), which would make
test_main_end_to_end_against_the_real_tree fail for a reason that has nothing
to do with the mechanism it verifies. The helpers below produce byte-identical
fixture CONTENT at runtime while keeping this file's own static source clear
of the trigger substrings.
"""

import os
import tempfile

import pytest

import check_no_criterion2_closure_imports as ccli

_PKG = "superslm_spike"


def _dotted_call(module: str, tail: str) -> str:
    return f"x = {_PKG}.{module}.{tail}\n"


def _from_import(names: str) -> str:
    return f"from {_PKG} import {names}\n"


def _from_import_trailing_comment(names: str, comment: str = "noqa: E402") -> str:
    return f"from {_PKG} import {names}  # {comment}\n"


def _from_import_trailing_semicolon(names: str, tail: str = "x = 1") -> str:
    return f"from {_PKG} import {names}; {tail}\n"


def _from_import_parenthesized(names: list) -> str:
    inner = "".join(f"    {n},\n" for n in names)
    return f"from {_PKG} import (\n{inner})\n"


def _from_import_backslash(name: str) -> str:
    return f"from {_PKG} import \\\n    {name}\n"


def _from_import_parenthesized_open_paren_comment(names: list, comment: str) -> str:
    inner = "".join(f"    {n},\n" for n in names)
    return f"from {_PKG} import (  # {comment}\n{inner})\n"


def _from_import_parenthesized_per_name_comment_before(names: list, comment_index: int, comment: str) -> str:
    lines = []
    for i, n in enumerate(names):
        if i == comment_index:
            lines.append(f"    {n},  # {comment}\n")
        else:
            lines.append(f"    {n},\n")
    inner = "".join(lines)
    return f"from {_PKG} import (\n{inner})\n"


def _string_literal_mentioning_the_import_form_with_open_paren() -> str:
    return f'DOC = "from {_PKG} import ("\n'


def _comment_mentioning_the_import_form_with_open_paren(module: str) -> str:
    return f"# see: from {_PKG} import ({module}\n"


def _write(tmpdir: str, rel_path: str, content: str) -> str:
    abs_path = os.path.join(tmpdir, rel_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)
    return abs_path


# --- find_banned_import_uses: rule coverage, both forms, all four modules. ---


def test_dotted_form_is_detected_for_every_banned_module():
    for module in ccli.BANNED_MODULES:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "site.py", _dotted_call(module, "forward_dynamic_vec(row)"))
            hits = ccli.find_banned_import_uses(path)
            assert hits == [(1, module)], f"expected exactly one hit on {module!r}, got {hits}"


def test_from_import_form_is_detected_for_every_banned_module():
    for module in ccli.BANNED_MODULES:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "site.py", _from_import(module))
            hits = ccli.find_banned_import_uses(path)
            assert hits == [(1, module)], f"expected exactly one hit on {module!r}, got {hits}"


def test_from_import_form_with_multiple_names_finds_only_the_banned_ones():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "site.py", _from_import("intmath, pipeline, rope"))
        hits = ccli.find_banned_import_uses(path)
        assert hits == [(1, "pipeline")]


def test_from_import_with_alias_is_still_recognized_by_its_real_name():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "site.py", _from_import("pipeline as p"))
        hits = ccli.find_banned_import_uses(path)
        assert hits == [(1, "pipeline")]


def test_the_narrow_excerpts_own_name_is_not_matched_by_the_pipeline_token():
    """pipeline_prob_width_ceiling is a DIFFERENT module, the deliberately
    narrow excerpt's own name (D-SLM367) -- neither import form may treat it
    as a hit on the banned 'pipeline' token."""
    with tempfile.TemporaryDirectory() as tmp:
        excerpt = "pipeline_prob_width_ceiling"
        content = _from_import(excerpt) + _dotted_call(excerpt, "NUMERATOR_CEILING")
        path = _write(tmp, "site.py", content)
        hits = ccli.find_banned_import_uses(path)
        assert hits == [], f"expected no hits on the narrow excerpt's own name, got {hits}"


def test_the_legitimate_intmath_rope_import_is_not_flagged():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "site.py", _from_import("intmath, rope"))
        assert ccli.find_banned_import_uses(path) == []


# --- The normalization gap Poirot found (D-SLM1058): the from-import matcher
# split a comma-separated names list on its own physical line and never
# removed a trailing comment or statement separator, so a spelling that
# splits the names across lines, or trails the last name with a comment or a
# ';', scanned clean. All four are the real spellings measured against
# tools/convert_model.py:48 and its formatter-produced neighbours. ---


def test_from_import_with_a_trailing_comment_is_detected():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "site.py", _from_import_trailing_comment("pipeline"))
        hits = ccli.find_banned_import_uses(path)
        assert hits == [(1, "pipeline")]


def test_from_import_with_a_trailing_semicolon_statement_is_detected():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "site.py", _from_import_trailing_semicolon("pipeline"))
        hits = ccli.find_banned_import_uses(path)
        assert hits == [(1, "pipeline")]


def test_the_parenthesized_multiline_form_is_detected():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "site.py", _from_import_parenthesized(["pipeline", "silu_lut"]))
        hits = ccli.find_banned_import_uses(path)
        assert hits == [(1, "pipeline"), (1, "silu_lut")]


def test_the_backslash_continuation_form_is_detected():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "site.py", _from_import_backslash("dynamic_engine"))
        hits = ccli.find_banned_import_uses(path)
        assert hits == [(1, "dynamic_engine")]


# --- The under-inclusion regressions Poirot found on the D-SLM1058 repair
# itself (findings B and C, 2414bd4-t1744-review-fold-confirmation.md): a
# comment on the opening-paren line of a wrapped import dropped every name,
# a per-name comment dropped every name after it, and the logical-line join
# swallowed a real import beneath any line merely mentioning the import form
# with an unbalanced '('. All four are formatter-stable (round-trip
# byte-identical through black 25.1.0) or, for the join regression, latent
# in exactly the file class that documents the ban. ---


def test_a_comment_on_the_opening_paren_line_does_not_drop_every_name():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "site.py",
            _from_import_parenthesized_open_paren_comment(["pipeline", "silu_lut"], "noqa: E402"),
        )
        hits = ccli.find_banned_import_uses(path)
        assert {m for _, m in hits} == {"pipeline", "silu_lut"}, f"expected both names caught, got {hits}"


def test_a_type_ignore_comment_on_the_opening_paren_line_does_not_drop_every_name():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "site.py",
            _from_import_parenthesized_open_paren_comment(["dynamic_engine"], "type: ignore"),
        )
        hits = ccli.find_banned_import_uses(path)
        assert {m for _, m in hits} == {"dynamic_engine"}, f"expected the name caught, got {hits}"


def test_a_per_name_comment_does_not_drop_every_name_after_it():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "site.py",
            _from_import_parenthesized_per_name_comment_before(
                ["intmath", "pipeline", "silu_lut"], comment_index=0, comment="noqa"
            ),
        )
        hits = ccli.find_banned_import_uses(path)
        assert {m for _, m in hits} == {"pipeline", "silu_lut"}, (
            f"expected both banned names after the comment caught, got {hits}"
        )


def test_a_string_literal_mentioning_the_import_form_does_not_swallow_a_real_import_below():
    with tempfile.TemporaryDirectory() as tmp:
        content = _string_literal_mentioning_the_import_form_with_open_paren() + _from_import("pipeline")
        path = _write(tmp, "site.py", content)
        hits = ccli.find_banned_import_uses(path)
        assert hits == [(2, "pipeline")], f"expected the real import on line 2 caught, got {hits}"


def test_a_comment_mentioning_the_import_form_does_not_swallow_a_real_import_two_lines_below():
    with tempfile.TemporaryDirectory() as tmp:
        content = (
            _comment_mentioning_the_import_form_with_open_paren("dynamic_engine")
            + "import os\n"
            + _from_import("dynamic_engine")
        )
        path = _write(tmp, "site.py", content)
        hits = ccli.find_banned_import_uses(path)
        assert hits == [(3, "dynamic_engine")], f"expected the real import on line 3 caught, got {hits}"


def test_a_file_that_does_not_parse_as_python_falls_back_to_the_text_scan():
    """A file ast.parse cannot read (a broken fixture, a template snippet
    saved with a .py suffix) must not silently report zero from-import hits
    -- the text-based logical-line scan is the fallback for exactly this
    case, per the module docstring."""
    with tempfile.TemporaryDirectory() as tmp:
        content = "def broken(\n" + _from_import("pipeline")
        path = _write(tmp, "site.py", content)
        with pytest.raises(SyntaxError):
            import ast as _ast

            _ast.parse(content)
        hits = ccli.find_banned_import_uses(path)
        assert (2, "pipeline") in hits, f"expected the fallback text scan to still catch line 2, got {hits}"


def test_tools_convert_model_with_a_trailing_noqa_comment_fails_clause_i():
    """Reproduces the exact real-file shape at tools/convert_model.py:48
    (D-SLM1058) -- a trailing `# noqa: E402` comment on the from-import line
    that the original matcher folded into the last imported name, so the
    line scanned clean over a live instance of the condition the check
    exists to ban."""
    with tempfile.TemporaryDirectory() as tmp:
        _write(
            tmp,
            "tools/convert_model.py",
            _from_import_trailing_comment("artifact_cache, pipeline", "noqa: E402"),
        )
        code = ccli.main(
            production_globs=("tools/convert_model.py",),
            test_globs=(),
            repo_root=tmp,
        )
        assert code == 1


# --- The plan's own four named red cells (item 4). ---


def test_a_scratch_tu_under_src_naming_the_dotted_form_fails_clause_i():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "src/forward/some_site.cpp.py", _dotted_call("dynamic_engine", "forward_dynamic_vec(row)"))
        code = ccli.main(
            production_globs=("src/forward/**/*.py",),
            test_globs=(),
            repo_root=tmp,
        )
        assert code == 1


def test_a_scratch_tu_under_include_naming_the_dotted_form_fails_clause_i():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "include/superslm/leak.py", _dotted_call("silu_lut", "LUT_TABLE"))
        code = ccli.main(
            production_globs=("include/**/*.py",),
            test_globs=(),
            repo_root=tmp,
        )
        assert code == 1


def test_tools_convert_model_adding_from_import_form_fails_clause_i():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "tools/convert_model.py", _from_import("pipeline"))
        code = ccli.main(
            production_globs=("tools/convert_model.py",),
            test_globs=(),
            repo_root=tmp,
        )
        assert code == 1


def test_a_scratch_fixture_generator_swapping_the_narrow_excerpt_for_the_wide_module_fails_clause_ii():
    """Mirrors the real gen_c32_softmax_row_width_gate_fixtures.py's own
    convention: today it imports the narrow pipeline_prob_width_ceiling
    excerpt; a hypothetical edit reaching for the wide pipeline module
    instead must fail."""
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "tests/gen_c32_softmax_row_width_gate_fixtures.py", _from_import("pipeline"))
        code = ccli.main(
            production_globs=(),
            test_globs=("tests/**/*.py",),
            repo_root=tmp,
        )
        assert code == 1


def test_a_brand_new_scratch_file_under_tests_importing_the_wide_module_fails_clause_ii_without_being_named():
    """Input coverage: the allowlist is a fixed, named set (the producer and
    comparator); ANY other file under tests/ -- including one that did not
    exist when this check was authored -- must fail if it imports the wide
    closure, with no update to this module required to catch it."""
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "tests/some_brand_new_future_generator.py", _from_import("constrain"))
        code = ccli.main(
            production_globs=(),
            test_globs=("tests/**/*.py",),
            repo_root=tmp,
        )
        assert code == 1


def test_all_four_red_cells_revert_to_pass():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "src/forward/some_site.cpp.py", "auto x = 1;\n")
        _write(tmp, "include/superslm/leak.py", "auto x = 1;\n")
        _write(tmp, "tools/convert_model.py", "import numpy\n")
        _write(
            tmp,
            "tests/gen_c32_softmax_row_width_gate_fixtures.py",
            _from_import("pipeline_prob_width_ceiling"),
        )
        _write(tmp, "tests/some_brand_new_future_generator.py", _from_import("intmath"))
        code = ccli.main(
            production_globs=("src/forward/**/*.py", "include/**/*.py", "tools/convert_model.py"),
            test_globs=("tests/**/*.py",),
            repo_root=tmp,
        )
        assert code == 0


# --- The allowlist itself: the two legitimate wide-package callers pass. ---


def test_the_allowlisted_producer_and_comparator_pass_even_though_they_import_the_wide_closure():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "tests/reference/run_criterion2_trace.py", _from_import("dynamic_engine"))
        _write(tmp, "tests/reference/compare_criterion2_traces.py", _from_import("pipeline"))
        code = ccli.main(
            production_globs=(),
            test_globs=("tests/**/*.py",),
            repo_root=tmp,
        )
        assert code == 0


def test_the_default_allowlist_names_exactly_four_files():
    """A named, sized set (matching this suite's sibling convention,
    e.g. check_no_forward_leaf_calls.py's _EXPECTED_REAL_FORWARD_FILES): item
    3's own producer and comparator, plus the two build-time tooling files
    T-1522's producer half adds under item 4's own stated principle. A fifth
    entry landing here without a matching test is caught by this equality
    assertion, not silently accepted."""
    assert set(ccli._DEFAULT_TEST_ALLOWLIST) == {
        "tests/reference/run_criterion2_trace.py",
        "tests/reference/compare_criterion2_traces.py",
        "tests/reference/precompute_criterion2_prompt_pack.py",
        "tests/reference/test_run_criterion2_trace.py",
    }


def test_the_precompute_script_and_its_test_file_pass_even_though_they_import_the_wide_closure():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "tests/reference/precompute_criterion2_prompt_pack.py", _from_import("pipeline"))
        _write(tmp, "tests/reference/test_run_criterion2_trace.py", _from_import("pipeline"))
        code = ccli.main(
            production_globs=(),
            test_globs=("tests/**/*.py",),
            repo_root=tmp,
        )
        assert code == 0


def test_a_third_file_alongside_the_allowlisted_two_still_fails():
    """Allowlist control: the allowlist exempts by exact path, not by
    directory -- a sibling file in the same tests/reference/ directory that
    is NOT one of the two named files still fails."""
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "tests/reference/run_criterion2_trace.py", _from_import("dynamic_engine"))
        _write(tmp, "tests/reference/some_other_file.py", _from_import("pipeline"))
        code = ccli.main(
            production_globs=(),
            test_globs=("tests/**/*.py",),
            repo_root=tmp,
        )
        assert code == 1


# --- The vendored closure's own directory is exempt from clause (ii): its
# internal imports of its sibling modules are the closure's own structure. ---


def test_the_vendored_closures_own_internal_sibling_imports_are_not_flagged():
    with tempfile.TemporaryDirectory() as tmp:
        _write(
            tmp,
            "tests/reference/superslm_spike/dynamic_engine.py",
            _from_import("intmath, pipeline, silu_lut"),
        )
        code = ccli.main(
            production_globs=(),
            test_globs=("tests/**/*.py",),
            repo_root=tmp,
        )
        assert code == 0


def test_a_file_just_outside_the_excluded_directory_with_a_similar_name_is_still_scanned():
    """The exclusion is a directory prefix, not a substring match --
    tests/reference/superslm_spike_extra/ (a different, similarly-named
    directory) must still be scanned and fail."""
    with tempfile.TemporaryDirectory() as tmp:
        _write(
            tmp,
            "tests/reference/superslm_spike_extra/not_really_vendored.py",
            _from_import("pipeline"),
        )
        code = ccli.main(
            production_globs=(),
            test_globs=("tests/**/*.py",),
            repo_root=tmp,
        )
        assert code == 1


# --- This file's own hygiene: FIXTURE CONTENT IS BUILT FROM PARTS applies to
# every string this file's own source contains, including an xfail reason,
# not only the helpers above (D-SLM1086; Poirot finding A,
# 2414bd4-t1744-review-fold-confirmation.md). A prior fix transcribed the
# literal banned import into the xfail reason below, which the module
# docstring's own convention forbids in capitals -- this cell is the red cell
# that convention was always owed. ---


def test_this_file_itself_scans_clean():
    """This file lives under tests/ci/ (clause ii) and is not on the
    allowlist. A literal contiguous banned substring anywhere in this file's
    own static source -- in a fixture-building helper, a docstring, or an
    xfail reason -- trips the checker's text scan the moment it scans
    itself, which is exactly the failure the module docstring names and this
    file's own top-of-file convention exists to prevent."""
    assert ccli.find_banned_import_uses(__file__) == []


# --- Present truth: the real tree, now that the closure is vendored. ---


@pytest.mark.xfail(
    reason=(
        "D-SLM1059 (OPEN, planner): tools/convert_model.py:48 genuinely "
        "imports the vendored closure (see that line and column for the "
        "exact statement -- not transcribed here, D-SLM1086, so this file "
        "does not trip its own text scan); the from-import parser repair "
        "(D-SLM1058) now detects it correctly, which is the intended "
        "effect of that repair. Whether this call site is a clause (i) "
        "violation or an accepted, allowlisted exception is a planner call "
        "under T-1745, not this suite's to decide. strict=True so this test "
        "re-surfaces for review the moment either disposition lands."
    ),
    strict=True,
)
def test_main_end_to_end_against_the_real_tree():
    """The wiring cell: the real production tree and the real test tree, both
    scanned under the module's own default globs/allowlist, pass end to end --
    nothing in include/, src/, or tools/convert_model.py imports the wide
    closure, and the only files under tests/ that do are the vendored
    closure's own internal imports (exempted by directory) and the four
    allowlisted files (the producer, the comparator, the precompute script,
    and the producer's own test file -- all four exist in this build's own
    scope and are exercised here against the real files, not only by the
    constructed cells above)."""
    code = ccli.main()
    assert code == 0
