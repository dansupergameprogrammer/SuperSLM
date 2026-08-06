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

import check_no_criterion2_closure_imports as ccli

_PKG = "superslm_spike"


def _dotted_call(module: str, tail: str) -> str:
    return f"x = {_PKG}.{module}.{tail}\n"


def _from_import(names: str) -> str:
    return f"from {_PKG} import {names}\n"


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


# --- Present truth: the real tree, now that the closure is vendored. ---


def test_main_end_to_end_against_the_real_tree():
    """The wiring cell: the real production tree and the real test tree, both
    scanned under the module's own default globs/allowlist, pass end to end --
    nothing in include/, src/, or tools/convert_model.py imports the wide
    closure, and the only files under tests/ that do are the vendored
    closure's own internal imports (exempted by directory) and, once T-1522
    lands, the allowlisted producer/comparator (neither exists yet in this
    build's own scope, so the allowlist is unexercised against real files
    here, proven instead by the constructed cells above)."""
    code = ccli.main()
    assert code == 0
