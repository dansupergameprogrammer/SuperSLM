"""Unit suite for _source_scan.py (T-1632 Minor 1): the shared comment/
literal masking, brace matching, function-definition finder, hash
normalization, and binary-currency primitives that both
check_test_registration_complete.py and check_test_definitions_registered.py
build on. _source_scan.py's own module docstring claimed this file exists
before it did (Minor 1); this is that file.

Every case below drives the primitives directly against small, constructed
strings -- never the real tests/ tree -- so a primitive's own behavior is
proven independent of whatever the real suite currently contains.
"""
from __future__ import annotations

import os
import tempfile

import pytest

import _source_scan as scan


# --- mask_comments_and_literals ---------------------------------------------


def test_mask_line_comment_is_blanked_but_length_and_newlines_preserved():
    src = "int x = 1; // a comment with a { brace\nint y = 2;\n"
    masked = scan.mask_comments_and_literals(src)
    assert len(masked) == len(src)
    assert "{" not in masked.split("\n")[0]
    assert masked.count("\n") == src.count("\n")
    assert masked.split("\n")[1] == "int y = 2;"


def test_mask_block_comment_spanning_multiple_lines_preserves_newlines_only():
    src = "a; /* line one\nline two { } \n*/ b;\n"
    masked = scan.mask_comments_and_literals(src)
    assert len(masked) == len(src)
    assert "{" not in masked
    assert "}" not in masked
    assert masked.count("\n") == src.count("\n")


def test_mask_string_literal_hides_braces_and_escaped_quote():
    src = 'const char* s = "a { brace and a \\" quote";\nint z;\n'
    masked = scan.mask_comments_and_literals(src)
    assert len(masked) == len(src)
    assert "{" not in masked.split("\n")[0]
    assert masked.split("\n")[1] == "int z;"


def test_mask_char_literal_hides_a_brace_character():
    src = "char c = '{';\n"
    masked = scan.mask_comments_and_literals(src)
    assert "{" not in masked


def test_mask_leaves_real_code_untouched():
    src = "void F() { return; }\n"
    assert scan.mask_comments_and_literals(src) == src


# --- match_braces ------------------------------------------------------------


def test_match_braces_simple_pair():
    masked = "void F() { return; }"
    pairs = scan.match_braces(masked)
    open_idx = masked.index("{")
    close_idx = masked.index("}")
    assert pairs[open_idx] == close_idx


def test_match_braces_nested_pairs_resolve_innermost_to_innermost():
    masked = "void F() { if (x) { y(); } z(); }"
    pairs = scan.match_braces(masked)
    outer_open = masked.index("{")
    outer_close = masked.rindex("}")
    inner_open = masked.index("{", outer_open + 1)
    inner_close = masked.index("}")
    assert pairs[outer_open] == outer_close
    assert pairs[inner_open] == inner_close


def test_match_braces_unbalanced_open_brace_has_no_entry():
    masked = "void F() { return;"
    pairs = scan.match_braces(masked)
    assert masked.index("{") not in pairs


# --- line_of -------------------------------------------------------------


def test_line_of_reports_one_indexed_lines_and_covers_eof():
    text = "a\nbb\nccc"
    lines = scan.line_of(text)
    assert lines[0] == 1          # 'a'
    assert lines[2] == 2          # first 'b'
    assert lines[5] == 3          # first 'c'
    assert lines[len(text)] == 3  # EOF entry


# --- find_function_definitions: is_niladic_void, six shapes + controls ------
# T-1632 Significant 1's own six-shape population, reproduced here as a
# regression suite for the widened pattern rather than a one-off script.

_NIADIC_VOID_POSITIVE_SHAPES = {
    "plain_static": "static void TestX() { CHECK(true); }",
    "multiline": "static\nvoid\nTestX\n(\n)\n{ CHECK(true); }",
    "inline": "inline void TestX() { CHECK(true); }",
    "noexcept": "static void TestX() noexcept { CHECK(true); }",
    "attribute": "[[maybe_unused]] static void TestX() { CHECK(true); }",
    "explicit_void_param": "static void TestX(void) { CHECK(true); }",
}

_NIADIC_VOID_NEGATIVE_SHAPES = {
    "takes_a_parameter": "void TestX(int y) { CHECK(y); }",
    "returns_bool": "bool TestX() { return true; }",
    "returns_int": "static int TestX() { return 0; }",
}


@pytest.mark.parametrize("shape_name", sorted(_NIADIC_VOID_POSITIVE_SHAPES))
def test_niladic_void_positive_shape_is_detected(shape_name):
    fns = scan.find_function_definitions(_NIADIC_VOID_POSITIVE_SHAPES[shape_name])
    assert len(fns) == 1, f"{shape_name}: expected exactly one definition, got {fns}"
    assert fns[0]["is_niladic_void"] is True, (
        f"{shape_name}: header {fns[0]['header']!r} was not recognized as niladic-void"
    )


@pytest.mark.parametrize("shape_name", sorted(_NIADIC_VOID_NEGATIVE_SHAPES))
def test_non_niladic_void_shape_is_not_flagged(shape_name):
    fns = scan.find_function_definitions(_NIADIC_VOID_NEGATIVE_SHAPES[shape_name])
    assert len(fns) == 1, f"{shape_name}: expected exactly one definition, got {fns}"
    assert fns[0]["is_niladic_void"] is False, (
        f"{shape_name}: header {fns[0]['header']!r} was incorrectly flagged niladic-void"
    )


def test_all_six_population_shapes_together_produce_the_two_caught_four_missed_before_widening():
    """The exact T-1632 Significant 1 measurement, reproduced as a standing
    regression check against the CURRENT (widened) pattern: all six shapes,
    compiled into one translation unit together, must now ALL be flagged --
    where the pre-widening pattern caught only two of six."""
    combined = "\n".join(_NIADIC_VOID_POSITIVE_SHAPES[k].replace("TestX", f"TestX{i}")
                          for i, k in enumerate(sorted(_NIADIC_VOID_POSITIVE_SHAPES)))
    fns = scan.find_function_definitions(combined)
    assert len(fns) == len(_NIADIC_VOID_POSITIVE_SHAPES)
    assert all(fn["is_niladic_void"] for fn in fns), (
        f"expected every one of the six shapes to be flagged niladic-void, got {fns}"
    )


# --- find_function_definitions: qualification and nesting -------------------


def test_free_function_is_unqualified():
    fns = scan.find_function_definitions("void Helper() { return; }")
    assert fns[0]["qualified"] == "Helper"


def test_in_class_member_function_is_qualified_by_its_struct():
    src = "struct Fixture {\n    void Build() { return; }\n};\n"
    fns = scan.find_function_definitions(src)
    member = [fn for fn in fns if fn["name"] == "Build"]
    assert len(member) == 1
    assert member[0]["qualified"] == "Fixture::Build"


def test_lambda_body_inside_a_function_is_not_reported_as_a_separate_definition():
    src = "void Outer() {\n    auto inner = []() { return 1; };\n    (void)inner;\n}\n"
    fns = scan.find_function_definitions(src)
    assert [fn["name"] for fn in fns] == ["Outer"]


def test_a_definition_inside_a_comment_is_not_matched():
    src = "// static void Fake() { CHECK(false); }\nint real = 1;\n"
    fns = scan.find_function_definitions(src)
    assert fns == []


def test_sslm_test_macro_call_is_not_niladic_void_shaped():
    """SSLM_TEST(Name, Order) { ... } reads, at the text level, as a call
    expression followed by a brace -- find_function_definitions' header
    regex requires an identifier directly followed by '(', so the macro
    invocation's own header text never matches _FUNC_HDR_RE the way a
    hand-written `void Name()` does. This is exactly the population split
    check_test_definitions_registered.py's own module docstring states:
    an SSLM_TEST-registered body is invisible to this scan, by construction."""
    fns = scan.find_function_definitions("SSLM_TEST(TestSomething, 1) { CHECK(true); }")
    assert fns == []


# --- normalized_hash ----------------------------------------------------


def test_normalized_hash_is_stable_under_whitespace_reformatting():
    a = scan.normalized_hash("void F() {\n  CHECK(true);\n}")
    b = scan.normalized_hash("void   F()   {  CHECK(true);  }")
    assert a == b


def test_normalized_hash_changes_when_the_body_actually_changes():
    a = scan.normalized_hash("void F() { CHECK(true); }")
    b = scan.normalized_hash("void F() { CHECK(false); }")
    assert a != b


def test_normalized_hash_is_sixteen_hex_chars():
    h = scan.normalized_hash("void F() { CHECK(true); }")
    assert len(h) == 16
    int(h, 16)  # raises ValueError if not valid hex


# --- checked_source_files / support_source_files -----------------------


def test_checked_source_files_always_includes_test_main_and_only_area_cpp_files():
    files = scan.checked_source_files()
    names = [os.path.basename(f) for f in files]
    assert "test_main.cpp" in names
    assert names[0] == "test_main.cpp"
    assert all(f.endswith(".cpp") for f in files)
    assert files == sorted(files[:1]) + sorted(files[1:])  # test_main.cpp first, rest sorted


def test_support_source_files_covers_only_h_and_cpp_under_support():
    files = scan.support_source_files()
    assert all(f.endswith((".h", ".cpp")) for f in files)
    assert all(os.path.join("tests", "support") in f.replace("/", os.sep) for f in files)


# --- candidate_binaries / locate_or_build_binary (T-1632 Significant 2) -----
# The cmake-rebuild recovery path (a stale binary present, `build/` configured,
# a real `cmake --build` invocation refreshes it) needs a real toolchain and is
# not exercised here -- it is exercised by execution in the T-1632 fold's own
# build log, which reproduces the review's exact stale-binary scenario. The
# three cases below drive the pure decision logic: pick a current binary,
# refuse on a stale one with nothing to rebuild from, and refuse when nothing
# exists at all.


def _make_file(path: str, mtime: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("x")
    os.utime(path, (mtime, mtime))


def test_candidate_binaries_lists_both_build_surfaces_under_the_given_root(monkeypatch, tmp_path):
    monkeypatch.setattr(scan, "REPO_ROOT", str(tmp_path))
    candidates = scan.candidate_binaries()
    rels = [os.path.relpath(p, str(tmp_path)) for p in candidates]
    assert os.path.join("build", "Release", "superslm_tests.exe") in rels
    assert os.path.join("out", "superslm_tests.exe") in rels


def test_locate_or_build_binary_returns_a_binary_at_least_as_new_as_every_scanned_source(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(scan, "REPO_ROOT", str(tmp_path))
    source = str(tmp_path / "tests" / "test_main.cpp")
    _make_file(source, mtime=1000)
    binary = str(tmp_path / "out" / "superslm_tests.exe")
    _make_file(binary, mtime=2000)
    assert scan.locate_or_build_binary([source]) == binary


def test_locate_or_build_binary_refuses_a_binary_older_than_a_scanned_source_with_no_build_dir_to_rebuild(
    monkeypatch, tmp_path
):
    """The T-1632 Significant 2 reproduction, in miniature: a binary that
    predates the source this check is about to scan must never be returned
    silently. No `build/` directory exists here, so there is nothing to
    rebuild from and the refusal must be loud (an exception), not a
    fallback to the stale binary."""
    monkeypatch.setattr(scan, "REPO_ROOT", str(tmp_path))
    binary = str(tmp_path / "out" / "superslm_tests.exe")
    _make_file(binary, mtime=1000)
    source = str(tmp_path / "tests" / "test_main.cpp")
    _make_file(source, mtime=2000)  # edited AFTER the binary was built
    with pytest.raises(scan.StaleBinaryError) as excinfo:
        scan.locate_or_build_binary([source])
    assert "superslm_tests.exe" in str(excinfo.value)


def test_locate_or_build_binary_raises_file_not_found_when_nothing_exists_and_no_build_dir(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(scan, "REPO_ROOT", str(tmp_path))
    source = str(tmp_path / "tests" / "test_main.cpp")
    _make_file(source, mtime=1000)
    with pytest.raises(FileNotFoundError):
        scan.locate_or_build_binary([source])


def test_locate_or_build_binary_prefers_the_first_current_candidate_in_order(monkeypatch, tmp_path):
    """build/Release is listed before out/ in candidate_binaries(); when both
    are current, the earlier one wins -- this is the exact ordering T-1632's
    real-repo state exercised (build/Release built after out/'s own last
    edit, both current relative to source, build/Release still preferred)."""
    monkeypatch.setattr(scan, "REPO_ROOT", str(tmp_path))
    source = str(tmp_path / "tests" / "test_main.cpp")
    _make_file(source, mtime=1000)
    cmake_binary = str(tmp_path / "build" / "Release" / "superslm_tests.exe")
    _make_file(cmake_binary, mtime=1500)
    out_binary = str(tmp_path / "out" / "superslm_tests.exe")
    _make_file(out_binary, mtime=1600)
    assert scan.locate_or_build_binary([source]) == cmake_binary
