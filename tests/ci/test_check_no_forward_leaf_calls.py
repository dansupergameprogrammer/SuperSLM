"""Curie's red suite for check_no_forward_leaf_calls.py (SuperSLM_S3a_Walking
SkeletonPlan.md Sec7.3, Sec11 S3.1; T-200, Sec17.3 cell 5).

Mirrors tests/test_check_no_pow_operator.py's own convention (constructed scratch
files, never the real production tree, per StandardsDocument Sec4's population-
validation requirement: a check shown able to FAIL on a fault it exists to catch,
not only shown to pass on unchanged input) and tests/ci/derive_bad_alloc_
membership.py's temp-file discipline (every scratch file is created under
tempfile.mkdtemp(), never written into the working tree, and removed in a
`finally`).

WHY SCRATCH FIXTURES, NOT THE REAL FORWARD DIRECTORY. The mechanism cells below
(the eight banned-leaf rule-coverage cells, the input-coverage cell, and the
allowlist-control cell) drive the check against constructed scratch
directories standing in for "a forward TU," never against the real
src/forward/ tree -- exactly as check_no_pow_operator.py's own self-test
stands in for "a generator that reintroduced **" with a scratch copy rather
than editing the real generator. This holds independent of whether real
forward-composition source exists: a real, unmodified
src/forward/checked_chain_funnel.cpp is not expected to name a banned leaf, so
scanning it alone could never show the check FAILING on a fault it exists to
catch (StandardsDocument Sec4's population-validation requirement -- a check
shown only to pass on unchanged input is not shown to catch anything). The
three cells the plan's own S3.1 red-cell list names --

    "a forward TU naming a leaf directly fails the CI check; a scratch TU
    added to the forward target and naming a banned leaf also fails it (input
    coverage); the funnel's own file and the leaf certification TUs pass
    (allowlist control)"

-- are realized here against those constructed scratch directories. The one
cell the scratch-only suite cannot realize is that the real glob and the real
allowlist, unmodified, agree on the real file that exists today -- the WIRING,
not the mechanism; that cell is realized separately, against the real tree, by
this module's own test_main_end_to_end_against_the_real_default_glob_is_no_
longer_vacuous (see its own docstring for why it is kept apart from the
mechanism cells above).
"""

import os
import re
import tempfile

import pytest

import check_no_forward_leaf_calls as cnfl

# T-1381 (D-SLM464): the door-count guard (find_leaf_forwarding_doors/
# check_door_count) is now Clang-AST-derived and needs a real clang++ on
# PATH. Every cell that exercises it is skipped, not failed, when no such
# clang++ is available -- an environment gap is not a mechanism defect,
# mirroring tests/ci/test_membership_check_population.py's own convention
# for the sibling AST-derived check.
def _clang_available() -> bool:
    try:
        cnfl._run_clang_ast_dump(
            os.path.join(cnfl._INCLUDE_DIR, "superslm", "sha256.h"),
            include_dir=cnfl._INCLUDE_DIR,
        )
        return True
    except cnfl.ClangUnavailable:
        return False


_HAVE_CLANG = _clang_available()
requires_clang = pytest.mark.skipif(
    not _HAVE_CLANG,
    reason="no clang++ on PATH capable of -Xclang -ast-dump=json "
    "(set SUPERSLM_CLANGXX to an explicit path)",
)


def _write(tmpdir: str, rel_path: str, content: str) -> str:
    abs_path = os.path.join(tmpdir, rel_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)
    return abs_path


# --- Rule coverage: every one of the eight banned leaves is individually detected. ---


def test_each_banned_leaf_is_individually_detected():
    for leaf in cnfl.BANNED_LEAVES:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "src/forward/site.cpp", f"auto d = {leaf}(ptr, n);\n")
            hits = cnfl.find_banned_leaf_uses(path)
            assert hits == [(1, leaf)], f"expected exactly one hit on {leaf!r}, got {hits}"


def test_a_leaf_name_appearing_only_as_a_substring_of_a_longer_identifier_is_not_flagged():
    """MaxAbsReduce is a strict PREFIX of MaxAbsReduceWide -- both are independently
    banned, and a call to the longer name must be reported as itself, never as a
    spurious extra hit on the shorter name it happens to start with."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "src/forward/site.cpp", "auto d = MaxAbsReduceWide(ptr, n);\n")
        hits = cnfl.find_banned_leaf_uses(path)
        assert hits == [(1, "MaxAbsReduceWide")], (
            f"a call to MaxAbsReduceWide must be reported once, as MaxAbsReduceWide, "
            f"not also as a hit on its prefix MaxAbsReduce: got {hits}"
        )


def test_a_clean_forward_file_reports_no_hits():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "src/forward/site.cpp",
            "auto status = RequantChainChecked(row, n, incoming, site_constant, out, &scale);\n",
        )
        assert cnfl.find_banned_leaf_uses(path) == []


# --- The plan's own three named S3.1 red cells. ---


def test_a_forward_tu_naming_a_banned_leaf_directly_fails_the_check():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "src/forward/site.cpp", "auto d = NormalizeScale(d_prime);\n")
        code = cnfl.main(
            globs=("src/forward/**/*.cpp",),
            allowlist=(),
            repo_root=tmp,
        )
        assert code == 1


def test_a_scratch_tu_added_to_the_forward_target_and_naming_a_leaf_is_also_caught():
    """Input coverage (Sec7.3): the check's file set is DERIVED from a glob, not
    enumerated by hand, so a file that did not exist when the check was authored is
    still scanned once it lands under the glob root -- proven by adding one here,
    after main() and the glob are already defined, and confirming it alone is what
    trips the check."""
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "src/forward/existing_site.cpp", "auto x = RowBoundsWide(row, n, &mx, &mn);\n")
        code_before = cnfl.main(globs=("src/forward/**/*.cpp",), allowlist=(), repo_root=tmp)
        assert code_before == 1, "the existing site itself already names a banned leaf"

        # Now add a SECOND, brand-new scratch TU naming a DIFFERENT banned leaf and
        # confirm the glob-derived scan picks it up too -- not just the file present
        # when the check was first pointed at this directory.
        _write(tmp, "src/forward/new_scratch_site.cpp", "auto y = RequantTokenCodeWide(x, r, s);\n")
        allow_only_existing = (os.path.relpath(
            os.path.join(tmp, "src/forward/existing_site.cpp"), tmp
        ),)
        failures = cnfl.scan_files(
            cnfl._glob_files(("src/forward/**/*.cpp",), tmp),
            allowlist=allow_only_existing,
            repo_root=tmp,
        )
        assert len(failures) == 1
        assert "new_scratch_site.cpp" in failures[0]
        assert "RequantTokenCodeWide" in failures[0]


def test_the_funnels_own_file_and_leaf_certification_tus_pass_the_allowlist_control():
    """Allowlist control: two scratch files at the ANTICIPATED allowlisted relative
    paths (the funnel's own file, and a leaf certification TU) each name a banned
    leaf directly and must NOT fail -- while a third, non-allowlisted sibling in the
    same directory, naming the same leaf, still fails. Proves the allowlist exempts
    by path, not by directory."""
    with tempfile.TemporaryDirectory() as tmp:
        funnel_rel = "src/forward/checked_chain_funnel.cpp"
        cert_rel = "tests/cert_intmath.cpp"
        sibling_rel = "src/forward/uses_leaf_directly.cpp"
        _write(tmp, funnel_rel, "int64_t D = MaxAbsReduceWide(row, n);\n")
        _write(tmp, cert_rel, "int64_t D = MaxAbsReduce(row32, n);\n")
        _write(tmp, sibling_rel, "int64_t D = MaxAbsReduceWide(row, n);\n")

        failures = cnfl.scan_files(
            [os.path.join(tmp, funnel_rel), os.path.join(tmp, cert_rel), os.path.join(tmp, sibling_rel)],
            allowlist=(funnel_rel, cert_rel),
            repo_root=tmp,
        )
        assert len(failures) == 1, f"expected exactly the non-allowlisted sibling to fail, got {failures}"
        assert "uses_leaf_directly.cpp" in failures[0]


def test_main_passes_when_every_scanned_file_is_allowlisted():
    with tempfile.TemporaryDirectory() as tmp:
        funnel_rel = "src/forward/checked_chain_funnel.cpp"
        _write(tmp, funnel_rel, "int64_t D = NarrowAccumulatorToI32(row, n, out);\n")
        code = cnfl.main(
            globs=("src/forward/**/*.cpp",),
            allowlist=(funnel_rel,),
            repo_root=tmp,
        )
        assert code == 0


# --- The self-test convention (test_check_no_pow_operator.py's own shape). ---


def test_missing_scanned_file_is_reported_distinctly_from_a_dirty_one():
    failures = cnfl.scan_files(["/does/not/exist/site.cpp"], allowlist=(), repo_root="/does/not/exist")
    assert len(failures) == 1
    assert "not found" in failures[0]


def test_nested_leaf_call_inside_an_expression_is_still_found():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "src/forward/site.cpp",
            "int8_t code = RequantTokenCode(NormalizeScale(d).dn, r, s);\n",
        )
        hits = cnfl.find_banned_leaf_uses(path)
        leaves_hit = sorted(leaf for _, leaf in hits)
        assert leaves_hit == ["NormalizeScale", "RequantTokenCode"]


# --- Present truth: the real forward directory, now that it exists. ---


@requires_clang
def test_main_end_to_end_against_the_real_default_glob_is_no_longer_vacuous():
    """UPDATED 2026-07-28 (S3.1 header-contract build, commit 32aca0c): this cell
    used to name and assert a vacuous pass (no forward-composition source existed
    in the real tree yet, Sec11 placing it at S3.2-S3.9). That premise no longer
    holds -- `src/forward/checked_chain_funnel.cpp` now exists and is the exact
    file the default glob was written in anticipation of. This test asserts the
    real, current state instead of the historical one: the default glob matches a
    nonempty, exact population (never grows silently uncounted -- a real
    forward-composition file landing without a matching update here is caught by
    the equality assertion, not just a >0 check), and the real production tree
    passes the check end-to-end.

    UPDATED AGAIN 2026-07-28 (S3.2 header-contract build, commit a594dd2, closed
    by this Curie pass): a second real forward-composition file,
    `src/forward_sites.cpp`, landed outside `src/forward/` -- exactly the
    situation the paragraph above named as "revisit again." The fix is not
    `len(files) > 0` (that would stop this cell from ever catching an
    unaccounted-for population change again, which is the whole property this
    cell exists to hold); it is naming the real, current set explicitly, sized to
    N rather than hardcoded to one. `cnfl._EXPECTED_REAL_FORWARD_FILES` is that
    named set, updated in the same commit that widened `_DEFAULT_FORWARD_GLOBS`
    to scan the new file, so this cell's equality assertion and the module's own
    scan stay in agreement rather than one silently drifting from the other.

    The funnel's own file passes here because `scan_files` skips reading an
    allowlisted path's content entirely (short-circuits on the path match before
    opening the file) -- so this cell does not, and cannot, prove the allowlist
    exempts real banned-leaf content; that half is proven by
    `test_the_funnels_own_file_and_leaf_certification_tus_pass_the_allowlist_control`
    against constructed scratch content standing in for it, per
    StandardsDocument Sec4's population-validation requirement (a vacuous
    production scan proves nothing; every other cell in this suite drives the
    mechanism directly). `src/forward_sites.cpp` is NOT allowlisted, so this cell
    DOES exercise real content through it -- it passes only because the file's
    real content today (S3.2's deliberately-wrong stub bodies) names no banned
    leaf, confirmed by direct read, not assumed.

    Revisit again the moment a third real forward-composition file lands, or the
    moment `src/forward_sites.cpp` is relocated under `src/forward/` (at which
    point it is matched by the directory glob on its own and its explicit entry
    in `_DEFAULT_FORWARD_GLOBS`/`_EXPECTED_REAL_FORWARD_FILES` becomes
    redundant, not wrong): this exact-match assertion will fail, correctly, and
    should be updated to name the new/moved file rather than loosened to
    `len(files) > 0`."""
    files = cnfl._glob_files(cnfl._DEFAULT_FORWARD_GLOBS, cnfl._REPO_ROOT)
    expected = sorted(
        os.path.normpath(os.path.join(cnfl._REPO_ROOT, rel))
        for rel in cnfl._EXPECTED_REAL_FORWARD_FILES
    )
    assert [os.path.normpath(f) for f in files] == expected, (
        f"expected the default forward-composition glob to match exactly "
        f"{expected}, got {files} -- a forward-composition file was added, "
        f"removed, or moved without updating _EXPECTED_REAL_FORWARD_FILES "
        f"and/or _DEFAULT_FORWARD_GLOBS"
    )
    code = cnfl.main()
    assert code == 0, (
        "the real forward-composition tree must pass the CI check end-to-end "
        "under the unmodified default glob and allowlist"
    )


# --- Significant 9 (Poirot e4b398c review): the door-count guard. ---


@requires_clang
def test_find_leaf_forwarding_doors_finds_a_one_line_forwarding_function():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "src/forward/checked_chain_funnel.cpp",
            "int64_t CarriedScaleReciprocal(int64_t m) { return DynamicScaleReciprocal(m); }\n",
        )
        assert cnfl.find_leaf_forwarding_doors(path) == ["CarriedScaleReciprocal"]


@requires_clang
def test_find_leaf_forwarding_doors_finds_a_multi_line_signature_door():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "src/forward/checked_chain_funnel.cpp",
            "int64_t CarriedScaleReciprocal(\n"
            "    int64_t m) {\n"
            "  return DynamicScaleReciprocal(m);\n"
            "}\n",
        )
        assert cnfl.find_leaf_forwarding_doors(path) == ["CarriedScaleReciprocal"]


@requires_clang
def test_find_leaf_forwarding_doors_excludes_the_funnels_own_entry_points():
    """RequantChainChecked and NarrowRowChecked call banned leaves internally as
    their whole job -- they must never be reported as forwarding doors, or this
    guard would fail on the funnel's own real file permanently."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "src/forward/checked_chain_funnel.cpp",
            "ChainResult RequantChainChecked(int n) {\n"
            "  auto D = MaxAbsReduceWide(nullptr, n);\n"
            "  return {};\n"
            "}\n"
            "SslmForwardStatus NarrowRowChecked(int n) {\n"
            "  auto x = NarrowAccumulatorToI32(nullptr, n, nullptr);\n"
            "  return {};\n"
            "}\n",
        )
        assert cnfl.find_leaf_forwarding_doors(path) == []


@requires_clang
def test_find_leaf_forwarding_doors_finds_a_second_door_alongside_the_first():
    """Mutation proof for the door-count guard's whole point: a SECOND function
    forwarding a different banned leaf must be found too, not just the first one
    the scanner happens to hit."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "src/forward/checked_chain_funnel.cpp",
            "int64_t CarriedScaleReciprocal(int64_t m) { return DynamicScaleReciprocal(m); }\n"
            "int64_t CarriedScaleNormalize(int64_t d) { return NormalizeScale(d).dn; }\n",
        )
        assert sorted(cnfl.find_leaf_forwarding_doors(path)) == [
            "CarriedScaleNormalize",
            "CarriedScaleReciprocal",
        ]


@requires_clang
def test_check_door_count_passes_when_the_scanned_file_matches_the_expected_set():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "src/forward/checked_chain_funnel.cpp",
            "int64_t CarriedScaleReciprocal(int64_t m) { return DynamicScaleReciprocal(m); }\n",
        )
        assert cnfl.check_door_count(path, expected=("CarriedScaleReciprocal",)) == []


@requires_clang
def test_check_door_count_fails_when_a_second_door_appears():
    """The mutation this guard exists to catch (Significant 9's own text): a
    second door opened alongside CarriedScaleReciprocal must fail loudly rather
    than silently widen the set the funnel's own file is allowed to export."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "src/forward/checked_chain_funnel.cpp",
            "int64_t CarriedScaleReciprocal(int64_t m) { return DynamicScaleReciprocal(m); }\n"
            "int64_t CarriedScaleNormalize(int64_t d) { return NormalizeScale(d).dn; }\n",
        )
        failures = cnfl.check_door_count(path, expected=("CarriedScaleReciprocal",))
        assert len(failures) == 1
        assert "CarriedScaleNormalize" in failures[0]


@requires_clang
def test_check_door_count_fails_when_the_door_is_silently_removed():
    """The other direction: the door disappearing (renamed or its forwarding
    call deleted) is also a change to what the funnel's own file exports, and
    must be caught the same way a second door appearing is."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "src/forward/checked_chain_funnel.cpp",
            "int64_t CarriedScaleReciprocal(int64_t m) { return 0; }\n",
        )
        failures = cnfl.check_door_count(path, expected=("CarriedScaleReciprocal",))
        assert len(failures) == 1


def test_check_door_count_is_silent_when_the_funnel_file_is_absent():
    """A scratch-directory mechanism test with no funnel file at all is
    exercising the leaf-ban scan, not this guard -- absence is not itself a
    door-count failure (see main()'s own repo_root gating for why the guard
    does not run at all against an arbitrary scratch repo_root)."""
    assert cnfl.check_door_count("/does/not/exist/checked_chain_funnel.cpp") == []


@requires_clang
def test_main_end_to_end_asserts_the_real_doors_hold_at_exactly_one():
    """The wiring cell for the door-count guard, mirroring
    test_main_end_to_end_against_the_real_default_glob_is_no_longer_vacuous
    immediately above: asserted against the real, current
    src/forward/checked_chain_funnel.cpp, not a scratch stand-in."""
    real_funnel = os.path.join(cnfl._REPO_ROOT, "src/forward/checked_chain_funnel.cpp")
    assert cnfl.check_door_count(real_funnel) == []
    doors = cnfl.find_leaf_forwarding_doors(real_funnel)
    assert doors == list(cnfl._EXPECTED_DOOR_FUNCTIONS), (
        f"the real funnel file's forwarding doors are {doors}, want exactly "
        f"{list(cnfl._EXPECTED_DOOR_FUNCTIONS)} -- update _EXPECTED_DOOR_FUNCTIONS "
        f"if a door was deliberately added, removed, or renamed"
    )


# --- T-1378 (Poirot 9d8b38e review, Significant 1): the door scanner is
# hardened against three shapes it missed and validated against a wider,
# independently-constructed population before being trusted. ---


def _old_find_leaf_forwarding_doors_for_comparison(path, leaves=cnfl.BANNED_LEAVES,
                                                     exclude=cnfl._FUNNEL_ENTRY_POINTS):
    """The PRE-T-1378 scanner, reproduced verbatim (not imported -- the module
    under test no longer carries this code) so this suite can independently
    confirm each new population member really was a miss, not merely assert
    it from the casebook. Column-0-anchored, no-nested-parens parameter list,
    raw (non-comment-stripped) brace counting -- exactly Significant 1's own
    quoted description of what shipped at `9d8b38e`."""
    import re as _re

    old_re = _re.compile(
        r"^[A-Za-z_][\w:*&<>,\s]*?\b([A-Za-z_]\w*)\s*\([^(){}]*\)\s*(?:noexcept\s*)?\{",
        _re.MULTILINE,
    )
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    doors = []
    for m in old_re.finditer(text):
        name = m.group(1)
        body_start = m.end() - 1
        depth = 0
        body_end = len(text)
        for i in range(body_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    body_end = i + 1
                    break
        body_text = text[body_start:body_end]
        if name not in exclude and any(
            cnfl._leaf_pattern(leaf).search(body_text) for leaf in leaves
        ):
            doors.append(name)
    return doors


# Five independently-constructed doors (StandardsDocument Sec4: a wider net
# cast deliberately, not the three original shapes' own author re-deriving
# them): the three shapes the code review found (paren-in-params, indented
# inside a namespace, brace inside a `//` comment), plus two more found by
# this pass's own wider sweep (brace inside a `/* */` block comment; the
# paren-in-params and indentation shapes combined on one door, nested TWO
# levels of `namespace { ... }` rather than one). Each on its own scratch copy
# of a minimal funnel file, `CarriedScaleReciprocal` present as the one
# EXPECTED door in every case so the new, second door is what each case is
# isolating.
_BASE_FUNNEL_PREFIX = (
    "int64_t CarriedScaleReciprocal(int64_t m) { return DynamicScaleReciprocal(m); }\n"
)

_POPULATION_CASES = {
    "paren_in_param_list": (
        _BASE_FUNNEL_PREFIX
        + "int64_t CarriedScaleDoorTwo(int64_t d, CarriedScale c = CarriedScale()) "
        "{ return DynamicScaleReciprocal(d); }\n"
    ),
    "indented_one_level_inside_namespace": (
        "namespace superslm {\n"
        + _BASE_FUNNEL_PREFIX
        + "    int64_t CarriedScaleDoorTwo(int64_t d) {\n"
        "        return DynamicScaleReciprocal(d);\n"
        "    }\n"
        "}  // namespace superslm\n"
    ),
    "closing_brace_inside_line_comment": (
        _BASE_FUNNEL_PREFIX
        + "int64_t CarriedScaleDoorTwo(int64_t d) {\n"
        "    // a closing brace inside a comment: }\n"
        "    return DynamicScaleReciprocal(d);\n"
        "}\n"
    ),
    "closing_brace_inside_block_comment": (
        _BASE_FUNNEL_PREFIX
        + "int64_t CarriedScaleDoorTwo(int64_t d) {\n"
        "    /* a closing brace inside a block comment: } */\n"
        "    return DynamicScaleReciprocal(d);\n"
        "}\n"
    ),
    "paren_in_params_and_two_levels_of_nested_namespace": (
        "namespace superslm {\n"
        "namespace {\n"
        + _BASE_FUNNEL_PREFIX
        + "int64_t CarriedScaleDoorTwo(int64_t d, CarriedScale c = CarriedScale()) {\n"
        "    return DynamicScaleReciprocal(d);\n"
        "}\n"
        "}  // namespace\n"
        "}  // namespace superslm\n"
    ),
}

# The false-positive control: a banned leaf's name AND a fake `{`/`}`-bearing
# signature, both inside a string literal. Must NOT be reported as a door
# either before or after T-1378 -- this hardening narrows what escapes
# detection, it must not widen what counts as a door.
_STRING_LITERAL_CONTROL = (
    _BASE_FUNNEL_PREFIX
    + 'const char* kNotADoor = "int64_t NormalizeScale(int64_t x) { return x; }";\n'
)


def test_population_each_case_was_a_miss_under_the_prior_scanner():
    """Independently confirms every population member above really was
    invisible to the pre-T-1378 scanner (not merely asserted from the
    casebook) -- the population-validation half of StandardsDocument Sec4:
    a check must be shown able to FAIL on the fault it exists to catch."""
    for case_name, content in _POPULATION_CASES.items():
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "src/forward/checked_chain_funnel.cpp", content)
            old_doors = sorted(_old_find_leaf_forwarding_doors_for_comparison(path))
            assert old_doors == ["CarriedScaleReciprocal"], (
                f"case {case_name!r}: expected the prior scanner to MISS the second "
                f"door (finding only CarriedScaleReciprocal), got {old_doors} -- "
                f"this case is not a valid population member if the prior scanner "
                f"already caught it"
            )


@requires_clang
def test_population_the_hardened_scanner_catches_every_case():
    """The hardening's own proof: every case the T-1378 text scanner missed
    is caught by the current cnfl.find_leaf_forwarding_doors, by name --
    updated by T-1381 to exercise the Clang-AST mechanism that now backs
    this function; the assertion is unchanged because the contract
    (find_leaf_forwarding_doors' return value) is unchanged."""
    for case_name, content in _POPULATION_CASES.items():
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "src/forward/checked_chain_funnel.cpp", content)
            new_doors = sorted(cnfl.find_leaf_forwarding_doors(path))
            assert new_doors == ["CarriedScaleDoorTwo", "CarriedScaleReciprocal"], (
                f"case {case_name!r}: expected the hardened scanner to catch both "
                f"doors, got {new_doors}"
            )


@requires_clang
def test_population_check_door_count_fails_on_every_case_too():
    """The wiring cell for the population above: `check_door_count`, the
    function the CI check actually calls, must fail on every case -- not
    merely `find_leaf_forwarding_doors` in isolation."""
    for case_name, content in _POPULATION_CASES.items():
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "src/forward/checked_chain_funnel.cpp", content)
            failures = cnfl.check_door_count(path, expected=("CarriedScaleReciprocal",))
            assert len(failures) == 1 and "CarriedScaleDoorTwo" in failures[0], (
                f"case {case_name!r}: expected check_door_count to name the new "
                f"door, got {failures}"
            )


@requires_clang
def test_a_banned_leaf_name_inside_a_string_literal_is_not_reported_as_a_door_before_or_after():
    """The false-positive control: hardening the scanner must not widen what
    counts as a door. Neither the prior nor the current scanner may report a
    door here -- the fake signature and the leaf name both live inside a
    string literal, not in real source."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "src/forward/checked_chain_funnel.cpp", _STRING_LITERAL_CONTROL)
        old_doors = sorted(_old_find_leaf_forwarding_doors_for_comparison(path))
        new_doors = sorted(cnfl.find_leaf_forwarding_doors(path))
        assert old_doors == ["CarriedScaleReciprocal"], old_doors
        assert new_doors == ["CarriedScaleReciprocal"], new_doors


def test_the_population_count_is_stated():
    """States, in one place, the count required by this campaign's own exit
    condition: five independently-constructed population members
    (_POPULATION_CASES) plus one false-positive control, all executed above."""
    assert len(_POPULATION_CASES) == 5


def test_a_scratch_forward_sites_cpp_naming_a_banned_leaf_is_caught_by_the_widened_glob():
    """Mutation proof for the S3.2 widening above: `src/forward_sites.cpp` sits
    outside `src/forward/` and was, before this pass, entirely outside
    _DEFAULT_FORWARD_GLOBS' scan root -- a direct banned-leaf call from it would
    have passed this check with nothing to catch it (the exact hole this pass
    closes). Proven here the same way every other mechanism cell in this suite
    proves its point -- a constructed scratch file at the real relative path,
    scanned under the real (module-default) globs, never the real production
    tree -- so this cell fails loudly if the widening in
    check_no_forward_leaf_calls.py is ever reverted or narrowed back to
    `src/forward/**` alone."""
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "src/forward_sites.cpp", "auto d = DynamicScaleReciprocal(dn);\n")
        code = cnfl.main(
            globs=cnfl._DEFAULT_FORWARD_GLOBS,
            allowlist=cnfl._DEFAULT_ALLOWLIST,
            repo_root=tmp,
        )
        assert code == 1, (
            "a banned leaf named directly in src/forward_sites.cpp must fail the "
            "check under the module's own default globs/allowlist -- if this "
            "passes, the S3.2 widening of _DEFAULT_FORWARD_GLOBS stopped "
            "covering this file"
        )


# ---------------------------------------------------------------------------
# T-1381 (Poirot 5af6ab5-t1377-t1378-review-2026-07-31.md, Significant 2;
# D-SLM464; SuperSLM_S3a_WalkingSkeleton_Plan.md Sec7.3a): the door-count
# guard's mechanism replaced a second time -- the T-1378 hardened text
# scanner above is now retired from check_no_forward_leaf_calls.py itself
# (find_leaf_forwarding_doors/check_door_count are Clang-AST-derived) and is
# reproduced verbatim here, exactly as _old_find_leaf_forwarding_doors_for_
# comparison preserves the PRE-T-1378 scanner above, so this suite can
# independently confirm the improvement rather than merely assert it
# (StandardsDocument Sec4). Validated against the SAME population this file
# already carries (five constructed doors plus the false-positive control,
# all still caught by the AST mechanism -- test_population_the_hardened_
# scanner_catches_every_case and test_a_banned_leaf_name_inside_a_string_
# literal_is_not_reported_as_a_door_before_or_after above, both now @requires
# _clang since cnfl.find_leaf_forwarding_doors is the AST mechanism) PLUS the
# three shapes the review found the T-1378 scanner still misses, below.
# ---------------------------------------------------------------------------


def _hardened_text_scan_find_leaf_forwarding_doors_for_comparison(
    path, leaves=cnfl.BANNED_LEAVES, exclude=cnfl._FUNNEL_ENTRY_POINTS
):
    """The T-1378 (post-hardening, pre-T-1381) scanner, reproduced verbatim --
    the module under test no longer carries this code, replaced by the
    Clang-AST walk T-1381 built. Comment/string stripping, a balanced-
    parenthesis parameter list, and brace-context tracking (namespace vs.
    other), exactly as it shipped and was validated against a five-case
    population at T-1378. Reproduced here, not imported, so this suite can
    independently confirm each of the three NEW shapes below really is a
    miss under THIS scanner specifically (not merely asserted from the
    casebook), mirroring how _old_find_leaf_forwarding_doors_for_comparison
    preserves the scanner one generation further back."""
    import re as _re

    def _strip_comments_and_strings(text):
        out = []
        i = 0
        n = len(text)
        while i < n:
            c = text[i]
            if c == "/" and i + 1 < n and text[i + 1] == "/":
                start = i
                while i < n and text[i] != "\n":
                    i += 1
                out.append(" " * (i - start))
                continue
            if c == "/" and i + 1 < n and text[i + 1] == "*":
                start = i
                i += 2
                while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                i = min(i + 2, n)
                chunk = text[start:i]
                out.append("".join(ch if ch == "\n" else " " for ch in chunk))
                continue
            if c in ("\"", "'"):
                quote = c
                start = i
                i += 1
                while i < n and text[i] != quote:
                    i += 2 if text[i] == "\\" and i + 1 < n else 1
                i = min(i + 1, n)
                out.append(" " * (i - start))
                continue
            out.append(c)
            i += 1
        return "".join(out)

    _SIGNATURE_HEAD_RE = _re.compile(r"[A-Za-z_][\w:*&<>,\s]*?\b([A-Za-z_]\w*)\s*\(")
    _SIGNATURE_TAIL_RE = _re.compile(r"\s*(?:noexcept\s*)?\{")
    _NAMESPACE_OPENER_RE = _re.compile(r"namespace(\s+[A-Za-z_]\w*)?\s*$")

    def _is_word_char(c):
        return c.isalnum() or c == "_"

    def _brace_tag(cleaned, brace_pos):
        look_start = max(0, brace_pos - 200)
        if _NAMESPACE_OPENER_RE.search(cleaned[look_start:brace_pos]):
            return "namespace"
        return "other"

    def _find_top_level_function_bodies(text):
        cleaned = _strip_comments_and_strings(text)
        n = len(cleaned)
        stack = []
        pending_names = []
        pending_starts = []
        results = []
        i = 0
        while i < n:
            ch = cleaned[i]
            if ch == "{":
                stack.append(_brace_tag(cleaned, i))
                i += 1
                continue
            if ch == "}":
                if stack:
                    tag = stack.pop()
                    if tag == "function":
                        results.append((pending_names.pop(), pending_starts.pop(), i + 1))
                i += 1
                continue
            starts_ident = _is_word_char(ch) and not ch.isdigit()
            prev_is_ident = i > 0 and _is_word_char(cleaned[i - 1])
            if starts_ident and not prev_is_ident:
                head = _SIGNATURE_HEAD_RE.match(cleaned, i)
                if head is not None:
                    depth = 1
                    j = head.end()
                    while j < n and depth > 0:
                        if cleaned[j] == "(":
                            depth += 1
                        elif cleaned[j] == ")":
                            depth -= 1
                        j += 1
                    if depth == 0:
                        tail = _SIGNATURE_TAIL_RE.match(cleaned, j)
                        if tail is not None:
                            brace_pos = tail.end() - 1
                            if all(t == "namespace" for t in stack):
                                stack.append("function")
                                pending_names.append(head.group(1))
                                pending_starts.append(brace_pos)
                            else:
                                stack.append("other")
                            i = brace_pos + 1
                            continue
            i += 1
        return results

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    doors = []
    for name, body_start, body_end in _find_top_level_function_bodies(text):
        body_text = text[body_start:body_end]
        if name not in exclude and any(cnfl._leaf_pattern(leaf).search(body_text) for leaf in leaves):
            doors.append(name)
    return doors


# The three shapes the review found the T-1378 scanner still misses -- two
# silently (a second door passes with nothing said), one (the digit
# separator) by corrupting the comment/string stripper's own output for the
# rest of the file. Each on its own scratch copy of a minimal funnel file,
# `CarriedScaleReciprocal` present as the one EXPECTED door, exactly matching
# _POPULATION_CASES' own convention above.
_T1381_NEW_SHAPES = {
    "trailing_return_type": (
        _BASE_FUNNEL_PREFIX
        + "auto CarriedScaleDoorTwo(int64_t d) -> int64_t {\n"
        "    return DynamicScaleReciprocal(d);\n"
        "}\n"
    ),
    "extern_c_linkage_block": (
        _BASE_FUNNEL_PREFIX
        + 'extern "C" {\n'
        "int64_t CarriedScaleDoorTwo(int64_t d) {\n"
        "    return DynamicScaleReciprocal(d);\n"
        "}\n"
        "}\n"
    ),
    "digit_separator_before_door": (
        _BASE_FUNNEL_PREFIX
        + "static const int64_t kBig = 1'000;\n"
        "int64_t CarriedScaleDoorTwo(int64_t d) {\n"
        "    return DynamicScaleReciprocal(d);\n"
        "}\n"
    ),
}


def test_t1381_new_shapes_were_misses_under_the_t1378_hardened_scanner():
    """Independently confirms each of the three new shapes really was invisible
    to the T-1378 hardened scanner -- StandardsDocument Sec4's population-
    validation requirement, applied one generation further: a check shown able
    to fail on the fault it exists to catch, not merely asserted to."""
    for case_name, content in _T1381_NEW_SHAPES.items():
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "src/forward/checked_chain_funnel.cpp", content)
            old_doors = sorted(_hardened_text_scan_find_leaf_forwarding_doors_for_comparison(path))
            assert old_doors == ["CarriedScaleReciprocal"], (
                f"case {case_name!r}: expected the T-1378 hardened scanner to MISS "
                f"the second door (finding only CarriedScaleReciprocal), got "
                f"{old_doors} -- this case is not a valid population member if "
                f"the T-1378 scanner already caught it"
            )


@requires_clang
def test_t1381_ast_mechanism_catches_the_new_shapes_too():
    """The AST-based replacement's own proof: every shape the T-1378 scanner
    missed above is caught by the current cnfl.find_leaf_forwarding_doors, by
    name -- measured against the SAME reproduced-verbatim T-1378 scanner
    immediately above, not merely asserted (StandardsDocument Sec4)."""
    for case_name, content in _T1381_NEW_SHAPES.items():
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "src/forward/checked_chain_funnel.cpp", content)
            new_doors = sorted(cnfl.find_leaf_forwarding_doors(path))
            assert new_doors == ["CarriedScaleDoorTwo", "CarriedScaleReciprocal"], (
                f"case {case_name!r}: expected the AST mechanism to catch both "
                f"doors, got {new_doors}"
            )


@requires_clang
def test_t1381_check_door_count_fails_on_every_new_shape_too():
    """The wiring cell for the three new shapes: check_door_count, the function
    the CI gate actually calls, must fail on every one -- not merely
    find_leaf_forwarding_doors in isolation."""
    for case_name, content in _T1381_NEW_SHAPES.items():
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "src/forward/checked_chain_funnel.cpp", content)
            failures = cnfl.check_door_count(path, expected=("CarriedScaleReciprocal",))
            assert len(failures) == 1 and "CarriedScaleDoorTwo" in failures[0], (
                f"case {case_name!r}: expected check_door_count to name the new "
                f"door, got {failures}"
            )


def test_t1381_population_count_is_stated():
    """States, in one place, the total population T-1381's construction was
    validated against: the five shapes _POPULATION_CASES already carried,
    plus the three new shapes this review found, plus the one false-positive
    control -- nine cases, each executed against BOTH the T-1378 hardened
    text scanner (reproduced verbatim above) and the new Clang-AST mechanism,
    per StandardsDocument Sec4's independent-population requirement."""
    assert len(_POPULATION_CASES) == 5
    assert len(_T1381_NEW_SHAPES) == 3
    total_population_plus_control = len(_POPULATION_CASES) + len(_T1381_NEW_SHAPES) + 1
    assert total_population_plus_control == 9


# --- Further sweep beyond the three named shapes (StandardsDocument Sec4:
# "add any further shapes your own sweep finds"). Two additional candidate
# shapes were checked, both confirmed to behave identically under the old
# and new mechanisms (no further miss found, no regression introduced):
#
#   1. A leaf call inside a LAMBDA defined within a door function's own body
#      (`auto inner = [d]() { return DynamicScaleReciprocal(d); }; return
#      inner();`). Against the REAL production file (compiled with its real
#      #include chain, exactly as the funnel's own file always is), the AST
#      mechanism finds both doors correctly -- proven directly below. Against
#      a FULLY DECLARATION-FREE scratch fixture with no #includes at all
#      (every type and identifier unresolved), Clang's error-recovery cannot
#      construct any body at all for the lambda's own `operator()` when its
#      return type additionally depends on `auto` deduction over an
#      unresolved call -- a residual specific to that adversarial,
#      declaration-free construction, not to the mechanism's use against
#      real code (which is what Sec7.3a's own property actually protects:
#      the funnel's real, always-fully-declared file). Not added as a
#      population member because it does not reflect how the real funnel
#      file is built, and is not one of the three shapes this review named.
#   2. A local variable shadowing a banned leaf's name inside a door's body
#      (`int64_t DynamicScaleReciprocal = d; return DynamicScaleReciprocal;`)
#      -- both the T-1378 scanner (a raw identifier match) and the AST
#      mechanism (which does not discriminate a variable DeclRefExpr from a
#      function DeclRefExpr by kind) treat this as a hit, consistently. Not
#      a new divergence; both over-approximate the same way, matching this
#      module's "a ban, not a classifier" convention (Sec7.3's own docstring
#      for the whole-tree text scan, which the door-count guard inherits for
#      this one edge case).
#
# Total independent sweep for T-1381: the nine required cases above plus
# these two further candidates -- eleven shapes swept.


@requires_clang
def test_t1381_ast_mechanism_finds_a_door_hidden_inside_a_lambda_in_real_declared_code():
    """Sweep candidate 1 above, proven against REAL declared code rather than
    a declaration-free scratch fixture: a copy of the real, current
    src/forward/checked_chain_funnel.cpp (compiled with its own real
    #include chain) with one appended door whose leaf call is hidden inside
    a lambda defined in the door's own body. The AST mechanism must find
    both the pre-existing real door and the new lambda-hiding one."""
    real_funnel = os.path.join(cnfl._REPO_ROOT, "src/forward/checked_chain_funnel.cpp")
    with open(real_funnel, "r", encoding="utf-8") as f:
        real_content = f.read()
    lambda_door = (
        "\nnamespace superslm {\n"
        "int64_t CarriedScaleDoorLambda(int64_t d) {\n"
        "    auto inner = [d]() { return DynamicScaleReciprocal(d); };\n"
        "    return inner();\n"
        "}\n"
        "}  // namespace superslm\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "checked_chain_funnel_with_lambda_door.cpp", real_content + lambda_door)
        doors = sorted(cnfl.find_leaf_forwarding_doors(path))
        assert doors == ["CarriedScaleDoorLambda", "CarriedScaleReciprocal"], (
            f"expected both the real door and the lambda-hiding one against "
            f"real declared code, got {doors}"
        )


# ---------------------------------------------------------------------------
# T-1383 (Poirot 5eff945-t1380-t1381-t1382-review-2026-07-31.md, Significant
# 1; D-SLM469 records the plan/decision-log correction, this section is the
# code-population correction): the T-1381 AST mechanism admitted `FunctionDecl`
# only, on a comment claiming this excluded a member function "exactly as the
# prior text scanner's brace-context tag 'other' did" -- true for an in-class
# member definition, false for an OUT-OF-LINE one, which both retired
# scanners caught and the shipped AST mechanism silently missed. The pre-fix
# mechanism (FunctionDecl-only kind filter, no lambda guard) is reproduced
# verbatim here, exactly as _old_find_leaf_forwarding_doors_for_comparison and
# _hardened_text_scan_find_leaf_forwarding_doors_for_comparison preserve the
# two mechanisms one and two generations further back, so this suite can
# independently confirm each new population member really was a miss under
# THIS module's own shipped code at `5eff945`, not merely asserted from the
# casebook.
# ---------------------------------------------------------------------------


def _pre_t1383_find_leaf_forwarding_doors_for_comparison(
    path,
    leaves=cnfl.BANNED_LEAVES,
    exclude=cnfl._FUNNEL_ENTRY_POINTS,
    clangxx=cnfl._DEFAULT_CLANGXX,
    include_dir=cnfl._INCLUDE_DIR,
):
    """The AS-SHIPPED-AT-`5eff945` mechanism, reproduced verbatim: kind
    filter admits `FunctionDecl` only, and the walk carries no lambda guard
    (both correspondingly widened/added by T-1383 in the module under test).
    Reproduced rather than imported, since the module under test no longer
    carries this shape, mirroring how this file already preserves the
    pre-T-1378 regex scanner and the T-1378 hardened text scanner one and two
    generations further back."""
    old_kinds = ("FunctionDecl",)
    root = cnfl._run_clang_ast_dump(path, clangxx, include_dir)
    target = os.path.normpath(os.path.abspath(path)).replace("\\", "/")
    state = {"file": None}
    doors: list[str] = []

    def walk(n):
        if not isinstance(n, dict):
            return
        loc = n.get("loc") or {}
        if "file" in loc:
            state["file"] = loc["file"]
        cur_file = str(state["file"] or "").replace("\\", "/")
        if (
            n.get("kind") in old_kinds
            and cur_file
            and os.path.normpath(os.path.abspath(cur_file)).replace("\\", "/") == target
        ):
            name = n.get("name")
            body = next(
                (c for c in (n.get("inner") or []) if isinstance(c, dict) and c.get("kind") == "CompoundStmt"),
                None,
            )
            if body is not None and name not in exclude and name not in doors and cnfl._body_calls_leaf(body, leaves):
                doors.append(name)
        for c in n.get("inner") or []:
            walk(c)

    walk(root)
    return doors


# Two population members: the out-of-line member door Significant 1 named
# directly, plus an in-class member door found by this pass's own wider
# sweep (StandardsDocument Sec4: cast a wider net than the one shape strictly
# required). Both surface as the identical `CXXMethodDecl` AST kind
# (confirmed by execution), so the single kind-filter widening in
# `check_no_forward_leaf_calls.py` fixes both at once.
_T1383_NEW_SHAPES = {
    "out_of_line_member_door": (
        _BASE_FUNNEL_PREFIX
        + "struct ScaleDoors { static int64_t CarriedScaleDoorTwo(int64_t d); };\n"
        "int64_t ScaleDoors::CarriedScaleDoorTwo(int64_t d) { return DynamicScaleReciprocal(d); }\n"
    ),
    "in_class_member_door": (
        _BASE_FUNNEL_PREFIX
        + "struct ScaleDoorsInClass {\n"
        "    static int64_t CarriedScaleDoorTwo(int64_t d) { return DynamicScaleReciprocal(d); }\n"
        "};\n"
    ),
}


@requires_clang
def test_t1383_new_shapes_were_misses_under_the_5eff945_shipped_mechanism():
    """Independently confirms each new population member really was invisible
    to the mechanism as shipped at `5eff945` -- StandardsDocument Sec4's
    population-validation requirement, applied one generation further: a
    check shown able to fail on the fault it exists to catch, not merely
    asserted to. Requires clang (unlike this file's two prior-generation
    comparison harnesses, both pure text scans): the mechanism being
    reproduced here for comparison is ITSELF Clang-AST-derived, since T-1381
    is what T-1383 corrects."""
    for case_name, content in _T1383_NEW_SHAPES.items():
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "src/forward/checked_chain_funnel.cpp", content)
            old_doors = sorted(_pre_t1383_find_leaf_forwarding_doors_for_comparison(path))
            assert old_doors == ["CarriedScaleReciprocal"], (
                f"case {case_name!r}: expected the 5eff945-shipped mechanism to MISS "
                f"the second door (finding only CarriedScaleReciprocal), got "
                f"{old_doors} -- this case is not a valid population member if the "
                f"shipped mechanism already caught it"
            )


@requires_clang
def test_t1383_ast_mechanism_catches_the_new_shapes_too():
    """The fix's own proof: both the out-of-line and in-class member-function
    doors are caught by the current cnfl.find_leaf_forwarding_doors, by name --
    measured against the SAME reproduced-verbatim 5eff945 mechanism
    immediately above, not merely asserted (StandardsDocument Sec4)."""
    for case_name, content in _T1383_NEW_SHAPES.items():
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "src/forward/checked_chain_funnel.cpp", content)
            new_doors = sorted(cnfl.find_leaf_forwarding_doors(path))
            assert new_doors == ["CarriedScaleDoorTwo", "CarriedScaleReciprocal"], (
                f"case {case_name!r}: expected the fixed mechanism to catch both "
                f"doors, got {new_doors}"
            )


@requires_clang
def test_t1383_check_door_count_fails_on_every_new_shape_too():
    """The wiring cell for the two new shapes: check_door_count, the function
    the CI gate actually calls, must fail on each -- not merely
    find_leaf_forwarding_doors in isolation."""
    for case_name, content in _T1383_NEW_SHAPES.items():
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "src/forward/checked_chain_funnel.cpp", content)
            failures = cnfl.check_door_count(path, expected=("CarriedScaleReciprocal",))
            assert len(failures) == 1 and "CarriedScaleDoorTwo" in failures[0], (
                f"case {case_name!r}: expected check_door_count to name the new "
                f"door, got {failures}"
            )


@requires_clang
def test_t1383_lambda_regression_control_still_reports_exactly_one_door():
    """The regression this fix must not reintroduce: widening the kind filter
    from `FunctionDecl` alone to include member-function kinds must not make
    a lambda's own closure-type call operator (a `CXXMethodDecl`, lexically
    nested inside a `LambdaExpr`) count as a SECOND door alongside the
    function it is defined inside. Re-runs
    test_t1381_ast_mechanism_finds_a_door_hidden_inside_a_lambda_in_real_
    declared_code's exact construction and asserts the count stays at two
    names total (the pre-existing real door plus the lambda-hiding one), not
    three."""
    real_funnel = os.path.join(cnfl._REPO_ROOT, "src/forward/checked_chain_funnel.cpp")
    with open(real_funnel, "r", encoding="utf-8") as f:
        real_content = f.read()
    lambda_door = (
        "\nnamespace superslm {\n"
        "int64_t CarriedScaleDoorLambda(int64_t d) {\n"
        "    auto inner = [d]() { return DynamicScaleReciprocal(d); };\n"
        "    return inner();\n"
        "}\n"
        "}  // namespace superslm\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "checked_chain_funnel_with_lambda_door.cpp", real_content + lambda_door)
        doors = sorted(cnfl.find_leaf_forwarding_doors(path))
        assert doors == ["CarriedScaleDoorLambda", "CarriedScaleReciprocal"], (
            f"expected exactly two doors (no spurious 'operator()' entry from "
            f"the lambda's own closure type), got {doors}"
        )


# ---------------------------------------------------------------------------
# T-1386 (Poirot 6f575df-t1383-t1384-t1385-review-2026-07-31.md, Significant
# 1): T-1383's lambda guard suppresses a nested candidate only when it is
# lexically inside a `LambdaExpr` -- named by AST kind. A local class (a
# `CXXRecordDecl`) declared inside a function body is the identical shape
# (a function-like body nested inside an already-counted enclosing door's
# body) under a different keyword, and the T-1383 guard does not recognize
# it: the local class's own member function surfaces as a THIRD, spurious
# door alongside the enclosing function and the pre-existing control door.
# The member is unreachable outside the enclosing function and its leaf call
# is already accounted for by the enclosing door's own `_body_calls_leaf`
# walk, which has no regard to nesting. This is the regression the routed
# remedy closes by carrying "inside any already-reported candidate" rather
# than "inside a LambdaExpr" down the walk.
#
# T-1427 (Poirot 9b0f938-t1411-t1415-t1416-t1386-t1388-confirmation-2026-07-31.md
# Minor 2): T-1386 landed with no population-validation structure of its own --
# neither a reproduction of the mechanism it replaced (this file's own
# established idiom, `_old_.../_hardened_text_scan_.../_pre_t1383_...`) nor an
# entry in the population-count total below. `_pre_t1386_find_leaf_forwarding_
# doors_for_comparison` and `_T1386_NEW_SHAPES` close that gap the same way
# the T-1383 section above does.
# ---------------------------------------------------------------------------


def _pre_t1386_find_leaf_forwarding_doors_for_comparison(
    path,
    leaves=cnfl.BANNED_LEAVES,
    exclude=cnfl._FUNNEL_ENTRY_POINTS,
    clangxx=cnfl._DEFAULT_CLANGXX,
    include_dir=cnfl._INCLUDE_DIR,
):
    """The AS-SHIPPED-AT-`6f575df` mechanism, reproduced verbatim: the nesting
    guard is `in_lambda`, set only by AST kind `LambdaExpr` (widened by
    T-1386, in the module under test, to `nested_in_candidate`, set by the
    walk's own `is_candidate` predicate so any function-like body nested
    inside another candidate is suppressed, not only a lambda's closure
    operator). Reproduced rather than imported, since the module under test
    no longer carries this shape, mirroring how this file already preserves
    the two mechanism generations further back
    (`_pre_t1383_find_leaf_forwarding_doors_for_comparison` and older)."""
    root = cnfl._run_clang_ast_dump(path, clangxx, include_dir)
    target = os.path.normpath(os.path.abspath(path)).replace("\\", "/")
    state = {"file": None}
    doors: list[str] = []

    def walk(n: object, in_lambda: bool) -> None:
        if not isinstance(n, dict):
            return
        kind = n.get("kind")
        loc = n.get("loc") or {}
        if "file" in loc:
            state["file"] = loc["file"]
        cur_file = str(state["file"] or "").replace("\\", "/")
        if (
            not in_lambda
            and kind in cnfl._TOP_LEVEL_FUNCTION_KINDS
            and cur_file
            and os.path.normpath(os.path.abspath(cur_file)).replace("\\", "/") == target
        ):
            name = n.get("name")
            body = next(
                (c for c in (n.get("inner") or []) if isinstance(c, dict) and c.get("kind") == "CompoundStmt"),
                None,
            )
            if body is not None and name not in exclude and name not in doors and cnfl._body_calls_leaf(body, leaves):
                doors.append(name)
        child_in_lambda = in_lambda or kind == "LambdaExpr"
        for c in n.get("inner") or []:
            walk(c, child_in_lambda)

    walk(root, False)
    return doors


_LOCAL_CLASS_IN_FUNCTION_BODY = (
    _BASE_FUNNEL_PREFIX
    + "int64_t OuterDoor(int64_t d) {\n"
    "    struct Helper { static int64_t Go(int64_t x) { return DynamicScaleReciprocal(x); } };\n"
    "    return Helper::Go(d);\n"
    "}\n"
)

# The one population member T-1386 adds via this file's established
# include-free-fixture convention (`_BASE_FUNNEL_PREFIX`, matching
# `_T1381_NEW_SHAPES`/`_T1383_NEW_SHAPES`'s own shape). The declared-code
# variant of the identical shape is confirmed separately, against real
# declared code, immediately below -- it is a further-sweep confirmation
# that the same shape is not an include-free-fixture artifact (this file's
# established convention, e.g. T-1381's own lambda-in-real-code sweep
# candidate), not a second dict member, since its content depends on a file
# read rather than being a static string.
_T1386_NEW_SHAPES = {
    "local_class_in_function_body": _LOCAL_CLASS_IN_FUNCTION_BODY,
}


@requires_clang
def test_t1386_new_shape_was_a_miss_under_the_pre_t1386_mechanism():
    """Independently confirms the new population member really was a miss
    (double-counted, not suppressed) under the mechanism as shipped at
    `6f575df` -- StandardsDocument Sec4's population-validation requirement,
    applied one generation further: a check shown able to fail on the fault
    it exists to catch, not merely asserted to."""
    for case_name, content in _T1386_NEW_SHAPES.items():
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "src/forward/checked_chain_funnel.cpp", content)
            old_doors = sorted(_pre_t1386_find_leaf_forwarding_doors_for_comparison(path))
            assert old_doors == ["CarriedScaleReciprocal", "Go", "OuterDoor"], (
                f"case {case_name!r}: expected the pre-T-1386 mechanism to report "
                f"THREE doors (the local class's member function double-counted "
                f"alongside the enclosing function), got {old_doors} -- this case "
                f"is not a valid population member if the pre-T-1386 mechanism "
                f"already suppressed it"
            )


@requires_clang
def test_t1386_local_class_member_inside_function_body_is_not_double_counted():
    """The regression, against an include-free scratch fixture in this
    suite's own idiom: a local class's member function must not surface as a
    door of its own alongside the enclosing function that declares it. Before
    the fix, this reports three names (`CarriedScaleReciprocal`, `Go`,
    `OuterDoor`); after it, exactly two."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "src/forward/checked_chain_funnel.cpp", _LOCAL_CLASS_IN_FUNCTION_BODY)
        doors = sorted(cnfl.find_leaf_forwarding_doors(path))
        assert doors == ["CarriedScaleReciprocal", "OuterDoor"], (
            f"expected exactly two doors (the local class's member function "
            f"'Go' is not itself a door -- it is unreachable outside "
            f"OuterDoor and its leaf call is already counted by OuterDoor's "
            f"own body), got {doors}"
        )


@requires_clang
def test_t1386_local_class_member_inside_function_body_is_not_double_counted_with_callee_declared():
    """The same regression against declared code (the callee is a real,
    resolved function rather than an `UnresolvedLookupExpr`), so the finding
    is not an artifact of the include-free fixture convention -- built on a
    copy of the real production file plus the same appended shape used
    above."""
    real_funnel = os.path.join(cnfl._REPO_ROOT, "src/forward/checked_chain_funnel.cpp")
    with open(real_funnel, "r", encoding="utf-8") as f:
        real_content = f.read()
    local_class_door = (
        "\nnamespace superslm {\n"
        "int64_t OuterDoorDeclared(int64_t d) {\n"
        "    struct HelperDeclared { static int64_t Go(int64_t x) { return DynamicScaleReciprocal(x); } };\n"
        "    return HelperDeclared::Go(d);\n"
        "}\n"
        "}  // namespace superslm\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "checked_chain_funnel_with_local_class.cpp", real_content + local_class_door)
        doors = sorted(cnfl.find_leaf_forwarding_doors(path))
        assert doors == ["CarriedScaleReciprocal", "OuterDoorDeclared"], (
            f"expected exactly two doors against declared code (no spurious "
            f"'Go' entry from the local class's member function), got {doors}"
        )


@requires_clang
def test_t1386_declared_code_shape_was_also_a_miss_under_the_pre_t1386_mechanism():
    """The declared-code counterpart of `test_t1386_new_shape_was_a_miss_
    under_the_pre_t1386_mechanism`: T-1427's population validation covers
    BOTH local-class shapes (Poirot 9b0f938-…-confirmation-2026-07-31.md
    Minor 2 names both), not only the include-free fixture -- confirming the
    miss is not an artifact of the include-free convention, the same
    distinction `test_t1386_local_class_member_inside_function_body_is_not_
    double_counted_with_callee_declared` draws for the fixed mechanism."""
    real_funnel = os.path.join(cnfl._REPO_ROOT, "src/forward/checked_chain_funnel.cpp")
    with open(real_funnel, "r", encoding="utf-8") as f:
        real_content = f.read()
    local_class_door = (
        "\nnamespace superslm {\n"
        "int64_t OuterDoorDeclared(int64_t d) {\n"
        "    struct HelperDeclared { static int64_t Go(int64_t x) { return DynamicScaleReciprocal(x); } };\n"
        "    return HelperDeclared::Go(d);\n"
        "}\n"
        "}  // namespace superslm\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "checked_chain_funnel_with_local_class.cpp", real_content + local_class_door)
        old_doors = sorted(_pre_t1386_find_leaf_forwarding_doors_for_comparison(path))
        assert old_doors == ["CarriedScaleReciprocal", "Go", "OuterDoorDeclared"], (
            f"expected the pre-T-1386 mechanism to report THREE doors against "
            f"declared code too (the local class's member function "
            f"double-counted alongside the enclosing function), got {old_doors}"
        )


@requires_clang
def test_t1386_local_class_inside_lambda_still_not_double_counted():
    """Regression control for the shape T-1383's own guard already handled
    correctly (Poirot's independent sweep, 'local class inside a lambda'):
    a local class declared inside a lambda's body must still not surface as
    a separate door once the walk carries the generalized 'inside any
    already-reported candidate' predicate rather than the LambdaExpr-named
    one."""
    with tempfile.TemporaryDirectory() as tmp:
        content = (
            _BASE_FUNNEL_PREFIX
            + "int64_t OuterDoor(int64_t d) {\n"
            "    auto inner = [d]() {\n"
            "        struct Helper { static int64_t Go(int64_t x) { return DynamicScaleReciprocal(x); } };\n"
            "        return Helper::Go(d);\n"
            "    };\n"
            "    return inner();\n"
            "}\n"
        )
        path = _write(tmp, "src/forward/checked_chain_funnel.cpp", content)
        doors = sorted(cnfl.find_leaf_forwarding_doors(path))
        assert doors == ["CarriedScaleReciprocal", "OuterDoor"], (
            f"expected exactly two doors (neither the lambda's closure "
            f"operator nor the local class's member function is a separate "
            f"door), got {doors}"
        )


# The documented parity case: a namespace-scope function pointer initialised
# to a leaf's address. Named by the review's own independent sweep as missed
# by both retired scanners AND the shipped AST mechanism identically -- not a
# regression, and not fixed by this pass (a `VarDecl` carries no
# `FunctionDecl`/`CXXMethodDecl` node for either generation of the mechanism
# to find; catching this shape is a different mechanism entirely, outside
# this ticket's scope). Executed here, both before and after the fix, so the
# claim is proven rather than merely asserted in a comment.
_FUNCTION_POINTER_TO_LEAF = (
    _BASE_FUNNEL_PREFIX
    + "int64_t (*CarriedScaleDoorPointer)(int64_t) = &DynamicScaleReciprocal;\n"
)


@requires_clang
def test_t1383_function_pointer_to_leaf_is_parity_not_regression():
    """Confirms the function-pointer shape is missed identically by the
    mechanism as shipped at `5eff945` and by the current (T-1383-fixed) one --
    parity, not a regression this pass introduces or is expected to close.
    Requires clang for the same reason as the miss-confirmation test above:
    the pre-T-1383 comparison harness is itself Clang-AST-derived."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "src/forward/checked_chain_funnel.cpp", _FUNCTION_POINTER_TO_LEAF)
        old_doors = sorted(_pre_t1383_find_leaf_forwarding_doors_for_comparison(path))
        assert old_doors == ["CarriedScaleReciprocal"], old_doors


@requires_clang
def test_t1383_function_pointer_to_leaf_is_still_missed_by_the_fixed_mechanism():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "src/forward/checked_chain_funnel.cpp", _FUNCTION_POINTER_TO_LEAF)
        new_doors = sorted(cnfl.find_leaf_forwarding_doors(path))
        assert new_doors == ["CarriedScaleReciprocal"], new_doors


#: The population containers this file expects `_t1_new_shape_containers` to
#: discover today, and the population containers this file expects to find by
#: name. Two independent nets now cover this population (T-1501, the union
#: D-SLM523 prescribes): `_t1_new_shape_containers` discovers by shape (a
#: non-empty module-level dict whose keys are all `str`), and
#: `_T1_CONTAINER_NAME_PATTERN` below discovers by name (any module-level
#: name matching the `_POPULATION_CASES`/`_T<digits>_NEW_SHAPES` convention,
#: whatever its type). Neither net alone was a superset of the other: a
#: container named exactly on the naming convention but whose values were not
#: all `str` -- a tuple, a nested dict, `bytes` -- escaped the value-type
#: predicate T-1484 shipped (T-1489 closed that), and a container named
#: exactly on the naming convention but shaped as a `list`, a `tuple`, or a
#: `dict` keyed by non-`str` values escapes the shape predicate above,
#: because it is absent from both the discovered set and the name-matched
#: set the assertions below compare (T-1501). `test_population_count_is_stated`
#: below asserts the shape-discovered set against this pin for equality, and
#: the name-matched set against the same pin for containment (`<=`, not
#: equality -- T-1506): a container that stops being discovered by shape, or
#: any module-level name that starts matching the name pattern without being
#: in this pin, turns the cell red instead of silently changing -- or
#: failing to change -- the total; a container registered here under a name
#: the pattern does not match is not required to appear in the name-matched
#: set, so registration does not depend on the naming convention. Registering
#: a population container is therefore two edits: define the dict, and add
#: its name here -- for any name, on-convention or not (T-1506; the equality
#: form of the second assertion briefly made this false for off-convention
#: names, measured by mutations O and P in Poirot
#: 066319d-t1501-t1503-confirmation-2026-07-31.md Significant 1).
_T1_EXPECTED_CONTAINER_NAMES = frozenset(
    {
        "_POPULATION_CASES",
        "_T1381_NEW_SHAPES",
        "_T1383_NEW_SHAPES",
        "_T1386_NEW_SHAPES",
    }
)

#: The name half of the union: any module-level name matching this pattern is
#: a claimed population container regardless of its type or contents. Paired
#: with `_t1_new_shape_containers`'s shape-based discovery in
#: `test_population_count_is_stated`, this closes the direction the shape
#: predicate cannot reach by construction -- an on-convention container that
#: is a `list`, a `tuple`, or a `dict` keyed by non-`str` values is invisible
#: to shape-based discovery (it is never a `dict` with all-`str` keys) but
#: matches this pattern, so it appears in the name-matched set and is absent
#: from `_T1_EXPECTED_CONTAINER_NAMES`, which fails the assertion below
#: (T-1501; measured against mutations constructed on-convention as a `list`,
#: a `tuple`, and an `int`-keyed `dict`).
_T1_CONTAINER_NAME_PATTERN = re.compile(r"^(_POPULATION_CASES|_T\d+_NEW_SHAPES)$")


def _t1_discover_population_containers(namespace):
    """The shape half of the union, factored out of `_t1_new_shape_containers`
    (T-1514) so the discovery predicate can be run against an arbitrary
    namespace mapping (name -> value), not only against the real module's
    `globals()`. `_t1_new_shape_containers` below is this function applied to
    `globals()`; `_t1_check_population_registration` below is the
    registration contract this file states in four places, applied to a
    caller-supplied namespace and pin. Both are used by
    `test_population_count_is_stated` against the real module, and by the
    T-1514 cells following it against constructed namespaces -- so those
    cells pin this predicate directly rather than a reimplementation of it.
    See `_t1_new_shape_containers`'s own docstring for what "shape" means and
    why."""
    return {
        name: value
        for name, value in namespace.items()
        if not name.startswith("__")
        and isinstance(value, dict)
        and value
        and all(isinstance(k, str) for k in value)
    }


def _t1_new_shape_containers():
    """Every module-level dict this file uses to enumerate a batch of doors a
    ticket adds to the population, discovered by shape -- a non-empty
    module-level dict whose keys are all `str` (every population container
    maps a case name to its expected contribution) -- rather than by matching
    the container's own name against a naming convention, or by constraining
    what its values are (T-1489: the values need not be `str` -- the obvious
    next population shape pairs a case's source with its expected doors as a
    tuple, which a value-type constraint would silently exclude). This
    function's own net does not reach an on-convention container shaped as a
    `list`, a `tuple`, or a `dict` keyed by non-`str` values -- that
    direction is closed by `_T1_CONTAINER_NAME_PATTERN`'s name-based check in
    `test_population_count_is_stated` instead, not by this function (T-1501:
    the union D-SLM523 prescribes, not a single wider predicate). Registering
    a new container follows the two-edit rule stated in the comment above
    `_T1_EXPECTED_CONTAINER_NAMES`. Defining the dict alone still reaches
    `test_population_count_is_stated` below through shape-based discovery
    when the container is a non-empty `str`-keyed dict, but its name is then
    missing from the pin, which fails that test's shape-set assertion until
    the second edit is made (T-1467, T-1484, T-1489, T-1501, T-1506; Poirot
    20f11c1-dslm497-t1425-t1430-fold-confirmation-2026-07-31.md Minor 4;
    82ff942-t1463-t1468-confirmation-2026-07-31.md Minor 3;
    2126ab1-t1482-t1484-confirmation-2026-07-31.md Significant 1;
    75692c0-t1489-t1492-confirmation-2026-07-31.md Significant 1, Minor 1;
    066319d-t1501-t1503-confirmation-2026-07-31.md Significant 1, Minor 1).
    This is what T-1386 needed and did not have: it added a new container
    outside the hand-written sum below, and the stated total did not move.
    Neither this function's shape check nor `_T1_CONTAINER_NAME_PATTERN`'s
    name check alone gates discovery completely; run together in
    `test_population_count_is_stated`, they close the on-convention and the
    off-convention escapes measured against three prior predicates. A
    container that is both off-convention in name and outside this
    function's structural shape -- an off-convention `list`, for instance --
    is caught by neither check; see that test's own docstring. Delegates to
    `_t1_discover_population_containers(globals())` (T-1514) -- the shape
    predicate itself is unchanged, only where it reads its namespace from."""
    return _t1_discover_population_containers(globals())


def _t1_name_matched_containers(namespace):
    """The name half of the union, factored out for the same reason as
    `_t1_discover_population_containers` (T-1514): every name in an
    arbitrary namespace mapping matching `_T1_CONTAINER_NAME_PATTERN`,
    regardless of the type or contents of the value it is bound to. See
    `_T1_CONTAINER_NAME_PATTERN`'s own comment for what the pattern is and
    why."""
    return {name for name in namespace if _T1_CONTAINER_NAME_PATTERN.match(name)}


def _t1_check_population_registration(namespace, pin):
    """The population-registration contract this file states in four places
    -- the module pin comment above `_T1_EXPECTED_CONTAINER_NAMES`,
    `_T1_CONTAINER_NAME_PATTERN`'s own comment, `_t1_new_shape_containers`'s
    docstring, and this function's caller `test_population_count_is_stated`
    -- pinned once, here, rather than left to those four statements alone
    (T-1514; Poirot 03a9413-t1506-t1507-confirmation-2026-07-31.md
    Significant 1). Two assertions, run in this order against `namespace`:
    the shape-discovered name set must equal `pin` exactly; the name-matched
    set must be a subset of `pin`. `test_population_count_is_stated` below
    calls this with `namespace=globals()` and `pin=_T1_EXPECTED_CONTAINER_
    NAMES`, which is the real cell; the T-1514 tests immediately following
    it call this same function against constructed namespaces built from the
    mutation battery in the Poirot casebook above, so those cells pin the
    real contract rather than a copy of it. Returns the discovered
    containers dict so a caller that also wants the population total (as
    `test_population_count_is_stated` does) does not discover twice."""
    containers = _t1_discover_population_containers(namespace)
    discovered_names = set(containers.keys())
    assert discovered_names == pin, (
        discovered_names - pin,
        pin - discovered_names,
    )
    name_matched = _t1_name_matched_containers(namespace)
    assert name_matched <= pin, (name_matched - pin)
    return containers


def test_population_count_is_stated():
    """States, in one place, the total accumulated population this campaign's
    own exit condition requires: the eleven shapes T-1381 already swept
    (nine required cases plus two further sweep candidates, all still
    re-verified against the T-1383-fixed mechanism above and by the existing
    T-1381 cells, none regressed) plus the two new shapes T-1383 adds to the
    test population (out-of-line member door, in-class member door) plus the
    one further shape T-1383's own sweep found and documents as parity
    rather than fixing (the function-pointer case) plus the one new shape
    T-1386 adds (local class member inside a function body) plus its
    declared-code further-sweep confirmation -- sixteen shapes total, at
    least, per StandardsDocument Sec4's population-validation requirement.

    `container_total` is derived, not hand-summed: `_t1_new_shape_containers`
    above discovers every population container by shape (a non-empty
    module-level dict keyed by `str`, values unconstrained), so a new
    container of that shape is caught automatically regardless of what it is
    named (T-1467, T-1484, T-1489, T-1506). Two independent assertions guard
    that discovery before the sum is taken, per D-SLM523's union (T-1501):
    the shape-discovered name set against `_T1_EXPECTED_CONTAINER_NAMES` for
    equality, so a container that stops being a non-empty `str`-keyed dict --
    or an unrelated module-level dict that starts being one -- fails this
    cell rather than silently changing, or failing to change, the total; and
    the name-matched set (every module-level name matching
    `_T1_CONTAINER_NAME_PATTERN`) against the same pin for containment
    (`<=`, not equality -- T-1506), so an on-convention container that is a
    `list`, a `tuple`, or a `dict` keyed by non-`str` values -- invisible to
    the shape check because it is never a `str`-keyed dict -- fails this
    cell too, because its name is new and unlisted, while a container
    registered under a name the pattern does not match is never required to
    appear in this set and so does not fail it (T-1501, T-1506; measured
    against mutations constructed on-convention as a `list`, a `tuple`, and
    an `int`-keyed `dict`, each escaping the shape check alone and each
    caught by the name check; and against mutations that register an
    off-convention container by the two-edit rule alone, which require no
    third edit to `_T1_CONTAINER_NAME_PATTERN` or to the container's own
    name -- T-1515: the stated total is still a third edit when the
    container is non-empty, on-convention or off, which is symmetric and not
    a defect the "two edits" framing ever claimed to remove; see mutations
    Y1 and Y2 in Poirot 03a9413-t1506-t1507-confirmation-2026-07-31.md).
    Both assertions are `_t1_check_population_registration`'s (T-1514); the
    T-1514 cells below pin the two assertions' behaviour directly against
    constructed namespaces, including the residual direction above -- a
    name the pattern does not match, bound to a value the shape check does
    not reach, is admitted by neither assertion (T-1517; mutations Q, R, S,
    T; a widening of `_T1_CONTAINER_NAME_PATTERN` that starts admitting one
    of those names turns the corresponding cell red) -- rather than leaving
    any of it to be re-verified in prose every round.
    `further_sweep_total` is NOT derived -- it sums four literals for five
    documented one-off cases that were never stored in a container, so
    neither check above can see them: T-1381's `+ 1` string-literal control
    (one case), T-1381's further-sweep `+ 2` (two cases -- lambda-in-real-code
    and shadowing-variable, both in the same literal), T-1383's `+ 1`
    function-pointer case, and T-1386's `+ 1` declared-code confirmation. A
    new one-off case added to this list without also incrementing
    `further_sweep_total` below is a shape this structure does not catch --
    inherent to a hand-summed literal bucket rather than a gap in container
    discovery, and not closed by T-1484's, T-1489's, or T-1501's fixes
    (documented rather than closed, per T-1467's remedy options). A second,
    independent shape also escapes both checks above: a container that is
    off-convention in its name and is not a non-empty module-level dict
    keyed by `str` -- an off-convention `list` or `tuple`, for instance --
    matches neither the shape predicate nor the name pattern, because it is
    absent from both sets the two assertions compare (T-1501; Poirot
    20f11c1-dslm497-t1425-t1430-fold-confirmation-2026-07-31.md Minor 4;
    82ff942-t1463-t1468-confirmation-2026-07-31.md Minor 2, Minor 3;
    2126ab1-t1482-t1484-confirmation-2026-07-31.md Significant 1;
    75692c0-t1489-t1492-confirmation-2026-07-31.md Significant 1). This
    escape stays uncaught by design -- it is not a claim the two assertions
    enforce -- but the T-1517 cell following `_t1_failing_assertion` below
    pins its behaviour under `_T1_CONTAINER_NAME_PATTERN` as shipped, so a
    widening of the pattern that starts admitting one of its fixture names
    turns that cell red instead of leaving the escape's stability to be
    re-verified in prose every round."""
    containers = _t1_check_population_registration(
        globals(), _T1_EXPECTED_CONTAINER_NAMES
    )
    container_total = sum(len(value) for value in containers.values())
    further_sweep_total = (
        1  # T-1381: +1 string-literal control
        + 2  # T-1381: lambda-in-real-code, shadowing variable (both documented parity)
        + 1  # T-1383: function pointer to leaf (documented parity, this test file)
        + 1  # T-1386: declared-code confirmation of the same shape (documented above)
    )
    total = container_total + further_sweep_total
    assert total == 16, (containers.keys(), container_total, further_sweep_total, total)


# --- T-1514: the population-registration contract, pinned directly against
# constructed namespaces, rather than left to the four prose statements
# above alone (the module pin comment, `_T1_CONTAINER_NAME_PATTERN`'s
# comment, `_t1_new_shape_containers`'s docstring, and
# `test_population_count_is_stated`'s own docstring). Every cell below
# exercises `_t1_check_population_registration` -- the same function
# `test_population_count_is_stated` calls above against the real module --
# so a cell here failing to fail is a cell exercising the real contract, not
# a reimplementation of it. Each cell reproduces one row of the mutation
# battery in Poirot 03a9413-t1506-t1507-confirmation-2026-07-31.md and
# 066319d-t1501-t1503-confirmation-2026-07-31.md, named in its own
# docstring; `Claude/Brunel/t1514-t1515-build-2026-07-31.md` reconciles
# every mutation in both battery tables (A-N, O, P, Q-T, W1-W5, X, O2, Y1,
# Y2) against the cell that exercises it. `Claude/Brunel/t1517-t1518-build-
# 2026-07-31.md` adds the four cells (Q, R, S, T) that close the residual
# the first build left as prose only (Poirot
# da69def-t1514-t1515-confirmation-2026-07-31.md Significant 1, T-1517);
# Y1 and Y2 remain outside this contract, pinned instead by the
# `total == 16` arithmetic assertion above, unchanged by either build. ---


def _t1_failing_assertion(excinfo):
    """Distinguishes which of `_t1_check_population_registration`'s two
    assertions raised, from the raised frame's own source statement rather
    than from surrounding context or from the exception's message -- pytest
    rewrites both assert statements to fold the tuple/set message into a
    single formatted string on failure (confirmed by execution: the
    original 2-tuple and single-set message shapes are not preserved on
    `AssertionError.args` once pytest's assertion-rewrite import hook has
    processed this module), so distinguishing by message shape is not
    reliable and the source line is used instead -- the same discipline
    Poirot 03a9413-t1506-t1507-confirmation-2026-07-31.md states: "the
    failing assertion is read from pytest's `>` source-line marker, not
    from surrounding context". Matches on each assertion's subject name
    alone (`discovered_names`, `name_matched`), not on the full statement
    text including the comparison operator (T-1518; Poirot
    da69def-t1514-t1515-confirmation-2026-07-31.md Minor 1): the operator is
    the part of these statements most likely to be edited -- T-1506 changed
    it once already -- and matching the whole expression means an operator
    edit alone, with the subject and the property it breaks unchanged,
    misattributes the failure to every cell that reaches the edited
    assertion, rather than to the one cell whose property the edit actually
    breaks (T-1542; Poirot 8762a30-t1517-t1518-confirmation-2026-07-31.md
    Minor 2 -- a namespace in this file is a mapping of names to values and
    contains no statement text; what actually misattributes is that every
    cell reaching the edited assertion asks this function which one raised
    and gets `unrecognized failing statement` once the source no longer
    matches the literal). Verified against the exact T-1506
    regression (`name_matched <= pin` reverted to `== pin`): matching the
    full expression reports 5 red cells, 4 of them spurious
    `unrecognized failing statement` failures; matching the subject name
    alone reports exactly the 1 cell whose property the regression breaks."""
    statement = str(excinfo.traceback[-1].statement)
    if "discovered_names" in statement:
        return "shape-set"
    if "name_matched" in statement:
        return "name-set"
    raise AssertionError(f"unrecognized failing statement: {statement!r}")


def test_registration_admits_an_off_convention_container_present_in_the_pin():
    """Pins mutations O and P (Poirot 03a9413 Significant 1, Execution
    evidence): an off-convention non-empty `str`-keyed dict, named in the
    pin, passes both assertions -- the property T-1506 restored, and the one
    T-1507's docstring states in the "for any name, on-convention or not"
    clause."""
    namespace = {"_CONTROL_CASES": {"case": "value"}}
    pin = frozenset({"_CONTROL_CASES"})
    _t1_check_population_registration(namespace, pin)  # must not raise


@pytest.mark.parametrize(
    "value",
    [[], (), {1: "x"}, {}],
    ids=["list", "tuple", "int-keyed-dict", "empty-dict"],
)
def test_registration_rejects_an_on_convention_container_of_the_wrong_shape(value):
    """Pins mutations H, J, K, L (Poirot 066319d Significant 1; reproduced
    unchanged in 03a9413's Execution evidence): an on-convention name --
    matches `_T1_CONTAINER_NAME_PATTERN` -- bound to a value that is not a
    non-empty `str`-keyed dict is invisible to shape-based discovery, so it
    is absent from the pin and the name-set assertion fires -- not the
    shape-set assertion, which is what closes the direction
    `_t1_new_shape_containers` cannot reach by construction."""
    namespace = {"_T9999_NEW_SHAPES": value}
    pin = frozenset()
    with pytest.raises(AssertionError) as excinfo:
        _t1_check_population_registration(namespace, pin)
    assert _t1_failing_assertion(excinfo) == "name-set"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("_ORPHAN_CASES", {"case": "value"}),
        ("_T9999_NEW_SHAPES", {"case": "value"}),
        ("_T9999_NEW_SHAPES", {"case": ("tuple", "value")}),
        ("_T9999_NEW_SHAPES", {"case": {"nested": "dict"}}),
        ("_T9999_NEW_SHAPES", {"case": b"bytes"}),
    ],
    ids=[
        "off-convention-name",
        "on-convention-name",
        "on-convention-name-tuple-values",
        "on-convention-name-nested-dict-values",
        "on-convention-name-bytes-values",
    ],
)
def test_registration_rejects_a_dict_absent_from_the_pin(name, value):
    """Pins mutations A, C, D, G (Poirot 066319d/03a9413 A-N battery) and E,
    F, I (the value-type independence T-1489 established): a non-empty
    `str`-keyed dict is discovered by shape regardless of its name's
    convention or its values' types, so failing to name it in the pin fails
    the shape-set assertion the same way in every case -- shape-based
    discovery does not depend on the naming convention or on what the
    values are, only on the keys being `str` and the dict being non-empty."""
    namespace = {name: value}
    pin = frozenset()
    with pytest.raises(AssertionError) as excinfo:
        _t1_check_population_registration(namespace, pin)
    assert _t1_failing_assertion(excinfo) == "shape-set"


@pytest.mark.parametrize(
    ("pin_name", "namespace"),
    [
        ("_T9999_NEW_SHAPES", {}),
        ("_PHANTOM_CASES", {}),
        ("_T9999_NEW_SHAPES", {"_T9999_NEW_SHAPES": []}),
        ("_T9999_NEW_SHAPES", {"_T9999_NEW_SHAPES": {}}),
        ("_T9999_NEW_SHAPES", {"_T9999_NEW_SHAPES": {1: "x"}}),
        ("_PHANTOM_CASES", {"_PHANTOM_CASES": []}),
        ("_PHANTOM_CASES", {"_PHANTOM_CASES": {}}),
        ("_PHANTOM_CASES", {"_PHANTOM_CASES": {1: "x"}}),
        ("_T9999_NEW_SHAPES", {"_T9999_RENAMED_SHAPES": {"case": "value"}}),
    ],
    ids=[
        "W1-undefined-on-convention",
        "W2-undefined-off-convention",
        "X-on-convention-bound-to-list",
        "N-analog-on-convention-bound-to-empty-dict",
        "M-on-convention-bound-to-int-keyed-dict",
        "W3-off-convention-bound-to-list",
        "W4-off-convention-bound-to-empty-dict",
        "W5-off-convention-bound-to-int-keyed-dict",
        "O2-pin-stale-after-off-convention-rename",
    ],
)
def test_registration_rejects_a_pinned_name_absent_from_the_namespace(pin_name, namespace):
    """Pins mutations W1-W5, X, M, N and O2 (Poirot 066319d/03a9413, "the
    removed direction, probed to exhaustion" and the A-N battery's shape
    mutations): a name present in the pin but not discovered by shape in
    the namespace -- because nothing is bound to it (W1, W2), because it is
    bound to something other than a non-empty `str`-keyed dict (M, N, W3,
    W4, W5, X), or because the container was renamed away from the pinned
    name without updating the pin (O2) -- fails the shape-set assertion
    before the name-set assertion is ever reached, whatever the pin's own
    name looks like and whatever the namespace's actual container is
    named."""
    pin = frozenset({pin_name})
    with pytest.raises(AssertionError) as excinfo:
        _t1_check_population_registration(namespace, pin)
    assert _t1_failing_assertion(excinfo) == "shape-set"


@pytest.mark.parametrize(
    "value",
    [[], (), {}, {1: "x"}],
    ids=[
        "Q-off-convention-list",
        "R-off-convention-tuple",
        "S-off-convention-empty-dict",
        "T-off-convention-int-keyed-dict",
    ],
)
def test_registration_admits_the_documented_off_convention_wrong_shape_residual(value):
    """Pins mutations Q, R, S, T (the file's own documented residual;
    `Claude/Brunel/t1514-t1515-build-2026-07-31.md` "What was not
    pinnable"; Poirot da69def-t1514-t1515-confirmation-2026-07-31.md
    Significant 1, T-1517): a name that does not match
    `_T1_CONTAINER_NAME_PATTERN` as shipped, bound to a value that is not a
    non-empty `str`-keyed dict, is invisible to both assertions -- absent
    from the shape-discovered set because of its shape, absent from the
    name-matched set because of its name -- which is the residual
    `test_population_count_is_stated`'s own docstring names and this cell
    now pins directly rather than leaving it to be inferred from every
    other cell's contrast.

    This is the one direction in this file that exercises
    `_T1_CONTAINER_NAME_PATTERN`'s admission side through the population-
    registration contract itself: every other cell here supplies a name the
    pattern already matches, or an off-convention name. Each fixture
    family's own convention -- on or off -- is asserted directly against the
    pattern by `test_fixture_family_convention_matches_its_stated_convention`
    below (T-1540; Poirot 8762a30-t1517-t1518-confirmation-2026-07-31.md
    Significant 1), rather than left to this cell's single fixture name or
    to prose. `_ORPHAN_CASES` is chosen for this cell because it does not
    match the pattern as shipped but does match both a broad widening
    (`^_[A-Z0-9_]*(CASES|SHAPES)$`) and a plausible convention extension
    (adding a `_[A-Z]+_CASES` alternative) -- measured directly: widening
    `_T1_CONTAINER_NAME_PATTERN` either way turns this cell red, because
    the name then joins the name-matched set while remaining absent from
    the pin; narrowing the pattern leaves it green, because this class was
    never reached by shape. No cell before this one changes verdict under a
    widening of the pattern; four change verdict under a narrowing --
    `test_registration_rejects_an_on_convention_container_of_the_wrong_shape`'s
    four cases (T-1541; Poirot 8762a30-t1517-t1518-confirmation-2026-07-31.md
    Minor 1), measured in `Claude/Brunel/t1517-t1518-build-2026-07-31.md`'s
    own verification table."""
    namespace = {"_ORPHAN_CASES": value}
    pin = frozenset()
    _t1_check_population_registration(namespace, pin)  # must not raise


@pytest.mark.parametrize(
    ("name", "on_convention"),
    [
        ("_T9999_NEW_SHAPES", True),
        ("_ORPHAN_CASES", False),
        ("_CONTROL_CASES", False),
        ("_PHANTOM_CASES", False),
        ("_T9999_RENAMED_SHAPES", False),
    ],
    ids=[
        "T9999_NEW_SHAPES-on-convention",
        "ORPHAN_CASES-off-convention",
        "CONTROL_CASES-off-convention",
        "PHANTOM_CASES-off-convention",
        "T9999_RENAMED_SHAPES-off-convention",
    ],
)
def test_fixture_family_convention_matches_its_stated_convention(name, on_convention):
    """Asserts each fixture family used elsewhere in this section against
    `_T1_CONTAINER_NAME_PATTERN` directly, rather than leaving the
    convention-ness of every fixture but `_ORPHAN_CASES` to prose (T-1540;
    Poirot 8762a30-t1517-t1518-confirmation-2026-07-31.md Significant 1 --
    the un-built third clause of the T-1517 ticket in
    `Claude/Poirot/da69def-t1514-t1515-confirmation-2026-07-31.md`, which
    asked for exactly this: "assert each fixture's convention against the
    pattern directly, so a cell whose docstring says 'off-convention' fails
    when its name stops being off-convention"). `_T9999_NEW_SHAPES` is the
    on-convention fixture used throughout this file's T-1514 cells above.
    `_ORPHAN_CASES` is
    `test_registration_admits_the_documented_off_convention_wrong_shape_
    residual`'s fixture, above. `_CONTROL_CASES` is
    `test_registration_admits_an_off_convention_container_present_in_the_
    pin`'s fixture. `_PHANTOM_CASES` is the W3, W4, W5 cases and
    `_T9999_RENAMED_SHAPES` is the O2 case, both in
    `test_registration_rejects_a_pinned_name_absent_from_the_namespace`
    above -- every one of those cells' own id or docstring calls its
    fixture off-convention, and none of those cells is sensitive to a
    widening of the pattern that starts admitting its own fixture's name,
    because none of them exercises the pattern's admission side (only
    `test_registration_admits_the_documented_off_convention_wrong_shape_
    residual` does, and only for `_ORPHAN_CASES`). This cell closes that:
    a widening that starts admitting any of the four off-convention names
    turns this cell red directly, independent of whether that widening also
    changes any other cell's verdict.

    Measured against all four widenings named or derived for this ticket,
    each applied by literal substitution to the real
    `_T1_CONTAINER_NAME_PATTERN` line, the file's full suite run, and the
    file restored and SHA-256-verified afterward: the three
    `Claude/Poirot/8762a30-t1517-t1518-confirmation-2026-07-31.md`
    Significant 1 names -- `^(_POPULATION_CASES|_T\\d+_(NEW|RENAMED)_
    SHAPES)$`, `^(_POPULATION_CASES|_T\\d+_NEW_SHAPES|_CONTROL_CASES)$`, and
    `^(_POPULATION_CASES|_T\\d+_NEW_SHAPES|_PHANTOM_CASES)$` -- each admit
    exactly one of `_T9999_RENAMED_SHAPES`, `_CONTROL_CASES`, or
    `_PHANTOM_CASES` and turn exactly that one case in this cell red,
    leaving this cell's other four cases and the file's other 65 cells
    green. A fourth widening, not named in that casebook,
    `^_[A-Z][A-Z0-9_]*_(CASES|SHAPES)$`, admits all four off-convention
    names in this cell's own parametrization at once -- including
    `_ORPHAN_CASES` -- and turns all four of this cell's off-convention
    cases red together, and also turns
    `test_registration_admits_the_documented_off_convention_wrong_shape_
    residual`'s four cases red, because that widening admits `_ORPHAN_CASES`
    too. This cell's on-convention case, `_T9999_NEW_SHAPES`, stays green
    under all four widenings, matching what `_T1_CONTAINER_NAME_PATTERN` is
    stated to admit as shipped."""
    assert bool(_T1_CONTAINER_NAME_PATTERN.match(name)) == on_convention
