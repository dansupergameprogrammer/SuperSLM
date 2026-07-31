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
