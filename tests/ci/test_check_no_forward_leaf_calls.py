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

import check_no_forward_leaf_calls as cnfl


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


def test_find_leaf_forwarding_doors_finds_a_one_line_forwarding_function():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "src/forward/checked_chain_funnel.cpp",
            "int64_t CarriedScaleReciprocal(int64_t m) { return DynamicScaleReciprocal(m); }\n",
        )
        assert cnfl.find_leaf_forwarding_doors(path) == ["CarriedScaleReciprocal"]


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


def test_check_door_count_passes_when_the_scanned_file_matches_the_expected_set():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            tmp,
            "src/forward/checked_chain_funnel.cpp",
            "int64_t CarriedScaleReciprocal(int64_t m) { return DynamicScaleReciprocal(m); }\n",
        )
        assert cnfl.check_door_count(path, expected=("CarriedScaleReciprocal",)) == []


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
