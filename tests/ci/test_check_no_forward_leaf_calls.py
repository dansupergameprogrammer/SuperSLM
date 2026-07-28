"""Curie's red suite for check_no_forward_leaf_calls.py (SuperSLM_S3a_Walking
SkeletonPlan.md Sec7.3, Sec11 S3.1; T-200, Sec17.3 cell 5).

Mirrors tests/test_check_no_pow_operator.py's own convention (constructed scratch
files, never the real production tree, per StandardsDocument Sec4's population-
validation requirement: a check shown able to FAIL on a fault it exists to catch,
not only shown to pass on unchanged input) and tests/ci/derive_bad_alloc_
membership.py's temp-file discipline (every scratch file is created under
tempfile.mkdtemp(), never written into the working tree, and removed in a
`finally`).

WHY SCRATCH FIXTURES, NOT THE REAL FORWARD DIRECTORY. As of S3.1's authoring
(2026-07-28) no forward-composition source exists in D:\\SuperSLM -- Sec11
places the site-composition sources this check polices at S3.2-S3.9, still
open. The three cells the plan's own S3.1 red-cell list names --

    "a forward TU naming a leaf directly fails the CI check; a scratch TU
    added to the forward target and naming a banned leaf also fails it (input
    coverage); the funnel's own file and the leaf certification TUs pass
    (allowlist control)"

-- are realized here against constructed scratch directories standing in for
"a forward TU" and "the forward target," exactly as check_no_pow_operator.py's
own self-test stands in for "a generator that reintroduced **" with a scratch
copy rather than editing the real generator. The one cell this suite cannot
realize against the real tree -- because there is no real forward-composition
directory yet -- is named and left for a follow-up once Brunel's build lands
one (see check_no_forward_leaf_calls.py's own module docstring).
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


# --- Present truth: the real (currently nonexistent) forward directory. ---


def test_main_end_to_end_against_the_real_default_glob_is_currently_vacuous():
    """As of 2026-07-28 (T-200, S3.1's authoring) no forward-composition source
    exists in the real D:\\SuperSLM tree -- Sec11 places it at S3.2-S3.9, still
    open. The default glob therefore matches zero real files and this end-to-end
    call passes vacuously. This is the current, honest state of the production
    wiring, not a claim that the check has been proven against real forward
    sources -- every cell above proves the mechanism against constructed
    fixtures instead, per StandardsDocument Sec4's population-validation
    requirement. This test must be revisited (a companion cell added asserting
    the glob is now NONEMPTY) the moment a forward-composition source lands."""
    files = cnfl._glob_files(cnfl._DEFAULT_FORWARD_GLOBS, cnfl._REPO_ROOT)
    assert files == [], (
        "the default forward-composition glob now matches real file(s) -- S3.2 or later "
        "has landed a forward-composition source. This test's own vacuous-pass premise no "
        "longer holds: add a companion assertion that the real files are clean, and update "
        "this test's docstring rather than deleting it."
    )
    code = cnfl.main()
    assert code == 0
