"""T-2380 (Curie) -- the red suite for the archive-based ship gate's own
COMPOSITION stage: design Sec4.1/Sec7 dimension 11's forty-seventh
population (corrected, D-SLM5069) and forty-eighth population
(D-SLM5068), the fold-42 membership-resolving-power commissioning.

THE CELL THAT MATTERS MOST. Fold round 41's own dead composition function
(T-2360), fold round 39's unreplaced membership demotion, and fold round
40's ungraded archive composition were each found only after being shipped
as "closed." T-2378's own strike (`Claude/Loki/t2378-fold41-acceptance-
form-strike-2026-08-28.md`) found the sharpest instance yet: eighteen
drivers identical in reader, classifier, and composition, each silently
dropping ONE yielded member, scored PASS on every population fold round 41
had -- 16 of 18 passing every existing cell and exiting 0 on a real archive
carrying a genuine floating-point object. The forty-eighth population
closes that gap by construction: it does not test whether the composition
CAN be graded (that is the corrected forty-seventh population's own job,
resolving power zero over membership); it tests whether the MEMBERSHIP
stage is graded at Δn = 1, swept across every one of the archive's eighteen
member positions, so a driver dropping exactly one position at ANY point
in that range fails, not only at the one position an existing fixture
happens to pin.

ADOPTED, NOT AUTHORED. Per this ticket's own contract ("adopt that
construction rather than authoring your own"), the constructions below are
T-2376's (`Claude/Loki/t2376-probe/fp_carrier.cpp`, the poisoned-archive
append) and T-2378's (`Claude/Loki/t2378-probe/gen_fixture_cmds.py`,
`membership_census.py`) own already-executed, adversary-authored
must-reject constructions, reproduced here via `archive_fixtures.py`
(fixture construction only -- see that module's own docstring) rather than
re-derived or read from the probe directories, per this ticket's own
"copy what you need into the suite's own fixture space; do not write into
the probe directory."

WHAT THIS FILE TESTS AGAINST. There is no production Python function named
for "the archive-based driver's own composition" -- the design describes it
behaviorally (Sec5.5's own disjunction: "for every OBJECT member
iterate_archive_members yields... the job fails exactly as (i) does").
Sec4.1's own platform-legs paragraph states the COFF/MSVC leg is "unchanged
in every respect except which artifact is read," and the real corpus this
session's own conftest.py fixture builds lands at
`<build_dir>/Release/superslm.lib` -- the exact layout that paragraph
names. Every cell below therefore invokes `tests/ci/scan_build_output.py`
as a subprocess against a fixture directory shaped that way: the same
production CLI `build.bat` and `.github/workflows/tests.yml` both already
invoke, unmodified. This is the composition stage's own graded observable
(the job's exit code).

BUILT (T-2381, Brunel, 2026-08-28). `scan_build_output.py`'s own `main()`
reads the archive at `<build_dir>/Release/superslm.lib` -- the exact layout
every fixture below produces -- via `find_target_archive`/
`enumerate_archive_objects`, and classifies every OBJECT-kind member it
yields; `find_target_objects`'s own directory glob never runs against these
fixtures, because an archive is present at the first candidate location
`find_target_archive` checks. Every cell below carries no `xfail` marker and
exercises the built interface directly.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS_ROOT = os.path.dirname(_HERE)
_CI_DIR = os.path.join(_TESTS_ROOT, "ci")

sys.path.insert(0, _HERE)
import archive_fixtures as af  # noqa: E402


def _place_as_release_archive(archive_path, tmp_path, subdir):
    build_dir = tmp_path / subdir
    release_dir = build_dir / "Release"
    release_dir.mkdir(parents=True, exist_ok=True)
    dest = release_dir / "superslm.lib"
    with open(archive_path, "rb") as src, open(dest, "wb") as dst:
        dst.write(src.read())
    return str(build_dir)


def _run_scan_build_output(build_dir):
    r = subprocess.run(
        [sys.executable, os.path.join(_CI_DIR, "scan_build_output.py"),
         "--build-dir", build_dir, "--target", "superslm", "--isa", "x86-64"],
        capture_output=True, text=True,
    )
    return r.returncode, r.stdout, r.stderr


@pytest.fixture(scope="session")
def real_coff_archive(real_build_dir):
    p = os.path.join(real_build_dir, "Release", "superslm.lib")
    if not os.path.exists(p):
        pytest.skip("real_build_dir produced no Release/superslm.lib at {}".format(p))
    return p


@pytest.fixture(scope="session")
def fp_carrier_obj(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("t2380_comp_carrier")
    obj = str(out_dir / "fp_carrier.obj")
    try:
        af.compile_fp_carrier(obj)
    except af.ToolUnavailable as e:
        pytest.skip(str(e))
    return obj


@pytest.fixture(scope="session")
def poisoned_archive(real_coff_archive, fp_carrier_obj, tmp_path_factory):
    """The forty-seventh population's own must-reject (D-SLM5053): the real
    archive plus the carrier appended last (object position 18 of 18)."""
    out_dir = tmp_path_factory.mktemp("t2380_poisoned")
    out_path = str(out_dir / "poisoned.lib")
    af.build_poisoned_archive(real_coff_archive, fp_carrier_obj, out_path, str(out_dir / "work"))
    return out_path


@pytest.fixture(scope="session")
def fp_at_k_archives(real_coff_archive, fp_carrier_obj, tmp_path_factory):
    """The forty-eighth population's own must-reject family (D-SLM5068):
    eighteen real archives, one per object-member position k in 1..18, each
    holding the same seventeen real objects plus the carrier at position k
    (k=18 is poisoned_archive's own construction, built independently here
    rather than shared, so this fixture stands alone). Built once per
    session -- eighteen real lib.exe invocations."""
    n_objects = len(af.raw_object_payloads(real_coff_archive))
    assert n_objects == 17, "expected 17 real objects, found {}".format(n_objects)
    out_dir = tmp_path_factory.mktemp("t2380_fp_at_k")
    archives = {}
    for k in range(1, n_objects + 2):  # 1..18
        out_path = str(out_dir / ("fp_at_%02d.lib" % k))
        af.build_archive_with_extra_member_at(
            real_coff_archive, fp_carrier_obj, k, out_path, str(out_dir / ("work_%02d" % k)))
        archives[k] = out_path
    return archives


# ===========================================================================
# Fixture verification -- confirms every fp_at_k.lib genuinely carries the
# carrier at its own claimed position and nowhere else, and genuinely has
# 18 members/18 objects. Not xfail (uses only archive_fixtures.py's own
# fixture-verification reader, never the production surface).
# ===========================================================================

def test_fp_at_k_fixtures_place_carrier_at_the_claimed_position(fp_at_k_archives):
    for k, path in fp_at_k_archives.items():
        total, counts = af.raw_member_counts(path)
        assert total == 21 and counts.get("OBJECT") == 18, (
            "fp_at_{:02d}.lib: expected 21 members / 18 objects, got {} / {} "
            "({})".format(k, total, counts.get("OBJECT"), counts)
        )
        objs = af.raw_object_payloads(path)
        assert objs[k - 1][0] == "fp_carrier.obj", (
            "fp_at_{:02d}.lib: expected the carrier at OBJECT position {}, "
            "found {!r}; object order was {}".format(
                k, k, objs[k - 1][0], [n for n, _ in objs])
        )
        others = [n for i, (n, _) in enumerate(objs, 1) if i != k]
        assert "fp_carrier.obj" not in others, (
            "fp_at_{:02d}.lib: the carrier appears at a position other than "
            "{} as well".format(k, k)
        )


def test_poisoned_archive_fixture_matches_forty_seventh_population(poisoned_archive):
    """Fixture verification: poisoned.lib is exactly the forty-seventh
    population's own claimed construction, measured (D-SLM5069's own
    correction): 21 members / 18 objects, the carrier last."""
    total, counts = af.raw_member_counts(poisoned_archive)
    assert total == 21 and counts.get("OBJECT") == 18, (total, counts)
    objs = af.raw_object_payloads(poisoned_archive)
    assert objs[-1][0] == "fp_carrier.obj", objs[-1][0]


# ===========================================================================
# Population 47, corrected (D-SLM5053/D-SLM5069) -- the composition stage.
# Resolving power: Δn = 0 over membership (the must-accept and must-reject
# differ in membership too, per the fold-42 correction) -- this population
# commissions ONLY whether a classified REJECT reaches the job's exit code,
# never which position carries it. The forty-eighth population below is
# what commissions membership at fine resolving power.
# ===========================================================================

def test_pop47_must_accept_real_archive_has_no_reject_under_the_built_gate(real_coff_archive, real_build_dir):
    """Fixture verification of the must-accept side, independent of the
    composition-level assertions below: the real build directory scans
    clean under the built ship gate -- which reads the archive at
    `Release/superslm.lib` (T-2381/T-2385, D-SLM5100/D-SLM5101), not the
    object directory -- 0 REJECT, 0 REFUSE, exit 0. Confirms the
    must-accept genuinely has no floating-point arithmetic before any
    archive-composition-level assertion is made against it, by reusing the
    already-commissioned gate rather than re-deriving a second reading."""
    rc, out, err = _run_scan_build_output(real_build_dir)
    assert rc == 0, (
        "the real build's own archive-based gate (the production path "
        "today) must exit 0 on this session's own real corpus; got {} "
        "stdout={!r} stderr={!r}".format(rc, out, err)
    )


def test_pop47_must_accept_real_archive_composition(real_coff_archive, tmp_path):
    """Must-accept: the real, unmodified archive placed at
    <build_dir>/Release/superslm.lib -- the job must exit 0."""
    build_dir = _place_as_release_archive(real_coff_archive, tmp_path, "clean")
    rc, out, err = _run_scan_build_output(build_dir)
    assert rc == 0, "expected exit 0 on the real clean archive; got {} stdout={!r} stderr={!r}".format(rc, out, err)


def test_pop47_must_reject_poisoned_archive_composition(poisoned_archive, tmp_path):
    """Must-reject: poisoned.lib (the real archive plus the carrier) placed
    identically -- the job must exit nonzero. This is the exact
    discrimination T-2376 found neither driver (HONEST/UNWIRED) separated
    under the pre-fold-41 form, and fold 41's own forty-seventh population
    now commissions: a driver whose exit code is a function of the
    classified REJECT differs from one whose exit code is not."""
    build_dir = _place_as_release_archive(poisoned_archive, tmp_path, "poisoned")
    rc, out, err = _run_scan_build_output(build_dir)
    assert rc == 1, "expected exit 1 on poisoned.lib; got {} stdout={!r} stderr={!r}".format(rc, out, err)


# ===========================================================================
# Population 48 (D-SLM5068) -- THE CELL THAT MATTERS MOST. Membership,
# swept at resolving power Δn = 1 across all eighteen archive positions.
#
# Discriminating coordinate: the archive member position k in 1..18 at
# which the carrier sits.
# Must-accept, fixed across every k: the real, unmodified archive.
# Must-reject, for every k: fp_at_k.lib -- eighteen members, exactly one
# classified REJECT, wherever k places it.
# Graded observable: whether the job's own exit code reflects the
# classified REJECT AT POSITION k, for every k -- not whether it reflects
# a REJECT at any single k a fixture happens to pin. A driver whose
# membership stage silently omits exactly one position exits 0 on the one
# construction whose carrier sits at the omitted position and exits 1 on
# the other seventeen; sweeping k across the full range is what this
# population's own resolving power actually is, and what T-2378's own
# census (16 of 18 positions invisible to fold round 41's own form) found
# missing.
# ===========================================================================

@pytest.mark.parametrize("k", list(range(1, 19)))
def test_pop48_membership_resolves_carrier_at_every_position(k, fp_at_k_archives, tmp_path):
    """Must-reject, position k of 18: fp_at_k.lib must make the job exit
    nonzero, whichever position k places the carrier at. A driver that
    silently drops object-member position k from classification -- byte-
    identical to the honest driver in every other respect -- exits 0 on
    exactly this one construction and would otherwise pass every cell in
    test_archive_gate.py untouched, reproducing T-2378's own SKIPPER_k
    finding (16 of 18 positions invisible to fold round 41's own form)."""
    build_dir = _place_as_release_archive(fp_at_k_archives[k], tmp_path, "fp_at_%02d" % k)
    rc, out, err = _run_scan_build_output(build_dir)
    assert rc == 1, (
        "position {}: expected exit 1 (the carrier at this position must "
        "be classified and reach the job's own exit code); got {} "
        "stdout={!r} stderr={!r}".format(k, rc, out, err)
    )


def test_pop48_resolving_power_is_stated_and_matches_the_sweep():
    """Per design Sec4.1's resolving-power clause (D-SLM5066): this
    population's own resolving power is Δn = 1, swept across all eighteen
    positions the real archive's own member count (17 clean objects + 1
    carrier) admits -- pinned here as a literal fact about the parametrize
    range above, so a future edit that narrows the sweep (e.g. back to a
    single pinned k) is caught by this cell rather than silently reducing
    the population's own resolving power."""
    markers = test_pop48_membership_resolves_carrier_at_every_position.pytestmark
    parametrize_markers = [m for m in markers if m.name == "parametrize"]
    assert len(parametrize_markers) == 1, markers
    argname, argvalues = parametrize_markers[0].args
    assert argname == "k"
    assert list(argvalues) == list(range(1, 19)), (
        "the forty-eighth population's own sweep no longer covers all "
        "eighteen positions 1..18 -- resolving power narrowed from Δn = 1 "
        "swept across every position to something coarser"
    )
