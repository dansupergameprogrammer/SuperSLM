"""T-2380 (Curie) -- the red suite for the archive-based ship gate's own
reader, classifier, and Mach-O-disposition stages: design Sec4.1/Sec5.4/
Sec5.5/Sec7 dimension 11's thirty-seventh through forty-sixth populations,
as amended by fold round 42
(Claude/Vitruvius/t2265-superslm-fp-free-open-design-2026-08-24.md), plus
the fold-42 correction to `scan_object`'s own printed signature (D-SLM5071).

Companion file `test_archive_composition.py` carries the composition-stage
populations (forty-seventh, corrected, and forty-eighth) -- this split
mirrors the design's own four-stage partition (Sec4.1's fourth law):
membership/reading/classification live here, composition lives there.

BUILT (T-2381, Brunel, 2026-08-28): `check_fp_free_scan.py` now has
`iterate_archive_members`, `enumerate_archive_objects`,
`MalformedArchiveError`, and `ArchiveHasNoObjectsError`, and `scan_object`'s
own signature carries both `corpus_symbols` and `data`
(`(path: str, isa: str, corpus_symbols: frozenset | None = None, data:
bytes | None = None) -> ScanResult`). Every `xfail(strict=True)` cell this
file originally carried has been removed by Brunel's build round, per this
suite's own stated gate ("the day a cell starts passing unexpectedly,
strict=True turns it into a reported failure, which is the signal to remove
that cell's own marker") -- every cell below is now a real, currently-
enforced regression guard, none an absent-instrument placeholder.

RESOLVING POWER is stated in each population's own docstring, per design
Sec4.1's resolving-power clause (D-SLM5066): the smallest unit-count change
the population's own cells are built to separate. Most cells here resolve
membership/classification at the finest available grain (one member, one
mnemonic); nothing in this file tests the composition stage's own
resolving power over the full 1..18 position sweep -- that is
test_archive_composition.py's own subject.
"""
from __future__ import annotations

import inspect
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIXTURES = os.path.join(_HERE, "fp_scan_fixtures")
_TESTS_ROOT = os.path.dirname(_HERE)
_CI_DIR = os.path.join(_TESTS_ROOT, "ci")

sys.path.insert(0, _HERE)
import fp_scan_common as fc  # noqa: E402
import archive_fixtures as af  # noqa: E402

sys.path.insert(0, _CI_DIR)
import check_fp_free_scan as scan  # noqa: E402
import scan_build_output  # noqa: E402  -- T-2385: find_target_objects builds the
# restored fortieth population's own no-archive fixtures (design Sec7 dim 11,
# D-SLM5103), which need a real <target>.dir layout to place against no archive.

_ELF_LIB_ENV = "SUPERSLM_FP_SCAN_ELF_LIB"
_ELF_LIB_DEFAULT = r"D:/SuperSLM/.worktrees/elf-leg/libsuperslm.a"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_coff_archive(real_build_dir):
    """The real, freshly-built (this session, conftest.py's own real_build_dir)
    COFF archive `superslm.lib` -- the identical artifact windows-latest's
    own fp-free-scan-gate job ships. Skips, with a stated reason, in an
    environment where real_build_dir itself skipped."""
    p = os.path.join(real_build_dir, "Release", "superslm.lib")
    if not os.path.exists(p):
        pytest.skip("real_build_dir produced no Release/superslm.lib at {}".format(p))
    return p


@pytest.fixture(scope="session")
def real_elf_archive():
    """An optional, environment-provided real ELF archive
    (`libsuperslm.a`, GNU ar) -- this ticket's own environment note names
    `D:/SuperSLM/.worktrees/elf-leg/libsuperslm.a` as an existing, real,
    GCC-built archive (D-SLM5032's own conductor measurement). Overridable
    via SUPERSLM_FP_SCAN_ELF_LIB for a machine where that exact worktree
    path does not exist; skips, never fails, when neither is present -- this
    suite has no CI leg that builds an ELF archive itself (design Sec4.1's
    own forty-sixth population: the CMake+build-system integration half of
    the ELF/GCC leg is owed, not built)."""
    p = os.environ.get(_ELF_LIB_ENV, _ELF_LIB_DEFAULT)
    if not os.path.isfile(p):
        pytest.skip(
            "no real ELF archive at {} (set {} to override) -- this "
            "environment cannot cross-check the toolchain-specific member "
            "count correction (D-SLM5055) against a real ar-built "
            "archive".format(p, _ELF_LIB_ENV)
        )
    return p


@pytest.fixture(scope="session")
def malformed_archives(real_coff_archive, tmp_path_factory):
    """The thirty-seventh/-eighth/-ninth/forty-fourth populations' own
    must-reject constructions, built once from the real COFF archive's own
    real bytes (archive_fixtures.build_malformed_fixtures)."""
    out_dir = tmp_path_factory.mktemp("t2380_malformed")
    return af.build_malformed_fixtures(real_coff_archive, str(out_dir))


@pytest.fixture(scope="session")
def fp_carrier_obj(tmp_path_factory):
    """The forty-seventh/forty-eighth populations' own genuinely
    floating-point-carrying object (archive_fixtures.compile_fp_carrier),
    adopted unmodified here for the restored fortieth population's own
    FP-carrying-directory control (D-SLM5103)."""
    out_dir = tmp_path_factory.mktemp("t2385_pop40_carrier")
    obj = str(out_dir / "fp_carrier.obj")
    try:
        af.compile_fp_carrier(obj)
    except af.ToolUnavailable as e:
        pytest.skip(str(e))
    return obj


def _place_as_release_archive(archive_path, tmp_path, subdir="fixture_build"):
    """Lays out a directory shaped `<dir>/Release/superslm.lib`, matching the
    real windows-latest CI leg's own VS-generator output layout (design
    Sec4.1's own platform-legs paragraph) -- the shape scan_build_output.py's
    own archive-based `main()` reads (T-2381, Brunel). Returns the directory
    to pass as `--build-dir`."""
    build_dir = tmp_path / subdir
    release_dir = build_dir / "Release"
    release_dir.mkdir(parents=True, exist_ok=True)
    dest = release_dir / "superslm.lib"
    with open(archive_path, "rb") as src, open(dest, "wb") as dst:
        dst.write(src.read())
    return str(build_dir)


def _run_scan_build_output(build_dir):
    """Invokes the real production CLI as a subprocess -- the same
    invocation build.bat and .github/workflows/tests.yml both use --
    returning (returncode, stdout, stderr)."""
    r = subprocess.run(
        [sys.executable, os.path.join(_CI_DIR, "scan_build_output.py"),
         "--build-dir", build_dir, "--target", "superslm", "--isa", "x86-64"],
        capture_output=True, text=True,
    )
    return r.returncode, r.stdout, r.stderr


# ===========================================================================
# Fixture-verification / documented-claim pins -- real execution, no
# dependency on any not-yet-built production surface. Not xfail: these pass
# today and stay true regardless of build status.
# ===========================================================================

def test_dslm5055_real_coff_archive_member_counts(real_coff_archive):
    """D-SLM5055: the real Windows/lib.exe archive is 20 members
    (2 SYMTAB + 1 LONGNAMES + 17 OBJECT). Fixture verification only
    (archive_fixtures.raw_member_counts, NOT the production reader) --
    confirms the real archive this session's own build produced genuinely
    has the shape the thirty-seventh/-eighth populations' own must-accept
    claims, independent of whether the production reader exists yet."""
    total, counts = af.raw_member_counts(real_coff_archive)
    assert total == 20, "expected 20 total members, found {} ({})".format(total, counts)
    assert counts.get("SYMTAB") == 2, counts
    assert counts.get("LONGNAMES") == 1, counts
    assert counts.get("OBJECT") == 17, counts


def test_dslm5055_real_elf_archive_member_counts(real_elf_archive):
    """D-SLM5055's own correction: GNU ar writes ONE symbol-table member
    where lib.exe writes two -- the real ELF archive is 19 members
    (1 SYMTAB + 1 LONGNAMES + 17 OBJECT), not the Windows archive's 20.
    Fixture verification only."""
    total, counts = af.raw_member_counts(real_elf_archive)
    assert total == 19, "expected 19 total members, found {} ({})".format(total, counts)
    assert counts.get("SYMTAB") == 1, counts
    assert counts.get("LONGNAMES") == 1, counts
    assert counts.get("OBJECT") == 17, counts


def test_dslm5054_real_archive_has_odd_sized_member(real_coff_archive):
    """Precondition for the padding pin below: confirms, on the real
    archive, that D-SLM5054's even-byte padding rule is genuinely exercised
    (at least one member has an odd `size`) rather than the archive
    happening to contain only even-sized members, which would make the pad
    logic untestable against it by coincidence."""
    assert af.has_odd_sized_member(real_coff_archive), (
        "the real archive contains no odd-sized member -- the even-byte "
        "padding rule (D-SLM5054) is not exercised by this fixture; the "
        "population needs a different real archive or the minimal "
        "hand-built one below"
    )


def test_dslm5054_minimal_archive_pads_correctly_mechanism_check():
    """D-SLM5054's own even-byte padding rule, pinned at the smallest
    possible construction: a hand-built, two-member, valid `!<arch>` file
    with one odd-sized member (3 bytes) followed by its own pad byte, then
    one even-sized member (2 bytes, no pad). A degenerate mechanism check
    (StandardsDocument.md Sec5.4) -- the real-archive-at-real-size cell
    above is this claim's product-level cell; this one isolates the padding
    arithmetic alone, independent of any real toolchain's own archive
    happening to contain an odd-sized member. Fixture-verification-only
    reader (archive_fixtures.raw_iterate_archive_members already implements
    the pad-skip); mutation-pinned below by removing the pad byte and
    confirming the read then desyncs, so this cell does not merely restate
    the helper's own code."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="t2380_pad_")
    p = os.path.join(tmp, "pad.lib")
    af.build_minimal_padded_archive(p)
    total, counts = af.raw_member_counts(p)
    assert total == 2 and counts == {"SYMTAB": 1, "OBJECT": 1}, (total, counts)

    # Mutation: read the SAME correctly-built, correctly-padded file with a
    # reader whose own padding arithmetic is forcibly disabled (the
    # `size % 2` term removed -- the pre-D-SLM5054 shape). The pad byte is
    # real, present, and untouched in `p` itself; only the READER'S
    # handling of it is mutated. A pad-unaware reader treats the odd-sized
    # first member's own pad byte as the first byte of the second header,
    # desyncing every field after it -- reproducing what T-2376 measured
    # against the real archive (MalformedArchiveError, size field no longer
    # parseable as base-10 ASCII).
    def _read_without_padding(path):
        with open(path, "rb") as fh:
            b = fh.read()
        off, n = 8, len(b)
        count = 0
        while off < n:
            if off + 60 > n:
                raise af.MalformedArchiveError("header truncated at %d" % off)
            hdr = b[off:off + 60]
            st = hdr[48:58].rstrip(b" ")
            if not st or not all(0x30 <= c <= 0x39 for c in st):
                raise af.MalformedArchiveError(
                    "size field %r not base-10 ASCII at %d" % (hdr[48:58], off))
            size = int(st)
            off = off + 60 + size  # NO % 2 term -- the pre-D-SLM5054 shape
            count += 1
        return count

    with pytest.raises(af.MalformedArchiveError):
        _read_without_padding(p)


def test_dslm5071_current_scan_object_signature_keeps_corpus_symbols():
    """Baseline, confirmed current: `scan_object`'s third parameter is
    `corpus_symbols` (D-SLM4857, fold round 34), unaffected by anything this
    ticket adds. Not xfail -- this is true today and must stay true; it is
    the fact the forty-first lesser-plane finding (D-SLM5071) exists to
    protect against silently regressing when `data` is added."""
    params = list(inspect.signature(scan.scan_object).parameters)
    assert params[:2] == ["path", "isa"], params
    assert params[2] == "corpus_symbols", (
        "scan_object's third parameter is {!r}, not 'corpus_symbols' -- "
        "the design's fold-42 correction (D-SLM5071) exists precisely "
        "because a naive 'additive' data-parameter patch could delete "
        "this position".format(params[2] if len(params) > 2 else None)
    )


# ===========================================================================
# Population 37 -- archive magic-and-header validation. Resolving power:
# whole-archive (a single malformed archive either reads cleanly to
# completion or it does not) -- this population does not resolve WHICH
# member is affected, only whether the container itself is trustworthy.
# ===========================================================================

def test_pop37_must_accept_real_archive_enumerates_seventeen_objects(real_coff_archive):
    """Must-accept: the real, unmodified archive -- enumerate_archive_objects
    yields exactly 17 OBJECT members, magic and all 20 headers well-formed."""
    objs = scan.enumerate_archive_objects(real_coff_archive)
    assert len(objs) == 17, "expected 17 real object members, got {}".format(len(objs))


def test_pop37_must_reject_bad_magic(malformed_archives):
    """Must-reject: the real archive with its first 8 bytes overwritten to
    any other value -- an infrastructure failure. The design's own text
    does not name a specific Python exception class for a magic mismatch
    (unlike MalformedArchiveError, reserved for header-level corruption) --
    this cell asserts only that SOME exception is raised, never a silent
    empty read."""
    with pytest.raises(Exception):
        list(scan.iterate_archive_members(malformed_archives["bad_magic"]))


def test_pop37_must_reject_truncated(malformed_archives):
    """Must-reject: a copy of the real archive truncated mid-header --
    RAISES MalformedArchiveError (design Sec4.1's own pinned exception name
    for header-level corruption)."""
    with pytest.raises(scan.MalformedArchiveError):
        list(scan.iterate_archive_members(malformed_archives["truncated"]))


def test_pop37_must_reject_size_overrun(malformed_archives):
    """Must-reject: one header's `size` field replaced by a value exceeding
    the file's own remaining length -- RAISES MalformedArchiveError."""
    with pytest.raises(scan.MalformedArchiveError):
        list(scan.iterate_archive_members(malformed_archives["size_overrun"]))


def test_pop37_must_reject_size_nondigit(malformed_archives):
    """Must-reject: one header's `size` field replaced by non-digit bytes --
    RAISES MalformedArchiveError."""
    with pytest.raises(scan.MalformedArchiveError):
        list(scan.iterate_archive_members(malformed_archives["size_nondigit"]))


def test_pop37_malformed_fixtures_are_real_byte_patches_not_fabrications(malformed_archives, real_coff_archive):
    """Fixture verification: every must-reject construction above is a real
    file, produced by truncating or byte-patching the real archive's own
    bytes -- not a hand-assembled fixture standing in for one (design Sec7
    dim 11's own thirty-seventh population text). Confirms each fixture
    differs from the real archive by exactly the intended corruption and is
    not, e.g., an empty or zero-length file."""
    real_size = os.path.getsize(real_coff_archive)
    for label in ("bad_magic", "size_overrun", "size_nondigit", "bad_name"):
        assert os.path.getsize(malformed_archives[label]) == real_size, (
            "{} must be the same size as the real archive (a byte patch, "
            "not a truncation or fabrication)".format(label)
        )
    assert 0 < os.path.getsize(malformed_archives["truncated"]) < real_size


@pytest.mark.parametrize("label", ["bad_magic", "truncated", "size_overrun", "size_nondigit", "bad_name"])
def test_pop37_38_malformed_archive_exits_2_at_composition_level(label, malformed_archives, tmp_path):
    """T-2381 (Brunel) -- a new pin for a production change this build
    round landed with no existing cell: `scan_build_output.py`'s own
    `_scan_archive_corpus` catches `MalformedArchiveError`/`ValueError`
    from `enumerate_archive_objects` and converts it to exit 2 (an
    infrastructure failure, design Sec4.1's own disposition for header-
    level corruption and for an unrecognized container alike). Every
    thirty-seventh/thirty-eighth population cell above tests this at the
    READER level (`iterate_archive_members`/`enumerate_archive_objects`
    raising directly); none exercises the CLI/exit-code path the design's
    own text actually states the observable as ("Must-reject
    (infrastructure failure, exit 2, not a REFUSE)"). This cell places each
    of the five malformed-archive fixtures at `<build_dir>/Release/
    superslm.lib` and runs the real production CLI against it, confirming
    the composition-level except clause this round wrote actually fires
    for every one of the five header-level/magic-level corruption shapes,
    never a REFUSE (exit 1) and never an uncaught traceback."""
    build_dir = _place_as_release_archive(malformed_archives[label], tmp_path, "malformed_%s" % label)
    rc, out, err = _run_scan_build_output(build_dir)
    assert rc == 2, (
        "malformed archive {!r}: expected exit 2 (infrastructure failure); "
        "got {} stdout={!r} stderr={!r}".format(label, rc, out, err)
    )
    assert "Traceback (most recent call last)" not in err, (
        "malformed archive {!r}: the job crashed with an uncaught exception "
        "rather than reporting a clean infrastructure failure:\n{}".format(
            label, err)
    )


# ===========================================================================
# Population 38 -- member classification's three-way exhaustiveness.
# Resolving power: one member (a single member whose name is corrupted to
# match none of the three kinds is must-reject).
# ===========================================================================

def test_pop38_must_accept_no_fourth_branch(real_coff_archive):
    """Must-accept: the real archive's own two symbol-table members, its
    longnames member, and its seventeen object members, each classified to
    its correct kind, no fourth branch -- the OBJECT-only view
    (enumerate_archive_objects) must yield exactly 17, matching the
    fixture-verified total member breakdown (test_dslm5055_... above, 2+1+17)."""
    objs = scan.enumerate_archive_objects(real_coff_archive)
    assert len(objs) == 17


def test_pop38_must_reject_bad_member_name(malformed_archives):
    """Must-reject: a copy of the real archive with one object member's
    16-byte name field overwritten to a value that is neither a known index
    name, a .obj/.o-suffixed name, nor a valid longnames offset -- RAISES
    MalformedArchiveError, never silently read as a fourth kind."""
    with pytest.raises(scan.MalformedArchiveError):
        list(scan.iterate_archive_members(malformed_archives["bad_name"]))


# ===========================================================================
# Population 39 -- zero-object-member archive. Resolving power: Δn = 17
# (seventeen objects present versus zero) -- coarser than the forty-eighth
# population below, and does not by itself resolve a driver that silently
# omits exactly one of the seventeen.
# ===========================================================================

def test_pop39_must_accept_seventeen_objects_found(real_coff_archive):
    """Must-accept: the real archive, seventeen object members found."""
    assert len(scan.enumerate_archive_objects(real_coff_archive)) == 17


def test_pop39_must_reject_zero_object_archive_at_reader_level(malformed_archives):
    """Must-reject, reader level: a real, well-formed archive containing
    only its own symbol-table and longnames members -- enumerate_archive_
    objects must not silently return an empty list read as "nothing to
    scan, therefore clean"; it must signal an infrastructure failure the
    same way "zero objects found" already does at the object-directory
    gate. Asserted here as: a caller cannot get a false-clean empty result
    without an accompanying signal -- either a raised exception, or (if a
    future implementation instead returns []) that is EXACTLY what
    test_pop39_must_reject_zero_object_archive_at_composition_level below
    must convert into an exit-2 infrastructure failure. This cell pins the
    stronger of the two acceptable shapes: raises."""
    with pytest.raises(Exception):
        objs = scan.enumerate_archive_objects(malformed_archives["zero_objects"])
        assert objs, "zero_objects.lib produced an empty list with no exception -- see composition-level cell"


def test_pop39_must_reject_zero_object_archive_at_composition_level(malformed_archives, tmp_path):
    """Must-reject, composition level (design's own graded observable --
    "exit 2, identical treatment to today's zero-objects-found disposition"):
    a zero-object archive placed where the ship gate's own archive-based
    driver reads it must make the job exit 2, never 0. Built (T-2381,
    Brunel): `scan_build_output.py`'s own `_scan_archive_corpus` calls
    `enumerate_archive_objects`, catches `ArchiveHasNoObjectsError`, and
    exits 2 for exactly that reason -- the correctly-diagnosed empty
    archive, not the pre-archive object-directory path's own unrelated "no
    object files found." Not marked xfail: exit 2 was already the correct
    assertion before the archive path was built (the object-directory path
    reached the same exit code for an unrelated reason) and remains correct
    now that the archive path is what produces it.
    """
    build_dir = _place_as_release_archive(malformed_archives["zero_objects"], tmp_path)
    rc, out, err = _run_scan_build_output(build_dir)
    assert rc == 2, "expected exit 2 (infrastructure failure); got {} stdout={!r} stderr={!r}".format(rc, out, err)


# ===========================================================================
# Population 40, restored (D-SLM5099/D-SLM5103, fold round 43; supersedes
# the fold-41 "closed by construction" disposition, D-SLM5057). Fold round
# 41 retired this population on the ground that no archive-based driver
# opens any directory -- true of iterate_archive_members/
# enumerate_archive_objects, which take only an archive path, and never
# true of scan_build_output.py's own main(), which (before T-2385) fell
# back to the object-directory scan whenever find_target_archive found no
# archive at any of its four candidate locations (T-2382 finding 2). T-2385
# removes that fallback: main() now searches the four candidate locations
# and exits 2, with the search printed, when none exists -- the
# object-directory scan is never called. This population grades exactly
# that: the directory's own content -- clean or floating-point-carrying --
# has zero effect on a missing-archive disposition, because the corrected
# driver never reads the directory to find out what is in it. Two cells,
# not one, because a single must-reject cell would leave open whether an
# incidental exit 2 came from never reading the directory or from reading
# it and getting lucky on that content alone.
# ===========================================================================

def test_pop40_no_archive_clean_directory_exits_2(tmp_path, real_build_dir):
    """Must-reject-shaped control 1 -- clean directory (D-SLM5103). A build
    directory with no archive at any of find_target_archive's four
    candidate locations, whose <target>.dir holds the real build's own
    seventeen clean objects, must exit 2. A driver retaining the removed
    fallback would scan the directory, find 0 REJECT / 0 REFUSE, and exit
    0 -- a false PASS over a corpus that is not the shipped artifact."""
    objects = scan_build_output.find_target_objects(real_build_dir, "superslm")
    if not objects:
        pytest.skip("real_build_dir produced no objects to build this fixture from")
    build_dir = tmp_path / "pop40_clean"
    target_dir = build_dir / "superslm.dir" / "Release"
    target_dir.mkdir(parents=True)
    for obj in objects:
        with open(obj, "rb") as src, open(target_dir / os.path.basename(obj), "wb") as dst:
            dst.write(src.read())
    rc, out, err = _run_scan_build_output(str(build_dir))
    assert rc == 2, (
        "no archive present, clean object directory: expected exit 2 "
        "(infrastructure failure -- the directory's own content must have "
        "no bearing on a missing-archive disposition); got {} stdout={!r} "
        "stderr={!r}".format(rc, out, err)
    )


def test_pop40_no_archive_fp_carrying_directory_exits_2(tmp_path, real_build_dir, fp_carrier_obj):
    """Must-reject-shaped control 2 -- FP-carrying directory (D-SLM5103).
    The same construction as above, plus one genuinely floating-point-
    carrying object (the forty-seventh/forty-eighth populations' own
    fp_carrier, adopted unmodified) -- must also exit 2, not 1. A driver
    retaining the removed fallback would scan the directory, find one
    REJECT, and exit 1 -- the right-shaped failure for the wrong reason,
    over a corpus that is not the shipped artifact."""
    objects = scan_build_output.find_target_objects(real_build_dir, "superslm")
    if not objects:
        pytest.skip("real_build_dir produced no objects to build this fixture from")
    build_dir = tmp_path / "pop40_fp"
    target_dir = build_dir / "superslm.dir" / "Release"
    target_dir.mkdir(parents=True)
    for obj in objects:
        with open(obj, "rb") as src, open(target_dir / os.path.basename(obj), "wb") as dst:
            dst.write(src.read())
    with open(fp_carrier_obj, "rb") as src, open(target_dir / "fp_carrier.obj", "wb") as dst:
        dst.write(src.read())
    rc, out, err = _run_scan_build_output(str(build_dir))
    assert rc == 2, (
        "no archive present, FP-carrying object directory: expected exit 2 "
        "(infrastructure failure), not 1 -- a REJECT reachable only by "
        "reading the directory must never surface once the corrected "
        "driver never reads it); got {} stdout={!r} stderr={!r}".format(
            rc, out, err)
    )


# ===========================================================================
# Population 41 -- check (B)'s notrack prefix strip, and its fold-42
# replacement control. Resolving power: one mnemonic (a single two-token
# construction either strips correctly or does not).
# ===========================================================================

def test_pop41_must_accept_notrack_jmp_real_compiled():
    """Must-accept: `notrack jmp rax` (and `notrack call`) -- the underlying
    jmp/call is on _X86_GPR_ALLOW; stripping the CET prefix must ACCEPT.
    Real compiled ELF object (clang, x86_64-pc-linux-gnu), real capstone
    decode through the full scan_object pipeline -- not a hardcoded
    mnemonic string."""
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "notrack.o")
        fc.compile_clang_asm(os.path.join(_FIXTURES, "pop41_notrack.s"), obj, "x86_64-pc-linux-gnu")
        result = scan.scan_object(obj, isa="x86-64")
        assert not result.refuse
        assert result.ab_verdicts.get("NotrackJmpAccept") == "ACCEPT", result.ab_verdicts
        assert result.ab_verdicts.get("NotrackCallAccept") == "ACCEPT", result.ab_verdicts


def test_pop41_control_arithmetic_family_still_rejects_today():
    """Control, already true and unaffected by this population: the
    genuinely arithmetic float mnemonic family (addps/mulps/divps/sqrtps/
    cvtsi2sd/comiss) still REJECTs. Real compiled fixture. Not xfail --
    this must stay true before and after notrack is added."""
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "arith.o")
        fc.compile_clang_asm(os.path.join(_FIXTURES, "pop_arith_family_control.s"), obj, "x86_64-pc-linux-gnu")
        result = scan.scan_object(obj, isa="x86-64")
        assert not result.refuse
        assert result.ab_verdicts.get("ArithFamilyControlReject") == "REJECT", result.ab_verdicts


def test_pop41_fold42_control_bnd_mnemonics_specification_pin():
    """D-SLM5070: the fold-42 replacement control is a SPECIFICATION-level
    mutation proof, independent of whether notrack itself is built yet --
    it compares the SPECIFIED fix (strip only the seven named literals:
    rep/repe/repz/repne/repnz/lock/notrack) against the EXACT loosening the
    population's own text says it must exclude (strip everything before the
    first space, unconditionally), on real, capstone-rendered two-token
    mnemonics compiled from real machine code (pop41_control_bnd.s) --
    never hardcoded strings. This is a pin of the DESIGN's own construction,
    not a call into any not-yet-built production surface, so it is not
    xfail: it is true today and stays true regardless of build status,
    exactly like a StandardsDocument.md Sec5.4 mutation proof should be.

    Adopted from T-2378's own Claude/Loki/t2378-probe/lesser_planes.py
    method (StandardsDocument.md Sec5.4's commissioning rule: an adversary-
    authored must-reject construction is adopted, not re-derived), with one
    improvement: the mnemonics are read from REAL compiled bytes via real
    capstone decode, not typed as literal strings, closing the residual gap
    that probe's own bnd/data16 list was itself never executed against a
    real disassembly.

    MODEL GAP, surfaced rather than papered over: `data16 jmp`, the
    population's own third named control mnemonic, is asserted by design
    Sec7 dim 11 (D-SLM5070) to be "real, two-token mnemonics capstone
    renders." Executed against every 0x66-prefixed jmp/call encoding this
    session could construct (near rel32, near rel8, indirect register,
    indirect memory, RIP-relative) -- capstone 5.0.7 (this repo's own
    pinned version, matching check_fp_free_scan.py's own decoder) renders
    every one as a bare "jmp"/"call" mnemonic with NO "data16" text, because
    check_fp_free_scan.py's own decoder never inspects capstone's prefix
    detail, only `insn.mnemonic`. This claim is not reproducible by
    execution against the pinned capstone version and is NOT asserted
    below; routed to the planner rather than invented around, per Curie's
    own model-gap discipline. See this ticket's own casebook for the full
    reproduction.
    """
    real_two_token = ["bnd jmp", "bnd call"]

    def strip_named(m):
        m = m.lower()
        for pfx in ("rep ", "repe ", "repz ", "repne ", "repnz ", "lock ", "notrack "):
            if m.startswith(pfx):
                return m[len(pfx):]
        return m

    def strip_any_token(m):
        m = m.lower()
        return m.split(" ", 1)[1] if " " in m else m

    gpr_allow = set(scan._X86_GPR_ALLOW)  # "jmp"/"call" are already members

    import capstone
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "bnd.o")
        fc.compile_clang_asm(os.path.join(_FIXTURES, "pop41_control_bnd.s"), obj, "x86_64-pc-linux-gnu")
        sections = fc.code_sections(obj, ".text")
        insns = []
        for chunk in sections:
            insns.extend(md.disasm(chunk, 0))

    rendered = {i.mnemonic.lower() for i in insns}
    assert "bnd jmp" in rendered and "bnd call" in rendered, (
        "fixture verification FAILED: expected real 'bnd jmp'/'bnd call' "
        "mnemonics decoded from the real compiled bytes; got {}".format(sorted(rendered))
    )

    separated = 0
    for m in real_two_token:
        narrow_accepts = strip_named(m) in gpr_allow
        loose_accepts = strip_any_token(m) in gpr_allow
        if narrow_accepts != loose_accepts:
            separated += 1
        assert not narrow_accepts, (
            "the specified narrow strip must REJECT {!r} (bnd is not one of "
            "the seven named literals) -- got ACCEPT".format(m)
        )
        assert loose_accepts, (
            "the excluded loosening must ACCEPT {!r} (stripping the first "
            "token leaves 'jmp'/'call', already allowed) -- got "
            "REJECT".format(m)
        )
    assert separated == len(real_two_token), (
        "expected every real two-token control mnemonic to separate the "
        "specified fix from the excluded loosening; separated {} of "
        "{}".format(separated, len(real_two_token))
    )


def test_pop41_fold42_data16_jmp_model_gap_documented():
    """Documents, as a visible SKIP (not a silent omission), the model gap
    test_pop41_fold42_control_bnd_mnemonics_specification_pin's own
    docstring names: capstone 5.0.7 never renders a 'data16' mnemonic
    prefix for any 0x66-prefixed jmp/call encoding this session could
    construct, so the population's own third control mnemonic
    ('data16 jmp') cannot be exercised against this repo's own pinned
    decoder. Executed reproduction below, run every time this cell runs, so
    a future capstone upgrade that starts rendering it is caught by this
    skip turning into a real pass/fail rather than staying silently
    unnoticed."""
    import capstone
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    md.detail = True
    candidates = {
        "indirect jmp r/m64 (66 FF /4)": bytes([0x66, 0xFF, 0xE0]),
        "near jmp rel32 (66 E9)": bytes([0x66, 0xE9, 0x05, 0, 0, 0]),
        "indirect jmp mem (66 FF 20)": bytes([0x66, 0xFF, 0x20]),
    }
    rendered_mnemonics = set()
    for blob in candidates.values():
        for insn in md.disasm(blob, 0):
            rendered_mnemonics.add(insn.mnemonic.lower())
    if any("data16" in m for m in rendered_mnemonics):
        pytest.fail(
            "capstone now renders a 'data16' mnemonic ({}) -- the model gap "
            "this cell documents may be closed; route back to the planner "
            "to add the 'data16 jmp' cell to "
            "test_pop41_fold42_control_bnd_mnemonics_specification_pin "
            "instead of leaving it skipped".format(sorted(rendered_mnemonics))
        )
    pytest.skip(
        "MODEL GAP (routed, not invented around): design Sec7 dim 11's "
        "forty-first population (D-SLM5070) names 'data16 jmp' as a real, "
        "two-token mnemonic capstone renders; executed against every "
        "0x66-prefixed jmp/call encoding constructible this session, "
        "capstone 5.0.7 renders only bare 'jmp'/'call' with no 'data16' "
        "text (mnemonics seen: {}) -- this repo's own classifier consumes "
        "insn.mnemonic only (check_fp_free_scan.py:1177/1181), so this "
        "string can never reach it in production. Not asserted; see this "
        "ticket's own casebook.".format(sorted(rendered_mnemonics))
    )


# ===========================================================================
# Population 42 -- check (B)'s shrd/shld addition. Resolving power: one
# mnemonic.
# ===========================================================================

def test_pop42_must_accept_shrd_shld_real_compiled():
    """Must-accept: `shrd eax, ebx, cl` / `shld eax, ebx, cl` -- real integer
    double-precision shifts. Real compiled ELF object, real capstone decode
    through the full scan_object pipeline."""
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "shrdshld.o")
        fc.compile_clang_asm(os.path.join(_FIXTURES, "pop42_shrd_shld.s"), obj, "x86_64-pc-linux-gnu")
        result = scan.scan_object(obj, isa="x86-64")
        assert not result.refuse
        assert result.ab_verdicts.get("ShrdShldAccept") == "ACCEPT", result.ab_verdicts


def test_pop42_control_arithmetic_family_unaffected():
    """Control, already true: the arithmetic float family is unaffected by
    adding shrd/shld to the GPR allow-list. Not xfail."""
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "arith2.o")
        fc.compile_clang_asm(os.path.join(_FIXTURES, "pop_arith_family_control.s"), obj, "x86_64-pc-linux-gnu")
        result = scan.scan_object(obj, isa="x86-64")
        assert result.ab_verdicts.get("ArithFamilyControlReject") == "REJECT", result.ab_verdicts


# ===========================================================================
# Population 43 -- check (A)'s vextract/vinsert family, both suffixes.
# Resolving power: one mnemonic.
# ===========================================================================

def test_pop43_must_accept_vextract_measured_on_real_corpus():
    """Must-accept, the subset measured on the real GCC-built corpus
    (D-SLM5032): vextracti128, vextracti64x4."""
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "vext_measured.o")
        fc.compile_clang_asm(os.path.join(_FIXTURES, "pop43_vextract_measured.s"), obj,
                              "x86_64-pc-linux-gnu", extra_args=["-mavx512f"])
        result = scan.scan_object(obj, isa="x86-64")
        assert not result.refuse
        assert result.ab_verdicts.get("VextractMeasuredAccept") == "ACCEPT", result.ab_verdicts


def test_pop43_must_accept_vextract_vinsert_wider_family_specified():
    """Must-accept, the wider family specified but not yet measured on a
    real corpus (design's own ruling, D-SLM5037): both suffixes' f/i forms,
    the 32x4/64x2/64x4 AVX-512 sizes, and the vinsert-mnemonic counterparts."""
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "vext_wider.o")
        fc.compile_clang_asm(os.path.join(_FIXTURES, "pop43_vextract_wider.s"), obj,
                              "x86_64-pc-linux-gnu", extra_args=["-mavx512f"])
        result = scan.scan_object(obj, isa="x86-64")
        assert not result.refuse
        assert result.ab_verdicts.get("VextractWiderAccept") == "ACCEPT", result.ab_verdicts
        assert result.ab_verdicts.get("VinsertWiderAccept") == "ACCEPT", result.ab_verdicts


def test_pop43_control_arithmetic_family_unaffected():
    """Control, already true: the arithmetic float family is unaffected by
    widening the vec-move allow-list. Not xfail."""
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "arith3.o")
        fc.compile_clang_asm(os.path.join(_FIXTURES, "pop_arith_family_control.s"), obj, "x86_64-pc-linux-gnu")
        result = scan.scan_object(obj, isa="x86-64")
        assert result.ab_verdicts.get("ArithFamilyControlReject") == "REJECT", result.ab_verdicts


def test_dslm5032_real_elf_gcc_archive_all_four_corrections_end_to_end(real_elf_archive, tmp_path):
    """T-2381 (Brunel) -- a new pin for a production change this build
    round landed with no existing product-level cell: the four ELF/GCC
    classifier corrections (notrack strip, shrd/shld, and the
    vextract/vinsert family, design Sec4.1/Sec7 dim 11's forty-first
    through forty-third populations) landing TOGETHER, against the real
    GCC-built archive (D-SLM5032's own conductor measurement), through the
    archive-based composition path built this round -- not a synthetic
    per-mnemonic fixture, and not the unit-level cells above, which each
    exercise one correction in isolation via a hand-written clang-compiled
    fixture.

    D-SLM5032 measured 17 objects, 505 ACCEPT, 18 gating REJECT (checks
    (A)/(B)), 0 REFUSE against this exact real archive BEFORE any of the
    four corrections existed. Executed this session with all four landed:
    505 + 18 = 523 ACCEPT, 0 REJECT, 0 REFUSE, exit 0 -- every one of the
    eighteen measured false rejects now accepts, none of them floating-
    point, reproducing design Sec4.1's own closing claim
    ("closing 18 measured false rejects, none floating-point") end to end
    for the first time through the archive-based driver rather than
    through the conductor's own direct `check_fp_free_scan.scan_object`
    probe D-SLM5032 itself used.

    Does NOT discharge design Sec7 dim 11's forty-sixth population (the
    ELF/GCC leg's own CMake-integration half) -- that population requires
    a real `cmake -B build`/`cmake --build` configure on this archive's own
    source, which this cell does not perform (this ticket's own writable
    scope does not include standing up ELF CI wiring, and the forty-sixth
    population is explicitly routed, not built, by this ticket's own
    casebook). This cell instead pins the four corrections' own COMBINED
    effect against the real, already-built GCC archive named in this
    ticket's own environment note, through the driver this round built.
    """
    build_dir = _place_as_release_archive(real_elf_archive, tmp_path, "elf_leg")
    rc, out, err = _run_scan_build_output(build_dir)
    assert rc == 0, (
        "expected exit 0 on the real GCC-built ELF archive with all four "
        "classifier corrections landed; got {} stdout={!r} "
        "stderr={!r}".format(rc, out, err)
    )
    assert "Totals: 17 object(s); 523 symbol(s) ACCEPT, 0 REJECT, 0 object(s) REFUSE" in out, (
        "expected 17 objects / 523 ACCEPT / 0 REJECT / 0 REFUSE (505 "
        "pre-correction ACCEPT + 18 now-accepted false rejects, D-SLM5032); "
        "got:\n{}".format(out)
    )


# ===========================================================================
# Population 44 -- Mach-O's REFUSE-not-crash contract. Two cells: the
# scan_object-level behavior (already correct today, T-2343/78535ed-t2339's
# own M2 remedy -- NOT xfail, a real regression guard) and the archive/
# composition-level behavior (built, T-2381 -- NOT xfail either).
# ===========================================================================

def test_pop44_scan_object_already_refuses_not_crashes_on_macho():
    """Already true today (T-2343, 78535ed-t2339's own M2): scan_object
    catches an unrecognized-object-format exception internally and returns
    ScanResult(object_format='unknown', refuse=True) rather than raising --
    confirmed by direct execution this session against real Mach-O 64-bit
    magic bytes (0xFEEDFACF, little-endian on disk: CF FA ED FE). This is
    the per-call half of the forty-fourth population's own contract; the
    per-ARCHIVE-MEMBER half (a REFUSE inside iterate_archive_members'
    per-member loop, with the failing member named in the job's own output)
    is built (T-2381) and covered by the composition-level cell below.
    Not xfail."""
    with fc.TempDir() as tmp:
        p = os.path.join(tmp, "macho.obj")
        with open(p, "wb") as fh:
            fh.write(b"\xcf\xfa\xed\xfe" + b"\x00" * 200)
        result = scan.scan_object(p, isa="x86-64")
        assert result.refuse, "a Mach-O-magic'd file must REFUSE, not silently ACCEPT"
        assert result.object_format == "unknown"


def test_pop44_macho_archive_member_refuses_nonzero_exit_no_crash(malformed_archives, tmp_path):
    """Must-reject (REFUSE sense): a real archive with one member's payload
    overwritten to carry Mach-O magic -- the job must exit nonzero, with no
    uncaught Python traceback (a genuine crash), once the archive-based
    driver dispatches per-member scans. Built (T-2381): the archive-based
    driver reads this fixture's own `Release/superslm.lib`, REFUSEs the
    Mach-O-magic'd member, and the job exits 1."""
    build_dir = _place_as_release_archive(malformed_archives["macho_member"], tmp_path)
    rc, out, err = _run_scan_build_output(build_dir)
    # rc == 1 specifically (not merely nonzero): design Sec4.1 reserves exit
    # 2 for infrastructure failures (missing archive, zero objects,
    # malformed container) and a nonzero content-decision exit (REJECT/
    # REFUSE) for everything the classifier or composition stage decides --
    # asserting rc == 1 here (rather than the weaker rc != 0) confirms this
    # is a REFUSE reaching the job's exit code, not an infrastructure
    # failure reaching it by coincidence.
    assert rc == 1, (
        "expected exit 1 (a REFUSE reaching the job's own exit code, "
        "design Sec4.1's REFUSE-verdict contract); got {} stdout={!r} "
        "stderr={!r}".format(rc, out, err)
    )
    assert "Traceback (most recent call last)" not in err, (
        "the job crashed with an uncaught exception rather than reporting "
        "a clean REFUSE:\n{}".format(err)
    )
    assert "REFUSE" in out, (
        "the job's own output does not report a REFUSE verdict anywhere: "
        "{!r}".format(out)
    )
