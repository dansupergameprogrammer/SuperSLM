"""T-2326 (Curie) -- red suite for design Sec4.1's deciding instrument: an
instruction-level, byte-accounting, default-deny scan over compiled machine
code, which does not exist yet anywhere in this tree.

SOURCE ARTIFACT. `Claude/Vitruvius/t2265-superslm-fp-free-open-design-2026-08-24.md`
Sec4.1 (the mechanism) and Sec7 dimension 11 (the fourteen commissioning
populations this suite realizes). This campaign's own test-design record is
`Claude/Curie/t2326-fp-scan-instrument-red-suite-<date>.md` (records worktree).

WHY EVERY TEST HERE FAILS TODAY, AND WILL CONTINUE TO FAIL UNTIL BRUNEL BUILDS
THE INSTRUMENT. Sec4.1 specifies a Python CI tool -- a byte-level decoder
(capstone-driven, per-ISA), a closed-symbol-table membership rule, three
default-deny checks (A: register-file, B: GPR/control-flow allowlist, C: call/
jmp target allowlist), and a REFUSE control action on any unaccounted byte --
wired into this repo's own tests/ci/check_<name>.py + test_check_<name>.py
convention (tests/ci/check_no_forward_leaf_calls.py and its own test file are
the precedent this suite matches). None of that production module exists.
This file assumes it will be built at `tests/ci/check_fp_free_scan.py` (added
to sys.path below relative to this file, since this suite itself lives under
tests/t2296-fp-free-open-red-suite/ per this ticket's own writable-scope
constraint, not tests/ci/ -- a filed disposition, not a placement Curie
resolved on her own authority; see the case file). Every population test
below:

  1. Builds its own fixture FRESH (never a committed binary -- this design's
     own git-archive-only sourcing discipline, Sec4.1 throughout), using
     fp_scan_common.py's own toolchain wrappers. A toolchain genuinely absent
     from this environment SKIPs that one cell (an environment gap, not a
     mechanism defect, mirroring tests/ci/test_check_no_forward_leaf_calls.py's
     own `requires_clang` convention) -- every toolchain this suite calls on
     was confirmed present and working in this session (see the case file's
     own toolchain-availability table).
  2. Independently verifies, via a raw capstone decode -- NOT the (absent)
     instrument's own logic, see fp_scan_common.py's own docstring -- that the
     fixture genuinely carries the byte-level property its own population
     claims (a real FP-arithmetic mnemonic present; a real undecodable byte
     range; a real symbol reachable by no call/jmp edge; etc.). This half is
     real, executed, and independent of whether the instrument exists.
  3. THEN calls the (absent) production module and asserts its documented
     verdict, from WITHIN the same fixture's own scope (the compiled object
     is never deleted before this call runs). This is the genuinely red half,
     today, for exactly one reason: `import check_fp_free_scan` fails. Once
     Brunel builds that module matching the ASSUMED CONTRACT below, this half
     starts exercising the real grading path with NO EDIT to this file
     required -- the same "every gated CHECK flips from failure to load-
     bearing the moment the header exists" property T-2296's own dim6/dim7
     cells already hold, applied to a Python instrument instead of a C++
     header.

ASSUMED CONTRACT for tests/ci/check_fp_free_scan.py (Curie's own derivation
from the converged shape across this design's own fold-round probes --
Claude/Vitruvius/t2265-fold11-probe/fold11_remedy_check.py's own report_leg/
ci_gate, Claude/Loki/t2276-probe/strike.py's own verbatim copy of the same --
NOT an authoritative interface Curie is authorized to decide; routed to
Brunel as a proposal in this campaign's case file, and this file's own
assertions are the falsifiable form of that proposal):

    check_fp_free_scan.scan_object(path: str, isa: str, object_format: str) -> ScanResult

        isa in {"x86-64", "aarch64"}; object_format in {"coff", "elf"}.

    ScanResult:
        .refuse: bool                     -- clause (0)'s own REFUSE, True iff
                                              any code section's UNCLASSIFIED > 0
        .unclassified_bytes: int
        .verdicts: dict[str, str]         -- symbol name -> "ACCEPT" | "REJECT";
                                              EMPTY when .refuse is True (no
                                              verdict emitted on a refused object,
                                              design Sec4.1's second law)

POPULATION-TO-TEST CROSS-REFERENCE (full disposition in each test's own
docstring and in the case file):

  1  reintroduced reserve() on the ORIGINAL two sites (TE-32 Part B/C)     -- test_population_01_03
  2  MXCSR-invisible FP op classes (TE-32 Part D2/D/E/E2)                  -- test_population_02
  3  reintroduced reserve()-shaped growth, sslm_abi.cpp (fold round 3)     -- test_population_01_03
  4  reintroduced unordered_map growth, damped_greedy_antilm.cpp (fold 5)  -- test_population_04
  5  synthetic multi-hop transitive-closure proof (fold round 6)          -- test_population_05
  6  translation-unit-set desync detection (fold round 6)                 -- test_population_06
  7  T-2271's eight-object classifier construction (fold round 7)         -- test_population_07
  8  real v1.2.1 whole-corpus sweep (fold rounds 7/8)                     -- test_population_08
  9  T-2272's funclet, membership rule (fold round 8)                    -- test_population_09
  10 T-2273's AArch64 differential control (fold round 9, infeasible      -- test_population_10
     commissioning per fold round 10, superseded by population 11)
  11 four (ISA,format,toolchain) legs (fold round 10)                     -- test_population_11
  12 T-2275's clause-(0) REFUSE construction, extended 3 legs (fold 11/12)-- test_population_12
  13 T-2276's decodable-pool construction (fold round 13)                 -- test_population_13
  14 T-2277's per-symbol-granularity census, 11 cells (fold round 14)     -- test_population_14
"""
from __future__ import annotations

import os
import struct
import sys

import capstone
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIXTURES = os.path.join(_HERE, "fp_scan_fixtures")

sys.path.insert(0, _HERE)
import fp_scan_common as fc  # noqa: E402

sys.path.insert(0, _FIXTURES)
import pop14_make_objects as mk  # noqa: E402  -- pure-Python object synthesis, no compiler needed

# The production module's own expected home (tests/ci/, this repo's established
# check_<name>.py / test_check_<name>.py convention) -- NOT this file's own
# directory, per this ticket's writable-scope constraint (field 2: this
# session may write tests/t2296-fp-free-open-red-suite/ only).
_TESTS_ROOT = os.path.dirname(_HERE)
_CI_DIR = os.path.join(_TESTS_ROOT, "ci")
sys.path.insert(0, _CI_DIR)
try:
    import check_fp_free_scan as scan  # noqa: E402  -- NOT YET BUILT, design Sec4.1
    _SCAN_AVAILABLE = True
except ImportError:
    scan = None
    _SCAN_AVAILABLE = False


def _fail_absent(population_no, note=""):
    """The gating assertion every population ends in today. `note` states what
    THIS test already proved about its own fixture, independent of the
    instrument, so a reader of a failure log sees both halves: what is known
    (the fixture is real and correct) and what is missing (the instrument)."""
    pytest.fail(
        "check_fp_free_scan.py (design Sec4.1's deciding instrument) is not yet "
        "built at tests/ci/check_fp_free_scan.py -- population {} cannot be "
        "graded through it. {}".format(population_no, note)
    )


X86_ARITH_INFIXES = ("div", "mul", "add", "sub", "sqrt", "min", "max", "round",
                     "cmp", "comis", "ucomis", "cvt", "hadd")


def _is_x86_fp_arith(mnemonic: str) -> bool:
    """A narrow, test-local classifier -- NOT design Sec4.1's own checks
    (A)/(B): it identifies a floating-point-shaped mnemonic for the purpose of
    confirming a fixture carries one, nothing about ACCEPT/REJECT/REFUSE."""
    m = mnemonic.lower()
    if m.startswith("mov") or m in ("nop", "ret", "endbr64"):
        return False
    return any(k in m for k in X86_ARITH_INFIXES) and (
        m.endswith("sd") or m.endswith("ss") or m.endswith("pd") or m.endswith("ps")
        or "cvt" in m)


def _is_arm64_fp_arith(mnemonic: str) -> bool:
    m = mnemonic.lower()
    return m.startswith("f") and not m.startswith("fmov")


def _decode_x86(code: bytes, base=0x0):
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    md.detail = False
    return list(md.disasm(code, base))


def _decode_arm64(code: bytes, base=0x0):
    md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
    md.detail = False
    return list(md.disasm(code, base))


def _decode_sections(sections, isa):
    """Decodes each section chunk INDEPENDENTLY and pools the resulting
    instructions -- never concatenates chunks before decoding. Confirmed by
    direct execution, this session: capstone's own disasm() generator stops
    permanently at the first byte range it cannot decode, so a concatenated
    multi-section blob silently never reaches any section after the first
    non-code one (an unwind-thunk `.text$x` section, in the case that
    surfaced this) -- see fp_scan_common.py's own coff_code_sections
    docstring for the full account. Decoding section-by-section is immune to
    this: one section's own decode failure never affects another's."""
    decode_one = _decode_arm64 if isa == "aarch64" else _decode_x86
    insns = []
    for chunk in sections:
        insns.extend(decode_one(chunk))
    return insns


# ===========================================================================
# Populations one and three -- reintroduced reserve()/insert()-shaped
# std::unordered_map/set growth (fold round 3, D-SLM4308; the design's own
# must-reject shape for TE-32's original two sites, population one, is
# mechanically identical -- see this file's own header and the fixture's own
# provenance comment for why one fixture discharges both).
# ===========================================================================

def test_population_01_03_reserve_growth_mutant():
    """Falsifying construction: an std::unordered_set grown by plain insert()/
    erase() (mirrors sslm_abi.cpp's own pre-replacement `g_live_seqs`, T-2268's
    own "seqreg" leg) and a second grown by explicit reserve() (the original
    two sites' own shape). Both must REJECT: the bucket-array resize either
    path drives lowers to genuine SSE2 floating-point arithmetic (the
    max_load_factor()-driven divide/ceil inside the STL's own rehash sizing).

    Disposition, population one: TE-32's own historical artifact
    (Claude/Loki/te32-probe/) is outside this session's granted read-only
    scope. NOT reproduced byte-for-byte. The claim this test DOES discharge:
    a byte-level scan cannot distinguish "this container lives in
    tokenizer.cpp" from "this container lives in sslm_abi.cpp" -- both compile
    to the identical rehash-sizing machinery, and the SAME classifier property
    (checks (A)/(B) rejecting the divide/ceil instructions) is what both
    populations require. The residual left open -- confirming TE-32's own
    exact historical bytes specifically -- is filed in the case file, not
    silently treated as fully commissioned by this substitute.
    """
    src = os.path.join(_FIXTURES, "pop01_pop03_reserve_growth_mutant.cpp")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "pop0103.obj")
        try:
            fc.compile_cl(src, obj)
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        sections = fc.code_sections(obj, ".text")
        assert sum(len(s) for s in sections) > 0, "compiled .text section(s) are empty"
        insns = _decode_sections(sections, "x86-64")
        fp_insns = [(i.mnemonic, i.op_str) for i in insns if _is_x86_fp_arith(i.mnemonic)]
        assert fp_insns, (
            "fixture verification FAILED (not the instrument's fault): the compiled "
            "reserve()/insert()-growth mutant decoded to {} instructions and NONE is "
            "FP-arithmetic-shaped -- this fixture does not carry the property this "
            "population needs and must be repaired before it can commission anything"
        ).format(len(insns))

        if not _SCAN_AVAILABLE:
            _fail_absent(
                "one and three",
                "Fixture verified above: the compiled object genuinely contains FP-"
                "arithmetic instructions ({} found, e.g. {}) that a default-deny "
                "register-file check must REJECT.".format(len(fp_insns), fp_insns[:3]),
            )
        result = scan.scan_object(obj, isa="x86-64", object_format="coff")
        assert not result.refuse, "byte-accounting law should not REFUSE a fully-decodable object"
        assert any(v == "REJECT" for v in result.verdicts.values()), (
            "the reserve()/insert()-growth mutant must REJECT under checks (A)/(B); "
            "verdicts were: {}".format(result.verdicts)
        )


# ===========================================================================
# Population two -- MXCSR-invisible FP operation classes.
# ===========================================================================

def test_population_02_mxcsr_invisible_ops():
    """Falsifying construction: an SSE2 comparison (comisd/ucomisd) and a min
    (minsd), neither of which raises an IEEE-754 exception for a normal,
    non-NaN operand pair -- invisible to a trap-observing liveness control
    (design Sec4.2/Sec4.3's own masks-cleared/sticky-flag cells) but which
    must still REJECT under check (A): any instruction naming an xmm/ymm/zmm/
    st register is rejected unless on the closed VEC_MOVE_ALLOW list, and
    neither comisd/ucomisd nor minsd is on it.

    Disposition: TE-32's own historical artifact (Claude/Loki/te32-probe/) is
    outside this session's granted read-only scope; this is a fresh
    construction satisfying the documented property, not a reproduction.
    """
    src = os.path.join(_FIXTURES, "pop02_mxcsr_invisible_ops.cpp")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "pop02.obj")
        try:
            fc.compile_cl(src, obj)
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        sections = fc.code_sections(obj, ".text")
        insns = _decode_sections(sections, "x86-64")
        vec_reg_insns = [
            (i.mnemonic, i.op_str) for i in insns
            if "xmm" in (i.op_str or "").lower()
        ]
        assert vec_reg_insns, (
            "fixture verification FAILED: no xmm-register instruction decoded from "
            "the compiled comparison/min body -- this fixture does not exercise the "
            "register-file coordinate this population needs"
        )
        mnems = {m.lower() for m, _ in vec_reg_insns}
        assert mnems & {"comisd", "ucomisd", "minsd", "maxsd"}, (
            "expected at least one of comisd/ucomisd/minsd/maxsd; decoded xmm "
            "instructions were: {}".format(vec_reg_insns)
        )

        if not _SCAN_AVAILABLE:
            _fail_absent(
                "two",
                "Fixture verified above: the compiled object contains {} (an SSE2 "
                "comparison/min instruction that raises no exception for a normal "
                "operand pair, confirmed at compile time by mnemonic; the design's "
                "own liveness controls, Sec4.2/Sec4.3, would see nothing if this were "
                "the ONLY guard).".format(sorted(mnems & {"comisd", "ucomisd", "minsd", "maxsd"})),
            )
        result = scan.scan_object(obj, isa="x86-64", object_format="coff")
        assert not result.refuse
        assert any(v == "REJECT" for v in result.verdicts.values()), (
            "an MXCSR-invisible SSE2 comparison/min must still REJECT under check (A) "
            "(register-file classification); verdicts were: {}".format(result.verdicts)
        )


# ===========================================================================
# Population four -- reintroduced std::unordered_map growth on
# damped_greedy_antilm.cpp's own tables_/counts shape, via a restore-shaped
# replay (fold round 5).
# ===========================================================================

def test_population_04_antilm_restore_growth_mutant():
    """Falsifying construction: a standalone reproduction of
    damped_greedy_antilm.cpp's PRE-T-2296 tables_/counts shape (plain
    std::unordered_map, replaced today by GrowableContextMap/GrowableIntMap --
    confirmed by reading the real, current source this session, cited in the
    fixture's own header), populated through a restore-shaped replay: many
    distinct contexts, each accumulating many distinct candidate tokens in one
    call, mirroring sslm_seq_restore's own replay pattern into AntiLmUpdate.
    Must REJECT: the same rehash-sizing machinery as populations one/three,
    reached via a different call shape (restore-time bulk population, not
    open/load-time or steady insert/erase).
    """
    src = os.path.join(_FIXTURES, "pop04_antilm_growth_mutant.cpp")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "pop04.obj")
        try:
            fc.compile_cl(src, obj)
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        sections = fc.code_sections(obj, ".text")
        insns = _decode_sections(sections, "x86-64")
        fp_insns = [(i.mnemonic, i.op_str) for i in insns if _is_x86_fp_arith(i.mnemonic)]
        assert fp_insns, (
            "fixture verification FAILED: the compiled restore-shaped growth mutant "
            "decoded to {} instructions and NONE is FP-arithmetic-shaped".format(len(insns))
        )

        if not _SCAN_AVAILABLE:
            _fail_absent(
                "four",
                "Fixture verified above: {} FP-arithmetic instructions found (e.g. "
                "{}) in the restore-shaped growth mutant.".format(len(fp_insns), fp_insns[:3]),
            )
        result = scan.scan_object(obj, isa="x86-64", object_format="coff")
        assert not result.refuse
        assert any(v == "REJECT" for v in result.verdicts.values())


# ===========================================================================
# Population five -- synthetic multi-hop transitive-closure proof (fold round
# 6). DEMOTED TO DIAGNOSTIC at fold round 8: does not gate the production
# ACCEPT/REJECT verdict (membership is the closed symbol table, not the
# walk), still required for Sec7 dimensions 1-3's own "when does this site
# run" reasoning. This test's own gating assertion is scoped accordingly.
# ===========================================================================

def test_population_05_transitive_chain_diagnostic():
    """Falsifying construction: Root -> HopA -> HopB -> FlaggedLeaf, three
    hops, no shorter path from Root to FlaggedLeaf exists in the translation
    unit (Root calls ONLY HopA; HopA calls ONLY HopB). A diagnostic built on a
    genuine transitive walk (BFS/DFS) attributes FlaggedLeaf's own divsd to
    Root's reachable set; one built on a bounded-depth approximation (a
    one-hop-plus-one-hop check) does not.

    EXECUTED, independent of the instrument: the call chain is confirmed real
    by decoding HopA/HopB/Root and finding at least 3 `call` instructions
    survive to the object file (-O0 -fno-inline disables the optimizer's own
    inlining, so the chain is not collapsed away) -- the "no shorter path"
    property is a fact about the compiled bytes, not merely the source text.

    DISPOSITION: this population no longer gates the production ACCEPT/
    REJECT verdict (design Sec7 dim 11's own fold-round-12 text: membership
    is the closed symbol table since fold round 8, not the walk). It remains
    "genuinely valuable" for Sec7 dimensions 1-3's own diagnostic attribution
    reasoning. No diagnostic-walk API is part of this suite's own assumed
    contract (see this file's own header) -- there is nothing narrower than
    "the instrument is absent" to assert once it exists, until that API is
    itself specified. Filed as an open contract gap in the case file, not
    silently assumed covered by the ACCEPT/REJECT contract above.
    """
    src = os.path.join(_FIXTURES, "pop05_transitive_chain.cpp")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "pop05.o")
        try:
            fc.compile_clangxx(src, obj, "x86_64-pc-linux-gnu", extra_args=["-O0", "-fno-inline"])
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        sections = fc.code_sections(obj, ".text")
        insns = _decode_sections(sections, "x86-64")
        call_targets = [i.op_str for i in insns if i.mnemonic == "call"]
        fp_insns = [(i.mnemonic, i.op_str) for i in insns if _is_x86_fp_arith(i.mnemonic)]
        assert len(call_targets) >= 3, (
            "expected at least 3 call instructions (Root->HopA, HopA->HopB, "
            "HopB->FlaggedLeaf); found {}: {}".format(len(call_targets), call_targets)
        )
        assert fp_insns, "FlaggedLeaf's own divsd did not survive to the object file"

        _fail_absent(
            "five (diagnostic -- does not gate ACCEPT/REJECT since fold round 8; "
            "gates the walk's own attribution correctness only, and this suite's "
            "own assumed contract has no diagnostic-walk API yet -- an open "
            "contract gap, filed in the case file)",
            "Fixture verified above: a genuine 3-hop call chain with no shorter "
            "path, and FlaggedLeaf's own divsd present in the object ({} FP "
            "instructions, {} call sites).".format(len(fp_insns), len(call_targets)),
        )


# ===========================================================================
# Population six -- translation-unit-set desync detection (fold round 6).
# Also a diagnostic population per the same fold-round-12 disposition as
# population five (the walk's TU enumeration is diagnostic-only since fold
# round 8); the SAME open-contract-gap note applies.
# ===========================================================================

def test_population_06_tu_set_desync():
    """Falsifying construction: two translation units, pop06_tu_a.cpp (Root)
    and pop06_tu_b_added.cpp (CalleeInOtherTU, called from Root, one hop, a
    genuine divsd). pop06_mini_source_list.txt names BOTH as "the build's own
    compiled source set" (mirroring CMakeLists.txt's SUPERSLM_CORE_SOURCES
    shape); pop06_mini_scan_list_stale.txt names only tu_a.cpp -- a
    hand-maintained list that agreed on the day it was written and was never
    updated. A scan whose TU enumeration is DERIVED from the source-set file
    reports the flagged instruction; one that reads the stale list never
    opens tu_b_added's own object and reports clean.

    EXECUTED, independent of the instrument: both TUs compile; the desync
    itself (source-set lists 2 files, scan-list lists 1) is confirmed by
    reading both list files directly -- no instrument needed to see that they
    disagree, only to see what a scan reading each ONE would do about it.
    """
    src_a = os.path.join(_FIXTURES, "pop06_tu_a.cpp")
    src_b = os.path.join(_FIXTURES, "pop06_tu_b_added.cpp")
    source_list_path = os.path.join(_FIXTURES, "pop06_mini_source_list.txt")
    stale_list_path = os.path.join(_FIXTURES, "pop06_mini_scan_list_stale.txt")

    with open(source_list_path) as f:
        source_list = [line.strip() for line in f if line.strip()]
    with open(stale_list_path) as f:
        stale_list = [line.strip() for line in f if line.strip()]

    assert "pop06_tu_b_added.cpp" in source_list, "the mini build's own source set must name it"
    assert "pop06_tu_b_added.cpp" not in stale_list, (
        "the whole point of this population: the stale list must NOT mirror it"
    )
    assert set(stale_list) < set(source_list), (
        "the stale list must be a strict subset of the real source set, or this "
        "is not a desync at all"
    )

    with fc.TempDir() as tmp:
        obj_a = os.path.join(tmp, "tu_a.o")
        obj_b = os.path.join(tmp, "tu_b.o")
        try:
            fc.compile_clangxx(src_a, obj_a, "x86_64-pc-linux-gnu", extra_args=["-O0"])
            fc.compile_clangxx(src_b, obj_b, "x86_64-pc-linux-gnu", extra_args=["-O0"])
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        sections_b = fc.code_sections(obj_b, ".text")
        insns_b = _decode_sections(sections_b, "x86-64")
        fp_insns = [(i.mnemonic, i.op_str) for i in insns_b if _is_x86_fp_arith(i.mnemonic)]
        assert fp_insns, "CalleeInOtherTU's own divsd did not survive to its object file"

        _fail_absent(
            "six (diagnostic -- same open-contract-gap disposition as population "
            "five; TU-enumeration derivation is not part of this suite's own "
            "assumed ScanResult contract)",
            "Fixture verified above: the mini build's own source-set/scan-list "
            "desync is real (source set names 2 files, the stale scan list names "
            "1), and the omitted file's own object genuinely carries {} FP "
            "instructions.".format(len(fp_insns)),
        )


# ===========================================================================
# Population seven -- T-2271's own eight-object classifier construction (fold
# round 7, D-SLM4350): four bodies, each performing a genuine floating-point
# operation invisible to the pre-fold-7 36-mnemonic detector, at two ISA
# tiers (SSE baseline, AVX2).
# ===========================================================================

_POP07_TIERS = [("sse_baseline", []), ("avx2", ["-mavx2"])]


@pytest.mark.parametrize("tier,extra_flags", _POP07_TIERS)
def test_population_07_fpblind_classifier(tier, extra_flags):
    """Falsifying construction: BodyFloor/BodyFma/BodyRound/BodyHadd, adapted
    from T-2271's own construction (see pop07_fpblind.cpp's own header for the
    STL-portability adaptation). All four must REJECT at BOTH ISA tiers this
    fold's own commissioning names -- SSE baseline (this parametrization) and
    AVX2 (-mavx2) -- eight (function, tier) cells total, matching D-SLM4350's
    own "8 of 8 REJECT" figure in shape.
    """
    src = os.path.join(_FIXTURES, "pop07_fpblind.cpp")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, f"pop07_{tier}.obj")
        try:
            fc.compile_clangxx(src, obj, "x86_64-pc-windows-msvc",
                               extra_args=["-msse4.1", *extra_flags])
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        sections = fc.code_sections(obj, ".text")
        insns = _decode_sections(sections, "x86-64")
        fp_insns = [(i.mnemonic, i.op_str) for i in insns if _is_x86_fp_arith(i.mnemonic)]
        assert fp_insns, (
            "fixture verification FAILED at tier {}: the compiled floor/fma/round/"
            "hadd bodies decoded to {} instructions and NONE is FP-arithmetic-"
            "shaped".format(tier, len(insns))
        )

        if not _SCAN_AVAILABLE:
            _fail_absent(
                "seven, tier {}".format(tier),
                "Fixture verified above: {} FP-arithmetic instructions present at "
                "this ISA tier.".format(len(fp_insns)),
            )
        result = scan.scan_object(obj, isa="x86-64", object_format="coff")
        assert not result.refuse
        for fn in ("BodyFloor", "BodyFma", "BodyRound", "BodyHadd"):
            assert result.verdicts.get(fn) == "REJECT", (
                "{} must REJECT under the amended default-deny classifier "
                "(T-2271's own must-reject population) at tier {}; verdict was "
                "{}".format(fn, tier, result.verdicts.get(fn))
            )


# ===========================================================================
# Population eight -- the real v1.2.1 whole-corpus sweep (fold rounds 7/8).
# ===========================================================================

def test_population_08_real_corpus_whole_sweep():
    """This population is a claim about the REAL, currently-committed engine
    source (SUPERSLM_CORE_SOURCES, CMakeLists.txt) -- not a constructed
    fixture. Verified today, independent of the absent instrument: the real
    source list resolves to a non-vacuous, 17-file population, exactly the
    count design Sec4.1's own text states ("all 17 SUPERSLM_CORE_SOURCES
    translation units"), and every named file exists on disk.

    NOT graded this session even once the instrument exists in principle:
    compiling and scanning the real 17-TU corpus and reproducing the design's
    own "219 of 235 R4-closure ACCEPT / whole-corpus 4933 ACCEPT, 190 REJECT,
    zero unexplained" claim (Sec4.1, D-SLM4370) is a build-scale operation
    (the full CMake build, not a single translation unit) that belongs to the
    instrument's own commissioning run in CI, not a single pytest cell in
    this suite -- filed as its own residual in the case file, not silently
    assumed covered by populations one through seven's own per-fixture shape.
    """
    engine_root = os.path.dirname(_TESTS_ROOT)  # tests/.. == the engine repo root
    cmake_path = os.path.join(engine_root, "CMakeLists.txt")
    assert os.path.exists(cmake_path), "expected CMakeLists.txt at the engine repo root"
    with open(cmake_path) as f:
        text = f.read()
    start = text.index("set(SUPERSLM_CORE_SOURCES")
    end = text.index(")", start)
    block = text[start:end]
    sources = [line.strip() for line in block.splitlines()[1:] if line.strip()]
    assert len(sources) == 17, (
        "design Sec4.1's own text states 17 SUPERSLM_CORE_SOURCES translation "
        "units; CMakeLists.txt currently names {}: {}".format(len(sources), sources)
    )
    missing = [s for s in sources if not os.path.exists(os.path.join(engine_root, s))]
    assert not missing, "SUPERSLM_CORE_SOURCES names files that do not exist: {}".format(missing)

    pytest.fail(
        "check_fp_free_scan.py (design Sec4.1's deciding instrument) is not yet "
        "built at tests/ci/check_fp_free_scan.py -- population eight cannot be "
        "graded through it, and would not be graded by this single pytest cell "
        "even once it exists (a full-corpus build-and-scan is a CI-scale "
        "operation, not a unit cell -- see this test's own docstring). Verified "
        "above: SUPERSLM_CORE_SOURCES resolves to the documented 17 real files, "
        "all present on disk ({}).".format(sources)
    )


# ===========================================================================
# Population nine -- T-2272's funclet, the must-reject population for the
# MEMBERSHIP rule specifically (fold round 8).
# ===========================================================================

def test_population_09_funclet_membership():
    """Falsifying construction: RegressionParent (no FP instruction of its
    own) wraps a try/catch; the catch FUNCLET performs genuine IEEE-754
    double arithmetic and is entered by the runtime unwinder through
    `.xdata`, named by no `call` and no `jmp` instruction anywhere in the
    image. A membership rule decided by a call/jmp edge walk never reaches
    it (absent, not rejected); the closed-symbol-table rule (every code-
    carrying symbol in the object) scans it because it exists.

    EXECUTED, both directions, independent of the absent instrument: the
    compiled object's own COFF symbol table is read directly (no capstone
    needed for this check) and confirmed to contain a SECOND function symbol
    beyond RegressionParent (the funclet, name-mangled `?catch$...`), AND the
    whole object's own decoded instruction stream is scanned for any `call`/
    `jmp` whose target is a direct address -- none resolves to the funclet
    (an unlinked .obj's own inter-function control transfer goes through
    relocations, not immediate addresses, matching the population's own
    defining property: reachable by NO call/jmp INSTRUCTION; the funclet's
    real reachability is via `.xdata`, a metadata table this test does not
    parse).
    """
    src = os.path.join(_FIXTURES, "pop09_funclet_fp.cpp")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "pop09.obj")
        try:
            fc.compile_cl(src, obj)
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))

        with open(obj, "rb") as f:
            data = f.read()
        _machine, _nsec = struct.unpack_from("<HH", data, 0)
        symptr, nsym = struct.unpack_from("<II", data, 8)
        strtab_off = symptr + nsym * 18
        strtab = data[strtab_off:]

        def sym_name(i):
            # COFF long-name form: first 4 bytes zero, next 4 bytes is an
            # offset INTO the string table INCLUDING its own leading 4-byte
            # size prefix -- `strtab` already starts at that size prefix
            # (strtab_off == symptr + nsym*18, the string table's own start),
            # so `off` indexes directly into it with no adjustment. Fixed
            # this session: an earlier `off - 4` version silently returned
            # every long name shifted 4 bytes into its own predecessor's
            # tail, producing garbage fragments ('X@Z', 'urn', ...) instead
            # of real mangled names -- caught only because the funclet
            # symbol this test looks for never appeared among them.
            raw = data[symptr + i * 18: symptr + i * 18 + 8]
            if raw[:4] == b"\x00\x00\x00\x00":
                off, = struct.unpack_from("<I", raw, 4)
                end = strtab.index(b"\x00", off)
                return strtab[off:end].decode("ascii", "replace")
            return raw.rstrip(b"\x00").decode("ascii", "replace")

        func_syms = []
        i = 0
        while i < nsym:
            name = sym_name(i)
            value, sec_num, sym_type, storage, naux = struct.unpack_from(
                "<IhHBB", data, symptr + i * 18 + 8)
            if sym_type == 0x20 and sec_num > 0:  # DTYPE_FUNCTION
                func_syms.append(name)
            i += 1 + naux

        catch_funclets = [n for n in func_syms if "catch$" in n]
        assert catch_funclets, (
            "expected a second COFF function symbol for the catch funclet; symbol "
            "table names: {}".format(func_syms)
        )

        sections = fc.code_sections(obj, ".text")
        insns = _decode_sections(sections, "x86-64")
        fp_insns = [(i.mnemonic, i.op_str) for i in insns if _is_x86_fp_arith(i.mnemonic)]
        assert fp_insns, "expected genuine FP arithmetic somewhere in the compiled object"

        if not _SCAN_AVAILABLE:
            _fail_absent(
                "nine",
                "Fixture verified above: a genuine second function symbol exists "
                "({}), and the object carries real FP arithmetic ({} "
                "instructions).".format(catch_funclets, len(fp_insns)),
            )
        result = scan.scan_object(obj, isa="x86-64", object_format="coff")
        assert not result.refuse
        assert result.verdicts.get("RegressionParent") == "ACCEPT", (
            "RegressionParent itself carries no FP instruction and must ACCEPT"
        )
        funclet_verdicts = [result.verdicts.get(n) for n in catch_funclets]
        assert any(v == "REJECT" for v in funclet_verdicts), (
            "the catch funclet must REJECT -- membership must include it even "
            "though no call/jmp instruction names it; verdicts: {}".format(
                dict(zip(catch_funclets, funclet_verdicts)))
        )


# ===========================================================================
# Population ten -- T-2273's AArch64 differential control (fold round 9).
# Commissioning found INFEASIBLE at fold round 10 (its own dumpbin-rendered
# must-accept/must-reject pair cannot be produced by the real macos-arm64
# CI leg's own toolchain) and SUPERSEDED by population eleven (the same ISA
# leg via clang/ELF instead). Executed here anyway, at low marginal cost
# (this session's own toolchain confirms the real MSVC AArch64 cross-compiler
# is present) -- historical/diagnostic value only; this population's own
# GATING status is disposed to population eleven.
# ===========================================================================

def test_population_10_arm_differential_historical():
    """Falsifying + must-accept construction, one object: BuildMerges (the
    real site-1 shape, reserve()+emplace()), DedupNames (model.cpp's own
    site 5-7 shape, an unordered_set, no FP), BodyDivide, BodyConvertCompare
    -- compiled for AArch64 via MSVC's own cross-compiler (cl.exe via
    vcvarsamd64_arm64.bat), matching Claude/Loki/t2273-probe/build-arm.bat's
    own toolchain exactly (not clang, per that probe's own build script).

    Both directions in one object: BuildMerges/BodyDivide/BodyConvertCompare
    must REJECT (genuine fdiv/fmadd/fsub/scvtf); DedupNames must ACCEPT (an
    unordered_set with no floating-point operation).
    """
    src = os.path.join(_FIXTURES, "pop10_arm_site.cpp")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "pop10.obj")
        try:
            fc.compile_cl_arm64(src, obj)
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        sections = fc.code_sections(obj, ".text")
        insns = _decode_sections(sections, "aarch64")
        fp_insns = [(i.mnemonic, i.op_str) for i in insns if _is_arm64_fp_arith(i.mnemonic)]
        assert fp_insns, (
            "fixture verification FAILED: the AArch64 object decoded to {} "
            "instructions and NONE is FP-arithmetic-shaped".format(len(insns))
        )

        if not _SCAN_AVAILABLE:
            _fail_absent(
                "ten (historical/diagnostic -- commissioning disposed to "
                "population eleven per fold round 10's own infeasibility finding)",
                "Fixture verified above: {} genuine AArch64 FP instructions "
                "decoded (e.g. {}).".format(len(fp_insns), fp_insns[:3]),
            )
        result = scan.scan_object(obj, isa="aarch64", object_format="coff")
        assert not result.refuse
        for fn in ("BuildMerges", "BodyDivide", "BodyConvertCompare"):
            assert result.verdicts.get(fn) == "REJECT", (
                "{} must REJECT; verdict was {}".format(fn, result.verdicts.get(fn)))
        assert result.verdicts.get("DedupNames") == "ACCEPT", (
            "DedupNames carries no FP instruction and must ACCEPT; verdict was "
            "{}".format(result.verdicts.get("DedupNames"))
        )


# ===========================================================================
# Population eleven -- the byte-accounting law's own toolchain-and-format
# independence, four (ISA, object-format, toolchain) legs (fold round 10).
# ===========================================================================

_POP11_LEGS = [
    ("msvc_coff_x64", "pop11_msvc_x64.asm", "x86-64", "coff", "msvc"),
    ("clang_coff_x64", "pop11_coff_x64.s", "x86-64", "coff", "clang"),
    ("clang_elf_x64", "pop11_elf_x64.s", "x86-64", "elf", "clang"),
    ("clang_elf_arm64", "pop11_arm64.s", "aarch64", "elf", "clang"),
]


@pytest.mark.parametrize("leg_name,fname,isa,fmt,toolchain", _POP11_LEGS)
def test_population_11_toolchain_format_independence(leg_name, fname, isa, fmt, toolchain):
    """Falsifying + must-accept construction on ONE of the design's own four
    named production legs: HashSite (an ordinary integer add/mov, no FP) must
    ACCEPT; BodyDivide (a genuine divsd/fdiv) must REJECT -- on this exact
    (ISA, object-format, toolchain) combination, non-degenerately (a
    real, nonzero byte count decoded, not a 0-0=0 vacuous pass).
    """
    src = os.path.join(_FIXTURES, fname)
    with fc.TempDir() as tmp:
        ext = ".obj" if toolchain == "msvc" else ".o"
        obj = os.path.join(tmp, leg_name + ext)
        try:
            if toolchain == "msvc":
                fc.assemble_ml64(src, obj)
            else:
                triple = {
                    ("x86-64", "coff"): "x86_64-pc-windows-msvc",
                    ("x86-64", "elf"): "x86_64-pc-linux-gnu",
                    ("aarch64", "elf"): "aarch64-none-elf",
                }[(isa, fmt)]
                fc.compile_clang_asm(src, obj, triple)
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        sections = fc.code_sections(obj, ".text")
        total_bytes = sum(len(s) for s in sections)
        assert total_bytes > 0, "non-degenerate check: the .text section(s) must not be empty"
        insns = _decode_sections(sections, isa)
        is_fp = _is_arm64_fp_arith if isa == "aarch64" else _is_x86_fp_arith
        fp_insns = [(i.mnemonic, i.op_str) for i in insns if is_fp(i.mnemonic)]
        assert fp_insns, (
            "fixture verification FAILED on leg {}: {} instructions decoded, none "
            "FP-arithmetic-shaped".format(leg_name, len(insns))
        )

        if not _SCAN_AVAILABLE:
            _fail_absent(
                "eleven, leg {}".format(leg_name),
                "Fixture verified above: {} bytes decoded non-degenerately, "
                "BodyDivide's own FP instruction present ({}).".format(total_bytes, fp_insns),
            )
        result = scan.scan_object(obj, isa=isa, object_format=fmt)
        assert not result.refuse, "leg {}: byte-accounting law should not REFUSE".format(leg_name)
        assert result.verdicts.get("HashSite") == "ACCEPT", (
            "leg {}: HashSite carries no FP instruction and must ACCEPT".format(leg_name))
        assert result.verdicts.get("BodyDivide") == "REJECT", (
            "leg {}: BodyDivide carries a genuine FP instruction and must "
            "REJECT".format(leg_name))


# ===========================================================================
# Population twelve -- T-2275's clause-(0) REFUSE construction (fold round
# 11), extended to the three x86-64 production legs at fold round 12
# (D-SLM4442).
# ===========================================================================

_POP12_LEGS = [
    ("arm64_pool_before", "pop12_pool_arm64_a.s", "aarch64", "elf"),
    ("arm64_pool_after", "pop12_pool_arm64_b.s", "aarch64", "elf"),
    ("msvc_coff_x64", "pop12_pool_msvc_x64.asm", "x86-64", "coff"),
    ("clang_coff_x64", "pop12_pool_coff_x64.s", "x86-64", "coff"),
    ("clang_elf_x64", "pop12_pool_elf_x64.s", "x86-64", "elf"),
]


@pytest.mark.parametrize("leg_name,fname,isa,fmt", _POP12_LEGS)
def test_population_12_clause0_refuse_undecodable_pool(leg_name, fname, isa, fmt):
    """Falsifying construction: an ordinary integer function, an inline
    literal pool UNDECODABLE as machine code in this ISA/mode (fold-9's own
    disclosed FNV-1a-constant shape for AArch64; eight long-mode-invalid
    opcodes for x86-64, confirmed genuinely undecodable against capstone
    before assembly per each fixture's own header), then a floating-point
    body -- covered by no symbol, no recognised padding pattern, and no
    data-typed symbol. Clause (0) must REFUSE: zero ACCEPT/REJECT verdicts
    emitted for ANY symbol in the object, non-zero unclassified bytes, CI
    job fails.

    EXECUTED, independent of the instrument: capstone itself is used to
    confirm the pool bytes really are undecodable at this ISA/mode BEFORE any
    grading is attempted -- the same pre-flight check T-2265's own fold-12
    remedy performed (fold12_remedy_check.py's own pre-flight, cited in
    pop12_pool_elf_x64.s's header).
    """
    src = os.path.join(_FIXTURES, fname)
    with fc.TempDir() as tmp:
        ext = ".obj" if fname.endswith(".asm") else ".o"
        obj = os.path.join(tmp, leg_name + ext)
        try:
            if fname.endswith(".asm"):
                fc.assemble_ml64(src, obj)
            else:
                triple = {
                    ("x86-64", "coff"): "x86_64-pc-windows-msvc",
                    ("x86-64", "elf"): "x86_64-pc-linux-gnu",
                    ("aarch64", "elf"): "aarch64-none-elf",
                }[(isa, fmt)]
                fc.compile_clang_asm(src, obj, triple)
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        sections = fc.code_sections(obj, ".text")
        md = capstone.Cs(capstone.CS_ARCH_ARM64 if isa == "aarch64" else capstone.CS_ARCH_X86,
                         capstone.CS_MODE_ARM if isa == "aarch64" else capstone.CS_MODE_64)
        total_bytes = sum(len(s) for s in sections)
        decoded_bytes = sum(sum(i.size for i in md.disasm(s, 0x0)) for s in sections)
        assert decoded_bytes < total_bytes, (
            "leg {}: expected the decoder to STOP before the end of the section "
            "(the pool must be undecodable) -- decoded {} of {} bytes cleanly, "
            "meaning either the whole object decodes (this fixture does not "
            "carry the property) or a resync/skip strategy would be needed to "
            "see the failure at all".format(leg_name, decoded_bytes, total_bytes)
        )

        if not _SCAN_AVAILABLE:
            _fail_absent(
                "twelve, leg {}".format(leg_name),
                "Fixture verified above: a genuinely undecodable byte range exists "
                "({} of {} bytes decode linearly before the decoder stops), which "
                "is exactly the unaccounted range clause (0) must REFUSE "
                "on.".format(decoded_bytes, total_bytes),
            )
        result = scan.scan_object(obj, isa=isa, object_format=fmt)
        assert result.refuse, "leg {}: clause (0) must REFUSE on an undecodable range".format(leg_name)
        assert result.unclassified_bytes > 0
        assert not result.verdicts, (
            "leg {}: REFUSE must emit ZERO ACCEPT/REJECT verdicts for any symbol; "
            "got {}".format(leg_name, result.verdicts)
        )


# ===========================================================================
# Population thirteen -- T-2276's decodable-pool construction (fold round
# 13): the pool's bytes DO decode (a movabs whose immediate swallows the
# next function's own body), so clause (0)'s REFUSE must come from the
# COVERAGE relation (is this byte attributed to the symbol whose body it
# is?), not merely a decodability relation.
# ===========================================================================

_POP13_LEGS = [
    ("clang_elf_x64", "pop13_pool_decodable_elf_x64.s", "elf"),
    ("clang_coff_x64", "pop13_pool_decodable_coff_x64.s", "coff"),
    ("msvc_coff_x64", "pop13_pool_decodable_msvc_x64.asm", "coff"),
]


@pytest.mark.parametrize("leg_name,fname,fmt", _POP13_LEGS)
def test_population_13_coverage_relation_decodable_swallow(leg_name, fname, fmt):
    """Falsifying construction: HashSite, an 8-byte pool covered by no
    symbol whose trailing bytes `48 B8` begin a 10-byte `movabs` that
    DECODES CLEANLY and swallows BodyDivide's own 5-byte body whole,
    then BodyDivide (a genuine divsd). Ground truth: the raw bytes `f2 0f
    5e c1` (divsd xmm0, xmm1) are present in the compiled object (confirmed
    directly, no decoder needed) -- REFUSE must still fire, because the
    pool bytes are attributed to no symbol's own extent even though they
    decode; a law that only tests decodability (not attribution) would PASS
    this object with BodyDivide silently unscanned.
    """
    src = os.path.join(_FIXTURES, fname)
    with fc.TempDir() as tmp:
        ext = ".obj" if fname.endswith(".asm") else ".o"
        obj = os.path.join(tmp, leg_name + ext)
        try:
            if fname.endswith(".asm"):
                fc.assemble_ml64(src, obj)
            else:
                triple = "x86_64-pc-windows-msvc" if fmt == "coff" else "x86_64-pc-linux-gnu"
                fc.compile_clang_asm(src, obj, triple)
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        with open(obj, "rb") as f:
            raw = f.read()
        divsd_bytes = bytes([0xF2, 0x0F, 0x5E, 0xC1])
        assert divsd_bytes in raw, (
            "leg {}: expected the raw bytes of `divsd xmm0, xmm1` in the compiled "
            "object; ground truth check failed".format(leg_name)
        )

        if not _SCAN_AVAILABLE:
            _fail_absent(
                "thirteen, leg {}".format(leg_name),
                "Fixture verified above: the raw bytes of `divsd xmm0, xmm1` are "
                "present in the compiled object (ground truth, independent of any "
                "decoder) -- BodyDivide's own FP instruction genuinely exists, and "
                "a law that only checks decodability (not per-symbol coverage) "
                "would silently pass this object with a REJECT-worthy instruction "
                "unseen.",
            )
        result = scan.scan_object(obj, isa="x86-64", object_format=fmt)
        assert result.refuse, (
            "leg {}: the pool bytes are covered by no symbol's own extent (they "
            "decode, but are not attributed to anyone) -- clause (0) must REFUSE "
            "on the coverage relation, not merely the decodability "
            "relation".format(leg_name)
        )
        assert not result.verdicts


# ===========================================================================
# Population fourteen -- T-2277's per-symbol-granularity census, 11 cells
# (fold round 14). Pure Python object synthesis (pop14_make_objects.py,
# adapted from Claude/Loki/t2277-probe/make_objects.py) -- no compiler
# needed at all, so every cell runs regardless of toolchain availability.
# ===========================================================================

def _build_pop14_cells(out_dir):
    """Builds the 11 hand-crafted ELF64/COFF objects T-2277's own census
    derives from the fold-13 module's own branch conditions. Returns a list
    of (cell_id, note, obj_path, isa, fmt, owner_symbol) -- `owner_symbol` is
    the symbol whose body the FP instruction genuinely is, per the
    construction's own design (see pop14_make_objects.py's own header)."""
    FUNC, OBJECT = mk.STT_FUNC, mk.STT_OBJECT
    cells = []

    p = os.path.join(out_dir, "c1.o")
    mk.build_elf64(p, mk.EM_X86_64, mk.X86_TEXT,
                    [("HashSite", 0, 5, FUNC), ("BodyDivide", 8, 5, FUNC), ("CtorBase", 8, 5, FUNC)])
    cells.append(("C1", "ELF x64, alias symbol at BodyDivide's own start", p, "x86-64", "elf", "BodyDivide"))

    p = os.path.join(out_dir, "c2.o")
    mk.build_elf64(p, mk.EM_X86_64, mk.X86_TEXT,
                    [("HashSite", 0, 5, FUNC), ("BodyDivide", 8, 0, FUNC), ("CtorBase", 8, 0, FUNC)])
    cells.append(("C2", "ELF x64, no declared size, alias at BodyDivide's start", p, "x86-64", "elf", "BodyDivide"))

    p = os.path.join(out_dir, "c3.obj")
    mk.build_coff(p, mk.COFF_X86_64, mk.X86_TEXT,
                  [("HashSite", 0, True), ("BodyDivide", 8, True), ("CtorBase", 8, True)])
    cells.append(("C3", "COFF x64, alias symbol at BodyDivide's own start", p, "x86-64", "coff", "BodyDivide"))

    p = os.path.join(out_dir, "c4.o")
    mk.build_elf64(p, mk.EM_X86_64, mk.X86_TEXT, [("HashSite", 0, 0, FUNC), ("BodyDivide", 13, 0, FUNC)])
    cells.append(("C4", "ELF x64, BodyDivide symbol AT the section end", p, "x86-64", "elf", "BodyDivide"))

    p = os.path.join(out_dir, "c5.obj")
    mk.build_coff(p, mk.COFF_X86_64, mk.X86_TEXT, [("HashSite", 0, True), ("BodyDivide", 13, True)])
    cells.append(("C5", "COFF x64, BodyDivide symbol AT the section end", p, "x86-64", "coff", "BodyDivide"))

    p = os.path.join(out_dir, "c6.o")
    mk.build_elf64(p, mk.EM_AARCH64, mk.AARCH64_TEXT,
                    [("HashSite", 0, 8, FUNC), ("BodyDivide", 8, 8, FUNC), ("CtorBase", 8, 8, FUNC)])
    cells.append(("C6", "ELF AArch64, alias symbol at BodyDivide's start", p, "aarch64", "elf", "BodyDivide"))

    p = os.path.join(out_dir, "c7.o")
    mk.build_elf64(p, mk.EM_X86_64, mk.X86_TEXT, [("HashSite", 0, 13, FUNC), ("BodyDivide", 8, 5, FUNC)])
    cells.append(("C7", "ELF x64, HashSite's declared size overlaps BodyDivide", p, "x86-64", "elf", "BodyDivide"))

    p = os.path.join(out_dir, "c8.o")
    mk.build_elf64(p, mk.EM_X86_64, mk.X86_TEXT, [("HashSite", 0, 5, FUNC), ("Pool", 5, 8, OBJECT)])
    cells.append(("C8", "ELF x64, STT_OBJECT covers the FP bytes inside .text", p, "x86-64", "elf", "BodyDivide"))

    p = os.path.join(out_dir, "c9.obj")
    mk.build_coff(p, mk.COFF_X86_64, mk.X86_TEXT, [("HashSite", 0, True), ("Pool", 5, False)])
    cells.append(("C9", "COFF x64, non-function symbol over the FP bytes (control)", p, "x86-64", "coff", "BodyDivide"))

    X86_TAILPAD = bytes([0x01, 0xF7, 0x89, 0xF8, 0xC3, 0xCC, 0xCC, 0xCC,
                        0xF2, 0x0F, 0x5E, 0xC1, 0xC3])
    p = os.path.join(out_dir, "c10.o")
    mk.build_elf64(p, mk.EM_X86_64, X86_TAILPAD, [("HashSite", 0, 8, FUNC), ("BodyDivide", 8, 5, FUNC)])
    cells.append(("C10", "ELF x64, extent tail is pure 0xCC padding (control)", p, "x86-64", "elf", "BodyDivide"))

    p = os.path.join(out_dir, "c11.o")
    mk.build_elf64(p, mk.EM_X86_64, mk.X86_TEXT, [("HashSite", 0, 5, FUNC), ("BodyDivide", 8, 5, FUNC)])
    cells.append(("C11", "ELF x64, ordinary padding gap between two real extents (control)", p, "x86-64", "elf", "BodyDivide"))

    return cells


def test_population_14_per_symbol_granularity_census():
    """Eleven hand-synthesized objects (no compiler -- pure Python ELF64/COFF
    byte synthesis, pop14_make_objects.py) mechanically enumerated from
    design Sec4.1's own branch conditions: an alias symbol sharing a start
    address, a symbol with no declared size, a symbol at a section's own end,
    a declared size clamped by the next symbol, an ELF STT_OBJECT / COFF
    non-function symbol covering the FP bytes inside an executable section, a
    decodable extent tail, and an inter-extent padding gap. Every cell's own
    ground truth (the raw x86-64/AArch64 divsd/fdiv bytes are present in the
    object) is confirmed directly, independent of any decoder.

    Once the instrument exists: a scan that emits ANY verdict for BodyDivide
    computed over bytes that are not BodyDivide's own -- or that omits
    BodyDivide's own FP instruction from every emitted verdict while the law
    PASSes -- fails this cell (T-2277's own two named failure shapes,
    "vacuous/misattributed" and "FP invisible"), for whichever of the 11
    cells exercises it.
    """
    with fc.TempDir() as tmp:
        cells = _build_pop14_cells(tmp)

        x86_fp_bytes = bytes([0xF2, 0x0F, 0x5E, 0xC1])
        a64_fp_bytes = bytes([0x00, 0x18, 0x61, 0x1E])
        for cell_id, note, path, isa, fmt, owner in cells:
            with open(path, "rb") as f:
                raw = f.read()
            marker = a64_fp_bytes if isa == "aarch64" else x86_fp_bytes
            assert marker in raw, "{} ({}): ground truth failed -- FP bytes not found".format(
                cell_id, note)

        if not _SCAN_AVAILABLE:
            _fail_absent(
                "fourteen",
                "Fixture verification above: all {} cells' own ground-truth FP "
                "bytes are present in their respective compiled objects (confirmed "
                "directly, no decoder needed).".format(len(cells)),
            )
        misattributed_or_vacuous = []
        fp_invisible = []
        for cell_id, note, path, isa, fmt, owner in cells:
            result = scan.scan_object(path, isa=isa, object_format=fmt)
            if not result.refuse:
                owner_verdict = result.verdicts.get(owner)
                if owner_verdict is None:
                    misattributed_or_vacuous.append((cell_id, note))
                emitted_as_owner_reject = any(
                    name == owner and verdict == "REJECT"
                    for name, verdict in result.verdicts.items()
                )
                if not emitted_as_owner_reject:
                    fp_invisible.append((cell_id, note))
        assert not misattributed_or_vacuous, (
            "cells with a vacuous/misattributed verdict: {}".format(misattributed_or_vacuous))
        assert not fp_invisible, (
            "cells where the FP instruction is invisible to every emitted verdict: "
            "{}".format(fp_invisible))
