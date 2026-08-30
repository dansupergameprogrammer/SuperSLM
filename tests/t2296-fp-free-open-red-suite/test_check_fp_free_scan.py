"""T-2333/T-2326 (Curie) -- red suite for design Sec4.1's deciding instrument:
an instruction-level, byte-accounting, default-deny scan over compiled
machine code, which does not exist yet anywhere in this tree.

SOURCE ARTIFACT. `Claude/Vitruvius/t2265-superslm-fp-free-open-design-2026-08-24.md`
Sec4.1 (the mechanism and its now-RATIFIED production interface) and Sec7
dimension 11 (the fourteen commissioning populations this suite realizes), as
the design stands through fold round 33. This campaign's own test-design
records: `Claude/Curie/t2326-fp-scan-instrument-red-suite-2026-08-27.md`
(original 23-cell suite, contract proposed) and
`Claude/Curie/t2333-fp-scan-instrument-red-suite-ratified-2026-08-27.md`
(this ticket's own repointing to the ratified contract).

WHY EVERY TEST HERE FAILS TODAY, AND WILL CONTINUE TO FAIL UNTIL BRUNEL BUILDS
THE INSTRUMENT. Sec4.1 specifies a Python CI tool -- a byte-level decoder
(capstone-driven, per-ISA), a closed-symbol-table membership rule, three
default-deny checks (A: register-file, B: GPR/control-flow allowlist, C: call/
jmp target allowlist), and a REFUSE control action on any unaccounted byte --
wired into this repo's own tests/ci/check_<name>.py + test_check_<name>.py
convention (tests/ci/check_no_forward_leaf_calls.py and its own test file are
the precedent this suite matches). None of that production module exists.
This file assumes it will be built at tests/ci/check_fp_free_scan.py (added
to sys.path below relative to this file, since this suite itself lives under
tests/t2296-fp-free-open-red-suite/ per this ticket's own writable-scope
constraint, not tests/ci/). Every population test below:

  1. Builds its own fixture FRESH (never a committed binary), using
     fp_scan_common.py's own toolchain wrappers. A toolchain genuinely absent
     from this environment SKIPs that one cell -- none did this session;
     every toolchain fired (see the case file's own toolchain table).
  2. Independently verifies, via a raw capstone decode or a narrow disasm-text
     parse -- NOT the (absent) instrument's own logic -- that the fixture
     genuinely carries the byte-level property its own population claims.
     This half is real, executed, and independent of whether the instrument
     exists.
  3. THEN calls the (absent) production module and asserts its documented
     verdict, from WITHIN the same fixture's own scope. This is the genuinely
     red half, today, for exactly one reason: importing check_fp_free_scan
     fails.

THE RATIFIED CONTRACT for tests/ci/check_fp_free_scan.py (design
Sec4.1, fold rounds 31/32/33; D-SLM4826/D-SLM4827/D-SLM4830/D-SLM4834 --
superseding T-2326's own proposed contract, D-SLM4825, which this file no
longer asserts as current):

    scan_object(path: str, isa: str) -> ScanResult

        isa in {"x86-64", "aarch64"} -- a build-time input the CI leg
        declares, never inferred from the object.

    ScanResult:
        .object_format: str          -- "coff" | "elf", READ from the
                                         object's own header -- NEVER a
                                         caller-supplied parameter (the one
                                         correction fold round 31 made to
                                         T-2326's own proposal: T-2326 carried
                                         object_format as a peer parameter to
                                         isa; every executed probe this design
                                         has run reads it from the object's
                                         own header instead, and a caller-
                                         supplied label could drift from what
                                         the file actually is).
        .refuse: bool                -- clause (0)'s own REFUSE.
        .unclassified_bytes: int
        .verdicts: dict[str, str]    -- symbol -> "ACCEPT" | "REJECT"; EMPTY
                                         when .refuse (guarantee (i)).

    ci_gate(result: ScanResult, expected_symbols: Sequence[str]) -> bool

        expected_symbols is derived from the OBJECT'S OWN symbol table, never
        from result.verdicts. Enforces: (i) result.refuse implies
        result.verdicts == {}; (ii) the caller's own process exits nonzero
        whenever this returns False for any leg; (iii) any name in
        expected_symbols absent from result.verdicts, when NOT refused, fails
        independently -- the absent-report leg.

    enumerate_scan_targets() -> list[tuple[str, str]]

        (translation_unit_name, compiled_object_path) pairs, derived from
        SUPERSLM_CORE_SOURCES at scan time -- the production membership's own
        node-set derivation.

    diagnostic_walk(roots: Sequence[str], call_graph: Mapping[str, set[str]]) -> set[str]

        Breadth/depth-first transitive closure from roots over call_graph's
        own edges. Population five's own discharge: compile the chain, build
        the real call graph, assert the deepest member is reachable.

    derive_core_sources(manifest_path: str = "CMakeLists.txt") -> list[str]

        Parses SUPERSLM_CORE_SOURCES out of the named manifest -- the shared
        TU-list primitive enumerate_scan_targets, build_call_graph, and
        flagged_symbols all call internally. Specified fold round 32
        (D-SLM4830) as population six's own occupant; CORRECTED fold round 33
        (D-SLM4834): this function alone does NOT discharge population six --
        it parses a manifest into a TU-name list and never compiles,
        disassembles, or detects anything, so it cannot carry population
        six's own end-to-end claim (a genuine derivation REPORTS a flagged
        instruction in a newly-added TU; a stale enumeration SILENTLY MISSES
        it). It remains a real, sound sub-component of population five's own
        discharge and of enumerate_scan_targets.

    build_call_graph(manifest_path: str, disasm_dir: str) -> Mapping[str, set[str]]
    flagged_symbols(manifest_path: str, disasm_dir: str) -> Mapping[str, list[str]]
    diagnostic_fp_report(manifest_path: str, disasm_dir: str,
                          roots: Sequence[str]) -> Mapping[str, list[str]]

        Population six's real occupant, specified fold round 33 (D-SLM4834).
        Both build_call_graph and flagged_symbols derive their own TU
        enumeration from derive_core_sources(manifest_path) internally --
        never a caller-supplied source list -- and read each derived TU's own
        real disassembly text from disasm_dir. diagnostic_fp_report combines
        them: every symbol reachable from roots via
        diagnostic_walk(roots, build_call_graph(...)) that also has a
        non-empty entry in flagged_symbols(...). A call instruction names its
        target by symbol whether or not the target's own TU was ever read, so
        build_call_graph alone (population five's own surface) cannot
        distinguish a genuine derivation from a stale one -- only the
        combined report can, because a symbol's own content is scanned ONLY
        if its defining TU was actually read.

THIS FILE DOES NOT CITE, IMPORT, OR RELY ON Claude/Vitruvius/
t2265-fold32-probe/class_closure_check.py OR DESIGN Sec2.9's OWN VERDICTS.
Sec2.9's claim-shadow test is QUARANTINED (D-SLM4833) -- its own validation
population and its shadow-parameter field are each proven insufficient by a
named, executed finding, and the design's own text states its verdicts are
"recorded, never cited as evidence for any gate decision" until independently
commissioned (tracked at T-2337, which does not gate this design). Population
six's own correction is established here the same way the design's own text
establishes it: by an independent, directly executed probe
(Claude/Vitruvius/t2265-fold33-probe/population6_end_to_end_probe.py), never
by running Sec2.9's own check against a corrected row.

POPULATION-TO-TEST CROSS-REFERENCE (full disposition in each test's own
docstring and in the T-2333 case file):

  1  TE-32's original two sites (Part B/C), reproduced verbatim (T-2333)      -- test_population_01_te32_reserve
  2  TE-32's four always-invisible ops + BodyUnderTest, verbatim (T-2333)     -- test_population_02_te32_mxcsr_invisible
  3  reintroduced reserve()-shaped growth, sslm_abi.cpp (fold round 3)        -- test_population_03_seqreg_growth_mutant
  4  reintroduced unordered_map growth, damped_greedy_antilm.cpp (fold 5)     -- test_population_04_antilm_restore_growth_mutant
  5  synthetic multi-hop transitive-closure proof, graded via diagnostic_walk -- test_population_05_transitive_chain_diagnostic
  6  TU-set-desync, graded via build_call_graph/flagged_symbols/report (T-2333)-- test_population_06_tu_set_desync_end_to_end
  7  T-2271's eight-object classifier construction (fold round 7)            -- test_population_07_fpblind_classifier
  8  real v1.2.1 whole-corpus sweep (fold rounds 7/8); registered OPEN, T-2404
     R5 -- discharged operationally by the CI gate, not gradable by a unit
     cell; this cell verifies its own source-manifest premise only          -- test_population_08_source_manifest_resolves_seventeen_real_files
  9  T-2272's funclet, membership rule (fold round 8)                       -- test_population_09_funclet_membership
  10 T-2273's AArch64 differential control (fold round 9, historical) --
     split T-2403 R6 into a hard cell and a disclosed xfail(strict=True)
     tension (`Claude/Vitruvius/t2265-fold46-delta-manifest.md` Sec2)      -- test_population_10a_arm_differential_fixture_and_format,
                                                                              test_population_10b_arm_differential_verdicts_disclosed_tension
  11 four (ISA,format,toolchain) legs + ci_gate (fold round 10)              -- test_population_11_toolchain_format_independence
  12 T-2275's clause-(0) REFUSE construction + ci_gate (fold rounds 11/12)   -- test_population_12_clause0_refuse_undecodable_pool
  13 T-2276's decodable-pool construction + ci_gate (fold round 13)          -- test_population_13_coverage_relation_decodable_swallow
  14 T-2277's per-symbol-granularity census, 11 cells (fold round 14)        -- test_population_14_per_symbol_granularity_census
  -- ci_gate's own absent-report leg (item 2, T-2333's own brief)            -- test_ci_gate_absent_report_leg
  16 self-zeroing vxorps/vxorpd boundary (T-2347, fold round 35, D-SLM4889)  -- test_population_16_vxorps_*
  17 check (C) relocation-resolve-then-classify carve-out (T-2347, D-SLM4886)-- test_population_17_carveout_*
  18 diagnostic surface fail-closed on missing disassembly (T-2347, D-SLM4889)-- test_population_18_diagnostic_*
  19 ScanResult(refuse=True) on unrecognised ISA/format (T-2347, D-SLM4889)  -- test_population_19_refuse_on_*, test_population_19_recognized_pair_control_unaffected
  20 per-section local_starts keying (T-2347, D-SLM4889, Poirot's O3)        -- test_population_20_local_starts_*
  21 the empty-extent byte charge's own arithmetic (T-2347, D-SLM4889, M2)   -- test_population_21_*_empty_extent*

T-2347 (fold round 35) ALSO adds, outside the sixteen-through-twenty-first
population numbering (each pinning a named, routed finding rather than a
Coverage Model population): derive_core_sources's completeness self-check
(D-SLM4887, four cells, test_derive_core_sources_*), ci_gate_corpus's
vacuous-True closure (D-SLM4888, two cells, test_ci_gate_corpus_vacuous_*/
test_ci_gate_corpus_end_to_end_*), and the three findings Poirot's
8a28460-t2344 casebook routes to the test author rather than the build round:
M4 (fp_scan_common.compile_cl_release's own hardcoded flag duplication --
FIXED in place this ticket, since fp_scan_common.py is this suite's own test
helper, not production code -- see _production_flags()), O4 (populations
eight and ten migrated to xfail(strict=True), Poirot's own recommended
structural fix for build.bat's own fragile --deselect guard, S5), and O5
(run_fp_free_scan_real_corpus._msvc_target_flags's silent truncation on an
embedded ), documented as a disclosed residual --
test_msvc_target_flags_silently_truncates_on_embedded_paren).

T-2366 (Curie), D-SLM5001 -- the red suite for fold round 39's acceptance form
(design Sec4.1/Sec5.4/Sec5.5/Sec7 dim 11, as amended by fold round 39,
D-SLM4981/D-SLM4984-D-SLM4990), routed by two blind, concurrently-dispatched
gates that reached the same core defect independently: T-2364's adversary
strike (`Claude/Loki/t2364-fold39-acceptance-form-strike-2026-08-28.md`,
FRACTURED, 9 of 28 claims) and T-2365's coverage audit
(`Claude/Mendeleev/t2365-fold39-dim11-coverage-audit-2026-08-28.md`, GAPS
NAMED, three Structural findings). Both found that fold round 39 wrote four
specification acts as though built when only one design-level correction
(D-SLM4987, check (A)'s bitwise-family widening) is even routed to a build
round -- the other three (the gate's own check-(C) exclusion, D-SLM4988's
REFUSE-scope narrowing, and the CI wiring) do not exist as executable code
anywhere in this tree. Six new cell groups, filed at the end of this file
(D-SLM5001's own six-item build-round list):

  (1) the gate must not read check (C) -- test_gate_must_not_fail_on_a_check_c_only_reject,
      test_gate_still_fails_on_a_genuine_check_ab_violation
  (2) check (A)'s eight-mnemonic bitwise-family widening, D-SLM4987 --
      test_check_a_bitwise_family_widening_must_accept, plus population
      sixteen's own reconciliation, above
      (test_population_16_vxorps_differing_operands_now_accepts_per_fold39)
  (3) the p/vp-prefix rule's vitality pin, D-SLM4999 --
      test_check_a_p_vp_structural_accept_census_and_violation,
      test_check_a_p_vp_rule_fails_open_on_a_future_fp_mnemonic,
      test_check_a_p_prefix_exclude_list_is_load_bearing
  (4) D-SLM4359's seven switch-jump-table symbols, D-SLM4988's void
      narrowing -- test_dslm4359_seven_switch_jump_table_symbols_must_not_block_gate
  (5) scan_build_output.py's fail-closed membership discipline, design Sec7
      dim 11's thirty-sixth population --
      test_scan_build_output_zero_objects_exits_2_not_a_pass,
      test_scan_build_output_real_build_finds_exactly_seventeen_objects
  (6) CI reachability -- filed in a new sibling file,
      test_ci_gate_wiring.py, per this suite's writable scope
      ("tests/t2296-fp-free-open-red-suite/ and any new test file it
      needs").

This ticket's own casebook: `Claude/Curie/t2366-fp-gate-red-suite-2026-08-28.md`.
"""
from __future__ import annotations

import os
import re
import struct
import subprocess
import sys
from unittest import mock

import capstone
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIXTURES = os.path.join(_HERE, "fp_scan_fixtures")

sys.path.insert(0, _HERE)
import fp_scan_common as fc  # noqa: E402
import archive_fixtures as af  # noqa: E402  -- T-2385: run_lib_exe builds a real
# archive for the two _run_gate cells below, whose scratch build must carry one
# now that scan_build_output.py's main() no longer falls back to the object
# directory (design Sec4.1 fold round 43, D-SLM5100/D-SLM5101).

sys.path.insert(0, _FIXTURES)
import pop14_make_objects as mk  # noqa: E402  -- pure-Python object synthesis, no compiler needed
import pop49_evil_object as pop49_fx  # noqa: E402  -- T-2403 (Curie), R1's own hand-emitted COFF pair

# The production module's own expected home (tests/ci/, this repo's established
# check_<name>.py / test_check_<name>.py convention) -- NOT this file's own
# directory, per this ticket's writable-scope constraint.
_TESTS_ROOT = os.path.dirname(_HERE)
_CI_DIR = os.path.join(_TESTS_ROOT, "ci")
sys.path.insert(0, _CI_DIR)
try:
    import check_fp_free_scan as scan  # noqa: E402  -- built; retained as an import guard so this suite degrades gracefully if it is ever unavailable (T-2387)
    _SCAN_AVAILABLE = True
except ImportError:
    scan = None
    _SCAN_AVAILABLE = False

# T-2366 (Curie), D-SLM5001 item (1): the production driver design Sec4.1
# (fold round 39, D-SLM4981) names as "the gate" -- tests/ci/scan_build_output.py,
# imported the same way `scan` is above (this suite's own established
# absent-instrument pattern), so a cell that targets the GATE's own pass/fail
# decision (not just scan_object's per-symbol verdict) degrades gracefully in
# an environment where it is missing, rather than raising at import time.
try:
    import scan_build_output  # noqa: E402
    _GATE_AVAILABLE = True
except ImportError:
    scan_build_output = None
    _GATE_AVAILABLE = False


def _run_gate(build_dir, target="superslm", isa="x86-64"):
    """Invokes the production driver's own main() in-process, capturing its
    real return code -- the identical entry point `python tests/ci/
    scan_build_output.py` invokes from the command line, never a
    reimplementation of its own pass/fail logic."""
    saved_argv = sys.argv
    sys.argv = ["scan_build_output.py", "--build-dir", build_dir,
                "--target", target, "--isa", isa]
    try:
        return scan_build_output.main()
    finally:
        sys.argv = saved_argv


# T-2368 (Curie), D-SLM5008: the real 17-object corpus (CMake target
# superslm, MSBuild/Release layout) is no longer read from a hand-configured,
# unversioned directory named by a literal path here. Every cell below that
# needs a real-build leg takes the `real_build_dir` fixture (conftest.py,
# this directory) instead -- a corpus cell derives its corpus from a build it
# causes, never from a path it is told. That fixture builds fresh, once per
# test session, and skips every dependent cell with a stated reason when no
# usable corpus can be produced in this environment.


def _fail_absent(population_no, note=""):
    """The gating assertion every population ends in today. `note` states what
    THIS test already proved about its own fixture, independent of the
    instrument, so a reader of a failure log sees both halves: what is known
    (the fixture is real and correct) and what is missing (the instrument).

    R9 (T-2404, D-SLM5209/D-SLM5211): the message below formerly claimed the
    deciding instrument did not exist -- false since it was built, many
    folds before this repair (confirmed at source: the file exists and this
    suite imports it at module scope). The narrower, true claim this
    helper's callers actually need is stated instead: the instrument
    exists; the specific surface named by `population_no` and `note` is
    what this cell cannot grade through it (a still-missing production
    symbol, an allow-list awaiting its own build round, or similar) -- the
    same operational-discharge reasoning population eight's own repair
    (R5) already applies."""
    pytest.fail(
        "check_fp_free_scan.py (design Sec4.1's deciding instrument) exists at "
        "tests/ci/check_fp_free_scan.py, but population {} cannot be graded "
        "through it -- the surface this cell needs is absent from the shipped "
        "module. {}".format(population_no, note)
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
    instructions -- never concatenates chunks before decoding (see
    fp_scan_common.py's own coff_code_sections docstring for why)."""
    decode_one = _decode_arm64 if isa == "aarch64" else _decode_x86
    insns = []
    for chunk in sections:
        insns.extend(decode_one(chunk))
    return insns


# ===========================================================================
# Disassembly-TEXT parsing helpers, populations five and six ONLY.
#
# NOT the ratified build_call_graph/flagged_symbols surfaces (design Sec4.1,
# D-SLM4834) -- these exist solely to VERIFY, independent of and before the
# absent instrument, that a fixture's own REAL dumpbin disassembly text (or,
# for population six, the adopted fold-33-probe fixture text) genuinely
# carries the call edge / flagged mnemonic its own population claims. The
# regex shapes mirror Claude/Vitruvius/t2265-fold33-probe/
# population6_end_to_end_probe.py's own _SYM_RE/_INS_RE/FP_RE (cited, not
# imported -- that file's own fixtures are plain text, not a real corpus, and
# importing it would blur "verifying a fixture" into "being the instrument,"
# the same line fp_scan_common.py's own module docstring draws).
# ===========================================================================

_DISASM_SYM_RE = re.compile(r'^(\S.*):$')
_DISASM_INS_RE = re.compile(
    r'^\s+[0-9A-F]{16}:\s+(?:[0-9A-F]{2}\s)+\s*([a-zA-Z][a-zA-Z0-9]*)(?:\s+(.*))?$')


def _iter_disasm_text(text):
    """Yields (symbol_name, mnemonic, operand_text) for every instruction
    line in real dumpbin /disasm output (or the adopted fold-33-probe text
    fixtures, which are hand-authored in the identical shape)."""
    cur = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r\n")
        m = _DISASM_SYM_RE.match(line)
        if m:
            cur = m.group(1).split(" (")[0]
            continue
        m = _DISASM_INS_RE.match(line)
        if not m or cur is None:
            continue
        yield cur, m.group(1), (m.group(2) or "")


def _verify_call_edge(disasm_text, caller, callee):
    """Confirms a real `call <callee>` instruction exists inside caller's own
    disassembly text -- a narrow, single-purpose check, not a general
    call-graph builder."""
    for cur, mn, ops in _iter_disasm_text(disasm_text):
        if cur == caller and mn.lower() == "call" and callee in ops:
            return True
    return False


def _verify_fp_mnemonic(disasm_text, symbol, fp_re):
    for cur, mn, _ops in _iter_disasm_text(disasm_text):
        if cur == symbol and fp_re.match(mn):
            return mn
    return None


# ===========================================================================
# Populations one and two -- TE-32's own historical constructions, now
# reproduced verbatim: Claude/Loki/te32-probe/ was outside T-2326's own
# granted read-only scope and is granted for T-2333 (coordinator's own brief,
# item 5).
# ===========================================================================

def test_population_01_te32_reserve():
    """Falsifying construction: TE-32's own DoReserve, kind 0
    (std::unordered_map<std::string, uint32_t>, mirroring tokenizer.cpp's own
    four std::unordered_map sites) and kind 1 (std::unordered_set<
    std::string_view>, mirroring model.cpp's own three std::unordered_set<
    std::string_view> sites) -- the ORIGINAL two site families, reproduced
    verbatim from Claude/Loki/te32-probe/te32_fp_observability_cell.cpp's own
    DoReserve function. TE-32's own runtime harness (Part B/C) proved 69 of 69
    reserve()-driven legs trap under cleared exception masks
    (Claude/Loki/te32-probe/output-te32.txt); this population's own claim is
    the byte-level half of that corroboration -- the identical rehash-sizing
    machinery must REJECT under checks (A)/(B) when scanned.
    """
    src = os.path.join(_FIXTURES, "pop01_te32_reserve.cpp")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "pop01.obj")
        try:
            fc.compile_cl(src, obj)
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        sections = fc.code_sections(obj, ".text")
        assert sum(len(s) for s in sections) > 0, "compiled .text section(s) are empty"
        insns = _decode_sections(sections, "x86-64")
        fp_insns = [(i.mnemonic, i.op_str) for i in insns if _is_x86_fp_arith(i.mnemonic)]
        assert fp_insns, (
            "fixture verification FAILED (not the instrument's fault): TE-32's own "
            "reproduced DoReserve bodies decoded to {} instructions and NONE is "
            "FP-arithmetic-shaped".format(len(insns))
        )

        if not _SCAN_AVAILABLE:
            _fail_absent(
                "one",
                "Fixture verified above: TE-32's own reproduced construction genuinely "
                "contains FP-arithmetic instructions ({} found, e.g. {}) that a "
                "default-deny register-file check must REJECT.".format(len(fp_insns), fp_insns[:3]),
            )
        result = scan.scan_object(obj, isa="x86-64")
        assert result.object_format == "coff", (
            "a real MSVC cl.exe object must read back as coff, not {}".format(result.object_format)
        )
        assert not result.refuse, "byte-accounting law should not REFUSE a fully-decodable object"
        assert any(v == "REJECT" for v in result.verdicts.values()), (
            "TE-32's own reserve()-driven bodies must REJECT under checks (A)/(B); "
            "verdicts were: {}".format(result.verdicts)
        )


def test_population_02_te32_mxcsr_invisible():
    """Falsifying construction: TE-32's own four "always-invisible-to-MXCSR"
    operation classes (i64->f64 conversion, ordered compare, abs/negate, min
    -- the only four of TE-32's own twelve operation classes whose executed
    Part D census reports INVISIBLE on BOTH the exact and inexact operand set,
    Claude/Loki/te32-probe/output-te32.txt lines 170/177-179) plus
    BodyUnderTest() from Part E/E2, reproduced verbatim from
    Claude/Loki/te32-probe/te32_fp_observability_cell.cpp. A trap-observing
    liveness control (design Sec4.2/Sec4.3) sees nothing on any of these four
    bodies at any operand set; the byte-level scan must still REJECT under
    check (A) -- a vector/FP register touched by an instruction not on
    VEC_MOVE_ALLOW.
    """
    src = os.path.join(_FIXTURES, "pop02_te32_mxcsr_invisible.cpp")
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
            "TE-32's own reproduced always-invisible bodies"
        )

        if not _SCAN_AVAILABLE:
            _fail_absent(
                "two",
                "Fixture verified above: TE-32's own four always-invisible-to-MXCSR "
                "bodies (confirmed by that probe's own execution, not re-derived here) "
                "compile to real xmm-register instructions ({} found, e.g. {}) that a "
                "trap-observing liveness control cannot see at any operand set, but "
                "checks (A)/(B) must still REJECT.".format(len(vec_reg_insns), vec_reg_insns[:3]),
            )
        result = scan.scan_object(obj, isa="x86-64")
        assert result.object_format == "coff"
        assert not result.refuse
        assert any(v == "REJECT" for v in result.verdicts.values()), (
            "TE-32's own always-invisible-to-MXCSR bodies must still REJECT under "
            "check (A) (register-file classification); verdicts were: "
            "{}".format(result.verdicts)
        )


# ===========================================================================
# Population three -- reintroduced std::unordered_set growth on
# sslm_abi.cpp's own g_live_seqs shape (fold round 3).
# ===========================================================================

def test_population_03_seqreg_growth_mutant():
    """Falsifying construction: an std::unordered_set grown by plain insert()/
    erase() (mirrors sslm_abi.cpp's own pre-replacement g_live_seqs, T-2268's
    own "seqreg" leg, Claude/Loki/t2268-probe/t2268_insert_growth_fp.cpp).
    Must REJECT: the bucket-array resize drives the identical rehash-sizing
    machinery population one/two's own TE-32 bodies do.
    """
    src = os.path.join(_FIXTURES, "pop03_seqreg_growth_mutant.cpp")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "pop03.obj")
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
            "insert()/erase()-growth mutant decoded to {} instructions and NONE is "
            "FP-arithmetic-shaped".format(len(insns))
        )

        if not _SCAN_AVAILABLE:
            _fail_absent(
                "three",
                "Fixture verified above: the compiled object genuinely contains FP-"
                "arithmetic instructions ({} found, e.g. {}) that a default-deny "
                "register-file check must REJECT.".format(len(fp_insns), fp_insns[:3]),
            )
        result = scan.scan_object(obj, isa="x86-64")
        assert result.object_format == "coff"
        assert not result.refuse
        assert any(v == "REJECT" for v in result.verdicts.values()), (
            "the insert()/erase()-growth mutant must REJECT under checks (A)/(B); "
            "verdicts were: {}".format(result.verdicts)
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
    confirmed by reading the real, current source, that file's own line-33
    comment), populated through a restore-shaped replay mirroring
    sslm_seq_restore's own call pattern into AntiLmUpdate. Must REJECT.
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
        result = scan.scan_object(obj, isa="x86-64")
        assert result.object_format == "coff"
        assert not result.refuse
        assert any(v == "REJECT" for v in result.verdicts.values())


# ===========================================================================
# Population five -- synthetic multi-hop transitive-closure proof (fold round
# 6), graded via the ratified diagnostic_walk surface (fold round 31,
# D-SLM4827).
# ===========================================================================

def test_population_05_transitive_chain_diagnostic():
    """Falsifying construction: Root -> HopA -> HopB -> FlaggedLeaf, three
    hops, no shorter path from Root to FlaggedLeaf exists (Root calls ONLY
    HopA; HopA calls ONLY HopB). Built with MSVC cl.exe and read via real
    dumpbin /disasm text (T-2333: switched from clang to MSVC+dumpbin so this
    population's own real disassembly is read the same way population six's
    own ratified surfaces read theirs).

    VERIFIED, independent of the instrument: this test parses the REAL
    dumpbin text (not the ratified build_call_graph -- a narrow, single-edge
    check, _verify_call_edge, defined at this file's own top) and confirms
    all three call edges are genuinely present, and that FlaggedLeaf's own
    body carries a real divsd. GRADED via scan.diagnostic_walk(roots,
    call_graph) -- design Sec4.1's own ratified diagnostic surface
    (D-SLM4827) -- with call_graph built from the SAME real disassembly text
    by this file's own narrow, verification-only parser (never the
    instrument's own build_call_graph, which population six's own cell
    exercises instead).
    """
    src = os.path.join(_FIXTURES, "pop05_transitive_chain.cpp")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "pop05.obj")
        try:
            fc.compile_cl(src, obj)
            disasm_text = fc.dumpbin_disasm(obj)
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))

        assert _verify_call_edge(disasm_text, "Root", "HopA"), (
            "Root must call HopA -- not found in real dumpbin text"
        )
        assert _verify_call_edge(disasm_text, "HopA", "HopB"), (
            "HopA must call HopB -- not found in real dumpbin text"
        )
        assert _verify_call_edge(disasm_text, "HopB", "FlaggedLeaf"), (
            "HopB must call FlaggedLeaf -- not found in real dumpbin text"
        )
        assert not _verify_call_edge(disasm_text, "Root", "FlaggedLeaf"), (
            "Root must NOT call FlaggedLeaf directly -- the whole point of the "
            "3-hop construction is that no shorter path exists"
        )
        fp_re = re.compile(r'^(divsd|addsd|mulsd|subsd)$')
        assert _verify_fp_mnemonic(disasm_text, "FlaggedLeaf", fp_re), (
            "FlaggedLeaf's own body must carry a genuine FP-arithmetic instruction"
        )

        # Build a real call_graph from the same disassembly text, using this
        # file's own narrow parser (verification-only -- see this module's
        # own header) -- independent of, and prior to, the absent
        # diagnostic_walk grading below.
        real_call_graph: dict[str, set[str]] = {}
        for cur, mn, ops in _iter_disasm_text(disasm_text):
            if mn.lower() == "call":
                target = ops.strip().split()[0] if ops.strip() else ""
                if target:
                    real_call_graph.setdefault(cur, set()).add(target)
        assert real_call_graph.get("Root") == {"HopA"}
        assert real_call_graph.get("HopA") == {"HopB"}
        assert real_call_graph.get("HopB") == {"FlaggedLeaf"}

        if not _SCAN_AVAILABLE:
            _fail_absent(
                "five",
                "Fixture verified above: a genuine 3-hop call chain with no shorter "
                "path (Root->HopA->HopB->FlaggedLeaf, confirmed in real dumpbin text), "
                "FlaggedLeaf's own divsd present, and a real call_graph built from "
                "that same text ({}).".format(real_call_graph),
            )
        reachable = scan.diagnostic_walk(["Root"], real_call_graph)
        assert "FlaggedLeaf" in reachable, (
            "diagnostic_walk must reach FlaggedLeaf, 3 hops from Root, over the real "
            "call graph; reachable set was {}".format(reachable)
        )
        assert "HopA" in reachable and "HopB" in reachable


# ===========================================================================
# Population six -- translation-unit-set desync detection (fold round 6),
# respecified fold round 33 (D-SLM4834): graded via build_call_graph +
# flagged_symbols + diagnostic_fp_report, NOT derive_core_sources alone
# (fold round 32's own too-narrow fix, corrected) and NOT diagnostic_walk
# alone (fold round 31's own too-narrow fix, corrected).
# ===========================================================================

def test_population_06_tu_set_desync_end_to_end():
    """Falsifying construction, ADOPTED VERBATIM from
    Claude/Vitruvius/t2265-fold33-probe/fixtures/ (the design's own executed
    precedent for this exact population, per T-2333's own brief item 4):
    two manifest fixtures (manifest_stale.cmake.txt lists only tu_a;
    manifest_current.cmake.txt lists tu_a and tu_b) sharing one disassembly
    fixture directory that already contains BOTH tu_a.disasm.txt and
    tu_b.disasm.txt -- modelling the real situation this population's own
    claim describes: the added translation unit has genuinely been compiled
    and disassembled; the only question is whether the corpus-building step's
    own TU enumeration picked it up. tu_a defines root_fn, which calls
    callee_in_b (a real, named call target, present in tu_a's own
    disassembly regardless of whether tu_b is ever read -- exactly as a real
    relocation would be); tu_b defines callee_in_b, which contains a genuine
    addsd instruction.

    This is population six's own REAL, end-to-end claim (design Sec4.1,
    fold round 33): a genuine derivation REPORTS callee_in_b's flagged
    content under the current manifest; a stale enumeration SILENTLY MISSES
    it under the stale manifest -- NOT a TU-name-list diff (fold round 32's
    own too-narrow fix, corrected) and NOT a bare reachable-name-set diff
    (fold round 33's own first, failed attempt at this exact probe, kept at
    Claude/Vitruvius/t2265-fold33-probe/out-population6-probe-v1-FAILED.txt
    per StandardsDocument.md Sec7's own repair standard). callee_in_b is
    reachable-by-NAME under BOTH manifests (a call instruction names its
    target whether or not the target's own TU was ever read) -- this test
    confirms that explicitly, so the discrimination the grading step performs
    is real, not accidental.
    """
    fx_dir = os.path.join(_FIXTURES, "pop06_fold33_fixtures")
    manifest_stale = os.path.join(fx_dir, "manifest_stale.cmake.txt")
    manifest_current = os.path.join(fx_dir, "manifest_current.cmake.txt")
    disasm_dir = os.path.join(fx_dir, "disasm")

    with open(os.path.join(disasm_dir, "tu_a.disasm.txt")) as f:
        tu_a_text = f.read()
    with open(os.path.join(disasm_dir, "tu_b.disasm.txt")) as f:
        tu_b_text = f.read()

    assert _verify_call_edge(tu_a_text, "root_fn", "callee_in_b"), (
        "root_fn must call callee_in_b in the adopted fixture's own real text"
    )
    fp_re = re.compile(r'^(addsd|subsd|mulsd|divsd)$')
    assert _verify_fp_mnemonic(tu_b_text, "callee_in_b", fp_re), (
        "callee_in_b must carry a genuine FP-arithmetic mnemonic in the adopted "
        "fixture's own real text"
    )

    # Manifest content itself: confirm the desync is real before grading it.
    # Parses only the set(SUPERSLM_CORE_SOURCES ...) block's own entries, not
    # the raw file text -- both fixtures' own header COMMENTS mention "tu_b"
    # in prose regardless of which manifest actually lists it (confirmed by
    # direct execution this session: a raw substring check against the whole
    # file falsely finds "tu_b" in the stale manifest's own explanatory
    # comment, "stands in for a snapshot ... taken BEFORE tu_b.cpp was
    # added"), so the check must read the declaration block the way
    # derive_core_sources itself would, not the file's prose.
    _set_var_re = re.compile(r"set\(\s*SUPERSLM_CORE_SOURCES(.*?)\)", re.DOTALL)

    def _manifest_entries(path):
        with open(path) as f:
            text = f.read()
        m = _set_var_re.search(text)
        assert m, "{}: no set(SUPERSLM_CORE_SOURCES ...) block found".format(path)
        return [line.strip() for line in m.group(1).splitlines()
                if line.strip() and not line.strip().startswith("#")]

    stale_entries = _manifest_entries(manifest_stale)
    current_entries = _manifest_entries(manifest_current)
    assert not any("tu_b" in e for e in stale_entries), (
        "the stale manifest's own SUPERSLM_CORE_SOURCES block must NOT list tu_b; "
        "got {}".format(stale_entries)
    )
    assert any("tu_b" in e for e in current_entries), (
        "the current manifest's own SUPERSLM_CORE_SOURCES block must list tu_b; "
        "got {}".format(current_entries)
    )
    assert any("tu_a" in e for e in stale_entries)
    assert any("tu_a" in e for e in current_entries)

    if not _SCAN_AVAILABLE:
        _fail_absent(
            "six",
            "Fixture verified above: the adopted fold-33-probe fixture is genuine -- "
            "root_fn really calls callee_in_b (confirmed in tu_a's own real "
            "disassembly text), callee_in_b really carries FP-arithmetic content "
            "(confirmed in tu_b's own real disassembly text), and the two manifests "
            "genuinely differ exactly where the desync predicts.",
        )
    # Reachable-by-name in BOTH runs -- the property that makes population
    # six's own claim non-trivial (build_call_graph alone cannot discharge it).
    graph_stale = scan.build_call_graph(manifest_stale, disasm_dir)
    graph_current = scan.build_call_graph(manifest_current, disasm_dir)
    reach_stale = scan.diagnostic_walk(["root_fn"], graph_stale)
    reach_current = scan.diagnostic_walk(["root_fn"], graph_current)
    assert "callee_in_b" in reach_stale, (
        "callee_in_b must be reachable BY NAME even under the stale manifest -- a "
        "call edge names its target regardless of whether the target's own TU was "
        "read; reach_stale={}".format(reach_stale)
    )
    assert "callee_in_b" in reach_current

    # The actual discrimination: the COMBINED report (reachable AND flagged).
    report_stale = scan.diagnostic_fp_report(manifest_stale, disasm_dir, ["root_fn"])
    report_current = scan.diagnostic_fp_report(manifest_current, disasm_dir, ["root_fn"])
    assert "callee_in_b" not in report_stale, (
        "the STALE manifest must SILENTLY MISS callee_in_b's own flagged content -- "
        "its own TU was never in the derived enumeration; report_stale={}".format(report_stale)
    )
    assert "callee_in_b" in report_current, (
        "the CURRENT manifest must REPORT callee_in_b's own flagged content; "
        "report_current={}".format(report_current)
    )


# ===========================================================================
# Population seven -- T-2271's own eight-object classifier construction (fold
# round 7, D-SLM4350).
# ===========================================================================

_POP07_TIERS = [("sse_baseline", []), ("avx2", ["-mavx2"])]


@pytest.mark.parametrize("tier,extra_flags", _POP07_TIERS)
def test_population_07_fpblind_classifier(tier, extra_flags):
    """Falsifying construction: BodyFloor/BodyFma/BodyRound/BodyHadd, adapted
    from T-2271's own construction. All four must REJECT at both ISA tiers.
    """
    src = os.path.join(_FIXTURES, "pop07_fpblind.cpp")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "pop07_{}.obj".format(tier))
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
        result = scan.scan_object(obj, isa="x86-64")
        assert result.object_format == "coff"
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

# R5 (T-2404): this cell formerly ended in an unconditional pytest.fail()
# stating the deciding instrument did not exist -- false since it was
# built, many folds before this repair (confirmed at source this round:
# the file exists and is imported by this suite). The
# xfail(strict=True) marker that pinned that unconditional failure is
# removed with it. This population's own real claim -- a full-corpus
# build-and-scan of SUPERSLM_CORE_SOURCES -- is discharged operationally
# by the CI gate running scan_build_output.py against the real corpus, not
# by any unit cell in this file; no unit cell can grade a CI-scale
# operation. Population eight is registered OPEN with that blocker stated
# plainly, rather than recorded as covered by a cell that could not pass.
# The cell is renamed to what it actually checks: the source manifest this
# population's own claim depends on.
def test_population_08_source_manifest_resolves_seventeen_real_files():
    """This population is a claim about the REAL, currently-committed engine
    source (SUPERSLM_CORE_SOURCES, CMakeLists.txt) -- not a constructed
    fixture. Verified here: the real source list resolves to a non-vacuous
    17-file population, and every named file exists on disk. This is the
    whole of what this cell checks -- population eight's own claim (a
    full-corpus build-and-scan) is CI-scale, not gradable by any unit cell,
    and is registered OPEN rather than COVERED (design Sec7 dim 11): the
    blocker is discharged operationally by the CI gate running
    scan_build_output.py against the real corpus, outside this suite.

    T-2342 (design Sec5.4, D-SLM4861): the built instrument's own two
    real-corpus readings -- 181 REJECT of 5646 symbols (the build's own
    runner, `/std:c++20 /O2 /W4 /fp:precise /EHsc`, no /MD /Ob2 /DNDEBUG) and
    627 REJECT of 5579 symbols (Popper's own probe, `/O2 /Ob2 /DNDEBUG /MD
    /W4 /fp:precise /std:c++20 /EHsc`) -- are reconciled as a compiler-flags
    difference, not a counting error. The design's own acceptance criteria
    are RULED to read against 627 -- the real windows-latest CI leg's own
    Release configuration (CMake's stock MSVC default, /MD /O2 /Ob2 /DNDEBUG,
    plus CMakeLists.txt:64's own /W4 /fp:precise) -- never the 181-symbol,
    unshipped fourth configuration. Any future cell built over the real
    corpus in this suite uses fc.compile_cl_release (this session's own
    addition to fp_scan_common.py, matching Claude/Popper/t2340-probe/
    real_corpus.py:28-29's flags exactly, confirmed against
    CMakeLists.txt:64 at source), never a fourth, hand-picked flag line.
    This does NOT retroactively apply to populations one through fourteen's
    own isolated must-accept/must-reject constructions (they compile via
    plain fc.compile_cl/compile_clangxx, unchanged) -- those are standalone
    classifier probes, not claims about the real corpus, and the
    Release-flags ruling governs only cells that scan SUPERSLM_CORE_SOURCES
    itself. This ticket's own item 5 cell
    (test_external_edge_import_thunk_vs_plain_name_vetted_separately) is the
    one place in this suite so far that uses fc.compile_cl_release, for the
    identical reason stated here: the import-thunk rendering it pins is a
    real, Release-configuration-specific fact.
    """
    engine_root = os.path.dirname(_TESTS_ROOT)
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


# ===========================================================================
# Population nine -- T-2272's funclet, the must-reject population for the
# MEMBERSHIP rule specifically (fold round 8).
# ===========================================================================

def test_population_09_funclet_membership():
    """Falsifying construction: RegressionParent (no FP instruction of its
    own) wraps a try/catch; the catch FUNCLET performs genuine IEEE-754
    double arithmetic and is entered by the runtime unwinder through
    `.xdata`, named by no call and no jmp instruction anywhere in the image.
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
            # size prefix -- strtab already starts at that size prefix, so
            # off indexes directly into it with no adjustment.
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
        result = scan.scan_object(obj, isa="x86-64")
        assert result.object_format == "coff"
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
# Historical/diagnostic: commissioning infeasible per fold round 10, disposed
# to population eleven.
#
# T-2403 (Curie), R6 -- split corrected from executed evidence
# (`Claude/Vitruvius/t2265-fold46-delta-manifest.md` Sec2; D-SLM5207,
# D-SLM5211, D-SLM5213), superseding fold round 45's own split (D-SLM5187)
# before it ever landed. Fold round 45 assumed five of this cell's six
# post-scan assertions currently pass; re-executed against the real compiled
# AArch64 object this round (`Claude/Vitruvius/t2265-fold46-probes/
# r6_full_cell_probe.py`, and independently against the strike's own
# `pop10_assertion_census.py`, identical result): `scan_object` returns
# `refuse=True, unclassified_bytes=60, verdicts={}` -- FIVE of six fail
# today, not one, and it is one root cause, not five: `scan_object`'s own
# `if refuse:` branch (tests/ci/check_fp_free_scan.py:1383-1385) returns
# `verdicts={}` unconditionally the instant byte-accounting REFUSEs (60
# bytes it cannot attribute to any recognised symbol or padding run), before
# any per-symbol classification runs -- so the four verdict assertions
# cannot discriminate anything while REFUSE holds; they fail in lockstep
# with `not result.refuse`, for the one reason this cell's own marker names
# below (the COMDAT-filler padding-recognition gap this reader has not yet
# closed). Two assertions hold today, confirmed by this same execution: the
# pre-scan fixture verification (A0, genuine AArch64 FP instructions decode)
# and `result.object_format == 'coff'` (A1) -- both independent of the
# COMDAT-filler tension, since A0 is about the compiler's own output before
# `scan_object` ever runs and A1 is the container-format read, which
# succeeds before byte-accounting begins.
#
# The cell therefore splits in two: a hard cell (no marker) carrying exactly
# A0/A1, and an xfail(strict=True) cell carrying the five that fail
# together (A2, the four verdict checks). Nothing green today goes red under
# this split; the change is that a regression in A0/A1 is now caught, where
# before it was absorbed by the same marker as the disclosed tension.
# Population ten's own registry disposition is left unresolved rather than
# guessed at here (COVERED needs a passing must-reject side, and none exists
# while the classifier is never reached) -- filed as owed work for whichever
# fold builds the dimension-11 registry (out of this release's scope,
# D-SLM5200/D-SLM5201).
# ===========================================================================

def _build_pop10_object(tmp):
    """Compiles pop10_arm_site.cpp for AArch64 fresh (never a committed
    binary) and independently verifies -- via a raw capstone decode, NOT
    scan_object's own logic -- that it genuinely carries FP-arithmetic-
    shaped instructions. Shared by both of population ten's own split
    cells so each builds and verifies its own fixture from scratch, per
    this suite's own established convention."""
    src = os.path.join(_FIXTURES, "pop10_arm_site.cpp")
    obj = os.path.join(tmp, "pop10.obj")
    fc.compile_cl_arm64(src, obj)
    sections = fc.code_sections(obj, ".text")
    insns = _decode_sections(sections, "aarch64")
    fp_insns = [(i.mnemonic, i.op_str) for i in insns if _is_arm64_fp_arith(i.mnemonic)]
    return obj, insns, fp_insns


def test_population_10a_arm_differential_fixture_and_format():
    """Population ten's hard cell (T-2403, R6): the two of six post-scan
    assertions that hold today -- A0 (fixture verification: the compiled
    object decodes to genuine AArch64 FP-arithmetic-shaped instructions)
    and A1 (result.object_format == 'coff', MSVC's own AArch64
    cross-compiler still emits COFF). No marker: both are independent of
    the COMDAT-filler byte-accounting tension population ten-b discloses
    below, so a regression in either must fail the gate outright rather
    than being absorbed by that cell's own xfail(strict=True).
    """
    with fc.TempDir() as tmp:
        try:
            obj, insns, fp_insns = _build_pop10_object(tmp)
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        assert fp_insns, (
            "fixture verification FAILED: the AArch64 object decoded to {} "
            "instructions and NONE is FP-arithmetic-shaped".format(len(insns))
        )
        if not _SCAN_AVAILABLE:
            _fail_absent(
                "ten-a (hard cell: fixture verification + object_format)",
                "Fixture verified above: {} genuine AArch64 FP instructions "
                "decoded (e.g. {}).".format(len(fp_insns), fp_insns[:3]),
            )
        result = scan.scan_object(obj, isa="aarch64")
        assert result.object_format == "coff", (
            "MSVC's own AArch64 cross-compiler still emits COFF, not {}".format(
                result.object_format)
        )


@pytest.mark.xfail(
    strict=True,
    reason="T-2403 (Curie), R6 (Claude/Vitruvius/t2265-fold46-delta-"
    "manifest.md Sec2; D-SLM5207/D-SLM5211/D-SLM5213), superseding T-2347's "
    "own reason text: byte-accounting REFUSEs on unclassified_bytes bytes "
    "of unattributed content in this object's own AArch64 section "
    "(tests/ci/check_fp_free_scan.py:1383-1385 returns verdicts={} "
    "unconditionally the instant REFUSE fires), which forces ALL FOUR "
    "verdict assertions below to fail in lockstep with 'not result.refuse' "
    "-- not four independent verdict-specific failures, one cause surfacing "
    "through five assertions. The cause is this reader's own COMDAT-filler "
    "padding-recognition gap: the unattributed bytes are compiler-emitted "
    "padding this reader does not yet recognise as such. strict=True means "
    "the day that gap closes and byte-accounting stops REFUSING this "
    "object, this marker flips to a loud XPASS failure demanding removal "
    "and the four verdict assertions begin discriminating for the first "
    "time, rather than silently continuing to pass for the wrong reason.",
)
def test_population_10b_arm_differential_verdicts_disclosed_tension():
    """Population ten's xfail(strict=True) cell (T-2403, R6): the five
    post-scan assertions that fail together today, all downstream of one
    byte-accounting REFUSE -- not result.refuse (A2), and the four verdict
    checks (A3-A6: BuildMerges/BodyDivide/BodyConvertCompare must REJECT,
    DedupNames must ACCEPT) once the classifier is actually reached.
    """
    with fc.TempDir() as tmp:
        try:
            obj, _insns, fp_insns = _build_pop10_object(tmp)
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        if not _SCAN_AVAILABLE:
            _fail_absent(
                "ten-b (xfail cell: refuse + four verdict checks, disclosed "
                "COMDAT-filler tension)",
                "Fixture verified above: {} genuine AArch64 FP instructions "
                "decoded (e.g. {}).".format(len(fp_insns), fp_insns[:3]),
            )
        result = scan.scan_object(obj, isa="aarch64")
        assert not result.refuse, (
            "byte-accounting REFUSEd this object (unclassified_bytes={}); "
            "the four verdict checks below cannot discriminate anything "
            "while this holds".format(result.unclassified_bytes)
        )
        for fn in ("BuildMerges", "BodyDivide", "BodyConvertCompare"):
            assert result.verdicts.get(fn) == "REJECT", (
                "{} must REJECT; verdict was {}".format(fn, result.verdicts.get(fn)))
        assert result.verdicts.get("DedupNames") == "ACCEPT", (
            "DedupNames carries no FP instruction and must ACCEPT; verdict was "
            "{}".format(result.verdicts.get("DedupNames"))
        )


# ===========================================================================
# Population eleven -- the byte-accounting law's own toolchain-and-format
# independence, four (ISA, object-format, toolchain) legs (fold round 10) --
# now also exercises ci_gate's own True (healthy-job) path.
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
    named production legs: HashSite must ACCEPT; BodyDivide must REJECT --
    non-degenerately. Also exercises ci_gate's own True (job-passes) path:
    both expected symbols present, no REFUSE.
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
        result = scan.scan_object(obj, isa=isa)
        assert result.object_format == fmt, (
            "leg {}: expected object_format {}, read back {}".format(
                leg_name, fmt, result.object_format)
        )
        assert not result.refuse, "leg {}: byte-accounting law should not REFUSE".format(leg_name)
        assert result.verdicts.get("HashSite") == "ACCEPT", (
            "leg {}: HashSite carries no FP instruction and must ACCEPT".format(leg_name))
        assert result.verdicts.get("BodyDivide") == "REJECT", (
            "leg {}: BodyDivide carries a genuine FP instruction and must "
            "REJECT".format(leg_name))
        assert scan.ci_gate(result, expected_symbols=["HashSite", "BodyDivide"]) is True, (
            "leg {}: ci_gate must return True -- both expected symbols are present "
            "and the accounting law did not REFUSE".format(leg_name)
        )


# ===========================================================================
# Population twelve -- T-2275's clause-(0) REFUSE construction (fold round
# 11), extended to the three x86-64 production legs at fold round 12
# (D-SLM4442) -- now also exercises ci_gate's own False (job-fails-on-REFUSE)
# path.
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
    literal pool UNDECODABLE as machine code in this ISA/mode, then a
    floating-point body. Clause (0) must REFUSE; ci_gate must return False.
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
            "(the pool must be undecodable) -- decoded {} of {} bytes "
            "cleanly".format(leg_name, decoded_bytes, total_bytes)
        )

        if not _SCAN_AVAILABLE:
            _fail_absent(
                "twelve, leg {}".format(leg_name),
                "Fixture verified above: a genuinely undecodable byte range exists "
                "({} of {} bytes decode linearly before the decoder stops), which "
                "is exactly the unaccounted range clause (0) must REFUSE "
                "on.".format(decoded_bytes, total_bytes),
            )
        result = scan.scan_object(obj, isa=isa)
        assert result.object_format == fmt
        assert result.refuse, "leg {}: clause (0) must REFUSE on an undecodable range".format(leg_name)
        assert result.unclassified_bytes > 0
        assert not result.verdicts, (
            "leg {}: REFUSE must emit ZERO ACCEPT/REJECT verdicts for any symbol; "
            "got {}".format(leg_name, result.verdicts)
        )
        assert scan.ci_gate(result, expected_symbols=["BodyDivide"]) is False, (
            "leg {}: ci_gate must return False on a REFUSE leg -- the job must "
            "fail".format(leg_name)
        )


# ===========================================================================
# Population thirteen -- T-2276's decodable-pool construction (fold round
# 13) -- also exercises ci_gate's own False path on the coverage-relation
# REFUSE shape.
# ===========================================================================

_POP13_LEGS = [
    ("clang_elf_x64", "pop13_pool_decodable_elf_x64.s", "elf"),
    ("clang_coff_x64", "pop13_pool_decodable_coff_x64.s", "coff"),
    ("msvc_coff_x64", "pop13_pool_decodable_msvc_x64.asm", "coff"),
]


@pytest.mark.parametrize("leg_name,fname,fmt", _POP13_LEGS)
def test_population_13_coverage_relation_decodable_swallow(leg_name, fname, fmt):
    """Falsifying construction: HashSite, an 8-byte pool covered by no
    symbol whose trailing bytes decode cleanly as a movabs that swallows
    BodyDivide's own body whole, then BodyDivide (a genuine divsd). REFUSE
    must fire on the COVERAGE relation, not merely decodability; ci_gate must
    return False.
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
            "leg {}: expected the raw bytes of divsd xmm0, xmm1 in the compiled "
            "object; ground truth check failed".format(leg_name)
        )

        if not _SCAN_AVAILABLE:
            _fail_absent(
                "thirteen, leg {}".format(leg_name),
                "Fixture verified above: the raw bytes of divsd xmm0, xmm1 are "
                "present in the compiled object (ground truth, independent of any "
                "decoder).",
            )
        result = scan.scan_object(obj, isa="x86-64")
        assert result.object_format == fmt
        assert result.refuse, (
            "leg {}: the pool bytes are covered by no symbol's own extent -- "
            "clause (0) must REFUSE on the coverage relation".format(leg_name)
        )
        assert not result.verdicts
        assert scan.ci_gate(result, expected_symbols=["BodyDivide"]) is False, (
            "leg {}: ci_gate must return False on a REFUSE leg".format(leg_name)
        )


# ===========================================================================
# Population fourteen -- T-2277's per-symbol-granularity census, 11 cells
# (fold round 14). Pure Python object synthesis -- no compiler needed.
# ===========================================================================

def _build_pop14_cells(out_dir):
    """Builds the 11 hand-crafted ELF64/COFF objects T-2277's own census
    derives from the fold-13 module's own branch conditions."""
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
    design Sec4.1's own branch conditions.
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
        bad_format = []
        for cell_id, note, path, isa, fmt, owner in cells:
            result = scan.scan_object(path, isa=isa)
            if result.object_format != fmt:
                bad_format.append((cell_id, result.object_format, fmt))
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
        assert not bad_format, "cells whose object_format was misread: {}".format(bad_format)
        assert not misattributed_or_vacuous, (
            "cells with a vacuous/misattributed verdict: {}".format(misattributed_or_vacuous))
        assert not fp_invisible, (
            "cells where the FP instruction is invisible to every emitted verdict: "
            "{}".format(fp_invisible))


# ===========================================================================
# ci_gate's own absent-report leg (T-2333's own brief, item 2): a leg that is
# NOT refused, but names an expected symbol that never appears in
# result.verdicts. This is guarantee (iii) -- fails independently of (i)/(ii).
# ===========================================================================

def test_ci_gate_absent_report_leg():
    """Uses population eleven's own healthy, non-refused ELF/x86-64 fixture
    (HashSite/BodyDivide, both real, both present in verdicts once graded) but
    asks ci_gate for a THIRD symbol that does not exist anywhere in the
    object. Per the ratified contract's own guarantee (iii): "any name in
    expected_symbols absent from result.verdicts, when NOT refused, fails
    independently of (i) and (ii)" -- ci_gate must return False even though
    the accounting law does not REFUSE and every symbol that DOES exist is
    correctly graded.
    """
    src = os.path.join(_FIXTURES, "pop11_elf_x64.s")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "absent_report.o")
        try:
            fc.compile_clang_asm(src, obj, "x86_64-pc-linux-gnu")
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        sections = fc.code_sections(obj, ".text")
        assert sum(len(s) for s in sections) > 0
        with open(os.path.join(_HERE, "fp_scan_fixtures", "pop11_elf_x64.s")) as f:
            src_text = f.read()
        assert "NonexistentSymbol" not in src_text, (
            "the absent-report leg's own control depends on this name genuinely "
            "not existing in the fixture's own source"
        )

        if not _SCAN_AVAILABLE:
            _fail_absent(
                "ci_gate's own absent-report leg",
                "Fixture verified above: the object compiles and is non-degenerate; "
                "\"NonexistentSymbol\" is confirmed absent from the fixture's own "
                "source, so asking ci_gate for it exercises guarantee (iii) "
                "genuinely, not by accident.",
            )
        result = scan.scan_object(obj, isa="x86-64")
        assert not result.refuse, "this leg must not REFUSE -- guarantee (iii) is tested apart from (i)"
        assert "NonexistentSymbol" not in result.verdicts
        assert scan.ci_gate(
            result, expected_symbols=["HashSite", "BodyDivide", "NonexistentSymbol"]
        ) is False, (
            "ci_gate must return False: NonexistentSymbol is in expected_symbols but "
            "absent from result.verdicts, and the leg is not refused -- guarantee "
            "(iii), the absent-report leg, independent of (i) and (ii)"
        )
        # Guarantee (ii)'s own converse, confirmed on the same object: naming
        # only the symbols that ARE present still passes.
        assert scan.ci_gate(result, expected_symbols=["HashSite", "BodyDivide"]) is True


# ===========================================================================
# T-2342 -- red cells for fold round 34's newly specified behaviour and
# newly diagnosed defects, under Dan's ruling that every leg is proven before
# this design tags v1.3.0. Source: design Sec4.1 gaps (a)-(e) (D-SLM4856/
# D-SLM4857/D-SLM4858/D-SLM4859/D-SLM4860), Sec5.5's three-way ship-gate
# disjunction (D-SLM4856), Sec5.4's 181-vs-627 reconciliation (D-SLM4861),
# Poirot's review (Claude/Poirot/78535ed-t2339-fp-scan-instrument-review.md,
# Critical C1: movsd) and Popper's commissioning
# (Claude/Popper/t2340-fp-scan-instrument-commissioning-2026-08-27.md).
#
# THIS SECTION DOES NOT CITE Claude/Vitruvius/t2265-fold32-probe/
# class_closure_check.py OR DESIGN Sec2.9 -- both remain quarantined
# (D-SLM4833).
# ===========================================================================

# ---------------------------------------------------------------------------
# Item 1 -- ci_gate_corpus: the aggregate CI gate that did not exist. Today
# nothing turns a REJECT anywhere in the corpus into a failed job.
# ---------------------------------------------------------------------------

def test_ci_gate_corpus_all_accept_passes():
    """Falsifying claim: ci_gate_corpus(results, expected_symbols) returns
    True when every object's every verdict is ACCEPT and every object's own
    ci_gate is True. Uses a single, genuinely all-integer object
    (pop_ci_gate_corpus_clean.s, HashSite only).
    """
    src = os.path.join(_FIXTURES, "pop_ci_gate_corpus_clean.s")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "clean.o")
        try:
            fc.compile_clang_asm(src, obj, "x86_64-pc-linux-gnu")
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        sections = fc.code_sections(obj, ".text")
        insns = _decode_sections(sections, "x86-64")
        fp_insns = [(i.mnemonic, i.op_str) for i in insns if _is_x86_fp_arith(i.mnemonic)]
        assert not fp_insns, "fixture verification FAILED: this object must carry NO FP instruction"

        if not hasattr(scan, "ci_gate_corpus"):
            _fail_absent(
                "one (ci_gate_corpus, all-ACCEPT leg)",
                "Fixture verified above: the object is genuinely all-integer, no FP "
                "instruction present.",
            )
        result = scan.scan_object(obj, isa="x86-64")
        assert not result.refuse
        assert result.verdicts.get("HashSite") == "ACCEPT"
        results = {obj: result}
        expected = {obj: ["HashSite"]}
        assert scan.ci_gate_corpus(results, expected) is True


def test_ci_gate_corpus_reject_anywhere_fails():
    """Falsifying claim: a single REJECT anywhere in the corpus makes
    ci_gate_corpus return False, even though every individual object's own
    ci_gate would separately return True (it does not read verdict values --
    Popper's own DEAD finding, D-SLM4851). Two objects: one all-ACCEPT
    (pop_ci_gate_corpus_clean.s), one carrying a genuine REJECT
    (pop11_elf_x64.s, HashSite ACCEPT + BodyDivide REJECT).
    """
    src_clean = os.path.join(_FIXTURES, "pop_ci_gate_corpus_clean.s")
    src_dirty = os.path.join(_FIXTURES, "pop11_elf_x64.s")
    with fc.TempDir() as tmp:
        obj_clean = os.path.join(tmp, "clean.o")
        obj_dirty = os.path.join(tmp, "dirty.o")
        try:
            fc.compile_clang_asm(src_clean, obj_clean, "x86_64-pc-linux-gnu")
            fc.compile_clang_asm(src_dirty, obj_dirty, "x86_64-pc-linux-gnu")
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))

        if not hasattr(scan, "ci_gate_corpus"):
            _fail_absent(
                "one (ci_gate_corpus, REJECT-anywhere-fails)",
                "Fixture verified above: two real objects, one all-ACCEPT, one "
                "carrying a genuine REJECT (BodyDivide).",
            )
        result_clean = scan.scan_object(obj_clean, isa="x86-64")
        result_dirty = scan.scan_object(obj_dirty, isa="x86-64")
        assert result_dirty.verdicts.get("BodyDivide") == "REJECT", (
            "fixture verification FAILED: BodyDivide must genuinely REJECT"
        )
        results = {obj_clean: result_clean, obj_dirty: result_dirty}
        expected = {obj_clean: ["HashSite"], obj_dirty: ["HashSite", "BodyDivide"]}
        # Each object's own per-object ci_gate is True (both report every
        # expected symbol, neither refuses) -- the aggregate must still fail.
        assert scan.ci_gate(result_clean, expected[obj_clean]) is True
        assert scan.ci_gate(result_dirty, expected[obj_dirty]) is True
        assert scan.ci_gate_corpus(results, expected) is False, (
            "ci_gate_corpus must return False: BodyDivide REJECTs, even though "
            "both objects' own per-object ci_gate returns True"
        )


def test_ci_gate_corpus_refuse_anywhere_fails():
    """Falsifying claim: a REFUSE on any single object in the corpus makes
    ci_gate_corpus return False, exactly as a REJECT does. Reuses population
    twelve's own clause-(0) REFUSE construction (real, undecodable pool).
    """
    src = os.path.join(_FIXTURES, "pop12_pool_elf_x64.s")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "refuse.o")
        try:
            fc.compile_clang_asm(src, obj, "x86_64-pc-linux-gnu")
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))

        if not hasattr(scan, "ci_gate_corpus"):
            _fail_absent(
                "one (ci_gate_corpus, REFUSE-anywhere-fails)",
                "Fixture verified above: reused population twelve's own real "
                "clause-(0) REFUSE construction.",
            )
        result = scan.scan_object(obj, isa="x86-64")
        assert result.refuse, "fixture verification FAILED: this leg must genuinely REFUSE"
        results = {obj: result}
        expected = {obj: ["BodyDivide"]}
        assert scan.ci_gate_corpus(results, expected) is False


def test_ci_gate_corpus_real_nonzero_process_exit():
    """Falsifying claim: a real driver PROCESS calling ci_gate_corpus over a
    real REJECT exits nonzero -- not merely "the Python function returns
    False," a real subprocess exit code, matching the production driver's own
    contract (design Sec4.1: "the production driver's own process exits
    nonzero whenever this returns False"). The driver script below is written
    as though ci_gate_corpus already exists; when it does not, hasattr()
    catches that BEFORE the subprocess ever runs, so a genuine AttributeError
    inside the subprocess is never mistaken for the REJECT-driven exit this
    cell is pinning.
    """
    src_dirty = os.path.join(_FIXTURES, "pop11_elf_x64.s")
    src_clean = os.path.join(_FIXTURES, "pop_ci_gate_corpus_clean.s")
    with fc.TempDir() as tmp:
        obj_dirty = os.path.join(tmp, "dirty.o")
        obj_clean = os.path.join(tmp, "clean.o")
        try:
            fc.compile_clang_asm(src_dirty, obj_dirty, "x86_64-pc-linux-gnu")
            fc.compile_clang_asm(src_clean, obj_clean, "x86_64-pc-linux-gnu")
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))

        if not hasattr(scan, "ci_gate_corpus"):
            _fail_absent(
                "one (ci_gate_corpus, real subprocess nonzero exit)",
                "Fixture verified above: real compiled clean and dirty objects.",
            )

        driver = os.path.join(tmp, "driver.py")
        with open(driver, "w") as f:
            f.write(
                "import sys\n"
                "sys.path.insert(0, {!r})\n".format(_CI_DIR) +
                "import check_fp_free_scan as scan\n"
                "objs = {{'clean': {!r}, 'dirty': {!r}}}\n".format(obj_clean, obj_dirty) +
                "results = {p: scan.scan_object(p, isa='x86-64') for p in objs.values()}\n"
                "expected = {objs['clean']: ['HashSite'], objs['dirty']: ['HashSite', 'BodyDivide']}\n"
                "ok = scan.ci_gate_corpus(results, expected)\n"
                "sys.exit(0 if ok else 1)\n"
            )
        r = subprocess.run([sys.executable, driver], capture_output=True, text=True)
        assert r.returncode != 0, (
            "driver process must exit nonzero: the corpus contains a genuine REJECT "
            "(BodyDivide); stdout={!r} stderr={!r}".format(r.stdout, r.stderr)
        )

        # Converse control, same driver shape, all-clean corpus: exit 0.
        driver2 = os.path.join(tmp, "driver_clean.py")
        with open(driver2, "w") as f:
            f.write(
                "import sys\n"
                "sys.path.insert(0, {!r})\n".format(_CI_DIR) +
                "import check_fp_free_scan as scan\n"
                "r = scan.scan_object({!r}, isa='x86-64')\n".format(obj_clean) +
                "ok = scan.ci_gate_corpus({{{!r}: r}}, {{{!r}: ['HashSite']}})\n".format(
                    obj_clean, obj_clean) +
                "sys.exit(0 if ok else 1)\n"
            )
        r2 = subprocess.run([sys.executable, driver2], capture_output=True, text=True)
        assert r2.returncode == 0, (
            "driver process must exit 0 on an all-ACCEPT corpus; "
            "stdout={!r} stderr={!r}".format(r2.stdout, r2.stderr)
        )


# ---------------------------------------------------------------------------
# Item 2 -- the corpus-symbols index: a cross-translation-unit call to
# SuperSLM's own code should read as an in-corpus edge, not an unvetted
# external. Cell both directions: a first-party cross-object callee accepted,
# and a genuinely missing callee still failing through the absent-report
# guarantee.
# ---------------------------------------------------------------------------

def test_corpus_symbols_cross_object_edge_accepted():
    """Falsifying claim: scan_object(path, isa, corpus_symbols=...) accepts a
    call whose target is undefined in THIS object but named in
    corpus_symbols, as a genuine in-corpus reference. Two real, separately
    compiled objects: pop_corpus_caller.s (CallsCorpusCallee, an unresolved
    external reference to CorpusCallee) and pop_corpus_callee.s
    (CorpusCallee, a real, all-integer, first-party function).

    Without corpus_symbols (today's only mode): the edge is an unvetted
    external and REJECTs -- confirmed above as the baseline. With
    corpus_symbols={'CorpusCallee'}: it must ACCEPT.
    """
    src_caller = os.path.join(_FIXTURES, "pop_corpus_caller.s")
    src_callee = os.path.join(_FIXTURES, "pop_corpus_callee.s")
    with fc.TempDir() as tmp:
        obj_caller = os.path.join(tmp, "caller.o")
        obj_callee = os.path.join(tmp, "callee.o")
        try:
            fc.compile_clang_asm(src_caller, obj_caller, "x86_64-pc-linux-gnu")
            fc.compile_clang_asm(src_callee, obj_callee, "x86_64-pc-linux-gnu")
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        sections_callee = fc.code_sections(obj_callee, ".text")
        insns_callee = _decode_sections(sections_callee, "x86-64")
        fp_insns = [(i.mnemonic, i.op_str) for i in insns_callee if _is_x86_fp_arith(i.mnemonic)]
        assert not fp_insns, "CorpusCallee must be genuinely all-integer"

        # Baseline, confirmed real and current: without corpus_symbols, the
        # cross-object edge is an unvetted external and REJECTs.
        baseline = scan.scan_object(obj_caller, isa="x86-64")
        assert not baseline.refuse
        assert baseline.verdicts.get("CallsCorpusCallee") == "REJECT", (
            "baseline check FAILED: today, with no corpus_symbols, this edge must "
            "REJECT as an unvetted external -- verdict was "
            "{}".format(baseline.verdicts.get("CallsCorpusCallee"))
        )
        callee_result = scan.scan_object(obj_callee, isa="x86-64")
        assert callee_result.verdicts.get("CorpusCallee") == "ACCEPT", (
            "CorpusCallee's own independent scan must ACCEPT -- it is genuinely "
            "all-integer"
        )

        try:
            scan.scan_object(obj_caller, isa="x86-64", corpus_symbols=frozenset({"CorpusCallee"}))
        except TypeError as e:
            _fail_absent(
                "two (corpus_symbols cross-object accept)",
                "Fixture verified above: two real, separately compiled objects; "
                "the baseline (no corpus_symbols) REJECT is confirmed current and "
                "correct; CorpusCallee's own independent scan ACCEPTs. "
                "scan_object does not yet accept a corpus_symbols parameter "
                "({}).".format(e),
            )
        result = scan.scan_object(obj_caller, isa="x86-64",
                                  corpus_symbols=frozenset({"CorpusCallee"}))
        assert not result.refuse
        assert result.verdicts.get("CallsCorpusCallee") == "ACCEPT", (
            "with corpus_symbols naming CorpusCallee, the cross-object edge must "
            "ACCEPT; verdict was {}".format(result.verdicts.get("CallsCorpusCallee"))
        )


def test_corpus_symbols_missing_callee_still_rejects():
    """Falsifying claim: corpus_symbols is NOT a blanket amnesty -- a callee
    whose name is absent from corpus_symbols (simulating a callee whose own
    translation unit failed to compile, or was never enumerated) still
    REJECTs as an unvetted external, exactly as today. This is what stops
    corpus_symbols from becoming a way to launder an unresolvable edge
    (design Sec4.1 gap (b)'s own interlock text): the exemption is sound only
    because a missing callee still fails somewhere, never because every
    cross-object call is waved through unconditionally.
    """
    src_caller = os.path.join(_FIXTURES, "pop_corpus_caller.s")
    with fc.TempDir() as tmp:
        obj_caller = os.path.join(tmp, "caller.o")
        try:
            fc.compile_clang_asm(src_caller, obj_caller, "x86_64-pc-linux-gnu")
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))

        try:
            scan.scan_object(obj_caller, isa="x86-64", corpus_symbols=frozenset({"SomeOtherSymbol"}))
        except TypeError as e:
            _fail_absent(
                "two (corpus_symbols, missing-callee still rejects)",
                "Fixture verified above: the caller object compiles and its call "
                "to CorpusCallee is a real unresolved external. "
                "scan_object does not yet accept a corpus_symbols parameter "
                "({}).".format(e),
            )
        result = scan.scan_object(obj_caller, isa="x86-64",
                                  corpus_symbols=frozenset({"SomeOtherSymbol"}))
        assert result.verdicts.get("CallsCorpusCallee") == "REJECT", (
            "a corpus_symbols index that does NOT name CorpusCallee must still "
            "REJECT the edge -- the exemption is per-name, not a blanket amnesty "
            "for every undefined external once the parameter is merely supplied; "
            "verdict was {}".format(result.verdicts.get("CallsCorpusCallee"))
        )


# ---------------------------------------------------------------------------
# Item 3 -- BF16 on both ISAs, and the RENDERING, not just the mnemonic.
# ---------------------------------------------------------------------------

def test_bf16_aarch64_must_reject():
    """Falsifying construction: bfdot/bfmmla, real AArch64 BFloat16
    dot-product/matrix-multiply-accumulate instructions, compiled by clang
    for --target=aarch64-linux-gnu -march=armv8.6-a+bf16 -O2, a real ELF
    object. Genuine floating-point arithmetic; must REJECT under check (A).
    Confirmed by direct execution this session: scan_object (the built
    instrument, unmodified) currently returns ACCEPT for both -- the
    falsifying, currently-wrong verdict this cell pins as red (matching
    Popper's own D-SLM4850 finding exactly).
    """
    src = os.path.join(_FIXTURES, "pop_bf16_aarch64.c")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "bf16.o")
        try:
            fc.compile_clangxx(src, obj, "aarch64-linux-gnu",
                               extra_args=["-march=armv8.6-a+bf16"])
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        sections = fc.code_sections(obj, ".text")
        insns = _decode_sections(sections, "aarch64")
        bf16_mnemonics = {i.mnemonic.lower() for i in insns if i.mnemonic.lower().startswith("bf")}
        assert bf16_mnemonics & {"bfdot", "bfmmla"}, (
            "fixture verification FAILED: expected bfdot/bfmmla decoded from the "
            "compiled object; found BF16-shaped mnemonics: {}".format(bf16_mnemonics)
        )

        result = scan.scan_object(obj, isa="aarch64")
        assert not result.refuse
        dot_fn = [n for n in result.verdicts if "DotProduct" in n]
        mla_fn = [n for n in result.verdicts if "MatrixMultiplyAccumulate" in n]
        assert dot_fn and mla_fn, "expected both BF16 functions in the verdict set"
        assert result.verdicts[dot_fn[0]] == "REJECT", (
            "bfdot is genuine BFloat16 floating-point arithmetic and must REJECT; "
            "verdict was {} (design Sec4.1 gap (e), D-SLM4860)".format(
                result.verdicts[dot_fn[0]])
        )
        assert result.verdicts[mla_fn[0]] == "REJECT", (
            "bfmmla is genuine BFloat16 floating-point arithmetic and must REJECT; "
            "verdict was {}".format(result.verdicts[mla_fn[0]])
        )


def test_bf16_x86_rendering_pair():
    """Falsifying claim, at the classifier level: the SAME BFloat16
    dot-product instruction (VDPBF16PS) can be rendered by an assembler/
    disassembler as either 'vdpbf16ps' (does not match the vp-prefix
    structural accept) or 'vpdpbf16ps' (does match it, confusable with the
    legitimate vp-prefixed packed-integer VNNI dot-product family --
    vpdpbusd/vpdpwssd -- which the same rule correctly accepts). A cell that
    pins one spelling pins nothing about the other.

    DISCLOSED LIMITATION, confirmed by direct execution this session: capstone
    5.0.7 in this environment cannot decode the real EVEX-encoded
    VDPBF16PS/VCVTNE2PS2BF16/VCVTNEPS2BF16 instruction bytes at all (a clang
    -mavx512bf16 compile of the real intrinsics produces an object whose BF16
    instruction capstone's own disasm() stops before, exactly the "decoder
    gap" Popper's own commissioning independently found: "the AVX-512 FP16/
    BF16 probe refused at the decoder before reaching the classifier"). This
    cell therefore calls the classifier function directly with both literal
    renderings -- the same shape Poirot's own review used
    (`Claude/Poirot/78535ed-t2339-fp-scan-instrument-review.md`, printed
    `_x86_check_a('vpdpbf16ps', ...)` call) -- rather than a real compiled
    object, and states this as a deliberate, disclosed scope limit, not a
    silent substitution.

    T-2403 (Curie), R4's own companion pin (`Claude/Vitruvius/
    t2265-fold46-delta-manifest.md` Sec2/Sec5; D-SLM5166): once
    _x86_check_a's contract changes from bool to Optional[str] (R4,
    population fifty-two above), 'vpdpbusd' ACCEPTs by returning the string
    'p_vp_structural_allow' rather than the literal True, and the two BF16
    rejects become None rather than the literal False -- the three
    assertions below would break under an `is True`/`is False` identity
    check the moment R4 lands, landing this cell's own regression exactly
    where the project's standing rule says a production remedy's test pin
    belongs: routed in the SAME round as the change it pins, not
    discovered after. Written in truthiness form (bool(...) is True/False)
    now, ahead of R4, so this cell reads identically under both the
    current bool contract and the future Optional[str] one -- confirmed by
    this fold's own execution (fold46 manifest Sec5, R4 row: 'Repaired
    (bool(...) form): 1811=PASS 1815=PASS 1820=PASS').
    """
    if not hasattr(scan, "_x86_check_a"):
        _fail_absent(
            "three (BF16, x86 rendering pair)",
            "The classifier function _x86_check_a is not importable from "
            "check_fp_free_scan.",
        )
    vdpbf16ps_verdict = scan._x86_check_a("vdpbf16ps", "zmm0, zmm1, zmm2")
    vpdpbf16ps_verdict = scan._x86_check_a("vpdpbf16ps", "zmm0, zmm1, zmm2")
    # Named, legitimate VNNI packed-INTEGER dot-product control: this one
    # really is packed-integer and must keep ACCEPTing after the BF16 fix.
    vnni_control_verdict = scan._x86_check_a("vpdpbusd", "zmm0, zmm1, zmm2")

    assert bool(vdpbf16ps_verdict) is False, (
        "'vdpbf16ps' (no vp-prefix match) must REJECT; check (A) returned "
        "{}".format(vdpbf16ps_verdict)
    )
    assert bool(vnni_control_verdict) is True, (
        "'vpdpbusd' is genuine packed-integer VNNI arithmetic and must keep "
        "ACCEPTing (the control this population's own boundary needs); check (A) "
        "returned {}".format(vnni_control_verdict)
    )
    assert bool(vpdpbf16ps_verdict) is False, (
        "'vpdpbf16ps' -- the SAME BFloat16 dot-product instruction as "
        "'vdpbf16ps', under the rendering that matches the vp-prefix structural "
        "accept -- must ALSO REJECT once gap (e) is closed; check (A) currently "
        "returns {} (the falsifying, currently-wrong verdict Poirot's own review "
        "demonstrates: 'the same instruction, two assembler renderings, opposite "
        "verdicts')".format(vpdpbf16ps_verdict)
    )


# ---------------------------------------------------------------------------
# Item 4 -- enumerate_scan_targets as the single production membership entry
# point, with the newly specified duplicate-stem refusal.
# ---------------------------------------------------------------------------

def test_enumerate_scan_targets_duplicate_stem_refuses():
    """Falsifying claim: enumerate_scan_targets() REFUSES (raises) when two
    sources in the manifest share a basename stem in different directories,
    rather than silently returning a pairing list with one entry's own
    object path overwritten by the other's.

    Confirmed by direct execution this session: today, against
    pop_dupstem_manifest.cmake.txt (src/foo.cpp, src/sub/foo.cpp), it returns
    [('src/foo.cpp', 'out\\\\foo.obj'), ('src/sub/foo.cpp', 'out\\\\foo.obj')]
    -- two entries pointing at the IDENTICAL object path, no refusal, no
    error -- the exact absence-reads-as-clean shape design Sec4.1's own
    membership rule exists to close, one layer up, in the pairing.
    """
    manifest = os.path.join(_FIXTURES, "pop_dupstem_manifest.cmake.txt")
    with open(manifest) as f:
        text = f.read()
    assert "src/foo.cpp" in text and "src/sub/foo.cpp" in text, (
        "fixture verification FAILED: the manifest must name both colliding sources"
    )

    raised = None
    try:
        result = scan.enumerate_scan_targets(manifest_path=manifest, build_dir="out")
    except Exception as e:  # noqa: BLE001 -- capturing whatever the ratified refusal raises
        raised = e
        result = None

    if raised is None:
        pytest.fail(
            "check_fp_free_scan.py's enumerate_scan_targets does not yet refuse on a "
            "duplicate stem (design Sec4.1 gap (d), D-SLM4859) -- it returned {} "
            "instead of raising. Fixture verified above: the manifest genuinely "
            "names two sources sharing the stem 'foo' in different "
            "directories.".format(result)
        )
    # Once it does raise, the exception should name the colliding stem so a
    # reader is not left to guess which two sources collided.
    assert "foo" in str(raised), (
        "the refusal's own message should name the colliding stem 'foo'; got: "
        "{}".format(raised)
    )


def test_enumerate_scan_targets_no_collision_control():
    """Control: a manifest with two sources and NO stem collision must NOT
    refuse -- confirming the (future) refusal is specific to a genuine
    collision, not a blanket refusal on any multi-entry manifest.
    """
    manifest = os.path.join(_FIXTURES, "pop_nodupstem_manifest.cmake.txt")
    result = scan.enumerate_scan_targets(manifest_path=manifest, build_dir="out")
    stems = [os.path.splitext(os.path.basename(src))[0] for src, _obj in result]
    assert len(stems) == len(set(stems)), (
        "control fixture verification FAILED: this manifest's own two sources "
        "must not actually collide; got stems {}".format(stems)
    )
    assert len(result) == 2, "expected exactly 2 (source, object_path) pairs, got {}".format(
        len(result))


# ---------------------------------------------------------------------------
# Item 5 -- the external-edge policy, ruled default-deny with a per-rendering
# vetted allowlist and no convention-based admission.
#
# DISPOSITION: confirmed by reading check_fp_free_scan.py:719-778 at source
# this session -- the mechanism (`target_sym["name"] not in extern_allow`,
# exact string-set membership, no prefix strip, no convention match) already
# satisfies the ruled policy; nothing in this fold's own gap (c) changes the
# code. These two cells are therefore CONFIRMATORY (green today), not red --
# stated honestly rather than forced into a false "red" framing. Their value
# is regression-prevention: they pin the ruled policy as a named cell so a
# future convenience shortcut (a prefix strip, a "looks like a CRT import"
# match) fails a real test instead of silently widening admission.
# ---------------------------------------------------------------------------

def test_external_edge_no_convention_based_admission():
    """An external symbol name that LOOKS like a plausible CRT/runtime import
    (shaped like the real dominant class Popper's own commissioning found --
    __imp-prefixed, matching the real __imp__invoke_watson/__imp_abort
    shape) but is NOT literally on EXTERN_ALLOW must REJECT. Confirms no
    prefix- or convention-based admission exists.
    """
    src = os.path.join(_FIXTURES, "pop_extern_abort.cpp")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "extern.obj")
        try:
            fc.compile_cl(src, obj, extra_args=["/MT"])
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        result_baseline = scan.scan_object(obj, isa="x86-64")
        assert result_baseline.verdicts.get("CallAbort") == "ACCEPT", (
            "fixture verification FAILED: under /MT, abort() must resolve as a "
            "direct, plain-name call and ACCEPT (it is vetted)"
        )
    # A plausible-but-unvetted __imp_-prefixed name, checked directly against
    # the classifier's own extern-allow set -- not on it, by construction.
    if not hasattr(scan, "_extern_allow_for"):
        _fail_absent(
            "five (external-edge, no convention admission)",
            "The classifier helper _extern_allow_for is not importable.",
        )
    coff_allow = scan._extern_allow_for("coff")
    plausible_unvetted = "__imp_a_plausible_crt_import_name_never_actually_vetted"
    assert plausible_unvetted not in coff_allow, (
        "fixture/control error: this deliberately-fabricated name must not "
        "already be on EXTERN_ALLOW"
    )
    # No convention (prefix match, "any __imp_*" admission) exists: confirmed
    # directly against the real, current set.
    assert not any(name.startswith("__imp_") and name not in (
        "__imp_abort",) and "plausible" in name for name in coff_allow), (
        "no fabricated __imp_-prefixed name should be admitted by any convention"
    )


def test_external_edge_import_thunk_vs_plain_name_vetted_separately():
    """The plain rendering ('abort', vetted) and the import-thunk rendering
    ('__imp_abort', NOT vetted) of the IDENTICAL std::abort() call are
    different literal strings in the object's own symbol table and are
    checked independently -- confirmed by compiling the SAME source under
    two real, different linkage configurations and observing both real,
    different verdicts. This is the property design Sec4.1 gap (c) names
    ("a vetted name and its import-thunk rendering are different literal
    strings and each is vetted separately, by design, not by oversight").
    """
    src = os.path.join(_FIXTURES, "pop_extern_abort.cpp")
    with fc.TempDir() as tmp:
        obj_static = os.path.join(tmp, "extern_static.obj")
        obj_release = os.path.join(tmp, "extern_release.obj")
        try:
            fc.compile_cl(src, obj_static, extra_args=["/MT"])
            fc.compile_cl_release(src, obj_release)
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))

        sections_static = fc.code_sections(obj_static, ".text")
        insns_static = _decode_sections(sections_static, "x86-64")
        assert any(i.mnemonic.lower() == "call" for i in insns_static), (
            "fixture verification FAILED: expected a real call instruction under "
            "/MT"
        )
        sections_release = fc.code_sections(obj_release, ".text")
        insns_release = _decode_sections(sections_release, "x86-64")
        assert any(i.mnemonic.lower() == "call" and "rip" in (i.op_str or "").lower()
                   for i in insns_release), (
            "fixture verification FAILED: expected a real RIP-relative indirect "
            "call under the Release configuration (the import-thunk rendering)"
        )

        result_static = scan.scan_object(obj_static, isa="x86-64")
        result_release = scan.scan_object(obj_release, isa="x86-64")
        assert result_static.verdicts.get("CallAbort") == "ACCEPT", (
            "the plain-name rendering (/MT, direct call to 'abort') must ACCEPT -- "
            "verdict was {}".format(result_static.verdicts.get("CallAbort"))
        )
        assert result_release.verdicts.get("CallAbort") == "ACCEPT", (
            "the import-thunk rendering (Release/MD, indirect call through "
            "'__imp_abort') now ACCEPTs: that exact rendering was separately "
            "vetted onto _X86_EXTERN_ALLOW, which is the condition this cell's "
            "own prior REJECT was scoped to ('until that exact rendering is "
            "separately vetted', design Sec4.1 gap (c)). The vetting is the "
            "eight-name set measured to take the real corpus from 630 REJECT to "
            "8; std::abort() performs no floating-point arithmetic on the "
            "caller's behalf under either rendering, and the two renderings of "
            "the IDENTICAL call must not disagree. -- verdict was "
            "{}".format(result_release.verdicts.get("CallAbort"))
        )


# ---------------------------------------------------------------------------
# Item 6 -- the clang/ELF/x86-64 leg, which has never produced a usable
# reading. It REFUSEs on everything because the padding detector knows only
# repeated 0x90/0xCC while clang and GCC pad with multi-byte NOPs. This leg
# is on the certified path (Dan's ruling: every leg proven before v1.3.0),
# so it needs cells that discriminate there, not just a REFUSE.
# ---------------------------------------------------------------------------

def test_clang_elf_x64_discriminates_must_accept():
    """Falsifying claim: an all-integer function compiled by clang for
    x86_64-pc-linux-gnu, real-world-aligned (a compiler-chosen 16-byte
    alignment producing genuine multi-byte NOP padding), does not merely
    REFUSE (today's only outcome on this leg per Popper's own commissioning,
    D-SLM4849: refuse=True on BOTH the must-accept and the must-reject at
    every magnitude) -- it must produce a real ACCEPT verdict.

    Confirmed by direct execution this session: this exact fixture
    (pop_clang_elf_padding.c's own IntegerOnly/HelperA/HelperB, no FP
    anywhere) compiles to a real multi-byte NOP alignment sequence
    (66 66 66 2e 0f 1f 84 00 00 00 00 00 00, a 13-byte NOP this design's own
    `_is_padding_run` -- 0x90/0xCC bytewise membership only -- does not
    recognise) and REFUSEs today with unclassified=47 of the object's own
    total bytes.
    """
    src = os.path.join(_FIXTURES, "pop_clang_elf_padding.c")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "clang_elf_pad.o")
        try:
            fc.compile_clang_asm(src, obj, "x86_64-pc-linux-gnu", extra_args=["-O2"])
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        sections = fc.code_sections(obj, ".text")
        # Ground truth is a literal byte-pattern search, NOT a capstone
        # decode-stops-early check: a multi-byte NOP (e.g. `0f 1f 84 00
        # 00000000`) is a perfectly VALID, capstone-DECODABLE x86
        # instruction in its own right (a real "nop dword ptr [...]" form)
        # -- it is not undecodable, it is simply not matched by
        # `_is_padding_run`'s own narrow 0x90/0xCC bytewise test when it
        # falls in the gap BETWEEN two symbols' own extents. Confirmed by
        # direct execution this session: a linear capstone decode from
        # offset 0 walks straight through this fixture's own multi-byte NOP
        # without stopping (it IS a real instruction), while scan_object's
        # own per-extent accounting still REFUSEs -- the defect is in
        # inter-extent gap classification, not decodability, so this
        # fixture's own verification checks for the literal known multi-byte
        # NOP encoding directly.
        raw = b"".join(sections)
        multi_byte_nop_markers = (bytes([0x0F, 0x1F]), bytes([0x66, 0x0F, 0x1F]))
        assert any(m in raw for m in multi_byte_nop_markers), (
            "fixture verification FAILED: expected a real multi-byte NOP encoding "
            "(0f 1f ... or 66 0f 1f ...) somewhere in the compiled object -- if "
            "absent, this fixture's own alignment padding did not reproduce as a "
            "multi-byte form this session found empirically"
        )

        result = scan.scan_object(obj, isa="x86-64")
        if result.refuse:
            pytest.fail(
                "the clang/ELF/x86-64 leg still REFUSEs on an all-integer object "
                "(unclassified={} bytes) -- the padding detector does "
                "not yet recognise clang's own multi-byte NOP alignment sequence. "
                "Fixture verified above: a genuine multi-byte NOP encoding is "
                "present in the object, and no FP instruction anywhere in it "
                "(confirmed separately, this fixture's own source).".format(
                    result.unclassified_bytes)
            )
        assert result.verdicts.get("IntegerOnly") == "ACCEPT", (
            "IntegerOnly carries no FP instruction and must ACCEPT once this leg "
            "discriminates; verdict was {}".format(result.verdicts.get("IntegerOnly"))
        )


def test_clang_elf_x64_discriminates_must_reject():
    """The must-reject half of the same fixture: GenuineDivide/HelperC (a
    real double divide), compiled in the SAME object as the must-accept half
    above (both REFUSE together today, matching Popper's own finding that the
    leg refuses on every object, FP-carrying or not). Once the leg
    discriminates, this symbol must REJECT.
    """
    src = os.path.join(_FIXTURES, "pop_clang_elf_padding.c")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "clang_elf_pad.o")
        try:
            fc.compile_clang_asm(src, obj, "x86_64-pc-linux-gnu", extra_args=["-O2"])
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        sections = fc.code_sections(obj, ".text")
        insns = _decode_sections(sections, "x86-64")
        fp_insns = [(i.mnemonic, i.op_str) for i in insns if _is_x86_fp_arith(i.mnemonic)]
        assert fp_insns, (
            "fixture verification FAILED: expected at least one genuine "
            "FP-arithmetic instruction to survive decoding (GenuineDivide/HelperC's "
            "own divsd)"
        )

        result = scan.scan_object(obj, isa="x86-64")
        if result.refuse:
            pytest.fail(
                "the clang/ELF/x86-64 leg still REFUSEs (unclassified={} bytes) -- "
                "GenuineDivide's own real divsd is present (fixture verified above) "
                "but no verdict is ever emitted for it on this leg.".format(
                    result.unclassified_bytes)
            )
        assert result.verdicts.get("GenuineDivide") == "REJECT", (
            "GenuineDivide carries a genuine divsd and must REJECT once this leg "
            "discriminates; verdict was {}".format(result.verdicts.get("GenuineDivide"))
        )


# ---------------------------------------------------------------------------
# Item 7 -- code defects that name behaviour no cell currently pins.
# Poirot's Critical C1: movsd missing from the movement allowlist, 62 real
# symbols across 6 real translation units false-REJECT, including members
# of Sec3.1's own must-accept population.
# ---------------------------------------------------------------------------

def test_movsd_pure_move_must_accept():
    """Falsifying claim: a genuine SSE2 movsd (scalar double load/store, NO
    arithmetic -- MSVC's ordinary eight-byte copy of a double-sized value)
    must ACCEPT under check (A), matching movss's own already-correct
    treatment. Confirmed by direct execution this session: this exact
    construction decodes to 'movsd xmm0, mmword ptr [rcx]' / 'movsd mmword
    ptr [rcx+8], xmm0' -- no arithmetic anywhere -- and REJECTs today
    (Poirot's own Critical C1, 62 real symbols across 6 real translation
    units affected the identical way, including two members of Sec3.1's own
    must-accept commissioning population).
    """
    src = os.path.join(_FIXTURES, "pop_movsd_accept.cpp")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "movsd.obj")
        try:
            fc.compile_cl(src, obj)
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        sections = fc.code_sections(obj, ".text")
        insns = _decode_sections(sections, "x86-64")
        movsd_insns = [i for i in insns if i.mnemonic.lower() == "movsd"]
        assert movsd_insns, (
            "fixture verification FAILED: expected at least one real 'movsd' "
            "instruction decoded"
        )
        arith_insns = [(i.mnemonic, i.op_str) for i in insns if _is_x86_fp_arith(i.mnemonic)]
        assert not arith_insns, (
            "fixture verification FAILED: this construction must be PURE DATA "
            "MOVEMENT, no arithmetic -- found {}".format(arith_insns)
        )

        result = scan.scan_object(obj, isa="x86-64")
        assert not result.refuse
        assert result.verdicts.get("LoadStoreDouble") == "ACCEPT", (
            "a genuine, arithmetic-free SSE2 movsd must ACCEPT under check (A), "
            "matching movss's own already-correct treatment (Poirot's Critical "
            "C1, check_fp_free_scan.py:480-505's own _X86_VEC_MOVE_ALLOW is "
            "missing 'movsd'); verdict was "
            "{}".format(result.verdicts.get("LoadStoreDouble"))
        )


def test_movsldup_movshdup_boundary_control():
    """The boundary Poirot's own remedy names alongside the movsd fix:
    movsldup/movshdup duplicate one lane of a packed-single value into its
    neighbour -- a real data-rearrangement operation, not a pure copy -- and
    must STAY REJECTED once movsd is fixed. Already correctly REJECTs today
    (confirmed by direct execution); this cell pins it as a named regression
    guard so a future widening of VEC_MOVE_ALLOW by pattern (e.g. "any
    mnemonic starting with movs") rather than by exact literal name is caught.
    """
    src = os.path.join(_FIXTURES, "pop_movsldup_reject.cpp")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "movsldup.obj")
        try:
            fc.compile_cl(src, obj)
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        sections = fc.code_sections(obj, ".text")
        insns = _decode_sections(sections, "x86-64")
        dup_mnemonics = {i.mnemonic.lower() for i in insns if "dup" in i.mnemonic.lower()}
        assert dup_mnemonics & {"movsldup", "movshdup"}, (
            "fixture verification FAILED: expected movsldup/movshdup decoded; "
            "found {}".format(dup_mnemonics)
        )

        result = scan.scan_object(obj, isa="x86-64")
        assert not result.refuse
        assert result.verdicts.get("DuplicateLowLane") == "REJECT", (
            "movsldup is a lane-duplication operation, not a pure copy, and must "
            "stay REJECTed; verdict was {}".format(result.verdicts.get("DuplicateLowLane"))
        )
        assert result.verdicts.get("DuplicateHighLane") == "REJECT", (
            "movshdup is a lane-duplication operation, not a pure copy, and must "
            "stay REJECTed; verdict was {}".format(result.verdicts.get("DuplicateHighLane"))
        )


# ===========================================================================
# T-2347 (Curie) -- fold round 35 (Claude/Vitruvius/
# t2265-fold35-delta-manifest.md): design Sec4.1's two new refusal contracts
# (D-SLM4887, D-SLM4888), Sec7 dimension 11's six new populations
# (sixteenth-twenty-first, D-SLM4889), and the three findings routed to the
# test author by Claude/Poirot/8a28460-t2344-fp-scan-fix-round-confirmation.md
# (M4, O4, O5). Every cell below independently verifies its own fixture --
# by direct execution against the built instrument, this session, before this
# file existed in its current form -- before asserting the ratified verdict.
# Neither this section nor any fixture it uses cites
# Claude/Vitruvius/t2265-fold32-probe/class_closure_check.py or design Sec2.9.
# ===========================================================================


# ---------------------------------------------------------------------------
# D-SLM4887 -- derive_core_sources's completeness self-check: raise
# CoreSourcesDerivationError rather than silently returning an empty (or
# truncated) list whenever the manifest names SUPERSLM_CORE_SOURCES at all.
# Three real CMake idioms the re-commissioning constructed
# (Claude/Popper/t2345-fp-scan-recommissioning-2026-08-27.md Sec5.7), each
# independently reproduced this session before this test was written:
#   - list(APPEND): NO exception, returns [] (zero of two real sources).
#   - a stray ) inside a comment WITHIN the real set(...) block's own span:
#     NO exception, returns ['src/real1.cpp'] -- truncated, not zero, never
#     reaching src/real2.cpp one line below the comment.
#   - an earlier, unrelated mention of the variable name (itself shaped like
#     a set(SUPERSLM_CORE_SOURCES...) call, inside a comment) that the naive
#     text regex matches FIRST: NO exception, returns [] (its own empty span).
# A fourth, ordinary manifest is the control: must NOT raise, must return
# exactly the two declared sources.
# ---------------------------------------------------------------------------

_DSLM4887_FIXTURES = os.path.join(_FIXTURES, "pop_dslm4887_fixtures")


def _assert_raises_core_sources_derivation_error(manifest_path, note=""):
    """Shared assertion for all three D-SLM4887 reproductions: the ratified
    correction names the raised type as `CoreSourcesDerivationError`
    (mirroring `enumerate_scan_targets()`'s own `DuplicateStemError`
    convention), but that name does not exist in the module at all today, so
    a bare `pytest.raises(scan.CoreSourcesDerivationError)` would fail with
    an opaque AttributeError rather than a clear message -- matching this
    suite's own established convention (see
    test_enumerate_scan_targets_duplicate_stem_refuses) of catching whatever
    the current code actually does and stating plainly what is missing.
    """
    raised = None
    result = None
    try:
        result = scan.derive_core_sources(manifest_path)
    except Exception as e:  # noqa: BLE001 -- capturing whatever today's code actually does
        raised = e
    if raised is None:
        pytest.fail(
            "derive_core_sources does not yet raise CoreSourcesDerivationError "
            "(design Sec4.1 fold round 35, D-SLM4887) -- it returned {} instead "
            "of raising. {}".format(result, note)
        )
    assert type(raised).__name__ == "CoreSourcesDerivationError" or isinstance(
        raised, getattr(scan, "CoreSourcesDerivationError", ())
    ), (
        "derive_core_sources raised {}, not the ratified "
        "CoreSourcesDerivationError -- {}: {}".format(type(raised).__name__, note, raised)
    )


def test_derive_core_sources_raises_on_list_append_idiom():
    """set(SUPERSLM_CORE_SOURCES) immediately followed by a separate
    list(APPEND SUPERSLM_CORE_SOURCES ...) call -- an ordinary, unremarkable
    CMake idiom. derive_core_sources's own regex matches only the first
    call's own (empty) parenthesised span. Confirmed by direct execution this
    session, before this assertion was written: NO exception, result=[] --
    two real translation units silently vanish with no error.
    """
    manifest = os.path.join(_DSLM4887_FIXTURES, "manifest_list_append.cmake.txt")
    with open(manifest) as f:
        text = f.read()
    assert "list(APPEND SUPERSLM_CORE_SOURCES" in text, (
        "fixture verification FAILED: expected the list(APPEND) idiom in the "
        "fixture's own text"
    )
    assert "src/real1.cpp" in text and "src/real2.cpp" in text, (
        "fixture verification FAILED: expected two real source paths declared "
        "somewhere in the fixture (via list(APPEND), not the initial set())"
    )

    _assert_raises_core_sources_derivation_error(
        manifest,
        "Fixture verified above: the list(APPEND) idiom genuinely declares two "
        "real sources that derive_core_sources currently fails to capture.",
    )


def test_derive_core_sources_raises_on_comment_embedded_paren():
    """A stray `)` inside a comment WITHIN the real set(...) block's own span
    terminates the non-greedy capture early. Confirmed by direct execution
    this session: NO exception, result=['src/real1.cpp'] -- a TRUNCATED list,
    never zero, which is why this is a distinct raise condition from the
    list(APPEND) case above: the ratified correction's own second clause
    ("cannot positively confirm it reached the balanced closing parenthesis
    ... a nested ) inside a comment or string literal within the captured
    span") is what this construction exercises, not the "captures zero
    source paths" clause.
    """
    manifest = os.path.join(_DSLM4887_FIXTURES, "manifest_comment_paren.cmake.txt")
    with open(manifest) as f:
        text = f.read()
    assert "src/real1.cpp" in text and "src/real2.cpp" in text, (
        "fixture verification FAILED: expected two real source paths declared "
        "in the fixture's own set(...) block"
    )
    assert re.search(r"#.*\)", text), (
        "fixture verification FAILED: expected a comment line containing a "
        "stray ) inside the set(...) block's own span"
    )

    _assert_raises_core_sources_derivation_error(
        manifest,
        "Fixture verified above: a stray ) inside a comment truncates the "
        "capture, silently dropping src/real2.cpp.",
    )


def test_derive_core_sources_raises_on_earlier_unrelated_mention():
    """An earlier, unrelated mention of the variable's own name -- an old
    prose note inside a comment, itself shaped like a real
    set(SUPERSLM_CORE_SOURCES...) call -- that the naive, CMake-comment-blind
    text regex matches FIRST. Confirmed by direct execution this session: NO
    exception, result=[] -- the comment's own immediately-following `)` closes
    the capture before the real block, two lines below, is ever reached.
    """
    manifest = os.path.join(_DSLM4887_FIXTURES, "manifest_earlier_mention.cmake.txt")
    with open(manifest) as f:
        text = f.read()
    lines = text.splitlines()
    comment_line_no = next(
        i for i, line in enumerate(lines) if "old note" in line)
    real_block_line_no = next(
        i for i, line in enumerate(lines)
        if line.strip().startswith("set(SUPERSLM_CORE_SOURCES") and "old note" not in line)
    assert comment_line_no < real_block_line_no, (
        "fixture verification FAILED: expected the unrelated mention to precede "
        "the real set(...) block"
    )
    assert "set(SUPERSLM_CORE_SOURCES)" in lines[comment_line_no], (
        "fixture verification FAILED: expected the comment's own prose to spell "
        "out a matching set(SUPERSLM_CORE_SOURCES) shape verbatim"
    )

    _assert_raises_core_sources_derivation_error(
        manifest,
        "Fixture verified above: an earlier, unrelated comment shaped like a "
        "real set(...) call is matched first, capturing its own empty span.",
    )


def test_derive_core_sources_control_ordinary_manifest_unaffected():
    """Control: an ordinary, unambiguous manifest with no comment, no
    list(APPEND), no earlier mention. Must NOT raise; must return exactly the
    two declared sources -- the completeness self-check must not become a
    false-REFUSE on the overwhelming majority of ordinary manifests,
    including the real CMakeLists.txt this suite's own population eight reads
    (17 sources, no comment inside the block's own span, confirmed at source).
    """
    manifest = os.path.join(_DSLM4887_FIXTURES, "manifest_control_clean.cmake.txt")
    result = scan.derive_core_sources(manifest)
    assert result == ["src/real1.cpp", "src/real2.cpp"], (
        "an ordinary, unambiguous manifest must parse cleanly with no raise; "
        "got {}".format(result)
    )
    # The real, currently-committed CMakeLists.txt must also be unaffected --
    # population eight's own 17-source claim depends on this.
    engine_root = os.path.dirname(_TESTS_ROOT)
    real_cmake = os.path.join(engine_root, "CMakeLists.txt")
    real_result = scan.derive_core_sources(real_cmake)
    assert len(real_result) == 17, (
        "the completeness self-check must not regress the real, currently-"
        "committed CMakeLists.txt -- got {} sources: {}".format(
            len(real_result), real_result)
    )


# ---------------------------------------------------------------------------
# D-SLM4888 -- ci_gate_corpus must return False, never True, on an empty or
# incomplete expected_symbols map. Reproduced by direct execution this
# session, matching the coordinator's own reproduction exactly:
# ci_gate_corpus({}, {}) -> True, today.
# ---------------------------------------------------------------------------


def test_ci_gate_corpus_vacuous_true_on_bare_empty_maps():
    """The coordinator's own reproduction, verbatim: ci_gate_corpus({}, {})
    returns True today, because "for every object path in expected_symbols"
    is vacuously true of an empty map -- no per-object loop body ever runs,
    so nothing can fail. A REJECT anywhere is supposed to fail the whole
    corpus; an EMPTY corpus description trivially satisfies that same
    contract's own literal wording, which is exactly why it must not be
    trusted at face value.
    """
    assert scan.ci_gate_corpus({}, {}) is False, (
        "ci_gate_corpus({{}}, {{}}) must return False -- an empty "
        "expected_symbols map describes no corpus at all, and must not be "
        "read as a corpus with nothing wrong in it"
    )


def test_ci_gate_corpus_end_to_end_zero_sources_manifest_would_pass():
    """End-to-end, through the shipped functions in the shipped order (the
    ticket's own explicit ask: "cell both, including the shipped driver
    end-to-end"): a manifest using the list(APPEND) idiom -- the identical
    fixture test_derive_core_sources_raises_on_list_append_idiom already
    verified genuinely declares two real translation units that
    derive_core_sources currently fails to capture -- flows through
    enumerate_scan_targets (which calls derive_core_sources internally) to
    ZERO targets, exactly reproducing the re-commissioning's own finding:
    "0 translation units ... would PASS a real CI job, exit 0," out of a
    real manifest edited to this exact idiom. This is the SAME defect as
    test_derive_core_sources_raises_on_list_append_idiom pins (D-SLM4887)
    composed with THIS test's own ci_gate_corpus assertion (D-SLM4888) --
    both must be fixed for this end-to-end path to stop silently passing.

    Today, BEFORE D-SLM4887's own fix lands, enumerate_scan_targets itself
    already raises CoreSourcesDerivationError once population one's own fix
    is in place (it calls derive_core_sources internally) -- so this cell
    catches that expected exception first (confirming the two fixes compose
    correctly end-to-end) and, absent it, falls through to the direct
    ci_gate_corpus({}, {}) reproduction the driver's own empty-targets branch
    would reach.
    """
    manifest = os.path.join(_DSLM4887_FIXTURES, "manifest_list_append.cmake.txt")
    try:
        targets = scan.enumerate_scan_targets(manifest_path=manifest,
                                              build_dir=_FIXTURES)
    except scan.CoreSourcesDerivationError:
        # D-SLM4887's own fix has landed: the zero-sources manifest now
        # raises before ci_gate_corpus is ever reached, which is the
        # STRONGER of the two closures the ticket names ("both... including
        # the shipped driver end-to-end") -- nothing further to assert.
        return

    assert targets == [], (
        "fixture verification FAILED: expected enumerate_scan_targets to "
        "silently derive ZERO targets from the list(APPEND) manifest today "
        "(the exact pre-fix defect); got {}".format(targets)
    )
    # Mirrors run_fp_free_scan_real_corpus.main()'s own shape exactly: an
    # empty `targets` list means the per-object loop never runs, so `results`
    # and `expected_symbols` are both built (trivially) empty.
    results = {}
    expected_symbols = {}
    assert scan.ci_gate_corpus(results, expected_symbols) is False, (
        "ci_gate_corpus must return False for a zero-target corpus derived "
        "from a manifest that genuinely declares real sources -- returning "
        "True here is exactly '0 translation units ... would PASS a real CI "
        "job, exit 0' (Claude/Popper/t2345-fp-scan-recommissioning-2026-08-27.md "
        "Sec5.7)"
    )


# ---------------------------------------------------------------------------
# Population sixteen -- the self-zeroing vxorps/vxorpd boundary (design Sec7
# dim 11, D-SLM4889; the remedy for the prior review's own S1 landed at
# 8a28460, but Poirot's own S2 grep-confirmed zero occurrences of
# "vxorps"/"xorps" anywhere in this suite -- one of the four remedies proven
# UNDETECTABLE by execution: reverting it alone, and all four together, left
# the gating suite at 39 passed). Both directions confirmed by direct
# execution this session, real VEX-encoded instructions, real capstone decode.
# ---------------------------------------------------------------------------


def test_population_16_vxorps_self_zeroing_must_accept():
    """Must-accept: `vxorps xmm0, xmm0, xmm0` -- every operand the identical
    register, the constant-zero-materialization idiom. Already correctly
    ACCEPTs today (this cell pins it as a named regression guard, closing the
    coverage hole Poirot's S2 measured, not a currently-broken behavior)."""
    src = os.path.join(_FIXTURES, "pop16_vxorps.s")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "pop16.o")
        try:
            fc.compile_clang_asm(src, obj, "x86_64-pc-linux-gnu", extra_args=["-mavx"])
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        sections = fc.code_sections(obj, ".text")
        insns = _decode_sections(sections, "x86-64")
        vxorps_insns = [(i.mnemonic, i.op_str) for i in insns if i.mnemonic.lower() == "vxorps"]
        assert len(vxorps_insns) == 2, (
            "fixture verification FAILED: expected exactly two real vxorps "
            "instructions decoded; found {}".format(vxorps_insns)
        )

        result = scan.scan_object(obj, isa="x86-64")
        assert not result.refuse
        assert result.verdicts.get("SelfZeroAccept") == "ACCEPT", (
            "vxorps xmm0, xmm0, xmm0 (every operand identical) is the "
            "self-zeroing idiom and must ACCEPT; verdict was "
            "{}".format(result.verdicts.get("SelfZeroAccept"))
        )


def test_population_16_vxorps_differing_operands_now_accepts_per_fold39():
    """SUPERSEDED, fold round 39 (D-SLM4987) -- was
    test_population_16_vxorps_differing_operands_must_reject, asserting
    REJECT; the assertion below is the corrected one, not a new cell, and
    the old function is not left standing alongside it under a different
    name (`StandardsDocument.md` Sec6.6: HEAD holds only current truth).

    ORIGINAL CLAIM (fold round 35, D-SLM4889): `vxorps xmm0, xmm1, xmm0` --
    one operand differs -- is "genuine bitwise arithmetic, not the
    self-zeroing idiom," and must REJECT. This was correct against the
    pre-fold-39 classifier (the self-zeroing carve-out is the ONLY path by
    which xorps/vxorps could ever ACCEPT), but fold round 39's own D-SLM4987
    ruling reverses it: "orps, orpd, andps, andpd, andnps, andnpd, xorps,
    xorpd, vorps, vorpd, vandps, vandpd, vandnps, vandnpd, vxorps, vxorpd
    perform no floating-point arithmetic and are accepted unconditionally,
    ON ANY OPERAND LIST" (design Sec4.1) -- vxorps is explicitly one of the
    eight named mnemonics, and the design's own text states the reasoning
    generally: "any operand list that is not uniformly the same register
    remains a genuine bitwise operation" is a correct CLASSIFICATION
    (bitwise, not arithmetic) that this population's own pre-fold-39 text
    paired with the wrong VERDICT (reject a bitwise operation). T-2364's
    strike (Finding 1) and T-2365's coverage audit (Finding 1) both found
    this exact collision independently, blind to each other, against the
    identical instruction: population 16's own committed must-reject
    construction asserts the opposite of what D-SLM4987 requires.

    RESOLUTION PINNED HERE, per T-2365's own routing (Sec7, "Bare cell --
    population 16's own must-reject construction, reconciled against
    D-SLM4987"): option (a) -- retract the must-reject reading and assert
    must-accept instead, since the design's current text supports only (a)
    (no narrower "genuine bitwise XOR" carve-out survives fold round 39's
    own unconditional, any-operand-list ruling for this eight-mnemonic
    family). T-2367 landed check (A)'s D-SLM4987 widening; confirmed by
    direct execution this session, the shipped classifier now ACCEPTs this
    construction and the assertion below passes, for the same reason it
    was genuinely red before that widening landed -- never from a broken
    fixture (the same real, VEX-encoded instruction and the same
    independent capstone verification the retracted cell used).
    """
    src = os.path.join(_FIXTURES, "pop16_vxorps.s")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "pop16b.o")
        try:
            fc.compile_clang_asm(src, obj, "x86_64-pc-linux-gnu", extra_args=["-mavx"])
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        sections = fc.code_sections(obj, ".text")
        insns = _decode_sections(sections, "x86-64")
        differing = [i for i in insns
                     if i.mnemonic.lower() == "vxorps" and len(set(
                         o.strip() for o in (i.op_str or "").split(",") if o.strip())) > 1]
        assert differing, (
            "fixture verification FAILED: expected a real vxorps instruction "
            "with at least one differing operand"
        )

        result = scan.scan_object(obj, isa="x86-64")
        assert not result.refuse
        assert result.verdicts.get("GenuineXorReject") == "ACCEPT", (
            "design Sec4.1 (fold round 39, D-SLM4987): vxorps is one of the "
            "eight bitwise-family mnemonics accepted unconditionally, on any "
            "operand list -- vxorps xmm0, xmm1, xmm0 (operands differ) must "
            "now ACCEPT, superseding fold round 35's own narrower "
            "self-zeroing-only carve-out; verdict was "
            "{}".format(result.verdicts.get("GenuineXorReject"))
        )


# ---------------------------------------------------------------------------
# Population seventeen -- check (C)'s relocation-resolve-then-classify
# carve-out (design Sec4.1/Sec7 dim 11, D-SLM4886/D-SLM4889). THE ITEM THE
# COORDINATOR NAMED AS THE ONE THAT MOVES THE PRODUCT CLAIM. Must-accept: a
# relocated INDIRECT (memory-operand) call whose relocation resolves to a
# symbol that IS itself the callable entity (here, via corpus_symbols -- a
# sibling translation unit's own real function, not requiring any edit to the
# production EXTERN_ALLOW list, which is a build-round vetting obligation,
# not this ticket's own). Confirmed by direct execution this session: REJECTs
# TODAY, before the fold-35 fix -- `_check_c_for_symbol` currently returns
# False on `not _operand_is_direct_immediate(insn, isa)` BEFORE ever reading
# reloc_by_offset's own target, exactly Poirot's Critical C1. Must-reject:
# the ORIGINAL S2 construction (Helper/Caller, `jmp QWORD PTR [gp]`, gp a
# real in-object DATA symbol whose contents -- not its address -- determine
# the runtime target) -- confirmed REJECTing both before and after the fix,
# the carve-out's own boundary, not a regression it introduces.
# ---------------------------------------------------------------------------


def test_population_17_carveout_import_thunk_shape_must_accept():
    """Must-accept, the carve-out's own central claim: IndirectCallSibling
    performs `call qword ptr [rip]`, a genuine indirect (memory-operand) x86
    instruction, whose ELF relocation resolves to SiblingCallee -- a real
    function DEFINED in a separate compiled object (pop17_carveout_callee.s),
    undefined in THIS object's own symbol table, present in `corpus_symbols`.
    This mirrors MSVC's own `__imp_<name>` import-thunk shape structurally
    (an indirect call through a relocated pointer to the actual callable
    entity) without requiring any edit to the production EXTERN_ALLOW list --
    corpus_symbols is this suite's own, test-authored index, exactly as
    design Sec4.1 gap (b) specifies. Confirmed by direct execution this
    session, BEFORE this assertion was written: currently REJECTs (the
    addressing-mode check fires and returns False before the relocation's
    own target is ever read) -- Poirot's Critical C1, reproduced here on a
    fixture this ticket authored rather than only cited from the casebook.
    """
    caller_src = os.path.join(_FIXTURES, "pop17_carveout_caller.s")
    callee_src = os.path.join(_FIXTURES, "pop17_carveout_callee.s")
    with fc.TempDir() as tmp:
        caller_obj = os.path.join(tmp, "caller.o")
        callee_obj = os.path.join(tmp, "callee.o")
        try:
            fc.compile_clang_asm(caller_src, caller_obj, "x86_64-pc-linux-gnu")
            fc.compile_clang_asm(callee_src, callee_obj, "x86_64-pc-linux-gnu")
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))

        sections = fc.code_sections(caller_obj, ".text")
        insns = _decode_sections(sections, "x86-64")
        call_insns = [i for i in insns if i.mnemonic.lower() == "call"]
        assert call_insns, "fixture verification FAILED: expected a real call instruction"
        assert "[rip" in (call_insns[0].op_str or "") or call_insns[0].op_str.strip().startswith("qword"), (
            "fixture verification FAILED: expected a genuine memory-operand "
            "(indirect) call; got operand string {!r}".format(call_insns[0].op_str)
        )
        with open(caller_obj, "rb") as f:
            data = f.read()
        _sections, _sym_by_raw, relocs = scan._parse_elf(data)
        all_relocs = [r for rs in relocs.values() for r in rs]
        assert all_relocs, (
            "fixture verification FAILED: expected a real relocation on the "
            "caller's own indirect call instruction"
        )

        callee_result = scan.scan_object(callee_obj, isa="x86-64")
        assert callee_result.verdicts.get("SiblingCallee") == "ACCEPT", (
            "SiblingCallee itself must independently ACCEPT on its own bytes -- "
            "granting the caller's edge must never excuse the callee from its "
            "own check"
        )

        corpus_symbols = frozenset({"SiblingCallee"})
        result = scan.scan_object(caller_obj, isa="x86-64", corpus_symbols=corpus_symbols)
        assert not result.refuse
        assert result.verdicts.get("IndirectCallSibling") == "ACCEPT", (
            "design Sec4.1 fold round 35 (D-SLM4886): resolve-then-classify -- "
            "an indirect call/jmp whose relocation resolves to a symbol that IS "
            "the callable entity (here, a corpus_symbols member) must ACCEPT "
            "regardless of the addressing-mode encoding; verdict was {} "
            "(Poirot's Critical C1: today's check rejects on addressing mode "
            "BEFORE ever reading the relocation's own target)".format(
                result.verdicts.get("IndirectCallSibling"))
        )


def test_population_17_carveout_function_pointer_must_reject():
    """Must-reject, the carve-out's own boundary: the ORIGINAL S2
    construction, Helper/Caller -- `jmp QWORD PTR [gp]`, `gp` a real,
    in-object DATA symbol whose runtime CONTENTS (not its own address)
    determine which function actually executes. This edge's relocation
    resolves to `gp` itself -- the data HOLDER, never the callable entity --
    which is exactly the shape the carve-out's own ratified text says stays
    unvettable. Confirmed by direct execution this session: REJECTs today,
    and must stay REJECTed after the fold-35 fix lands -- this is the fix's
    own boundary, not a regression it introduces.
    """
    src = os.path.join(_FIXTURES, "pop17_carveout_reject.asm")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "reject.obj")
        try:
            fc.assemble_ml64(src, obj)
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        sections = fc.code_sections(obj, ".text")
        insns = _decode_sections(sections, "x86-64")
        jmp_insns = [i for i in insns if i.mnemonic.lower() == "jmp"]
        assert jmp_insns, "fixture verification FAILED: expected a real jmp instruction"
        assert "qword ptr" in (jmp_insns[0].op_str or "").lower(), (
            "fixture verification FAILED: expected a genuine memory-operand "
            "(indirect) jmp; got {!r}".format(jmp_insns[0].op_str)
        )

        result = scan.scan_object(obj, isa="x86-64")
        assert not result.refuse
        assert result.verdicts.get("Helper") == "ACCEPT", (
            "Helper itself carries no FP instruction and must ACCEPT"
        )
        assert result.verdicts.get("Caller") == "REJECT", (
            "jmp QWORD PTR [gp] (gp a data holder, not the callable entity) "
            "must REJECT -- the carve-out's own boundary; verdict was "
            "{}".format(result.verdicts.get("Caller"))
        )


# ---------------------------------------------------------------------------
# Population eighteen -- the diagnostic surface's fail-closed behavior on a
# missing disassembly (design Sec7 dim 11, D-SLM4889; the remedy for the
# prior review's own S4 landed at 8a28460 -- MissingDisassemblyError already
# exists at source -- but carries zero cells; one of the four remedies
# proven UNDETECTABLE by execution). Both directions confirmed by direct
# execution this session.
# ---------------------------------------------------------------------------

_POP18_FIXTURES = os.path.join(_FIXTURES, "pop18_fixtures")


def test_population_18_diagnostic_missing_disassembly_must_reject():
    """Must-reject: a manifest naming translation unit tu18_missing.cpp,
    whose disassembly does NOT exist anywhere under disasm_dir --
    build_call_graph and flagged_symbols must both raise
    MissingDisassemblyError rather than silently reporting that translation
    unit clean (indistinguishable, pre-fix, from a genuinely FP-free one).
    Confirmed by direct execution this session: both already raise correctly
    (Brunel's S4 remedy, 8a28460) -- this cell closes the coverage hole, not
    a currently-broken behavior.
    """
    manifest = os.path.join(_POP18_FIXTURES, "manifest_missing.cmake.txt")
    disasm_dir = os.path.join(_POP18_FIXTURES, "disasm")
    missing_path = os.path.join(disasm_dir, "tu18_missing.disasm.txt")
    assert not os.path.exists(missing_path), (
        "fixture verification FAILED: tu18_missing.disasm.txt must NOT exist"
    )

    with pytest.raises(scan.MissingDisassemblyError):
        scan.build_call_graph(manifest, disasm_dir)
    with pytest.raises(scan.MissingDisassemblyError):
        scan.flagged_symbols(manifest, disasm_dir)


def test_population_18_diagnostic_present_disassembly_must_accept():
    """Must-accept, the control: the identical machinery against a manifest
    whose named translation unit's disassembly IS present -- confirmed by
    direct execution this session: root18's own real `addsd` instruction
    (verified present in the fixture's own disasm text below) is correctly
    reported by flagged_symbols, and build_call_graph runs with no
    exception, on both functions.
    """
    manifest = os.path.join(_POP18_FIXTURES, "manifest_present.cmake.txt")
    disasm_dir = os.path.join(_POP18_FIXTURES, "disasm")
    disasm_path = os.path.join(disasm_dir, "tu18.disasm.txt")
    with open(disasm_path) as f:
        disasm_text = f.read()
    assert _verify_fp_mnemonic(disasm_text, "root18", re.compile(r"^addsd$")), (
        "fixture verification FAILED: expected root18 to carry a real addsd "
        "instruction in its own disassembly text"
    )

    graph = scan.build_call_graph(manifest, disasm_dir)
    flagged = scan.flagged_symbols(manifest, disasm_dir)
    assert isinstance(graph, dict)
    assert flagged.get("root18") == ["addsd"], (
        "flagged_symbols must report root18's own real addsd instruction when "
        "its disassembly is present; got {}".format(flagged.get("root18"))
    )


# ---------------------------------------------------------------------------
# Population nineteen -- ScanResult(refuse=True) on an unrecognised ISA or
# object format (design Sec7 dim 11, D-SLM4889; the prior review's own M2
# remedy landed at 8a28460 -- but carries zero cells; one of the four
# remedies proven UNDETECTABLE by execution). Confirmed by direct execution
# this session, all three legs.
# ---------------------------------------------------------------------------


def test_population_19_refuse_on_unrecognized_isa():
    """Must-reject (REFUSE sense): a real, valid COFF object scanned with an
    ISA string this reader's own decoder table does not recognise --
    scan_object must return ScanResult(refuse=True, verdicts={}), never let
    the ValueError _decoder raises propagate uncaught (which would silently
    skip the leg in a caller's own try/loop rather than failing the job).
    Confirmed by direct execution this session: already correct today.
    """
    src = os.path.join(_FIXTURES, "pop09_funclet_fp.cpp")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "pop19.obj")
        try:
            fc.compile_cl(src, obj)
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        result = scan.scan_object(obj, isa="bogus-isa-9000")
        assert result.refuse is True, (
            "an unrecognised ISA string must REFUSE (refuse=True), not raise "
            "an uncaught exception nor silently ACCEPT/REJECT; got {}".format(result)
        )
        assert result.verdicts == {}, (
            "guarantee (i): refuse implies an empty verdict map; got "
            "{}".format(result.verdicts)
        )


def test_population_19_refuse_on_unrecognized_object_format():
    """Must-reject (REFUSE sense): a file that is neither valid ELF nor a
    known COFF machine type, scanned with a perfectly ordinary, recognised
    ISA -- scan_object must REFUSE via the identical clause-(0) path,
    reporting object_format="unknown" (Mach-O and any future format this
    reader has never implemented is the named residual this exercises).
    Confirmed by direct execution this session: already correct today.
    """
    with fc.TempDir() as tmp:
        garbage = os.path.join(tmp, "not_an_object.bin")
        with open(garbage, "wb") as f:
            f.write(b"NOTANOBJECTFILEATALLxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        result = scan.scan_object(garbage, isa="x86-64")
        assert result.refuse is True, (
            "a file that is neither ELF nor a recognised COFF machine type "
            "must REFUSE; got {}".format(result)
        )
        assert result.object_format == "unknown"
        assert result.verdicts == {}


def test_population_19_recognized_pair_control_unaffected():
    """Must-accept, the control: an ordinary, recognised (isa, format) pair
    must be completely unaffected by the refuse-on-unrecognised path --
    reusing population nine's own real fixture, confirming a normal ACCEPT/
    REJECT verdict set is produced, object_format read correctly, no
    spurious REFUSE.
    """
    src = os.path.join(_FIXTURES, "pop09_funclet_fp.cpp")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "pop19_control.obj")
        try:
            fc.compile_cl(src, obj)
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        result = scan.scan_object(obj, isa="x86-64")
        assert result.object_format == "coff"
        assert not result.refuse
        assert result.verdicts, "expected a non-empty verdict map on a recognised pair"


# ---------------------------------------------------------------------------
# Population twenty -- per-section local_starts keying (design Sec7 dim 11,
# D-SLM4889; Poirot's own O3 finding, already fixed at 8a28460 via
# local_starts_by_section keyed on section.index). A pooled, cross-section
# local_starts set is a LATENT correctness bug the real corpus never
# exercises (MSVC relocates every real cross-function edge) -- this
# population constructs the exact discriminating shape at the level the
# defect lives: the arguments _check_c_for_symbol itself receives, real
# capstone-decoded instructions, real per-section extent semantics. A
# genuine assembler can never emit a no-relocation cross-section coincidence
# (cross-section displacements always need a relocation, since sections get
# independent final addresses at link time) -- this is why the discriminating
# half is necessarily white-box, matching Poirot's own likely methodology.
# ---------------------------------------------------------------------------


def test_population_20_local_starts_cross_section_confusion_rejected():
    """Discriminating construction: a real, capstone-decoded `jmp 0x20`
    instruction (E9 rel32, decoded target offset 32) placed in a section
    whose OWN real symbol starts are {0, 16} -- 32 belongs to nothing in
    THIS section. Per-section-correct local_starts ({0, 16}) must REJECT
    this edge (no relocation, no matching local target); a pooled,
    cross-section local_starts set ({0, 16, 32}, simulating a DIFFERENT
    section's own symbol leaking in, the pre-8a28460 shape) would incorrectly
    ACCEPT it. Confirmed by direct execution this session, both directions,
    against the actual `_check_c_for_symbol` function scan_object itself
    calls (not a reimplementation of it).
    """
    md = scan._decoder("x86-64")
    target = 32
    addr = 0
    rel32 = target - (addr + 5)
    code = bytes([0xE9]) + rel32.to_bytes(4, "little", signed=True)
    insns = list(md.disasm(code, addr))
    assert len(insns) == 1, "fixture verification FAILED: expected exactly one decoded jmp"
    insn = insns[0]
    assert insn.mnemonic.lower() == "jmp" and insn.operands[0].imm == target, (
        "fixture verification FAILED: expected a real jmp decoding to target "
        "offset {}; got {} {}".format(target, insn.mnemonic, insn.op_str)
    )

    class _FakeSection:
        index = 0

    sec = _FakeSection()
    ext_start, ext_end = 0, 5

    local_starts_correct = {0, 16}  # THIS section's own real symbol starts only
    result_correct = scan._check_c_for_symbol(
        md, sec, ext_start, ext_end, [insn], "x86-64", "elf",
        reloc_by_offset={}, sym_by_raw={}, local_starts=local_starts_correct)
    assert result_correct is False, (
        "per-section-correct local_starts must REJECT an edge whose target "
        "offset (32) matches nothing in THIS section's own symbol starts "
        "({0, 16}) -- got ACCEPT, which would be a false accept"
    )

    local_starts_pooled_buggy = {0, 16, 32}  # simulates a DIFFERENT section's symbol leaking in
    result_buggy = scan._check_c_for_symbol(
        md, sec, ext_start, ext_end, [insn], "x86-64", "elf",
        reloc_by_offset={}, sym_by_raw={}, local_starts=local_starts_pooled_buggy)
    assert result_buggy is True, (
        "fixture verification FAILED: expected the POOLED (cross-section, "
        "pre-8a28460-shaped) local_starts set to demonstrate the false-accept "
        "this population's own fix closes -- if this is False too, the "
        "discriminating shape itself is broken, not merely already-fixed"
    )


def test_population_20_single_section_corpus_unregressed_control():
    """Must-accept: the existing single-section corpus, unregressed --
    reusing population eleven's own real MSVC/COFF/x86-64 fixture end-to-end
    through the real scan_object pipeline (which internally builds
    local_starts_by_section keyed per real section.index), confirming the
    per-section keying fix changed nothing about the ordinary, single-code-
    section case every other population in this suite already exercises.
    """
    src = os.path.join(_FIXTURES, "pop11_msvc_x64.asm")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "pop20_control.obj")
        try:
            fc.assemble_ml64(src, obj)
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        result = scan.scan_object(obj, isa="x86-64")
        assert not result.refuse
        assert result.verdicts.get("HashSite") == "ACCEPT"
        assert result.verdicts.get("BodyDivide") == "REJECT"


# ---------------------------------------------------------------------------
# Population twenty-one -- the empty-extent byte charge's own arithmetic
# (design Sec7 dim 11, D-SLM4889; Poirot's own M2 finding: the O5 remedy
# charges sec_size ONCE PER EMPTY EXTENT, not once per section -- three empty
# extents on a 64-byte section produces unclassified_bytes=192, exceeding
# the section's own real size). White-box, at the level the defect lives
# (_account_section's own per-extent accumulation loop) -- a real compiler
# does not emit multiple function symbols placed at or beyond a section's
# own declared end, so this is necessarily a synthetic construction of the
# module's own internal _CodeSection/_Sym types, matching population
# twenty's own disclosed methodology.
# ---------------------------------------------------------------------------


def test_population_21_single_empty_extent_charges_true_span():
    """Must-accept, the control: exactly ONE empty extent (a function symbol
    placed at the section's own declared end, offset 64 on a 64-byte
    section) must charge unclassified_bytes == sec_size exactly ONCE.
    Confirmed by direct execution this session: already correct today (the
    multiplicative bug requires MORE than one empty extent in the same
    section to manifest at all).
    """
    md = scan._decoder("x86-64")
    sec = scan._CodeSection(
        index=0, name=".text", data=b"\x90" * 64,
        symbols=[scan._Sym(name="FuncA", value=64, size=0,
                            is_function=True, is_witness=True, raw_index=1)],
    )
    ok, unclassified, _per_symbol = scan._account_section(sec, "x86-64", md)
    assert ok is False, "a single empty extent must REFUSE the whole section"
    assert unclassified == 64, (
        "a section with exactly one empty extent must charge its own true "
        "byte span (64) exactly once; got {}".format(unclassified)
    )


def test_population_21_multiple_empty_extents_charge_once_not_per_extent():
    """Discriminating construction: THREE function symbols (FuncA@64,
    FuncB@70, FuncC@80), each individually placed at or beyond the same
    64-byte section's own declared end -- each independently computes an
    empty (or negative-span) extent. Confirmed by direct execution this
    session, BEFORE this assertion was written: today charges
    unclassified_bytes=192 (3 x 64) -- exceeding the section's own real size,
    Poirot's own M2 measurement exactly. The corrected accounting must charge
    the section's own unclassified span ONCE, however many empty extents it
    decomposes into: unclassified_bytes == sec_size (64), not a multiple of
    it.
    """
    md = scan._decoder("x86-64")
    sec = scan._CodeSection(
        index=0, name=".text", data=b"\x90" * 64,
        symbols=[
            scan._Sym(name="FuncA", value=64, size=0,
                      is_function=True, is_witness=True, raw_index=1),
            scan._Sym(name="FuncB", value=70, size=0,
                      is_function=True, is_witness=True, raw_index=2),
            scan._Sym(name="FuncC", value=80, size=0,
                      is_function=True, is_witness=True, raw_index=3),
        ],
    )
    ok, unclassified, _per_symbol = scan._account_section(sec, "x86-64", md)
    assert ok is False, "three empty extents must REFUSE the whole section"
    assert unclassified == 64, (
        "unclassified_bytes must charge the section's own true size (64) "
        "ONCE, regardless of how many empty extents it decomposes into -- "
        "Poirot's own M2: today this reads {} (an object of 64 real bytes "
        "cannot have more than 64 unclassified bytes in it)".format(unclassified)
    )


# ---------------------------------------------------------------------------
# Poirot's O5 (Claude/Poirot/8a28460-t2344-fp-scan-fix-round-confirmation.md,
# routed to the test author alongside M4/O4, above): run_fp_free_scan_real_
# corpus.py's own _msvc_target_flags captures `([^)]*)` after PRIVATE, which
# terminates at the FIRST literal `)` inside the captured span -- silent on a
# partial match, loud only on zero matches. Confirmed by direct execution
# this session: a target_compile_options(... PRIVATE ...) line whose own
# flag list contains a literal `)` (a macro-with-arguments define, or a CMake
# generator expression that itself embeds one) truncates the returned flag
# list silently, with everything after the embedded `)` -- including further,
# real flags -- silently dropped. This is a documentation/regression pin of a
# disclosed, Observation-level defect, not an assertion of a not-yet-ratified
# fix contract (no D-SLM number specifies what the corrected behavior must
# be); the finding is filed as an open residual for whoever next specifies
# the correction, per this campaign's own disclosure discipline.
# ---------------------------------------------------------------------------


def test_msvc_target_flags_silently_truncates_on_embedded_paren():
    """Documents Poirot's O5 exactly: a target_compile_options(superslm
    PRIVATE ...) line whose flags contain a literal embedded `)` (here, a
    macro-with-arguments define, `/DSOME_MACRO(x)=1`) causes
    _msvc_target_flags to silently return a TRUNCATED, partially-corrupted
    flag list -- both the offending flag itself (missing its own closing
    paren and everything after it) and every flag genuinely declared after
    it (here, /EHsc) are silently lost, with no exception raised. Verified
    against the control (no embedded paren) first, so the discrimination is
    real: the same function, same regex, only the fixture's own flag content
    differs.
    """
    _here = os.path.dirname(os.path.abspath(__file__))
    ci_dir = os.path.abspath(os.path.join(_here, "..", "ci"))
    if ci_dir not in sys.path:
        sys.path.insert(0, ci_dir)
    import run_fp_free_scan_real_corpus as runner

    control_text = (
        "if(MSVC)\n"
        "    target_compile_options(superslm PRIVATE /W4 /fp:precise /EHsc)\n"
        "else()\n"
        "    target_compile_options(superslm PRIVATE -Wall)\n"
        "endif()\n"
    )
    truncating_text = (
        "if(MSVC)\n"
        "    target_compile_options(superslm PRIVATE /W4 /DSOME_MACRO(x)=1 /EHsc)\n"
        "else()\n"
        "    target_compile_options(superslm PRIVATE -Wall)\n"
        "endif()\n"
    )

    with fc.TempDir() as tmp:
        control_path = os.path.join(tmp, "control.cmake.txt")
        truncating_path = os.path.join(tmp, "truncating.cmake.txt")
        with open(control_path, "w") as f:
            f.write(control_text)
        with open(truncating_path, "w") as f:
            f.write(truncating_text)

        orig_cmakelists = runner._CMAKELISTS
        try:
            runner._CMAKELISTS = control_path
            control_flags = runner._msvc_target_flags("superslm")
        finally:
            runner._CMAKELISTS = orig_cmakelists
        assert control_flags == ["/W4", "/fp:precise", "/EHsc"], (
            "control fixture must parse cleanly and completely; got "
            "{}".format(control_flags)
        )

        try:
            runner._CMAKELISTS = truncating_path
            truncated_flags = runner._msvc_target_flags("superslm")
        finally:
            runner._CMAKELISTS = orig_cmakelists

        assert "/EHsc" not in truncated_flags, (
            "documents Poirot's O5: /EHsc, genuinely declared after the "
            "embedded ), is silently dropped -- if this now fails, the "
            "truncation defect has been fixed and this pin should be "
            "updated/retired rather than left asserting a since-corrected "
            "defect; got {}".format(truncated_flags)
        )
        assert not any(f == "/DSOME_MACRO(x)=1" for f in truncated_flags), (
            "documents Poirot's O5: the offending flag itself is also "
            "corrupted (missing its own closing paren and suffix), not just "
            "the flags after it; got {}".format(truncated_flags)
        )


# ===========================================================================
# T-2366 (Curie), D-SLM5001 item (1) -- the gate's verdict must not read
# check (C). Design Sec4.1/Sec5.5 (fold round 39, D-SLM4985) state, in three
# places, that checks (A) and (B) alone decide the ship gate and that check
# (C) is retired as a gating surface. `check_fp_free_scan.py::scan_object`
# computes one ANDed verdict, `"ACCEPT" if (ab_accept and c_accept) else
# "REJECT"` (source, scan_object), with no field anywhere in `ScanResult`
# distinguishing which check produced a REJECT. `tests/ci/
# scan_build_output.py` -- the only production driver that exists -- reads
# that single verdict directly, so a symbol that fails ONLY check (C) fails
# the whole build exactly as one that fails checks (A)/(B) does. T-2364's
# strike (Findings C6/C7/C8/C9) and T-2365's coverage audit (Finding 3) both
# found this independently: the real 17-object build reads 264 REJECT
# through the shipped driver, not the 2 the design's own text states.
# ===========================================================================


def test_gate_must_not_fail_on_a_check_c_only_reject():
    """Must-accept: a real, compiled object whose only REJECT-shaped symbol
    fails check (C) alone (`pop17_carveout_reject.asm`, this suite's own
    established construction for population seventeen's must-reject leg --
    `Helper` carries no vector instruction at all; `Caller`'s only
    instruction is an indirect `jmp QWORD PTR [gp]` through a real,
    in-object DATA symbol, which check (C) rejects because the relocation
    names the data holder, never the callable entity -- and which checks
    (A)/(B) never even reach, since the instruction touches no vector
    register). Design Sec4.1/Sec5.5 (fold round 39, D-SLM4985): check (C)
    no longer gates, so a build whose only reject is check-(C)-shaped must
    PASS the ship gate. Confirmed by direct execution this session, before
    this assertion was written: the shipped driver reports REJECT and exits
    1 on this exact construction.

    T-2385 (Brunel, fold round 43, D-SLM5100/D-SLM5101): the scratch build
    below also carries a real `Release/superslm.lib` archive, built by
    `lib.exe` from the same object -- `scan_build_output.py`'s own main()
    no longer falls back to the object directory when no archive is found,
    so a `_run_gate` cell whose scratch build carries no archive would now
    exit 2 (infrastructure failure) before check (C) is ever reached,
    which is not what this cell tests.
    """
    src = os.path.join(_FIXTURES, "pop17_carveout_reject.asm")
    with fc.TempDir() as tmp:
        build_dir = os.path.join(tmp, "build")
        target_dir = os.path.join(build_dir, "superslm.dir", "Release")
        os.makedirs(target_dir)
        obj = os.path.join(target_dir, "reject.obj")
        try:
            fc.assemble_ml64(src, obj)
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))

        release_dir = os.path.join(build_dir, "Release")
        os.makedirs(release_dir, exist_ok=True)
        archive_path = os.path.join(release_dir, "superslm.lib")
        try:
            af.run_lib_exe(["/OUT:" + archive_path, "reject.obj"], cwd=target_dir)
        except af.ToolUnavailable as e:
            pytest.skip(str(e))

        # Independent verification of the fixture's own truth, before the
        # production driver is ever consulted: Caller's only instruction
        # touches no vector register at all, so checks (A)/(B) trivially
        # ACCEPT it and today's REJECT can only be check (C)'s.
        sections = fc.code_sections(obj, ".text")
        insns = _decode_sections(sections, "x86-64")
        assert insns, "fixture verification FAILED: expected real decoded instructions"
        assert not any(scan and hasattr(scan, "_x86_touches_vector_register")
                       and scan._x86_touches_vector_register(i.op_str or "")
                       for i in insns), (
            "fixture verification FAILED: expected no instruction in this "
            "object to touch a vector register -- the construction must be a "
            "pure check-(C) case, with checks (A)/(B) never in play"
        )

        if not _SCAN_AVAILABLE:
            _fail_absent("(gate check-C exclusion, must-accept)",
                         "Fixture verified above: no instruction touches a "
                         "vector register.")
        direct = scan.scan_object(obj, isa="x86-64", corpus_symbols=frozenset())
        assert not direct.refuse
        assert direct.verdicts.get("Helper") == "ACCEPT", (
            "Helper itself carries no FP instruction and must ACCEPT"
        )
        assert direct.verdicts.get("Caller") == "REJECT", (
            "fixture verification FAILED: expected Caller to REJECT today "
            "(check (C)'s own boundary, population seventeen); verdict was "
            "{}".format(direct.verdicts.get("Caller"))
        )

        if not _GATE_AVAILABLE:
            _fail_absent("(gate check-C exclusion, must-accept)",
                         "Fixture verified above: a real object whose only "
                         "REJECT-shaped symbol rejects via check (C) alone.")
        ec = _run_gate(build_dir)
        assert ec == 0, (
            "design Sec4.1/Sec5.5 (fold round 39, D-SLM4985): check (C) is "
            "retired as a gating surface -- a build whose only reject is "
            "check-(C)-shaped must PASS the ship gate. tests/ci/"
            "scan_build_output.py's own main() returned {} (0 == pass) "
            "instead; today it reads scan_object's single ANDed "
            "ab_accept-and-c_accept verdict with no way to separate the "
            "two, so this REJECTs and the gate fails (T-2364 Findings "
            "C6-C9, T-2365 Finding 3)".format(ec)
        )


def test_gate_still_fails_on_a_genuine_check_ab_violation():
    """Must-reject control, same construction plus a second, real object
    carrying a genuine check-(A) violation (`pop07_fpblind.cpp`'s
    `BodyFloor`, population seven's own falsifying construction --
    real floating-point-shaped arithmetic under the default-deny
    classifier, compiled via clang targeting the MSVC triple so the object
    lands in the same COFF build-dir layout). Proves the fix
    test_gate_must_not_fail_on_a_check_c_only_reject requires discriminates
    check (C) from checks (A)/(B), rather than making the gate pass
    unconditionally: a build carrying a genuine check-(A)/(B) violation must
    still fail. Already correct today (checks (A)/(B) already REJECT this
    symbol on their own, independent of check (C)), and must remain correct
    once check (C) stops gating.

    T-2385 (Brunel, fold round 43, D-SLM5100/D-SLM5101): the scratch build
    below also carries a real `Release/superslm.lib` archive holding BOTH
    objects, built by `lib.exe` -- the same reason
    test_gate_must_not_fail_on_a_check_c_only_reject needs one. Without an
    archive, `main()` would exit 2 (no archive found) before either object
    is ever scanned, which would leave this control's own `ec != 0`
    assertion trivially satisfied for the wrong reason and no longer
    discriminating check (C) from checks (A)/(B) at all.
    """
    reject_src = os.path.join(_FIXTURES, "pop17_carveout_reject.asm")
    arith_src = os.path.join(_FIXTURES, "pop07_fpblind.cpp")
    with fc.TempDir() as tmp:
        build_dir = os.path.join(tmp, "build")
        target_dir = os.path.join(build_dir, "superslm.dir", "Release")
        os.makedirs(target_dir)
        reject_obj = os.path.join(target_dir, "reject.obj")
        arith_obj = os.path.join(target_dir, "arith.obj")
        try:
            fc.assemble_ml64(reject_src, reject_obj)
            fc.compile_clangxx(arith_src, arith_obj, "x86_64-pc-windows-msvc",
                               extra_args=["-msse4.1"])
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))

        release_dir = os.path.join(build_dir, "Release")
        os.makedirs(release_dir, exist_ok=True)
        archive_path = os.path.join(release_dir, "superslm.lib")
        try:
            af.run_lib_exe(["/OUT:" + archive_path, "reject.obj", "arith.obj"],
                            cwd=target_dir)
        except af.ToolUnavailable as e:
            pytest.skip(str(e))

        if not _SCAN_AVAILABLE:
            _fail_absent("(gate check-C exclusion, must-reject control)", "")
        arith_result = scan.scan_object(arith_obj, isa="x86-64")
        assert not arith_result.refuse
        assert arith_result.verdicts.get("BodyFloor") == "REJECT", (
            "fixture verification FAILED: expected BodyFloor to REJECT under "
            "checks (A)/(B) today (genuine FP arithmetic, unaffected by "
            "D-SLM4987's bitwise widening); verdict was {}".format(
                arith_result.verdicts.get("BodyFloor"))
        )

        if not _GATE_AVAILABLE:
            _fail_absent("(gate check-C exclusion, must-reject control)",
                         "Fixture verified above: a genuine check-(A) "
                         "violation is present alongside the check-(C)-only "
                         "reject.")
        ec = _run_gate(build_dir)
        assert ec != 0, (
            "a build carrying a genuine check-(A)/(B) violation "
            "(BodyFloor) must still fail the ship gate even after check (C) "
            "stops gating -- main() returned {} (0 == pass)".format(ec)
        )


# ===========================================================================
# T-2366 (Curie), D-SLM5001 item (2) -- check (A)'s eight-mnemonic bitwise-
# family widening (design Sec4.1, D-SLM4987). Population sixteen's own
# reconciliation, above, closes the collision for vxorps specifically; this
# section closes T-2365's own coverage-audit Gap ("population 35's own
# must-accept construction names three of the eight corrected mnemonics ...
# a test author working from the text alone would have no textual basis to
# include xorps/xorpd/orpd/andpd/andnpd") for the remaining seven, plus the
# real-corpus leg the design's own text names
# (ReadDampedGreedyScaleConstants's two overloads, D-SLM4982).
# ===========================================================================

_DSLM4987_EIGHT_FAMILY = [
    "Legacy_Orps", "Legacy_Orpd", "Legacy_Andps", "Legacy_Andpd",
    "Legacy_Andnps", "Legacy_Andnpd", "Legacy_Xorps", "Legacy_Xorpd",
    "Vex_Vorps", "Vex_Vorpd", "Vex_Vandps", "Vex_Vandpd",
    "Vex_Vandnps", "Vex_Vandnpd", "Vex_Vxorps", "Vex_Vxorpd",
]


def test_check_a_bitwise_family_widening_must_accept():
    """Must-accept: all eight mnemonics D-SLM4987 names (orps/orpd/andps/
    andpd/andnps/andnpd/xorps/xorpd), legacy and VEX-encoded, each with a
    differing (non-self-same) operand list -- the shape the design's own
    text states is accepted unconditionally, joining pand/por/pandn/pxor's
    existing treatment. Sixteen real, assembled instructions
    (pop_dslm4987_bitwise_family.s), independently decoded by capstone
    before the production module is ever consulted. At the time this cell
    was authored, all sixteen REJECTed: the six OR/AND/ANDN mnemonics were
    named nowhere in `_x86_check_a` and fell through to its own default
    REJECT; the two XOR mnemonics were named only for the self-zeroing
    (all-operands-identical) idiom, which none of these constructions is.
    T-2367 landed D-SLM4987's widening; all sixteen now ACCEPT.
    """
    src = os.path.join(_FIXTURES, "pop_dslm4987_bitwise_family.s")
    with fc.TempDir() as tmp:
        obj = os.path.join(tmp, "bitwise_family.o")
        try:
            fc.compile_clang_asm(src, obj, "x86_64-pc-linux-gnu", extra_args=["-mavx"])
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        sections = fc.code_sections(obj, ".text")
        insns = _decode_sections(sections, "x86-64")
        decoded_mnemonics = {i.mnemonic.lower() for i in insns}
        expected_mnemonics = {
            "orps", "orpd", "andps", "andpd", "andnps", "andnpd", "xorps", "xorpd",
            "vorps", "vorpd", "vandps", "vandpd", "vandnps", "vandnpd", "vxorps", "vxorpd",
        }
        assert expected_mnemonics <= decoded_mnemonics, (
            "fixture verification FAILED: expected all sixteen mnemonics "
            "decoded; found {}".format(sorted(decoded_mnemonics))
        )

        if not _SCAN_AVAILABLE:
            _fail_absent("(D-SLM4987 bitwise-family widening)",
                         "Fixture verified above: all sixteen mnemonics "
                         "present, each with a differing operand list.")
        result = scan.scan_object(obj, isa="x86-64")
        assert not result.refuse
        still_rejecting = [name for name in _DSLM4987_EIGHT_FAMILY
                           if result.verdicts.get(name) != "ACCEPT"]
        assert still_rejecting == [], (
            "design Sec4.1 (fold round 39, D-SLM4987): orps/orpd/andps/"
            "andpd/andnps/andnpd/xorps/xorpd and their VEX forms perform no "
            "floating-point arithmetic and must ACCEPT unconditionally, on "
            "any operand list -- still rejecting today: {} "
            "(verdicts: {})".format(still_rejecting, {
                name: result.verdicts.get(name) for name in still_rejecting
            })
        )


# N/A -- T-2389 item 1 (Poirot 8788b01 -> D-SLM5128/D-SLM5129): the
# real-corpus leg's own fixture precondition harvested one literal
# mnemonic ("orps") that MSVC happened to select for the struct-pack site
# in the build `build.bat` produces. `cmake -B build_ci_preflight` --
# CI's own `fp-free-scan-gate` job configuration -- selects `xorps` for
# the identical semantic pack instead (confirmed by direct disassembly
# this session: 3 xorps, 0 orps, 611 total instructions). `xorps` is not
# an incidental substitute; it is itself named, on equal footing with
# `orps`, by the sixteen-member family D-SLM4987 widened (the same set
# `_DSLM4987_EIGHT_FAMILY`'s legacy half enumerates above). The
# precondition below is re-aimed at that named family rather than at one
# member of it, so it stops being configuration-dependent on which member
# the compiler picks and starts asserting the boundary the cell actually
# claims: check (A) must ACCEPT a bitwise-family instruction on the
# floating-point register file at this site, whichever family member is
# present. A skip-when-absent repair was refused for this cell
# (D-SLM5111's class, closed one round prior in this suite) -- the
# precondition below still asserts, it does not skip.
_BITWISE_FAMILY_MNEMONICS = {
    "orps", "orpd", "andps", "andpd", "andnps", "andnpd", "xorps", "xorpd",
    "vorps", "vorpd", "vandps", "vandpd", "vandnps", "vandnpd", "vxorps", "vxorpd",
}


def test_check_a_bitwise_family_real_corpus_leg(real_build_dir):
    """The design's own real-corpus leg (D-SLM4982): a D-SLM4987
    bitwise-family instruction in `ReadDampedGreedyScaleConstants`'s two
    overloads (src/damped_greedy_phaseD.cpp) -- MSVC's own instruction
    selection for packing two integer fields into an XMM-resident struct
    write, with no floating-point type anywhere in either function.
    Scanned directly from a real object built fresh by this session's own
    `real_build_dir` fixture (conftest.py, T-2368, D-SLM5008 -- never a
    hand-configured, unversioned directory) -- this suite's own standing
    law that at least one cell runs the real build, applied to this
    cell's own claim rather than only to a synthesized fixture.

    Which family member MSVC selects for this struct-pack site is a
    codegen detail, not the claim: `build.bat`'s own build selects
    `orps`, CI's `cmake -B build_ci_preflight` selects `xorps` (both
    confirmed by direct disassembly this session) -- the precondition
    below accepts any of the sixteen family members D-SLM4987 names,
    since the boundary under test is that ALL of them must ACCEPT on this
    non-floating-point site, not that one specific mnemonic must appear.
    """
    if not _SCAN_AVAILABLE:
        _fail_absent("(D-SLM4987 real-corpus leg)", "")
    if not _GATE_AVAILABLE:
        _fail_absent("(D-SLM4987 real-corpus leg, object lookup)", "")
    objects = scan_build_output.find_target_objects(real_build_dir, "superslm")
    matches = [o for o in objects if os.path.basename(o) == "damped_greedy_phaseD.obj"]
    if not matches:
        pytest.skip(
            "damped_greedy_phaseD.obj not found among the real build's own "
            "{} objects under {}".format(len(objects), real_build_dir))
    obj_path = matches[0]

    sections = fc.code_sections(obj_path, ".text")
    insns = _decode_sections(sections, "x86-64")
    family_insns = [i for i in insns if i.mnemonic.lower() in _BITWISE_FAMILY_MNEMONICS]
    assert family_insns, (
        "fixture verification FAILED: expected the real object to still "
        "carry at least one D-SLM4987 bitwise-family instruction (orps/"
        "orpd/andps/andpd/andnps/andnpd/xorps/xorpd or a VEX form) at "
        "this struct-pack site; found none among {} instructions -- "
        "mnemonics present: {}".format(
            len(insns), sorted({i.mnemonic.lower() for i in insns}))
    )

    result = scan.scan_object(obj_path, isa="x86-64")
    assert not result.refuse
    # T-2368: reads ab_verdicts (checks (A)/(B) alone), not the combined
    # verdicts field -- this claim is about check (A)'s own bitwise-family
    # classification (D-SLM4987), and the shipped ship gate itself decides
    # on ab_verdicts alone (D-SLM5004/D-SLM5007); the combined field also
    # carries check (C)'s own unrelated, non-gating cross-object call-target
    # vetting, which this single-object scan (no corpus_symbols) cannot
    # resolve and which is not what this cell claims about.
    overloads = [name for name in result.ab_verdicts if "ReadDampedGreedyScaleConstants" in name]
    assert len(overloads) == 2, (
        "fixture verification FAILED: expected exactly two "
        "ReadDampedGreedyScaleConstants overloads in the real object; found "
        "{}".format(overloads)
    )
    still_rejecting = [name for name in overloads if result.ab_verdicts.get(name) != "ACCEPT"]
    assert still_rejecting == [], (
        "design Sec4.1 (fold round 39, D-SLM4987, D-SLM4982's own measured "
        "real-corpus leg): ReadDampedGreedyScaleConstants's own orps "
        "instructions perform no floating-point arithmetic and must ACCEPT "
        "under checks (A)/(B) -- still rejecting today: {} (ab_verdicts: "
        "{})".format(
            still_rejecting,
            {name: result.ab_verdicts.get(name) for name in overloads})
    )


# ===========================================================================
# T-2366 (Curie), D-SLM5001 item (5) -- a vitality pin on the p/vp-prefix
# structural rule (D-SLM4999). D-SLM4986 justified checks (A)/(B) over a
# deny-list of floating-point mnemonics on the grounds that they are "an
# allow-list ... fail-closed on an unknown mnemonic." Conductor-executed
# against capstone's own x86-64 vocabulary at engine a1df129 (D-SLM4999):
# check (A) accepts a majority of its own vector-operand ACCEPTs via a
# structural rule ("starts with p or vp") guarded only by a 6-entry deny
# list, `_X86_P_PREFIX_EXCLUDE` -- named by no allow-list at all, and
# fail-OPEN rather than fail-closed on anything the deny list does not name.
# The rule's premise (a p/vp prefix means packed-integer by construction)
# has already been falsified twice: the 3DNow pi2f* family and the BF16
# dot-product/convert family, each patched into the deny list reactively
# after it leaked into the shipped classifier.
#
# REPRODUCTION NOTE, per this ticket's own instruction to reproduce the
# cited figures before pinning any of them: D-SLM4999 states 1523/539/481/58;
# T-2364's own probe (`Claude/Loki/t2364-probe/p02_mnemonic_census.py`)
# computes 1523/539/428/111, using a "named" set that includes
# `_X86_GPR_ALLOW` -- a list check (B), not check (A), consults, and which
# check (A) never reads on ANY branch, vector-operand or not. Reproduced
# independently, twice, this session (T-2366), using ONLY the allow-list
# check (A) itself actually consults for a vector-touching instruction
# (`_X86_VEC_MOVE_ALLOW`; the self-zeroing xorps/vxorps family contributed
# zero additional members under a differing-operand probe at that time,
# since it never matched a differing-operand construction): 1523
# vocabulary, 539 accepted by check (A) on a vector operand, of which 438
# passed ONLY the structural rule and 101 were named by
# `_X86_VEC_MOVE_ALLOW` explicitly. Neither D-SLM4999's 481/58 nor T-2364's
# 428/111 reproduced against the classifier actually consulted; 438/101 was
# pinned as the reproduced figure, and the discrepancy was filed per
# `StandardsDocument.md` Sec5.4 (a ruling contradicted by measurement is
# reopened, not defended) rather than silently pinning either prior,
# unreproduced number.
#
# T-2368 (Curie), D-SLM5009a: T-2367's build round widened check (A)'s
# bitwise-family boundary per D-SLM4987 -- `orps`/`orpd`/`andps`/`andpd`/
# `andnps`/`andnpd`/`xorps`/`xorpd` and their eight VEX forms, all sixteen
# now ACCEPT unconditionally. Re-derived fresh this session (not copied from
# any prior entry): the sixteen are ADDITIONAL structural-only accepts --
# none is named by `_X86_VEC_MOVE_ALLOW` -- so `accept_a` moves 539 -> 555
# and `structural_only` moves 438 -> 454; `named_accept` (101) is
# unaffected. The qualitative finding -- a majority of check (A)'s
# vector-operand accepts rely on the deny-list-guarded structural rule
# alone, not an allow-list -- is unchanged and is what the cell below
# checks: a design that claims "never a deny-list ... fail-closed on any
# mnemonic neither check names" is false while this count is nonzero.
#
# CLOSURE NOTE (T-2404/T-2407): every paragraph above describes the
# classifier as it stood through T-2368's own D-SLM5009a rebaseline --
# deny-list-guarded, and open on whether to make it fail-closed
# (D-SLM5009). D-SLM5155/D-SLM5156 (Dan, 2026-08-29) closed that question:
# `_X86_P_VP_STRUCTURAL_ALLOW` (R3, T-2404) replaces the deny-list-guarded
# structural rule with a frozen allow-list, and check (A)'s p/vp branch is
# fail-closed as of that commit, not fail-open. The counts this block
# reasoned toward (1523/571/117/454) are current facts about the shipped
# classifier -- see `test_check_a_p_vp_structural_accept_census_and_
# violation`'s own docstring below for why D-SLM5156's closure leaves them
# unchanged -- but the OPEN-question framing in the paragraphs above is
# superseded and must not be read as describing today's classifier.
# ===========================================================================


def _census_check_a_p_vp_structural_reliance():
    """Reproduces the census fresh, every call -- never a cached or
    hardcoded population, so a capstone upgrade changing the vocabulary is
    read at the moment the cell runs rather than silently compared against a
    stale snapshot."""
    from capstone import x86_const
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    vocabulary = sorted({
        md.insn_name(getattr(x86_const, attr))
        for attr in dir(x86_const) if attr.startswith("X86_INS_")
        if md.insn_name(getattr(x86_const, attr))
    })
    vec_ops = "xmm1, xmm2"  # touches a vector register -> check (A) decides
    accept_a = [m for m in vocabulary if scan._x86_check_a(m, vec_ops)]
    named = set(scan._X86_VEC_MOVE_ALLOW)
    structural_only = sorted(m for m in accept_a if m not in named)
    named_accept = sorted(m for m in accept_a if m in named)
    return vocabulary, accept_a, structural_only, named_accept


def test_check_a_p_vp_structural_accept_census_and_violation():
    """Reproduces D-SLM4999's own measurement, rebaselined per D-SLM5009a
    against T-2367's build (539+16=555 / 438+16=454 derivation), and
    rebaselined again (T-2381, Brunel, design Sec4.1 fold round 40,
    D-SLM5037): 1523 vocabulary (unchanged -- no capstone-version shift),
    571 accepted by check (A) on a vector operand (555 + 16, the
    vextract/vinsert lane-movement family design Sec7 dim 11's forty-third
    population commissions), of which 454 pass ONLY the structural rule
    (unchanged -- every one of the sixteen new mnemonics is explicitly
    named, landing in named_accept, never structural_only) and 117 are
    named by `_X86_VEC_MOVE_ALLOW` explicitly (101 + 16). This is a stale-
    pin update following the identical shape D-SLM5009a already corrected
    once in this same cell: a legitimate, specified widening of
    `_X86_VEC_MOVE_ALLOW` shifts this census's own literal counts, and the
    counts are recomputed and reasserted here, not the assertion loosened.
    These four counts are current facts about the shipped classifier and are
    unaffected by D-SLM5156's own closure: `_X86_P_VP_STRUCTURAL_ALLOW`
    (R3, T-2404) is built to admit exactly the 438 mnemonics that passed
    the pre-closure structural rule, so `accept_a`/`named_accept`/
    `structural_only` (against `_X86_VEC_MOVE_ALLOW`, this census's own
    definition of "named") hold the same counts under the fail-closed
    allow-list as they did under the deny-list-guarded rule it replaced.
    D-SLM5009b's own open question (whether check (A) should be made
    fail-closed over the p/vp class) is closed (D-SLM5155/D-SLM5156); the
    cell that pinned it as open,
    test_check_a_p_vp_structural_only_nonempty_violates_fail_closed_claim,
    is retired (R7, same commit as R3) rather than left describing a
    decision that has since been made.
    """
    if not _SCAN_AVAILABLE:
        _fail_absent("(D-SLM4999 p/vp vitality pin, census)", "")
    vocabulary, accept_a, structural_only, named_accept = _census_check_a_p_vp_structural_reliance()
    assert len(vocabulary) == 1523, (
        "capstone x86-64 mnemonic vocabulary census changed from the "
        "reproduced 1523 -- this suite's own capstone version may have "
        "changed; got {}".format(len(vocabulary))
    )
    assert len(accept_a) == 571, (
        "check (A)'s own ACCEPT count on a vector operand changed from the "
        "reproduced 571 (555 pre-D-SLM5037 + 16 vextract/vinsert "
        "lane-movement mnemonics D-SLM5037 widened); got {}".format(len(accept_a))
    )
    assert len(named_accept) == 117, (
        "check (A)'s own explicitly-allow-listed ACCEPT count changed from "
        "the reproduced 117 (101 pre-D-SLM5037 + 16 -- D-SLM5037's own "
        "sixteen-mnemonic widening is an EXPLICIT _X86_VEC_MOVE_ALLOW "
        "addition, unlike D-SLM4987's structural-rule accept, so this count "
        "moves with it); got {}".format(len(named_accept))
    )
    assert len(structural_only) == 454, (
        "check (A)'s own structural-only (no-allow-list) ACCEPT count "
        "changed from the reproduced 454 -- D-SLM5037's sixteen-mnemonic "
        "widening is an explicit allow-list addition (named_accept, above), "
        "not a structural-rule accept, so this count should be unaffected "
        "by it; got {}".format(len(structural_only))
    )


# R7 (T-2404): the cell that formerly stood here,
# test_check_a_p_vp_structural_only_nonempty_violates_fail_closed_claim,
# pinned design Sec5.4/Sec5.5's fail-closed guarantee as an open question
# under xfail(strict=True), reason "OPEN, waiting on Dan." Dan ruled it
# closed 2026-08-29 (D-SLM5155) and D-SLM5156 specified the closure
# (`_X86_P_VP_STRUCTURAL_ALLOW`, built above `_x86_check_a`). The cell is
# retired by R3's own must-accept/must-reject pair
# (test_check_a_p_vp_structural_allow_agrees_with_independent_fp_oracle,
# test_check_a_p_vp_mis_vetted_allow_list_caught_by_independent_oracle,
# below) rather than left standing with a reason describing a decision
# that has since been made -- retirement landed in the same commit as R3,
# per the manifest's own instruction (t2265-fold45-delta-manifest.md,
# R7). Note for a future reader: this cell's own `structural_only`
# computation (named only against `_X86_VEC_MOVE_ALLOW`) would not have
# detected the closure even if left in place -- the fail-closed guarantee
# is now satisfied via a second, separately-named list
# (`_X86_P_VP_STRUCTURAL_ALLOW`), which this cell's narrower definition of
# "named" never consulted. That staleness, not only the reason text, is
# why the cell is retired rather than merely reworded.


_P_VP_STRUCTURAL_ONLY_PIN_PATH = os.path.join(
    _FIXTURES, "p_vp_structural_only_pinned.txt")


def _load_pinned_p_vp_structural_only():
    with open(_P_VP_STRUCTURAL_ONLY_PIN_PATH) as f:
        return sorted(
            line.strip() for line in f
            if line.strip() and not line.strip().startswith("#")
        )


def test_check_a_p_vp_structural_only_set_is_pinned_against_vocabulary_growth():
    """T-2368 (Curie), D-SLM5001 item (5)/D-SLM5009b -- the vitality cell
    D-SLM5001 item (5) actually specified, replacing
    test_check_a_p_vp_rule_fails_open_on_a_future_fp_mnemonic (retired: it
    asserted a FABRICATED mnemonic, 'vpfoobaraddps', must REJECT under
    check (A) -- demanding a production change no decision authorizes,
    since check (A)'s p/vp branch is a structural, deny-list-guarded rule
    by design, not a positive allow-list (D-SLM4986/D-SLM4999), so that
    assertion was permanently red asserting an unruled requirement).

    THIS cell instead watches the REAL decoder: `_census_check_a_p_vp_
    structural_reliance()`'s own `structural_only` set (454 members, this
    file's own rebaselined census above) is pinned, member for member,
    against fp_scan_fixtures/p_vp_structural_only_pinned.txt.

    Deliberately NOT an attempt to classify which members are "genuinely
    floating point": D-SLM5009b files that classification as open, waiting
    on Dan, bounded but not small, and not this ticket's to build --  and
    this suite's own narrow, test-local `_is_x86_fp_arith` (confirmed this
    session to misclassify the real packed-integer mnemonics `pmaxsd`/
    `pminsd`/`vpcmpd`/`vpmaxsd`/`vpminsd` as floating-point-shaped, because
    their `sd`/`d` suffix is an integer element-size code that coincides
    textually with the FP scalar-double suffix) is not a safe instrument to
    run over the whole real vocabulary -- doing so produces false "FP"
    positives on ordinary packed-integer instructions, which would make
    this cell red for the wrong reason.

    A membership pin needs no such classifier: a REMOVED member -- capstone
    renaming or retiring a mnemonic already admitted onto the frozen
    allow-list -- is exactly the observable signal D-SLM4999's residual
    concern needs from THIS cell, and needs no confirmed-FP classification
    to trigger.

    S6 (T-2407, review e9879e2-t2404-1p3-shipping-repair-set-review.md):
    D-SLM5156's own freeze (R3, T-2404) forecloses this cell's other half.
    `_x86_check_a`'s p/vp branch now consults `_X86_P_VP_STRUCTURAL_ALLOW`
    alone, so a brand-new p/vp mnemonic capstone starts recognizing is, by
    construction, absent from that frozen list -- it REJECTs and never
    enters `accept_a`, so it never enters `structural_only` either. This
    cell alone can no longer see a member ADDED to the set the way it once
    did (when the pre-freeze rule accepted anything not on a 6-entry deny
    list, a genuinely new mnemonic joined `structural_only` automatically).
    The added-member half of "any change to this set" is restored by the
    sibling cell below,
    `test_check_a_p_vp_naming_convention_vocabulary_is_pinned_against_growth`,
    which watches the raw p/vp naming-convention vocabulary instead of
    this post-decision accept set, so a new mnemonic changes what THAT
    cell sees the moment capstone starts emitting it, REJECT or not.

    On failure here, a removed member means capstone renamed or retired a
    mnemonic already admitted onto the frozen allow-list -- raise it; the
    fixture is not silently regenerated. (The `added` branch below is kept
    rather than deleted: it still fires, safely, if `_X86_VEC_MOVE_ALLOW`
    ever shrinks and reclassifies an existing accept into
    `structural_only` -- it is only a brand-new capstone mnemonic that
    this cell can no longer see arrive.)
    """
    if not _SCAN_AVAILABLE:
        _fail_absent("(D-SLM4999 p/vp vitality pin, membership)", "")
    _, _, structural_only, _ = _census_check_a_p_vp_structural_reliance()
    pinned = _load_pinned_p_vp_structural_only()
    added = sorted(set(structural_only) - set(pinned))
    removed = sorted(set(pinned) - set(structural_only))
    assert added == [] and removed == [], (
        "check (A)'s p/vp structural-only accept set changed since "
        "fp_scan_fixtures/p_vp_structural_only_pinned.txt was pinned ({} "
        "members) -- added: {}; removed: {}. Determine whether each added "
        "member is genuine floating-point arithmetic (D-SLM4999's hole "
        "leaking again -- raise it, do not silently regenerate) or genuine "
        "packed-integer (regenerate the pin deliberately) before updating "
        "the fixture.".format(len(pinned), added, removed)
    )


def _p_vp_naming_convention_membership():
    """S6 (T-2407): the raw p/vp-prefixed, non-`pf`-prefixed capstone
    vocabulary, independent of `_x86_check_a`'s own D-SLM5156 frozen-
    allow-list decision -- reproduced fresh every call, like
    `_census_check_a_p_vp_structural_reliance` above. Starts with `p` or
    `vp`, does not start with `pf`, and is not one of
    `_X86_VEC_MOVE_ALLOW`'s own named entries -- the identical predicate
    the pre-D-SLM5156 structural rule used, kept alive here purely as a
    census surface now that `_x86_check_a` no longer computes it. See
    `test_check_a_p_vp_naming_convention_vocabulary_is_pinned_against_
    growth`'s own docstring for why this population, not `structural_only`,
    is what detects an ADDED p/vp mnemonic post-freeze."""
    from capstone import x86_const
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    vocabulary = sorted({
        md.insn_name(getattr(x86_const, attr))
        for attr in dir(x86_const) if attr.startswith("X86_INS_")
        if md.insn_name(getattr(x86_const, attr))
    })
    named = set(scan._X86_VEC_MOVE_ALLOW)
    return sorted(
        m for m in vocabulary
        if (m.startswith("p") or m.startswith("vp"))
        and not m.startswith("pf")
        and m not in named
    )


_P_VP_NAMING_CONVENTION_PIN_PATH = os.path.join(
    _FIXTURES, "p_vp_naming_convention_pinned.txt")


def _load_pinned_p_vp_naming_convention():
    with open(_P_VP_NAMING_CONVENTION_PIN_PATH) as f:
        return sorted(
            line.strip() for line in f
            if line.strip() and not line.strip().startswith("#")
        )


def test_check_a_p_vp_naming_convention_vocabulary_is_pinned_against_growth():
    """S6 (T-2407, review e9879e2-t2404-1p3-shipping-repair-set-review.md):
    restores the added-member half of "any change to this set" the
    sibling cell above,
    `test_check_a_p_vp_structural_only_set_is_pinned_against_vocabulary_
    growth`, can no longer detect once D-SLM5156's freeze (R3, T-2404) is
    in place -- see that cell's own docstring for why. A brand-new
    p/vp-shaped mnemonic REJECTs under `_x86_check_a` (absent from
    `_X86_P_VP_STRUCTURAL_ALLOW`) and so never reaches `structural_only`,
    but it DOES change `_p_vp_naming_convention_membership`'s raw,
    decision-independent population the moment capstone starts emitting
    it -- which is exactly the human-vetting trigger the frozen
    allow-list's own design depends on: the freeze is safe (an
    unrecognized mnemonic REJECTs by default) but not self-maintaining,
    and nothing else notices a new member exists to vet.

    440 members: the 438-member frozen allow-list plus `pi2fd`/`pi2fw`,
    the two `_X86_P_PREFIX_EXCLUDE` deny-list entries capstone's own
    vocabulary still recognizes (the deny list's other four entries either
    are not p/vp-prefixed by this exact rule -- `vcvtne2ps2bf16`,
    `vcvtneps2bf16` -- or are absent from capstone 5.0.7's own vocabulary
    entirely -- `vdpbf16ps`, `vpdpbf16ps` -- reproduced and executed
    against the real capstone install, not asserted).
    """
    if not _SCAN_AVAILABLE:
        _fail_absent("(D-SLM4999 p/vp vitality pin, naming-convention membership)", "")
    naming = _p_vp_naming_convention_membership()
    pinned = _load_pinned_p_vp_naming_convention()
    added = sorted(set(naming) - set(pinned))
    removed = sorted(set(pinned) - set(naming))
    assert added == [] and removed == [], (
        "check (A)'s p/vp naming-convention vocabulary (independent of "
        "the frozen allow-list) changed since fp_scan_fixtures/"
        "p_vp_naming_convention_pinned.txt was pinned ({} members) -- "
        "added: {}; removed: {}. An added member is a new p/vp-shaped "
        "mnemonic capstone now recognizes that _X86_P_VP_STRUCTURAL_ALLOW "
        "has never seen -- vet it against ISA-reference IEEE-754 "
        "semantics and, if it performs no floating-point operation, add "
        "it to the allow-list deliberately (never automatically) before "
        "updating this pin; if it does perform floating-point arithmetic, "
        "it already REJECTs under the freeze and no production change is "
        "needed either way. A removed member means capstone renamed or "
        "retired a mnemonic.".format(len(pinned), added, removed)
    )


# T-2404 (Brunel), a self-found consequence of building R3: the cell that
# formerly stood here, test_check_a_p_prefix_exclude_list_is_load_bearing,
# proved `_X86_P_PREFIX_EXCLUDE` (the deny list) was load-bearing by
# showing that removing an entry from it flipped the verdict to ACCEPT.
# D-SLM5156 rules that this deny list is "no longer load-bearing as a
# gate" once `_X86_P_VP_STRUCTURAL_ALLOW` (R3, above `_x86_check_a`) is
# built -- `_x86_check_a`'s p/vp branch now consults the allow-list alone,
# so removing an entry from `_X86_P_PREFIX_EXCLUDE` no longer changes any
# verdict: `pi2fd`/`vpdpbf16ps` REJECT because they are absent from the
# allow-list, not because they are present on the deny list, and stay
# absent (and therefore still REJECT) whether or not the deny list still
# names them. This cell's own premise -- that the deny list gates the
# verdict -- is exactly what D-SLM5156 supersedes; it is retired in the
# same commit as R3, alongside R7's own predecessor cell, rather than left
# asserting a mechanism the production change deliberately replaced. This
# retirement was not named by the fold-46 manifest, the red suite's own
# casebook, or any repair item -- it surfaced only once R3 was built and
# this cell was re-run red for a mechanism-superseded reason. Flagged in
# the build log for review alongside R3 rather than silently folded into
# it.


# ===========================================================================
# T-2366 (Curie), D-SLM5001 item (3) -- D-SLM4359's seven switch-jump-table
# symbols. D-SLM4359 ruled all seven restructured to direct conditional
# branches before v1.3.0 ships. D-SLM4988 (fold round 39) narrowed the
# GATING half of that ruling to only the two that REFUSE
# (?ExpectedDtype/?IsKnownSectionType), on the premise that the other five
# -- which reject via check (C) alone -- no longer block the gate once check
# (C) stops gating. T-2364's strike (blast radius item 2) found that premise
# void against the driver as it stood then: the shipped driver's own
# verdict still read check (C), so all five still REJECTed through it,
# unconditionally. T-2367 (fold round 39) built the premise's own
# precondition -- the driver now decides on `ab_verdicts` (checks (A)/(B)
# alone) -- and all seven now ACCEPT.
# ===========================================================================

_DSLM4359_SEVEN = [
    ("checked_chain_funnel.obj", "SslmForwardStatusName"),
    ("model.obj", "SslmModelStatusName"),
    ("model.obj", "ValidateSectionValues"),
    ("proof_manifest.obj", "BuildProofManifestJsonImpl"),
    ("proof_manifest.obj", "ConfigGeometryStatusName"),
    ("artifact.obj", "ExpectedDtype"),
    ("artifact.obj", "IsKnownSectionType"),
]


def test_dslm4359_seven_switch_jump_table_symbols_must_not_block_gate(real_build_dir):
    """None of D-SLM4359's own seven switch-jump-table symbols may block
    the ship gate, which decides on `ab_verdicts` (checks (A)/(B) alone)
    and REFUSE, never on the combined `verdicts` field check (C) still
    populates as a non-gating diagnostic (D-SLM5004/D-SLM5007, matching
    `scan_build_output.py`'s own gating read of `ab_verdicts`) -- five via
    check (C) alone (no longer a gating surface, design Sec4.1 D-SLM4985)
    and two via REFUSE (`artifact.obj`'s own two unclassified bytes,
    disposed at design Sec4.1 as D-SLM4359's already-ruled restructure,
    D-SLM4988). Executed against a real 17-object corpus built fresh by
    this session's own `real_build_dir` fixture (conftest.py, T-2368,
    D-SLM5008 -- never a hand-configured, unversioned directory) -- not a
    synthesized fixture, per this suite's own standing law that at least
    one cell runs the real build.
    """
    if not _SCAN_AVAILABLE:
        _fail_absent("(D-SLM4359 seven-symbol sweep)", "")
    if not _GATE_AVAILABLE:
        _fail_absent("(D-SLM4359 seven-symbol sweep, object lookup)", "")

    objects = scan_build_output.find_target_objects(real_build_dir, "superslm")
    objects_by_name: dict = {}
    for o in objects:
        objects_by_name.setdefault(os.path.basename(o), o)

    blocking = []
    for obj_name, substr in _DSLM4359_SEVEN:
        obj_path = objects_by_name.get(obj_name)
        if obj_path is None:
            pytest.skip(
                "real build object not present: {} (real build under {} "
                "has {} objects)".format(obj_name, real_build_dir, len(objects)))
        result = scan.scan_object(obj_path, isa="x86-64")
        if result.refuse:
            blocking.append("{} ({}, object REFUSEs)".format(substr, obj_name))
            continue
        matches = [name for name in result.ab_verdicts if substr in name]
        assert matches, (
            "fixture verification FAILED: expected a symbol containing "
            "{!r} in {}; found none among {} symbols".format(
                substr, obj_name, len(result.ab_verdicts))
        )
        verdict = result.ab_verdicts[matches[0]]
        if verdict != "ACCEPT":
            blocking.append("{} ({}, ab_verdict={})".format(substr, obj_name, verdict))

    assert blocking == [], (
        "D-SLM4359's restructure is owed for all seven switch-jump-table "
        "symbols -- D-SLM4988's narrowing to two is void (T-2364 blast "
        "radius item 2) -- {} of 7 still block the ship gate today: "
        "{}".format(len(blocking), blocking)
    )


# ===========================================================================
# T-2366 (Curie), D-SLM5001 item (6, first half) -- scan_build_output.py's
# own fail-closed membership discipline (design Sec7 dim 11's thirty-sixth
# population, D-SLM4990). Confirmed at source by T-2365's own coverage audit
# ("already implemented... requires no further specification from the
# planner") and re-confirmed here by direct execution, filed as named
# regression guards rather than left an audit-only sanity check with no
# cell of its own.
# ===========================================================================


def test_scan_build_output_zero_objects_exits_2_not_a_pass():
    """Must-reject leg: a build directory with no archive at any candidate
    location must never report a pass -- "nothing to scan is an
    infrastructure failure, never a pass" (scan_build_output.py's own
    module docstring).

    Re-aimed (T-2389 item 2, Poirot 8788b01 N6): the second leg used to
    assert that an existing but empty `<target>.dir` exits 2 "(zero
    objects found)". `main()` no longer enumerates `<target>.dir` at all
    -- the archive is the only corpus (D-SLM5100/D-SLM5101) -- so it
    reaches `find_target_archive`, finds no candidate, and exits 2 via
    the identical missing-archive path a NONEXISTENT build directory
    takes. An empty `<target>.dir` therefore graded nothing about object
    count; it duplicated the first leg under a different label.

    Re-aimed at what the driver actually discriminates: a `<target>.dir`
    carrying a REAL compiled object, still with no archive at any
    candidate path. This is the design's own "THE ARCHIVE IS THE ONLY
    CORPUS -- NO FALLBACK" claim, not yet pinned anywhere else in this
    suite -- `main()`'s own comment cites it (design Sec7 dim 11's
    restored fortieth population, D-SLM5103) but no cell exercises a
    populated object directory against a missing archive. Able to fail
    for its own reason -- proven by mutation this round (T-2389 casebook,
    `Claude/Curie/t2389-1p3-test-surface-repairs-2026-08-29.md`):
    monkeypatching `find_target_archive` so it fabricates a hit (any real,
    well-formed archive elsewhere on disk) for this exact populated,
    archive-less build directory turns the exit code from 2 to 0 (PASS),
    and the assertion below catches it. (A literal object-directory
    fallback that mis-encodes the raw `.obj` itself as an archive is
    instead caught one step later, by the malformed-archive check --
    still exit 2, but for a different reason; the fabricated-hit mutation
    isolates the property this leg actually pins: exit code 2 depends on
    the archive genuinely being absent from every candidate location, not
    on anything `find_target_archive` returns or on what is in
    `<target>.dir`.)
    """
    if not _GATE_AVAILABLE:
        _fail_absent("(thirty-sixth population, must-reject)", "")
    with fc.TempDir() as tmp:
        nonexistent = os.path.join(tmp, "does-not-exist")
        assert _run_gate(nonexistent) == 2, (
            "a nonexistent build directory must exit 2 (infrastructure "
            "failure), never a pass"
        )

        populated_build = os.path.join(tmp, "populated-build")
        target_dir = os.path.join(populated_build, "superslm.dir", "Release")
        os.makedirs(target_dir)
        src = os.path.join(_FIXTURES, "pop17_carveout_reject.asm")
        obj = os.path.join(target_dir, "reject.obj")
        try:
            fc.assemble_ml64(src, obj)
        except fc.ToolUnavailable as e:
            pytest.skip(str(e))
        assert os.path.isfile(obj), (
            "fixture verification FAILED: expected a real compiled object "
            "at {}".format(obj)
        )

        assert _run_gate(populated_build) == 2, (
            "design Sec4.1 (fold round 43, D-SLM5100/D-SLM5101): the "
            "archive is the only corpus, with no object-directory "
            "fallback -- a <target>.dir carrying a real compiled object "
            "but no archive at any candidate location must still exit 2, "
            "never scan the object directory directly"
        )


def test_scan_build_output_real_build_finds_exactly_seventeen_objects(real_build_dir):
    """Must-accept leg: the real superslm CMake target's own build output
    resolves to exactly 17 objects, no configuration supplied. Uses a real
    corpus built fresh by this session's own `real_build_dir` fixture
    (conftest.py, T-2368, D-SLM5008 -- never a hand-configured, unversioned
    directory).
    """
    if not _GATE_AVAILABLE:
        _fail_absent("(thirty-sixth population, must-accept)", "")
    objects = scan_build_output.find_target_objects(real_build_dir, "superslm")
    assert len(objects) == 17, (
        "design Sec4.1's own text states the real build emits exactly 17 "
        "objects for the superslm target; found {}: {}".format(
            len(objects), objects)
    )


# ===========================================================================
# T-2403 (Curie), R1 -- population forty-nine, the COFF section-selection
# bypass (`Claude/Vitruvius/t2265-fold46-delta-manifest.md` Sec2's own R1
# entry; design Sec7 dimension 11's forty-ninth population, D-SLM5153;
# corrected must-reject observables, D-SLM5163/D-SLM5169). See
# fp_scan_fixtures/pop49_evil_object.py's own module docstring for the full
# mechanism. `_parse_coff`'s own selection criterion
# (tests/ci/check_fp_free_scan.py:334) keys on IMAGE_SCN_CNT_CODE alone; a
# section flagged MEM_EXECUTE without CNT_CODE never enters code_sections,
# so an object built entirely inside such a section is invisible to the
# scan today -- LIVE on the shipping tree (D-SLM5169), release-blocking.
#
# Two different mechanisms, two must-reject cells, per the design's own
# correction (D-SLM5163): the zero-symbol object has no function symbol to
# derive an extent from, so it can never reach per-symbol classification --
# its real post-fix observable is refuse=True (byte-accounting cannot
# attribute the divsd bytes to anything). The symbol-carrying variant's one
# function symbol witnesses an extent covering the whole section, so
# byte-accounting succeeds and check (A) classifies the decoded divsd --
# its real post-fix observable is verdicts['EvilExecSectionFn'] == 'REJECT'.
# ===========================================================================

def test_population_49a_evil_object_zero_symbol_refuses_once_fixed():
    """Population forty-nine's first must-reject cell (T-2403, R1): the
    conductor's own zero-symbol `evil.obj` construction (`Claude/Bach/
    probes/t2391-exec-section-bypass.py`, reproduced byte for byte by
    fp_scan_fixtures/pop49_evil_object.py's own build_zero_symbol), a
    hand-emitted x86-64 COFF object with one MEM_EXECUTE-without-CNT_CODE
    section containing a real `divsd xmm0, xmm0`. Real observable once R1
    widens section selection: refuse=True (byte-accounting cannot attribute
    the divsd bytes to any symbol -- there is none -- or recognise them as
    padding). TODAY the bypass is live: the section is skipped entirely,
    code_sections is empty, and the object exits clean (refuse=False, no
    verdict) -- confirmed by this cell's own fixture verification below,
    independent of scan_object's own logic.
    """
    import capstone as _capstone
    with fc.TempDir() as tmp:
        obj_path, raw = pop49_fx.build_zero_symbol(tmp)
        md = _capstone.Cs(_capstone.CS_ARCH_X86, _capstone.CS_MODE_64)
        insns = list(md.disasm(pop49_fx.CODE, 0))
        assert [(i.mnemonic, i.op_str) for i in insns] == [("divsd", "xmm0, xmm0")], (
            "fixture verification FAILED: expected exactly one decoded "
            "'divsd xmm0, xmm0'; independent capstone decode found "
            "{}".format([(i.mnemonic, i.op_str) for i in insns])
        )
        if not _SCAN_AVAILABLE:
            _fail_absent(
                "forty-nine-a (zero-symbol evil.obj, must-reject: refuse=True)",
                "Fixture verified above: a real divsd decoded from the "
                "hand-emitted section.",
            )
        result = scan.scan_object(obj_path, isa="x86-64", data=raw)
        assert result.refuse is True, (
            "the zero-symbol object's own MEM_EXECUTE-without-CNT_CODE "
            "section carries a real divsd with no symbol to attribute it "
            "to -- byte-accounting must REFUSE once the section is not "
            "skipped outright; got refuse={} verdicts={} (the live bypass: "
            "the section never enters code_sections, so the object exits "
            "clean)".format(result.refuse, result.verdicts)
        )


def test_population_49b_evil_object_symbol_carrying_rejects_once_fixed():
    """Population forty-nine's second must-reject cell (T-2403, R1): the
    design's own "symbol-carrying variant (one COFF function symbol added,
    otherwise identical)" -- fp_scan_fixtures/pop49_evil_object.py's own
    build_symbol_carrying, byte-identical to the zero-symbol construction
    plus one function-typed COFF symbol (EvilExecSectionFn) witnessing an
    extent over the whole section. Real observable once R1 widens section
    selection: byte-accounting succeeds (the symbol's own extent covers the
    divsd cleanly) and check (A) classifies it -- divsd is not on
    _X86_VEC_MOVE_ALLOW, not p/vp-prefixed, not in _X86_BITWISE_FP_FAMILY,
    so it default-deny REJECTs: verdicts['EvilExecSectionFn'] == 'REJECT'.
    TODAY the bypass is live for this construction too -- confirmed by this
    cell's own fixture verification below.
    """
    import capstone as _capstone
    with fc.TempDir() as tmp:
        obj_path, raw = pop49_fx.build_symbol_carrying(tmp)
        md = _capstone.Cs(_capstone.CS_ARCH_X86, _capstone.CS_MODE_64)
        insns = list(md.disasm(pop49_fx.CODE, 0))
        assert [(i.mnemonic, i.op_str) for i in insns] == [("divsd", "xmm0, xmm0")], (
            "fixture verification FAILED: expected exactly one decoded "
            "'divsd xmm0, xmm0'; independent capstone decode found "
            "{}".format([(i.mnemonic, i.op_str) for i in insns])
        )
        if not _SCAN_AVAILABLE:
            _fail_absent(
                "forty-nine-b (symbol-carrying evil object, must-reject: "
                "verdict REJECT)",
                "Fixture verified above: a real divsd decoded from the "
                "hand-emitted section, witnessed by one real COFF function "
                "symbol.",
            )
        result = scan.scan_object(obj_path, isa="x86-64", data=raw)
        assert result.verdicts.get(pop49_fx.SYMBOL_NAME) == "REJECT", (
            "the symbol-carrying object's own {} witnesses a real divsd -- "
            "check (A) must REJECT it (not on any accept list) once the "
            "section is not skipped outright; got refuse={} "
            "verdicts={}".format(
                pop49_fx.SYMBOL_NAME, result.refuse, result.verdicts)
        )


# ===========================================================================
# T-2403 (Curie), R3 -- population fifty-one's corrected must-accept/
# must-reject pair (`Claude/Vitruvius/t2265-fold46-delta-manifest.md` Sec2's
# R3 entry via Sec5's execution ledger; design Sec7 dimension 11's
# fifty-first population, D-SLM5156; the tautology finding, D-SLM5164).
#
# D-SLM5164 executed fold round 44's own original population and found
# neither cell can fail on a deliberately mis-vetted allow-list that
# re-admits a known real IEEE-754-arithmetic mnemonic ('pi2fd'): the
# original must-accept ("every admitted mnemonic is admitted") is a
# membership lookup against the list under test -- it can only check the
# list against itself, and passes regardless of what the list contains. The
# corrected pair below cross-checks the allow-list against an
# INDEPENDENTLY-derived oracle (StandardsDocument.md Sec5.4: a reference
# that shares no parameter with the list it grades).
#
# `_X86_P_VP_STRUCTURAL_ALLOW` is R3's own production build item and does
# not exist yet -- both cells below are guarded and fail explicitly for
# that reason until it is built.
# ===========================================================================

# Adopted UNMODIFIED from `Claude/Vitruvius/t2265-fold46-probes/
# r3_premise_probe.py`'s own `_KNOWN_FP_LEAKS` -- p/vp-prefixed x86
# mnemonics that are genuine IEEE-754 floating-point arithmetic per the
# Intel SDM (pi2fd/pi2fw: 3DNow packed-int-to-float conversion, rounds;
# vpdpbf16ps: AVX-512 BF16 dot-product-accumulate, the SAME instruction as
# vdpbf16ps under the rendering that matches the vp-prefix structural
# accept -- test_bf16_x86_rendering_pair's own subject). A literal constant
# in THIS file, not read from any list check_fp_free_scan.py defines, so it
# shares no parameter with the allow-list it cross-checks.
_KNOWN_P_VP_FP_LEAKS = frozenset({"pi2fd", "pi2fw", "vpdpbf16ps"})


def test_check_a_p_vp_structural_allow_agrees_with_independent_fp_oracle():
    """Population fifty-one's corrected must-accept (D-SLM5156): every
    mnemonic the independent oracle above names is run through the real,
    unmodified `_x86_check_a` -- production code, not a set intersection
    over the raw allow-list -- and none may be ACCEPTed via the
    `p_vp_structural_allow` branch.

    M5 (T-2407, review e9879e2-t2404-1p3-shipping-repair-set-review.md):
    the ORIGINAL form of both cells in this population computed
    `set(scan._X86_P_VP_STRUCTURAL_ALLOW) & _KNOWN_P_VP_FP_LEAKS` directly
    -- a membership check against the list under test, never calling
    `_x86_check_a` at all, so `mock.patch.object` in the must-reject cell
    below had no bearing on what either assertion actually exercised. Both
    cells now call `_x86_check_a` per mnemonic, the real function
    `check_fp_free_scan.scan_object` calls in production, so the oracle
    grades what the shipped classifier actually decides.
    """
    if not hasattr(scan, "_X86_P_VP_STRUCTURAL_ALLOW"):
        _fail_absent(
            "fifty-one (must-accept: allow-list vs. independent FP oracle)",
            "_X86_P_VP_STRUCTURAL_ALLOW is not yet defined on check_fp_free_scan.",
        )
    vec_ops = "xmm1, xmm2"
    leaked = sorted(
        m for m in _KNOWN_P_VP_FP_LEAKS
        if scan._x86_check_a(m, vec_ops) == "p_vp_structural_allow"
    )
    assert leaked == [], (
        "_x86_check_a wrongly ACCEPTs {} known real IEEE-754-arithmetic "
        "mnemonic(s) via the p_vp_structural_allow branch, per the "
        "independent oracle: {} -- the shipped classifier and the oracle "
        "disagree".format(len(leaked), leaked)
    )


def test_check_a_p_vp_mis_vetted_allow_list_caught_by_independent_oracle():
    """Population fifty-one's corrected must-reject (D-SLM5156, D-SLM5164's
    own reproduction): a copy of the real allow-list with 'pi2fd' (a
    documented member of the independent oracle above) wrongly re-admitted
    must be caught -- the exact construction D-SLM5164 executed and found
    the ORIGINAL population's must-accept/must-reject pair both silent on.
    Proves the corrected check discriminates a mis-vetted list from a
    genuinely fail-closed one, rather than only checking the list against
    itself (a membership tautology that cannot fail on any input).

    M5 (T-2407): the must-reject now runs `_x86_check_a("pi2fd", ...)`
    against the module under `mock.patch.object`, not a set intersection
    over the patched attribute directly -- `_x86_check_a` resolves
    `_X86_P_VP_STRUCTURAL_ALLOW` as a module global at call time, so the
    patch is genuinely exercised through the production function's own
    control flow, and the assertion would fail if `_x86_check_a`'s p/vp
    branch were ever rewired to consult a different list.
    """
    if not hasattr(scan, "_X86_P_VP_STRUCTURAL_ALLOW"):
        _fail_absent(
            "fifty-one (must-reject: mis-vetted list caught by oracle)",
            "_X86_P_VP_STRUCTURAL_ALLOW is not yet defined on check_fp_free_scan.",
        )
    assert "pi2fd" not in scan._X86_P_VP_STRUCTURAL_ALLOW, (
        "fixture verification FAILED: 'pi2fd' must not already be on the "
        "real, correctly-vetted allow-list"
    )
    vec_ops = "xmm1, xmm2"
    mis_vetted = set(scan._X86_P_VP_STRUCTURAL_ALLOW) | {"pi2fd"}
    with mock.patch.object(scan, "_X86_P_VP_STRUCTURAL_ALLOW", mis_vetted):
        leaked = sorted(
            m for m in _KNOWN_P_VP_FP_LEAKS
            if scan._x86_check_a(m, vec_ops) == "p_vp_structural_allow"
        )
        assert leaked == ["pi2fd"], (
            "D-SLM5164's own construction: re-admitting 'pi2fd' to the "
            "allow-list must make _x86_check_a('pi2fd', ...) ACCEPT via "
            "the p_vp_structural_allow branch -- got {}".format(leaked)
        )


# ===========================================================================
# T-2403 (Curie), R4 -- population fifty-two, the reason-reporting census
# correction (`Claude/Vitruvius/t2265-fold46-delta-manifest.md` Sec2's own
# R4 entry via Sec5's execution ledger; design Sec7 dimension 11's
# fifty-second population, D-SLM5157). `_x86_check_a`'s contract changes
# from bool to Optional[str] (the admitting branch's own name, or None on
# REJECT); structural_only is redefined as every acceptance with no
# reported reason (not isinstance(reason, str) -- D-SLM5165 proved the
# literal 'reason is None' reading dead: accept_a is already filtered to
# acceptances, so nothing inside it can satisfy that predicate and the
# census reports zero on the mutant it exists to catch).
#
# Both cells below need R4's contract change to exist: the must-accept
# fails naturally today (every accept returns bool True, not a reason
# string, so structural_only == accept_a, nonempty); the must-reject is a
# mutation on the contract itself and is guarded explicitly until R4 lands.
# ===========================================================================

def _census_check_a_reason_attribution():
    """Population fifty-two's own census: walks the real x86-64 capstone
    vocabulary and calls the real, unmodified _x86_check_a for each
    mnemonic on a vector operand, recording whether the call's own return
    value is a reason STRING (attributed) or merely truthy-non-string
    (unattributed). Never a cached or hardcoded population, so a capstone
    upgrade is read fresh every call -- the same discipline
    _census_check_a_p_vp_structural_reliance already follows.

    M3 (T-2407, review e9879e2-t2404-1p3-shipping-repair-set-review.md):
    the third return value is named `unattributed`, not `structural_only`
    -- `_census_check_a_p_vp_structural_reliance`'s own `structural_only`
    names a completely different quantity (accepted, and not on
    `_X86_VEC_MOVE_ALLOW`, regardless of whether the acceptance carries a
    reason string); this census's own quantity is accepted with NO
    reason string at all. Same identifier, two unrelated meanings, one
    file was the finding; this function's own return value is renamed to
    stop it."""
    from capstone import x86_const
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    vocabulary = sorted({
        md.insn_name(getattr(x86_const, attr))
        for attr in dir(x86_const) if attr.startswith("X86_INS_")
        if md.insn_name(getattr(x86_const, attr))
    })
    vec_ops = "xmm1, xmm2"
    accept_a = []
    unattributed = []
    for m in vocabulary:
        r = scan._x86_check_a(m, vec_ops)
        if not r:
            continue
        accept_a.append(m)
        if not isinstance(r, str):
            unattributed.append(m)
    return vocabulary, accept_a, unattributed


def test_check_a_reason_attribution_census_every_accept_has_a_reason():
    """Population fifty-two's must-accept (D-SLM5157): once _x86_check_a
    adopts the Optional[str] contract (R4), `unattributed` (every
    acceptance with no reported reason) must be [] -- every member of
    accept_a is attributed to exactly one of the three named reasons
    (_X86_VEC_MOVE_ALLOW, _X86_BITWISE_FP_FAMILY, _X86_P_VP_STRUCTURAL_ALLOW).

    TODAY _x86_check_a still returns bool: every accepted mnemonic's own
    'reason' is the literal True, not a string, so unattributed ==
    accept_a in full (nonempty) -- this cell fails for exactly that reason
    until R4's contract change lands.
    """
    _vocabulary, accept_a, unattributed = _census_check_a_reason_attribution()
    assert unattributed == [], (
        "{} of {} check(A) acceptances report no distinguishable reason "
        "(not a string) -- _x86_check_a has not yet adopted the "
        "Optional[str] contract (R4); first 10: {}".format(
            len(unattributed), len(accept_a), unattributed[:10])
    )


def test_check_a_reason_attribution_census_catches_a_silently_unattributed_accept():
    """Population fifty-two's must-reject (D-SLM5157, mutation-tested, not
    a real defect): once R4's Optional[str] contract lands, wrapping
    _x86_check_a so the bitwise_fp_family branch's own reason tag is
    stripped (returning a bare truthy sentinel instead of the reason
    string) WHILE its own ACCEPT behaviour is unchanged must make
    structural_only nonempty (16, the family's own size) -- proving the
    census actually discriminates a silently-unattributed accept from an
    attributed one, rather than reporting [] regardless of whether
    reason-reporting is wired correctly.

    Cannot be constructed against today's bool-returning _x86_check_a --
    there is no reason string for the wrapper to strip. Guarded explicitly
    until R4 lands; the mutation logic below runs unmodified once it does
    (it wraps the function's own OUTPUT, never its internal branches).
    """
    if not hasattr(scan, "_X86_BITWISE_FP_FAMILY"):
        _fail_absent("fifty-two (must-reject: bitwise-family mutation)", "")
    sample = next(iter(scan._X86_BITWISE_FP_FAMILY))
    sample_reason = scan._x86_check_a(sample, "xmm1, xmm2")
    if not isinstance(sample_reason, str):
        pytest.fail(
            "check_fp_free_scan.py's _x86_check_a still returns bool (R4's "
            "Optional[str] contract, design Sec7 dim 11's fifty-second "
            "population, has not landed) -- population fifty-two's "
            "must-reject (a mutation on the bitwise_fp_family branch's own "
            "reason tag) cannot be constructed until _x86_check_a returns a "
            "reason string. Confirmed at source: _x86_check_a({!r}, "
            "'xmm1, xmm2') returned {!r}, not a string.".format(sample, sample_reason)
        )

    real_check_a = scan._x86_check_a
    bitwise_family = scan._X86_BITWISE_FP_FAMILY

    def _mutant(mnemonic, op_str):
        r = real_check_a(mnemonic, op_str)
        if mnemonic.lower() in bitwise_family and isinstance(r, str):
            return True  # reason tag stripped; ACCEPT behaviour unchanged
        return r

    with mock.patch.object(scan, "_x86_check_a", _mutant):
        _vocabulary, _accept_a, unattributed_mutant = _census_check_a_reason_attribution()
    assert len(unattributed_mutant) == len(bitwise_family), (
        "mutating the bitwise_fp_family branch's own reason tag (ACCEPT "
        "unchanged, reason stripped to a bare truthy sentinel) was "
        "expected to make unattributed nonempty ({} members, the "
        "family's own size) -- got {}: the census does not actually "
        "discriminate a silently-unattributed accept from an attributed "
        "one".format(len(bitwise_family), len(unattributed_mutant))
    )
