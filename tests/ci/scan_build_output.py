"""Scan the real archive a CMake build produced for a target, and fail if any
object member it contains carries floating-point arithmetic.

WHY THIS SHAPE. The question this answers is "does the shipped library contain
floating-point math?", and the only trustworthy list of what is in the shipped
library is the one the build system itself produced. Earlier versions of this
check derived that list by parsing `CMakeLists.txt` and then tried to prove the
derivation was complete; four successive attempts at that proof each had a hole,
because a derived list is only as good as the parser and the parser is what you
are trying to check. A later version globbed the object files CMake emitted into
the target's own object directory -- no derivation, but two sources of truth
(the directory, the archive) that a stale incremental build could let disagree
(design Sec4.1, D-SLM5034). THE CORPUS IS THE ARCHIVE (T-2381, Brunel, design
Sec4.1/Sec5.4/Sec5.5 fold round 40, D-SLM5034-D-SLM5039): the static archive
CMake actually links (`superslm.lib` on MSVC, `libsuperslm.a` wherever `ar`
produces one) IS the shipped artifact, read member-by-member by
`check_fp_free_scan.iterate_archive_members`/`enumerate_archive_objects` --
there is no derivation step and no second corpus for a future build-layout
change to desync, because reading the archive's own member table is reading
what the shipped artifact contains, not a claim about it.

FAILS CLOSED. An archive that cannot be opened or parsed, or that opens
cleanly but carries zero OBJECT-kind members, is an infrastructure failure
(exit 2), never a pass -- "nothing to scan" is what a mis-pointed build
directory looks like, and it must never read as clean. A recognized object
member with an unrecognized container format (e.g. a Mach-O-magic'd member)
REFUSEs per member -- `check_fp_free_scan.scan_object` already catches this
internally and returns a REFUSE `ScanResult` rather than raising (T-2343,
78535ed-t2339's own M2) -- so a single unrecognized member fails the job with
the failing member named in this driver's own output, never an uncaught
Python traceback propagating to the top of the process (design Sec4.1,
D-SLM5036).

THE GATE READS CHECKS (A)/(B) ALONE (T-2367, design Sec4.1/Sec5.5 fold round
39, D-SLM4985/D-SLM4996). `check_fp_free_scan.scan_object` runs checks (A),
(B), AND (C) and reports two per-symbol surfaces: `ab_verdicts` (checks
(A)/(B) only) and `verdicts` (the combined ab_accept-and-c_accept verdict,
unchanged). This driver's own pass/fail decision is a function of
`ab_verdicts` and `refuse` alone -- a call/tail-jmp edge check (C) cannot vet
(an external target absent from the vetted list, or a first-party indirect
tail jump) is reported below as a non-gating diagnostic, never as a build
failure. A genuine check-(A)/(B) violation still fails the job exactly as
before EXCEPT on one open, named axis: an unvetted vector-register mnemonic
fails unless it is `p`/`vp`-prefixed, in which case check (A) is
structural-only and currently fail-OPEN on that whole class (measured:
capstone 5.0.7's own vocabulary gives 555 accepted `p`/`vp` mnemonics, of
which 454 are named by no allow-list at all) -- whether to close that class
is OPEN, waiting on Dan (D-SLM5009), and is pinned in the suite as
`xfail(strict=True)` rather than asserted here as settled. Check (C)'s
retirement narrows what can fail checks (A)/(B) alone (a call/tail-jmp edge
check (C) alone used to reject is now a non-gating diagnostic).

THE OBJECT-DIRECTORY PATH IS RETAINED, NOT AS A FALLBACK OVER THE SAME BUILD
(T-2381, Brunel). Design Sec4.1 rules out reading BOTH the archive and the
directory for one build's own corpus -- that is the two-sources-of-truth
shape the archive retargeting exists to close, and this driver never does
it: whenever an archive exists at any of `find_target_archive`'s own
candidate locations, it is read and the directory is never consulted for
that run. `find_target_objects`/`_scan_object_directory_corpus` are kept,
used only when NO archive is found at any candidate location, because
several pre-archive unit constructions in
`tests/t2296-fp-free-open-red-suite/test_check_fp_free_scan.py` (read-only
to this ticket) build a bare `<target>.dir` object layout with no archive at
all and invoke this driver's own `main()` directly against it -- deleting
the directory path would fail those currently-green, already-committed
cells for a reason unrelated to what this round built. Every real CI leg
(`.github/workflows/tests.yml`'s `fp-free-scan-gate` job, `build.bat`'s own
gating invocation) always produces an archive, so this fallback is inert on
every real leg; it exists solely for the pre-archive synthetic fixtures
named above.

WHAT IT DOES NOT ANSWER. Scanning is per-member and per-ISA. This module
reports what `check_fp_free_scan.scan_object` returns for each object member,
on the ISA named on the command line; it does not itself decide whether the
scanner is correct. The scanner's own per-symbol verdict is independently
commissioned on five production legs at a one-instruction floor
(`Claude/Popper/t2345-fp-scan-recommissioning-2026-08-27.md`); the
archive-based driver's own re-commissioning is a separate, owed obligation
(design Sec5.4, D-SLM5041/D-SLM5068/D-SLM5069) that this build round does not
itself discharge -- it builds the driver design Sec7 dimension 11's
thirty-seventh through forty-eighth populations grade, it does not stand in
for the re-commissioning those populations' own execution is.

Usage:
    python tests/ci/scan_build_output.py --build-dir <cmake-build-dir>
                                         [--target superslm] [--isa x86-64]

Exit codes: 0 every object's checks (A)/(B) accept, on every symbol, and no
object REFUSEs; 1 a checks-(A)/(B) REJECT or a REFUSE; 2 nothing to scan, the
build directory was not found, or the archive/object-directory corpus itself
could not be read (an infrastructure failure, kept distinct from a scan
finding).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_fp_free_scan as scan  # noqa: E402

_OBJ_EXTS = (".obj", ".o")

# T-2381 (Brunel), design Sec4.1's platform-legs paragraph: candidate
# locations for the target's own static archive, checked in order -- the
# first that exists wins. The COFF/MSVC leg (`windows-latest`,
# `fp-free-scan-gate`) is unchanged in every respect except which artifact
# is read: `cmake -B build` on that runner defaults to the Visual Studio
# generator, producing `build/Release/<target>.lib`, the first candidate.
# The remaining candidates are defensive, covering a single-config Windows
# generator and a GNU-`ar`-toolchain layout (`lib<target>.a`) for whichever
# leg next wires the ELF/GCC archive path CI-side (design Sec7 dim 11's
# forty-sixth population, routed, not built by this round).
_ARCHIVE_CANDIDATES_TEMPLATE = (
    os.path.join("Release", "{target}.lib"),
    "{target}.lib",
    "lib{target}.a",
    os.path.join("Release", "lib{target}.a"),
)


def find_target_archive(build_dir: str, target: str):
    """The static-library archive CMake's own build produced for `target`,
    at the first of `_ARCHIVE_CANDIDATES_TEMPLATE`'s own locations (relative
    to `build_dir`) that exists. Returns the path, or None if none of the
    candidates exists -- callers read None as "no archive at this build
    directory," never as an error on its own (a caller may fall back to the
    object-directory path, below, or may itself treat a missing archive as
    an infrastructure failure)."""
    for template in _ARCHIVE_CANDIDATES_TEMPLATE:
        candidate = os.path.join(build_dir, template.format(target=target))
        if os.path.isfile(candidate):
            return candidate
    return None


def find_target_objects(build_dir: str, target: str) -> list:
    """Every object file CMake emitted for `target`, from the build tree itself.

    Both generator layouts are matched, because the CI matrix uses more than one:
      - MSBuild / Visual Studio: <build>/<target>.dir/<config>/*.obj
      - Ninja, Makefiles:        <build>/CMakeFiles/<target>.dir/**/*.o
    Nothing is parsed and nothing is derived; the directory name is the build
    system's own record of which objects belong to the target.

    T-2381 (Brunel): retained as the object-directory corpus's own reader,
    used by `_scan_object_directory_corpus` only when `find_target_archive`
    finds no archive at all -- see this module's own docstring ("THE
    OBJECT-DIRECTORY PATH IS RETAINED..."). Also called directly by several
    read-only cells in `tests/t2296-fp-free-open-red-suite/
    test_check_fp_free_scan.py` that predate the archive-based corpus.
    """
    wanted = target + ".dir"
    found = []
    for root, _dirs, files in os.walk(build_dir):
        parts = os.path.normpath(root).split(os.sep)
        if wanted not in parts:
            continue
        for name in files:
            if name.endswith(_OBJ_EXTS):
                found.append(os.path.join(root, name))
    return sorted(found)


def _report_and_tally(label: str, result, counters: dict) -> None:
    """Prints one object member's own scan result and updates the shared
    running totals -- the identical accounting and print shape both the
    archive-based and object-directory corpus paths use (T-2381, Brunel),
    factored out so the report format is defined once rather than twice."""
    if result.refuse:
        counters["refuse"] += 1
        print("  REFUSE   {}  (unclassified_bytes={}, format={})".format(
            label, result.unclassified_bytes, result.object_format))
        return

    # Gating decision: checks (A)/(B) alone (design Sec4.1/Sec5.5 fold
    # round 39, D-SLM4985/D-SLM4996). Check (C) keeps running and keeps
    # reporting through `result.verdicts` (the combined verdict,
    # unaffected) -- read below only for the non-gating diagnostic line,
    # never for the pass/fail decision.
    ab_rejects = sorted(s for s, v in result.ab_verdicts.items() if v == "REJECT")
    check_c_only = sorted(
        s for s, v in result.verdicts.items()
        if v == "REJECT" and result.ab_verdicts.get(s) == "ACCEPT"
    )
    counters["accept"] += sum(1 for v in result.ab_verdicts.values() if v == "ACCEPT")
    counters["reject"] += len(ab_rejects)
    counters["check_c_only"] += len(check_c_only)

    if ab_rejects:
        print("  REJECT   {}  ({} symbol(s), format={})".format(
            label, len(ab_rejects), result.object_format))
        for s in ab_rejects:
            print("             {}".format(s))
    else:
        print("  clean    {}  ({} symbol(s), format={})".format(
            label, len(result.ab_verdicts), result.object_format))
    if check_c_only:
        print("             (non-gating diagnostic: {} symbol(s) reject under "
              "check (C) alone -- an unvetted external call target or an "
              "unresolved indirect edge; checks (A)/(B) accept them, and "
              "check (C) does not gate)".format(len(check_c_only)))
        for s in check_c_only:
            print("               [check-C-only] {}".format(s))


def _finish(counters: dict, n_units: int) -> int:
    """Prints the closing totals line and returns the job's own exit code --
    shared by both corpus paths (T-2381, Brunel)."""
    print()
    print("Totals: {} object(s); {} symbol(s) ACCEPT, {} REJECT, {} object(s) REFUSE "
          "(checks (A)/(B), gating); {} symbol(s) reject under check (C) alone "
          "(non-gating diagnostic)".format(
              n_units, counters["accept"], counters["reject"], counters["refuse"],
              counters["check_c_only"]))

    if counters["reject"] or counters["refuse"]:
        print("FAIL: the scan did not come back clean.")
        return 1
    print("PASS: no floating-point arithmetic found in any object of this target "
          "(checks (A)/(B); check (C) is a non-gating diagnostic).")
    return 0


def _archive_member_symbol_names(member) -> set:
    """The corpus_symbols index's own per-member contribution, read from an
    in-memory archive member payload rather than a file path (T-2381,
    Brunel). A member whose own container format this reader does not
    recognize (e.g. a Mach-O-magic'd member, design Sec4.1's REFUSE-
    not-crash contract) contributes nothing to the index -- it REFUSEs at
    scan time below and never resolves a check-(C) edge that would need it
    named as an in-corpus callee, so silently excluding it from the index
    here does not excuse it from its own bytes being checked."""
    try:
        return scan._function_symbol_names_from_bytes(member.payload)
    except ValueError:
        return set()


def _scan_archive_corpus(archive_path: str, args) -> int:
    """The archive-based corpus path (T-2381, Brunel, design Sec4.1's
    archive member-iterator contract, D-SLM5034/D-SLM5035, as amended by
    fold rounds 41/42): every OBJECT-kind member `enumerate_archive_objects`
    yields from `archive_path` is handed to `scan_object` by byte range (the
    member's own `payload`), never extracted to a temporary file. Malformed
    or unreadable archives, and archives with zero object members, are
    infrastructure failures (exit 2), never a pass -- design Sec4.1's own
    disposition for both."""
    try:
        members = scan.enumerate_archive_objects(archive_path)
    except scan.ArchiveHasNoObjectsError as exc:
        print("ERROR: {}".format(exc))
        print("       Nothing to scan is an infrastructure failure, never a pass.")
        return 2
    except (scan.MalformedArchiveError, ValueError) as exc:
        print("ERROR: archive {!r} could not be read: {}".format(archive_path, exc))
        print("       A malformed or unrecognized archive is an infrastructure "
              "failure, never a pass.")
        return 2

    print("Scanning {} object member(s) of archive {} for target {!r} (isa={})".format(
        len(members), archive_path, args.target, args.isa))

    # Built once, from every member's own symbol table, so each scan sees
    # the whole in-corpus index rather than a growing prefix of it -- the
    # identical discipline the object-directory path already used, applied
    # to archive-member payloads instead of object-file paths.
    corpus_symbols = frozenset().union(
        *(_archive_member_symbol_names(m) for m in members)
    ) if members else frozenset()

    counters = {"accept": 0, "reject": 0, "refuse": 0, "check_c_only": 0}
    print()
    for member in members:
        # design Sec4.1, D-SLM5036: a recognized object member with an
        # unrecognized container format REFUSEs -- scan_object's own
        # existing try/except around _read_object_format already converts
        # that exception into ScanResult(refuse=True) for ANY caller,
        # archive-based or not, so calling it here (rather than reading the
        # header ourselves) is what makes "no leg reaches PASS by not
        # looking, and no leg crashes uncaught" true by construction.
        result = scan.scan_object(path=member.name, isa=args.isa,
                                  corpus_symbols=corpus_symbols, data=member.payload)
        _report_and_tally(member.name, result, counters)

    return _finish(counters, len(members))


def _scan_object_directory_corpus(args) -> int:
    """The pre-archive object-directory corpus path (T-2367/T-2371),
    retained per this module's own docstring ("THE OBJECT-DIRECTORY PATH IS
    RETAINED...") -- used only when `find_target_archive` finds no archive
    at any candidate location."""
    objects = find_target_objects(args.build_dir, args.target)
    if not objects:
        print("ERROR: no object files found for target {!r} under {}".format(
            args.target, args.build_dir))
        print("       Nothing to scan is an infrastructure failure, never a pass.")
        return 2

    print("Scanning {} object(s) emitted by the build for target {!r} (isa={})".format(
        len(objects), args.target, args.isa))

    # Built once, from every object's own symbol table, so each scan sees the
    # whole in-corpus index rather than a growing prefix of it.
    # `_read_function_symbol_names` lives in `check_fp_free_scan` itself
    # (T-2371, D-SLM5018 M2): this driver's only production dependency is
    # the scanner module already imported above as `scan`, not the retired
    # `run_fp_free_scan_real_corpus.py` driver, which is not load-bearing
    # for the ship gate and must not become a hard import of it.
    corpus_symbols = frozenset().union(
        *(scan._read_function_symbol_names(o) for o in objects)
    )

    counters = {"accept": 0, "reject": 0, "refuse": 0, "check_c_only": 0}
    print()
    for obj in objects:
        result = scan.scan_object(obj, isa=args.isa, corpus_symbols=corpus_symbols)
        rel = os.path.relpath(obj, args.build_dir)
        _report_and_tally(rel, result, counters)

    return _finish(counters, len(objects))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--target", default="superslm")
    ap.add_argument("--isa", default="x86-64")
    args = ap.parse_args()

    if not os.path.isdir(args.build_dir):
        print("ERROR: build directory not found: {}".format(args.build_dir))
        return 2

    # T-2381 (Brunel), design Sec4.1/Sec5.4 (D-SLM5034/D-SLM5038): the
    # archive is the corpus whenever one exists at any candidate location --
    # never consulted alongside the object directory for the same build, per
    # this module's own docstring. The directory path below runs only when
    # no archive is found at all.
    archive_path = find_target_archive(args.build_dir, args.target)
    if archive_path is not None:
        return _scan_archive_corpus(archive_path, args)
    return _scan_object_directory_corpus(args)


if __name__ == "__main__":
    raise SystemExit(main())
