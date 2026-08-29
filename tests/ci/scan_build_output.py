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

THE ARCHIVE IS THE ONLY CORPUS -- NO FALLBACK (T-2385, Brunel, fold round 43,
D-SLM5100/D-SLM5101, correcting T-2381's own retained fallback found by
T-2382 finding S1/S4: the stated six-cell justification for keeping it
measured one cell at source, not six). `main()` searches
`find_target_archive`'s own four candidate locations, in order; if none
exists, this is an infrastructure failure -- the search is printed and the
job exits 2 -- and the object-directory scan is NEVER called. This design
has retired every other dual-corpus mechanism outright rather than keeping
the retired one as a named fallback (fold rounds 8, 10, 39, 40), and the
same reasoning applies here: a fallback means the two corpora can still
disagree on some future build layout neither author anticipated, which is
exactly the two-sources-of-truth shape the archive retargeting exists to
close. `find_target_objects` remains in the tree, unretired and
non-load-bearing, matching this design's own scrub convention for every
mechanism it retires (`run_fp_free_scan_real_corpus.py`, fold rounds
39/40): three read-only cells in
`tests/t2296-fp-free-open-red-suite/test_check_fp_free_scan.py` still call
it directly against a build that already has an archive. The driver
function that used to wrap it for `main()`'s own use,
`_scan_object_directory_corpus`, is REMOVED (T-2388): unlike
`find_target_objects`, it had acquired zero callers and zero test coverage
anywhere in the tree once `main()` stopped calling it (fold round 43), so
the retention precedent above -- a retired mechanism stays because
something still exercises it -- did not cover it; keeping it would have
been dead code with nothing to catch it going stale against `scan_object`'s
own signature. Design Sec7 dimension 11's restored fortieth population
(D-SLM5099/D-SLM5103) grades exactly this contract: a build directory with
no archive at any candidate location must exit 2 regardless of whether its
own `<target>.dir` is clean or floating-point-carrying, because the
corrected driver never opens the directory to find out which.

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
build directory was not found, no archive was found at any candidate
location, or the archive itself could not be read (an infrastructure
failure, kept distinct from a scan finding).
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


def _archive_candidate_paths(build_dir: str, target: str) -> list:
    """Every location `_ARCHIVE_CANDIDATES_TEMPLATE` names for `target`'s
    archive under `build_dir`, in search order -- the single source both
    `find_target_archive` (which searches them) and `main`'s missing-archive
    error path (which prints them) read from, so the list `main` prints can
    never diverge from the list it actually searched (T-2388, O1)."""
    return [os.path.join(build_dir, template.format(target=target))
            for template in _ARCHIVE_CANDIDATES_TEMPLATE]


def find_target_archive(build_dir: str, target: str):
    """The static-library archive CMake's own build produced for `target`,
    at the first of `_archive_candidate_paths`'s own locations that exists.
    Returns the path, or None if none of the candidates exists -- `main`
    reads None as an infrastructure failure and exits 2, printing every
    location searched; there is no object-directory fallback (module
    docstring, "THE ARCHIVE IS THE ONLY CORPUS -- NO FALLBACK",
    D-SLM5100/D-SLM5101)."""
    for candidate in _archive_candidate_paths(build_dir, target):
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

    T-2385 (Brunel, fold round 43, D-SLM5100/D-SLM5101): kept in the tree as
    a non-load-bearing utility -- `main()` never calls it -- see this
    module's own docstring ("THE ARCHIVE IS THE ONLY CORPUS -- NO
    FALLBACK"). Called directly by three read-only cells in
    `tests/t2296-fp-free-open-red-suite/test_check_fp_free_scan.py` that
    predate the archive-based corpus and run against a build that already
    has an archive, independent of this module's own dispatch. The driver
    function that used to wrap it for `main()`'s own use,
    `_scan_object_directory_corpus`, is removed (T-2388) -- once `main()`
    stopped calling it, it had no caller anywhere, unlike this function.
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
    running totals -- factored out of `_scan_archive_corpus` (T-2381,
    Brunel) so the report format is defined once. `_scan_object_directory_
    corpus`, the only other caller this was ever factored out for, is
    removed (T-2388, O2); this function is single-caller today."""
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
    """Prints the closing totals line and returns the job's own exit code
    (T-2381, Brunel). Single-caller today -- see `_report_and_tally`'s own
    docstring (T-2388, O2)."""
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--target", default="superslm")
    ap.add_argument("--isa", default="x86-64")
    args = ap.parse_args()

    if not os.path.isdir(args.build_dir):
        print("ERROR: build directory not found: {}".format(args.build_dir))
        return 2

    # T-2385 (Brunel, fold round 43, D-SLM5100/D-SLM5101): the archive is
    # the only corpus. `find_target_archive` searches its own four
    # candidate locations, in order; if none exists, this is an
    # infrastructure failure -- the search is printed and the job exits 2 --
    # and the object-directory scan is never called. Design Sec7 dimension
    # 11's restored fortieth population (D-SLM5103) grades exactly this: the
    # build directory's own content, clean or floating-point-carrying, has
    # zero effect on a missing-archive disposition.
    archive_path = find_target_archive(args.build_dir, args.target)
    if archive_path is None:
        print("ERROR: no archive found for target {!r} under {} -- "
              "searched:".format(args.target, args.build_dir))
        # T-2388, O1: the printed list is `find_target_archive`'s own search
        # list, read from the same `_archive_candidate_paths` helper rather
        # than rebuilt from `_ARCHIVE_CANDIDATES_TEMPLATE` here -- the two
        # can no longer diverge by construction.
        for candidate in _archive_candidate_paths(args.build_dir, args.target):
            print("       {}".format(candidate))
        print("       Nothing to scan is an infrastructure failure, never a pass.")
        return 2
    return _scan_archive_corpus(archive_path, args)


if __name__ == "__main__":
    raise SystemExit(main())
