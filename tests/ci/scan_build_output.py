"""Scan every object file a real CMake build produced for a target, and fail
if any of them contains floating-point arithmetic.

WHY THIS SHAPE. The question this answers is "does the shipped library contain
floating-point math?", and the only trustworthy list of what is in the shipped
library is the one the build system itself produced. Earlier versions of this
check derived that list by parsing `CMakeLists.txt` and then tried to prove the
derivation was complete; four successive attempts at that proof each had a hole,
because a derived list is only as good as the parser and the parser is what you
are trying to check. This module does not derive anything. It globs the object
files CMake actually emitted into the target's own object directory. If a source
is added to the target, its object appears there because the build put it there.

FAILS CLOSED. Finding zero objects is an error, not a pass -- "nothing to scan"
is what a mis-pointed build directory looks like, and it must never read as
clean.

THE GATE READS CHECKS (A)/(B) ALONE (T-2367, design Sec4.1/Sec5.5 fold round
39, D-SLM4985/D-SLM4996). `check_fp_free_scan.scan_object` runs checks (A),
(B), AND (C) and reports two per-symbol surfaces: `ab_verdicts` (checks
(A)/(B) only) and `verdicts` (the combined ab_accept-and-c_accept verdict,
unchanged). This driver's own pass/fail decision is a function of
`ab_verdicts` and `refuse` alone -- a call/tail-jmp edge check (C) cannot vet
(an external target absent from the vetted list, or a first-party indirect
tail jump) is reported below as a non-gating diagnostic, never as a build
failure. A genuine check-(A)/(B) violation (real floating-point arithmetic,
or an unvetted vector-register mnemonic) still fails the job exactly as
before -- check (C)'s retirement narrows what can fail the gate, it does not
widen what can pass it.

WHAT IT DOES NOT ANSWER. Scanning is per-object and per-ISA. This module reports
what `check_fp_free_scan.scan_object` returns for each object, on the ISA named
on the command line; it does not itself decide whether the scanner is correct.
The scanner's own per-symbol verdict is independently commissioned on five
production legs at a one-instruction floor
(`Claude/Popper/t2345-fp-scan-recommissioning-2026-08-27.md`).

Usage:
    python tests/ci/scan_build_output.py --build-dir <cmake-build-dir>
                                         [--target superslm] [--isa x86-64]

Exit codes: 0 every object's checks (A)/(B) accept, on every symbol, and no
object REFUSEs; 1 a checks-(A)/(B) REJECT or a REFUSE; 2 nothing to scan, or
the object directory was not found (an infrastructure failure, kept distinct
from a scan finding).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_fp_free_scan as scan  # noqa: E402

_OBJ_EXTS = (".obj", ".o")


def find_target_objects(build_dir: str, target: str) -> list:
    """Every object file CMake emitted for `target`, from the build tree itself.

    Both generator layouts are matched, because the CI matrix uses more than one:
      - MSBuild / Visual Studio: <build>/<target>.dir/<config>/*.obj
      - Ninja, Makefiles:        <build>/CMakeFiles/<target>.dir/**/*.o
    Nothing is parsed and nothing is derived; the directory name is the build
    system's own record of which objects belong to the target.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--target", default="superslm")
    ap.add_argument("--isa", default="x86-64")
    args = ap.parse_args()

    if not os.path.isdir(args.build_dir):
        print("ERROR: build directory not found: {}".format(args.build_dir))
        return 2

    objects = find_target_objects(args.build_dir, args.target)
    if not objects:
        print("ERROR: no object files found for target {!r} under {}".format(
            args.target, args.build_dir))
        print("       Nothing to scan is an infrastructure failure, never a pass.")
        return 2

    print("Scanning {} object(s) emitted by the build for target {!r} (isa={})".format(
        len(objects), args.target, args.isa))

    # Built once, from every object's own symbol table, so each scan sees the
    # whole in-corpus index rather than a growing prefix of it. The reader is
    # imported from the existing driver rather than restated here -- one
    # derivation, two callers.
    import run_fp_free_scan_real_corpus as driver  # noqa: E402
    corpus_symbols = frozenset().union(
        *(driver._read_function_symbol_names(o) for o in objects)
    )

    n_reject = 0          # gating: checks (A)/(B) alone
    n_refuse = 0          # gating: byte-accounting REFUSE
    n_accept = 0          # gating: checks (A)/(B) ACCEPT
    n_check_c_only = 0    # non-gating diagnostic: check (C) alone
    print()
    for obj in objects:
        result = scan.scan_object(obj, isa=args.isa, corpus_symbols=corpus_symbols)
        rel = os.path.relpath(obj, args.build_dir)
        if result.refuse:
            n_refuse += 1
            print("  REFUSE   {}  (unclassified_bytes={}, format={})".format(
                rel, result.unclassified_bytes, result.object_format))
            continue

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
        n_accept += sum(1 for v in result.ab_verdicts.values() if v == "ACCEPT")
        n_reject += len(ab_rejects)
        n_check_c_only += len(check_c_only)

        if ab_rejects:
            print("  REJECT   {}  ({} symbol(s), format={})".format(
                rel, len(ab_rejects), result.object_format))
            for s in ab_rejects:
                print("             {}".format(s))
        else:
            print("  clean    {}  ({} symbol(s), format={})".format(
                rel, len(result.ab_verdicts), result.object_format))
        if check_c_only:
            print("             (non-gating diagnostic: {} symbol(s) reject under "
                  "check (C) alone -- an unvetted external call target or an "
                  "unresolved indirect edge; checks (A)/(B) accept them, and "
                  "check (C) does not gate)".format(len(check_c_only)))
            for s in check_c_only:
                print("               [check-C-only] {}".format(s))

    print()
    print("Totals: {} object(s); {} symbol(s) ACCEPT, {} REJECT, {} object(s) REFUSE "
          "(checks (A)/(B), gating); {} symbol(s) reject under check (C) alone "
          "(non-gating diagnostic)".format(
              len(objects), n_accept, n_reject, n_refuse, n_check_c_only))

    if n_reject or n_refuse:
        print("FAIL: the scan did not come back clean.")
        return 1
    print("PASS: no floating-point arithmetic found in any object of this target "
          "(checks (A)/(B); check (C) is a non-gating diagnostic).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
