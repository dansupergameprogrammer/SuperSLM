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

WHAT IT DOES NOT ANSWER. Scanning is per-object and per-ISA. This module reports
what `check_fp_free_scan.scan_object` returns for each object, on the ISA named
on the command line; it does not itself decide whether the scanner is correct.
The scanner's own per-symbol verdict is independently commissioned on five
production legs at a one-instruction floor
(`Claude/Popper/t2345-fp-scan-recommissioning-2026-08-27.md`).

Usage:
    python tests/ci/scan_build_output.py --build-dir <cmake-build-dir>
                                         [--target superslm] [--isa x86-64]

Exit codes: 0 every object scanned clean; 1 a REJECT or a REFUSE; 2 nothing to
scan, or the object directory was not found (an infrastructure failure, kept
distinct from a scan finding).
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

    n_reject = 0
    n_refuse = 0
    n_accept = 0
    print()
    for obj in objects:
        result = scan.scan_object(obj, isa=args.isa, corpus_symbols=corpus_symbols)
        rel = os.path.relpath(obj, args.build_dir)
        if result.refuse:
            n_refuse += 1
            print("  REFUSE   {}  (unclassified_bytes={}, format={})".format(
                rel, result.unclassified_bytes, result.object_format))
            continue
        rejects = sorted(s for s, v in result.verdicts.items() if v == "REJECT")
        n_accept += sum(1 for v in result.verdicts.values() if v == "ACCEPT")
        n_reject += len(rejects)
        if rejects:
            print("  REJECT   {}  ({} symbol(s), format={})".format(
                rel, len(rejects), result.object_format))
            for s in rejects:
                print("             {}".format(s))
        else:
            print("  clean    {}  ({} symbol(s), format={})".format(
                rel, len(result.verdicts), result.object_format))

    print()
    print("Totals: {} object(s); {} symbol(s) ACCEPT, {} REJECT, {} object(s) REFUSE".format(
        len(objects), n_accept, n_reject, n_refuse))

    if n_reject or n_refuse:
        print("FAIL: the scan did not come back clean.")
        return 1
    print("PASS: no floating-point arithmetic found in any object of this target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
