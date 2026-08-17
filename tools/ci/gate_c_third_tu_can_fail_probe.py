#!/usr/bin/env python3
"""T-2139 sixth confirmation review O1 (Claude/Poirot/5fbd04d-t2139-sixth-confirmation-review.md;
the fifth casebook's O1 asked for one of two closers -- a scripted mutate-compile-revert step, or
naming the residual on the record. This is the scripted closer.)

Gate C's THIRD TU (tools/t2139_gate_c_real_suite_side_check.cpp) is a MUST-ACCEPT construction --
it has no committed must-reject sibling, so HEAD carries no repeatable, re-runnable proof that it
can fail at all. The sixth casebook's reviewer hand-proved this twice already (once compiling a
mirrored governed append, once a one-sided append, both against scratch copies outside the repo)
and both times the reviewer had to rebuild the constructions by hand. This script commits that
proof as a repeatable battery step instead of leaving it to be re-derived by hand every round.

Mechanism: copy include/superslm/sslm_abi.h into a SCRATCH include tree under out/ (gitignored,
never touches the tracked file), append ONE new enumerator to the scratch copy's own
SSLM_STATUS_ENUM_LIST X-macro -- a ONE-SIDED append, since tests/t2130-g5-red-suite/sslm_g5.h is
left completely untouched -- then compile the real, unmodified
tools/t2139_gate_c_real_suite_side_check.cpp with the scratch include directory placed AHEAD of the
real one on the include path (/I<scratch>/include /Iinclude /Itests), so the compiler resolves
superslm/sslm_abi.h to the mutated scratch copy while every other header (including the real
tests/t2130-g5-red-suite/sslm_g5.h) resolves normally.

Required outcome: the compile FAILS, and the captured output names this construction's own
registry-top sentinel assertion (the "registry-top divergence" static_assert message,
tools/t2139_gate_c_real_suite_side_check.cpp:88-93) -- not merely any failure, the SAME
marker-text discrimination this round's S1/S3 fix requires of the CMake-side negatives. A clean
compile means the third TU's only can-it-fail evidence has regressed; a failure without the marker
means it failed for an unrelated reason. Either is reported as a hard failure. The scratch tree is
removed afterward regardless of outcome, and `git status --porcelain` is asserted empty both before
mutating and after cleanup, so this step can never leave the tracked tree dirty.
"""
import pathlib
import shutil
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
REAL_HEADER = REPO_ROOT / "include" / "superslm" / "sslm_abi.h"
SCRATCH_ROOT = REPO_ROOT / "out" / "t2139" / "gate_c_third_tu_mutation_probe"
SCRATCH_INCLUDE_DIR = SCRATCH_ROOT / "include"
SCRATCH_HEADER = SCRATCH_INCLUDE_DIR / "superslm" / "sslm_abi.h"
PROBE_SOURCE = REPO_ROOT / "tools" / "t2139_gate_c_real_suite_side_check.cpp"
PROBE_LOG = REPO_ROOT / "out" / "t2139" / "gate_c_third_tu_mutation_probe.log"
PROBE_EXE = REPO_ROOT / "out" / "t2139_gate_c_real_suite_side_check_mutation_probe.exe"

MARKER = "registry-top divergence"

# The new enumerator is inserted ahead of the closing X(SSLM_ALLOCATION_FAILED) line so the
# X-macro list stays syntactically valid (each line but the last needs its own trailing
# backslash-continuation); this exact anchor line is asserted present before mutating, so a future
# reshuffle of sslm_abi.h that removes or renames it fails LOUDLY here instead of silently mutating
# nothing and reporting a false pass.
ANCHOR = "    X(SSLM_ALLOCATION_FAILED) /* 25 */"
INSERTED = (
    "    X(T2139_MUTATION_PROBE_ONE_SIDED_APPEND) /* scratch-only, inserted by "
    "tools/ci/gate_c_third_tu_can_fail_probe.py -- never in the real tree */ \\\n" + ANCHOR
)


def git_status_clean():
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    return result.stdout.strip() == ""


def run_cl(extra_include_dirs):
    PROBE_EXE.parent.mkdir(parents=True, exist_ok=True)
    include_flags = []
    for d in extra_include_dirs:
        include_flags += ["/I", str(d)]
    cmd = (
        ["cl", "/nologo", "/std:c++20", "/O2", "/W4", "/EHsc"]
        + include_flags
        + [str(PROBE_SOURCE), "/Fo:" + str(PROBE_EXE.parent) + "\\",
           "/Fe:" + str(PROBE_EXE)]
    )
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def main():
    if not REAL_HEADER.is_file() or not PROBE_SOURCE.is_file():
        print("gate_c_third_tu_can_fail_probe.py: source files not found -- skipping (non-fatal)")
        return 0

    if not git_status_clean():
        print("gate_c_third_tu_can_fail_probe.py FAILED -- git status is not clean BEFORE this "
              "probe touched anything; refusing to run against a dirty tree (this probe's own "
              "revert-and-verify contract depends on knowing what state was clean).")
        return 1

    if SCRATCH_ROOT.exists():
        shutil.rmtree(SCRATCH_ROOT)

    try:
        header_text = REAL_HEADER.read_text(encoding="utf-8")
        if ANCHOR not in header_text:
            print(f"gate_c_third_tu_can_fail_probe.py FAILED -- expected anchor line not found "
                  f"in {REAL_HEADER}:")
            print(f"  {ANCHOR!r}")
            print("sslm_abi.h's SSLM_STATUS_ENUM_LIST shape has changed; update this probe's "
                  "anchor rather than let it silently mutate nothing.")
            return 1

        SCRATCH_HEADER.parent.mkdir(parents=True, exist_ok=True)
        mutated_text = header_text.replace(ANCHOR, INSERTED, 1)
        SCRATCH_HEADER.write_text(mutated_text, encoding="utf-8")

        # Scratch include dir placed FIRST so the compiler resolves "superslm/sslm_abi.h" to the
        # mutated copy; /Iinclude and /Itests still resolve everything else (including the REAL
        # tests/t2130-g5-red-suite/sslm_g5.h) normally -- this is the one-sided-append shape.
        returncode, output = run_cl([
            SCRATCH_INCLUDE_DIR,
            REPO_ROOT / "include",
            REPO_ROOT / "tests",
        ])
        PROBE_LOG.parent.mkdir(parents=True, exist_ok=True)
        PROBE_LOG.write_text(output, encoding="utf-8")

        if returncode == 0:
            print("gate_c_third_tu_can_fail_probe.py FAILED -- the one-sided-append mutation "
                  f"COMPILED CLEAN. Gate C's third TU has REGRESSED: it no longer catches a "
                  f"one-sided registry-top append. See {PROBE_LOG}.")
            return 1

        if MARKER not in output:
            print("gate_c_third_tu_can_fail_probe.py FAILED -- the mutation failed to compile, "
                  f"but NOT for its own assertion -- marker text {MARKER!r} not found in the "
                  f"build output. See {PROBE_LOG}.")
            return 1

        print("gate_c_third_tu_can_fail_probe.py: PASS -- the one-sided-append mutation failed "
              f"to compile for its own reason (marker {MARKER!r} confirmed present). "
              "Gate C's third TU is proven able to fail its registry-top sentinel assertion.")
        return 0
    finally:
        if SCRATCH_ROOT.exists():
            shutil.rmtree(SCRATCH_ROOT)
        if PROBE_EXE.exists():
            PROBE_EXE.unlink()
        obj = PROBE_EXE.parent / (PROBE_SOURCE.stem + ".obj")
        if obj.exists():
            obj.unlink()
        if not git_status_clean():
            print("gate_c_third_tu_can_fail_probe.py: WARNING -- git status is not clean AFTER "
                  "cleanup. Tracked-tree revert may have failed.")
            sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
