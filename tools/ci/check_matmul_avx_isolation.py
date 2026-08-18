#!/usr/bin/env python3
"""Persisted CI gate for T-2149's AVX-isolation claim (Claude/Vitruvius/
t2149-avx-kernel-design-2026-08-18.md §6.4, §10 dimension 7 item (d), D-SLM3510;
red-suite realization: Claude/Curie/t2158-t2149-avx-red-suite-2026-08-18.md).

The design's own claim: `DotRowAvx2`/`DotRowAvx512` (src/matmul.cpp) are the ONLY two
functions in the whole project that may compile with AVX2/AVX-512 instructions enabled,
via GCC/Clang's per-function `__attribute__((target("avx2")))` /
`__attribute__((target("avx512bw")))` mechanism -- never via a translation-unit-wide
compiler flag (`-mavx2`, `-mavx512*` on GCC/Clang; `/arch:AVX2`, `/arch:AVX512*` on
MSVC/ClangCL). A TU-wide flag on the matmul target (or any target compiling
src/matmul.cpp) would let the compiler's own auto-vectorizer use AVX2/AVX-512
instructions ANYWHERE in that translation unit, silently reopening the SIGILL risk on
hardware without those instruction sets that the whole force-macro/dispatch story
(design §6) exists to prevent -- with nothing red to catch it, since the two
function-attributed sites would keep compiling fine either way.

This is a CMakeLists.txt-level check, not a source-level one: the two attributed sites
live inside src/matmul.cpp as C++ function attributes, which never appear in
CMakeLists.txt at all -- so the check does not need to special-case them. It fails if
ANY `target_compile_options(...)` call anywhere in CMakeLists.txt contains one of the
AVX-family flags, regardless of which target -- a TU-wide flag on ANY target that
compiles src/matmul.cpp (today: superslm, superslm_test_injection, superslm_scalar_
forced, and every SIMD-tier-forced target T-2149 adds) would reopen the exact risk this
guard exists to catch, and a flag on a target that does NOT compile matmul.cpp is still
worth refusing outright: this project has never needed one (design §6.4, grounded via
web search: GCC/Clang require a TU-wide `-mavx2`/`-mavx512*` flag OR a per-function
attribute, and MSVC/ClangCL gate no intrinsic by `/arch:` at all), so a future TU-wide
AVX flag anywhere is itself a design-contradicting event this check should surface
rather than silently permit off to one side.

Mutation-provable (Curie's own "pin the documented claim" discipline,
~/.claude/personas/Implement/Curie/Curie.md): deliberately adding
`target_compile_options(superslm PRIVATE -mavx2)` (or the MSVC/ClangCL equivalent)
to a throwaway copy of CMakeLists.txt must flip this script's exit code from 0 to 1;
removing it again must flip it back. Executed and recorded in
Claude/Curie/t2158-t2149-avx-red-suite-2026-08-18.md -- this script's own vitality is
not asserted, it is run both ways.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_DEFAULT_CMAKELISTS = os.path.join(_REPO_ROOT, "CMakeLists.txt")

# Every TU-wide spelling of "enable AVX2 or AVX-512 for this whole translation unit"
# across the four toolchains design §6.4 names (GCC, Clang, MSVC, ClangCL). Matched
# case-sensitively against CMakeLists.txt's literal flag text -- these are the exact
# strings a real -m/-arch flag would use; a false negative from a novel spelling is a
# gap in this list, not in the check's own logic, and is exactly the "narrower than the
# real variation" failure mode StandardsDocument.md §4 names for a new structure --
# routed here as a residual, checked against real compiler documentation at authoring
# time rather than invented.
_AVX_FAMILY_FLAG_PATTERNS = [
    re.compile(r"-mavx2\b"),
    re.compile(r"-mavx512\w*"),
    re.compile(r"/arch:AVX2\b", re.IGNORECASE),
    re.compile(r"/arch:AVX512\w*", re.IGNORECASE),
]

# `target_compile_options(<target> ...)` calls, capturing the target name and the full
# argument list up to the matching close-paren (CMake calls never nest parens inside
# this argument list in this project -- confirmed by reading every existing
# target_compile_options call in CMakeLists.txt before writing this pattern).
_TARGET_COMPILE_OPTIONS_RE = re.compile(
    r"target_compile_options\s*\(\s*([A-Za-z0-9_]+)([^)]*)\)", re.MULTILINE
)


def find_avx_family_flags(cmakelists_text: str) -> list[dict]:
    """Returns one dict per `target_compile_options` call whose argument text
    contains a TU-wide AVX-family flag: {"target": str, "flag": str, "line": int}."""
    hits: list[dict] = []
    for m in _TARGET_COMPILE_OPTIONS_RE.finditer(cmakelists_text):
        target_name = m.group(1)
        args_text = m.group(2)
        for pattern in _AVX_FAMILY_FLAG_PATTERNS:
            flag_match = pattern.search(args_text)
            if flag_match:
                line_no = cmakelists_text.count("\n", 0, m.start()) + 1
                hits.append({
                    "target": target_name,
                    "flag": flag_match.group(0),
                    "line": line_no,
                })
    return hits


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cmakelists",
        default=_DEFAULT_CMAKELISTS,
        help="Path to the CMakeLists.txt to scan (default: the repo root's own).",
    )
    args = parser.parse_args(argv)

    with open(args.cmakelists, "r", encoding="utf-8") as f:
        text = f.read()

    hits = find_avx_family_flags(text)
    if hits:
        sys.stderr.write(
            "check_matmul_avx_isolation: FAIL -- {} translation-unit-wide AVX-family "
            "compile flag(s) found. design §6.4's isolation claim is that ONLY "
            "DotRowAvx2/DotRowAvx512 (src/matmul.cpp) may use AVX2/AVX-512 instructions, "
            "via a per-function __attribute__((target(\"...\"))) -- never a TU-wide "
            "flag, which would let the compiler use those instructions anywhere else in "
            "the same translation unit and reopen the SIGILL risk on hardware without "
            "them.\n".format(len(hits))
        )
        for hit in hits:
            sys.stderr.write(
                "  CMakeLists.txt:{}: target '{}' sets '{}'\n".format(
                    hit["line"], hit["target"], hit["flag"]
                )
            )
        return 1

    print("check_matmul_avx_isolation: OK -- no translation-unit-wide AVX-family "
          "compile flag found in any target_compile_options call.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
