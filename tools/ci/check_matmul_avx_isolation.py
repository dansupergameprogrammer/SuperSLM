#!/usr/bin/env python3
"""Persisted CI gate for T-2149's AVX-isolation claim (Claude/Vitruvius/
t2149-avx-kernel-design-2026-08-18.md §6.4, §10 dimension 7 item (d), D-SLM3510;
red-suite realization: Claude/Curie/t2158-t2149-avx-red-suite-2026-08-18.md).

The design's own claim: `DotRowAvx2`/`DotRowAvx512` (src/matmul.cpp) are the ONLY two
functions in the whole project that may compile with AVX2/AVX-512 instructions enabled,
via GCC/Clang's per-function `__attribute__((target("avx2")))` /
`__attribute__((target("avx512f,avx512bw")))` mechanism -- never via a translation-unit-
wide compiler flag (`-mavx2`, `-mavx512*` on GCC/Clang; `/arch:AVX2`, `/arch:AVX512*` on
MSVC/ClangCL). A TU-wide flag on the matmul target (or any target compiling
src/matmul.cpp) would let the compiler's own auto-vectorizer use AVX2/AVX-512
instructions ANYWHERE in that translation unit, silently reopening the SIGILL risk on
hardware without those instruction sets that the whole force-macro/dispatch story
(design §6) exists to prevent -- with nothing red to catch it, since the two
function-attributed sites would keep compiling fine either way.

**Required coverage widened, fold round 4 (D-SLM3570).** Code review (T-2170, Finding 4)
executed five mutations against the pre-remedy checker and found it caught two of five
real spellings (`target_compile_options` and its `COMPILE_LANGUAGE` generator-expression
form) and was structurally blind to the `-DCMAKE_CXX_FLAGS=`/`/DCMAKE_CXX_FLAGS=`
command-line channel three of this repo's own CI jobs already use
(`linux-x64-asan`, `linux-x64-tsan`, `branch-coverage`), and to `add_compile_options`
(directory-scoped -- the widest-blast-radius spelling, applying to every target in the
directory including every forced-tier library at once). This checker now closes a
spellings x channels matrix, both axes required before the guard is trusted:

Spellings (any one of these landing a TU-wide AVX-family flag is a hit):
  1. `target_compile_options(<target> ...)`             -- CMakeLists.txt
  2. same, wrapped in a `$<$<COMPILE_LANGUAGE:CXX>:...>` generator expression
  3. `add_compile_options(...)`                          -- CMakeLists.txt (directory-scoped)
  4. `set_target_properties(<target> PROPERTIES COMPILE_OPTIONS "...")` -- CMakeLists.txt
  5. `set(CMAKE_CXX_FLAGS[_<CONFIG>] "...")`             -- CMakeLists.txt

Channels (any one of these carrying an AVX-family flag is a hit, independent of spelling):
  A. The `CMakeLists.txt` file itself (spellings 1-5 above, scanned directly).
  B. A `cmake -B`/`cmake --build`/direct-compiler-invocation command line in
     `.github/workflows/tests.yml` or `build.bat` carrying `-DCMAKE_CXX_FLAGS=`/
     `/DCMAKE_CXX_FLAGS=` (a global CMake cache variable -- TU-wide for every target the
     configured project builds, which always includes src/matmul.cpp, so no
     target-name correlation is needed) or, in `build.bat`, a direct compiler
     invocation whose source-file list names `matmul.cpp` in the same logical
     (`^`-continued) statement as an AVX-family flag.

Mutation-provable (Curie's own "pin the documented claim" discipline,
~/.claude/personas/Implement/Curie/Curie.md): eight representative mutations span this
matrix (the five CMakeLists.txt spellings; the CLI CMAKE_CXX_FLAGS channel in both its
`-D` and `/D` forms, against `tests.yml`; and the build.bat direct-invocation channel)
-- CMAKE_CXX_FLAGS is the only spelling with a genuine CLI-settable form, which is why
the matrix is not a full 5x2 cross-product. Every mutation must flip this script's exit
code from 0 to 1; removing it again must flip it back, and the unmutated real tree must
stay green throughout. Executed and recorded, this fold, per remedy (build log).
"""
from __future__ import annotations

import argparse
import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_DEFAULT_CMAKELISTS = os.path.join(_REPO_ROOT, "CMakeLists.txt")
_DEFAULT_WORKFLOW = os.path.join(_REPO_ROOT, ".github", "workflows", "tests.yml")
_DEFAULT_BUILD_BAT = os.path.join(_REPO_ROOT, "build.bat")

# Every TU-wide spelling of "enable AVX2 or AVX-512 for this whole translation unit"
# across the four toolchains design §6.4 names (GCC, Clang, MSVC, ClangCL). Matched
# case-sensitively (case-insensitively for the /arch: spellings, which MSVC accepts in
# any case) against literal flag text -- these are the exact strings a real -m/-arch
# flag would use; a false negative from a novel spelling is a gap in this list, not in
# the check's own logic (StandardsDocument.md §4's "narrower than the real variation"
# residual), routed here rather than invented.
_AVX_FAMILY_FLAG_PATTERNS = [
    re.compile(r"-mavx2\b"),
    re.compile(r"-mavx512\w*"),
    re.compile(r"/arch:AVX2\b", re.IGNORECASE),
    re.compile(r"/arch:AVX512\w*", re.IGNORECASE),
]


def _contains_avx_flag(text: str) -> str | None:
    """Returns the matched flag text, or None."""
    for pattern in _AVX_FAMILY_FLAG_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(0)
    return None


# --- Channel A: CMakeLists.txt, five spellings -----------------------------------

# Spellings 1+2: target_compile_options(<target> ...) -- captures the full argument
# list up to the matching close-paren (this project never nests parens inside this
# call's argument list -- confirmed by reading every existing call before writing this
# pattern); a generator-expression-wrapped flag (spelling 2) is still substring-matched
# inside that same argument text, so one pattern covers both spellings.
_TARGET_COMPILE_OPTIONS_RE = re.compile(
    r"target_compile_options\s*\(\s*([A-Za-z0-9_]+)([^)]*)\)", re.MULTILINE
)

# Spelling 3: add_compile_options(...) -- directory-scoped, no target name to capture.
_ADD_COMPILE_OPTIONS_RE = re.compile(r"add_compile_options\s*\(([^)]*)\)", re.MULTILINE)

# Spelling 4: set_target_properties(<target> ... PROPERTIES ... COMPILE_OPTIONS "...")
_SET_TARGET_PROPERTIES_RE = re.compile(
    r"set_target_properties\s*\(\s*([A-Za-z0-9_]+)([^)]*)\)", re.MULTILINE | re.DOTALL
)

# Spelling 5: set(CMAKE_CXX_FLAGS ...) / set(CMAKE_CXX_FLAGS_<CONFIG> ...)
_SET_CMAKE_CXX_FLAGS_RE = re.compile(
    r"set\s*\(\s*(CMAKE_CXX_FLAGS(?:_[A-Za-z0-9_]+)?)\s+([^)]*)\)", re.MULTILINE
)


def find_cmakelists_hits(cmakelists_text: str) -> list[dict]:
    """Returns one dict per CMakeLists.txt construct carrying a TU-wide AVX-family
    flag: {"spelling": str, "target": str, "flag": str, "line": int}."""
    hits: list[dict] = []

    def _line_of(pos: int) -> int:
        return cmakelists_text.count("\n", 0, pos) + 1

    for m in _TARGET_COMPILE_OPTIONS_RE.finditer(cmakelists_text):
        flag = _contains_avx_flag(m.group(2))
        if flag:
            hits.append({
                "spelling": "target_compile_options",
                "target": m.group(1),
                "flag": flag,
                "line": _line_of(m.start()),
            })

    for m in _ADD_COMPILE_OPTIONS_RE.finditer(cmakelists_text):
        flag = _contains_avx_flag(m.group(1))
        if flag:
            hits.append({
                "spelling": "add_compile_options",
                "target": "(directory-scoped)",
                "flag": flag,
                "line": _line_of(m.start()),
            })

    for m in _SET_TARGET_PROPERTIES_RE.finditer(cmakelists_text):
        args_text = m.group(2)
        if "COMPILE_OPTIONS" not in args_text.upper():
            continue
        flag = _contains_avx_flag(args_text)
        if flag:
            hits.append({
                "spelling": "set_target_properties(...COMPILE_OPTIONS...)",
                "target": m.group(1),
                "flag": flag,
                "line": _line_of(m.start()),
            })

    for m in _SET_CMAKE_CXX_FLAGS_RE.finditer(cmakelists_text):
        flag = _contains_avx_flag(m.group(2))
        if flag:
            hits.append({
                "spelling": "set({})".format(m.group(1)),
                "target": "(directory-scoped, every target)",
                "flag": flag,
                "line": _line_of(m.start()),
            })

    return hits


# --- Channel B: command-line -DCMAKE_CXX_FLAGS= / direct compiler invocation -------

_CMAKE_CXX_FLAGS_CLI_RE = re.compile(
    r"[-/]DCMAKE_CXX_FLAGS(?:_[A-Za-z0-9_]+)?=(\"[^\"]*\"|'[^']*'|\S+)"
)


def find_cli_hits(text: str, source_label: str) -> list[dict]:
    """Channel B, first form: a `-DCMAKE_CXX_FLAGS[_<CONFIG>]=...` command-line
    argument (any CMake invocation) carrying an AVX-family flag -- TU-wide for the
    whole configured project, so no target correlation is needed."""
    hits: list[dict] = []
    for m in _CMAKE_CXX_FLAGS_CLI_RE.finditer(text):
        flag = _contains_avx_flag(m.group(1))
        if flag:
            line_no = text.count("\n", 0, m.start()) + 1
            hits.append({
                "spelling": "-DCMAKE_CXX_FLAGS= (command line)",
                "target": "(every target in the configured project)",
                "flag": flag,
                "line": line_no,
                "source": source_label,
            })
    return hits


def find_build_bat_direct_invocation_hits(build_bat_text: str) -> list[dict]:
    """Channel B, second form: build.bat's direct compiler-invocation recipes list
    source files across multiple `^`-continued lines. Join each continued statement
    into one logical line, and flag any statement whose source-file list names
    matmul.cpp while an AVX-family flag also appears in the same statement."""
    hits: list[dict] = []
    lines = build_bat_text.split("\n")
    i = 0
    stmt_start_line = 0
    stmt_parts: list[str] = []
    while i < len(lines):
        raw = lines[i]
        stripped = raw.rstrip("\r")
        if not stmt_parts:
            stmt_start_line = i + 1
        stmt_parts.append(stripped)
        if stripped.rstrip().endswith("^"):
            i += 1
            continue
        statement = "\n".join(stmt_parts)
        stmt_parts = []
        if "matmul.cpp" in statement:
            flag = _contains_avx_flag(statement)
            if flag:
                hits.append({
                    "spelling": "direct compiler invocation (build.bat)",
                    "target": "(matmul.cpp named in this recipe's source list)",
                    "flag": flag,
                    "line": stmt_start_line,
                    "source": "build.bat",
                })
        i += 1
    return hits


def _display_path(path: str) -> str:
    """Relative-to-repo-root for a clean report line; falls back to the path as
    given when it is not under the repo root at all (e.g. a scratch copy used for
    this checker's own mutation-vitality test, off the repo's drive/tree)."""
    try:
        return os.path.relpath(path, _REPO_ROOT).replace("\\", "/")
    except ValueError:
        return path.replace("\\", "/")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cmakelists",
        default=_DEFAULT_CMAKELISTS,
        help="Path to the CMakeLists.txt to scan (default: the repo root's own).",
    )
    parser.add_argument(
        "--workflow",
        default=_DEFAULT_WORKFLOW,
        help="Path to the CI workflow YAML to scan (default: .github/workflows/tests.yml).",
    )
    parser.add_argument(
        "--build-bat",
        default=_DEFAULT_BUILD_BAT,
        help="Path to build.bat to scan (default: the repo root's own).",
    )
    args = parser.parse_args(argv)

    with open(args.cmakelists, "r", encoding="utf-8") as f:
        cmakelists_text = f.read()
    hits = find_cmakelists_hits(cmakelists_text)
    for h in hits:
        h["source"] = _display_path(args.cmakelists)

    if os.path.isfile(args.workflow):
        with open(args.workflow, "r", encoding="utf-8") as f:
            workflow_text = f.read()
        hits.extend(find_cli_hits(
            workflow_text, _display_path(args.workflow)))

    if os.path.isfile(args.build_bat):
        with open(args.build_bat, "r", encoding="utf-8") as f:
            build_bat_text = f.read()
        hits.extend(find_cli_hits(
            build_bat_text, _display_path(args.build_bat)))
        hits.extend(find_build_bat_direct_invocation_hits(build_bat_text))

    if hits:
        sys.stderr.write(
            "check_matmul_avx_isolation: FAIL -- {} translation-unit-wide AVX-family "
            "compile flag(s) found. design §6.4's isolation claim is that ONLY "
            "DotRowAvx2/DotRowAvx512 (src/matmul.cpp) may use AVX2/AVX-512 instructions, "
            "via a per-function __attribute__((target(\"...\"))) -- never a TU-wide "
            "flag (any spelling, any channel), which would let the compiler use those "
            "instructions anywhere else in the same translation unit and reopen the "
            "SIGILL risk on hardware without them.\n".format(len(hits))
        )
        for hit in hits:
            sys.stderr.write(
                "  {}:{}: [{}] target '{}' sets '{}'\n".format(
                    hit["source"], hit["line"], hit["spelling"], hit["target"], hit["flag"]
                )
            )
        return 1

    print("check_matmul_avx_isolation: OK -- no translation-unit-wide AVX-family "
          "compile flag found in any of the five CMakeLists.txt spellings or the "
          "command-line/direct-invocation channels.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
