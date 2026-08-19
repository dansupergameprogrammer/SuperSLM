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
directory including every forced-tier library at once).

**Required coverage widened a second time, fold round 5 (D-SLM3583).** Confirmation
review (T-2171, Finding 1) built an independently-found six-mutation population and found
the fold-round-4 matrix caught only its own control mutation. Two coverage dimensions,
per StandardsDocument.md §4: a **flag-text axis** the fold-round-4 matrix never widened
-- `_AVX_FAMILY_FLAG_PATTERNS` matched only the four literal `-mavx2`/`-mavx512*`/
`/arch:AVX2`/`/arch:AVX512*` spellings and was blind to the entire `-march=`/`-mtune=`
family (`-march=native`, `-march=x86-64-v3`/`-v4`, and any named microarchitecture that
selects AVX2/AVX-512, e.g. `-march=skylake-avx512`) -- the first thing a person reaches
for when they want this file to go faster; a sixth CMake spelling,
`string(APPEND|PREPEND CMAKE_CXX_FLAGS ...)`; and a third channel, an `env:` block's
`CXXFLAGS`/`CFLAGS` entry in a workflow YAML file (TU-wide for every compiler invocation
started under that environment, same as the existing `-DCMAKE_CXX_FLAGS=` CLI channel).
And an **input-set axis** -- `main()` read a fixed, hand-maintained two-path list
(the workflow file, one `build.bat`) behind an `os.path.isfile` guard that printed `OK`
and skipped silently on a missing/renamed path, while 15 files in this repository carry
a direct compiler invocation naming `src/matmul.cpp` (`tools/build_verify.bat` among
them) and the checker opened exactly one. Both axes are closed: `_AVX_FAMILY_FLAG_PATTERNS`
is widened, the CMakeLists.txt scan gains the sixth spelling, the workflow scan gains the
`env:` channel, and the checker now derives its own scanned-file population at check time
by walking the whole tree for direct-compiler-invocation sites naming `src/matmul.cpp`
(`scan_repo`) rather than reading one hand-maintained path -- a
file this walk finds but cannot open is a **checker failure** (`CheckerFailure`, distinct
stderr tag), not a silent skip; the `os.path.isfile`-guarded silent-skip path this
replaces is removed for the workflow file too (it is now opened unconditionally via
`_read_required`, which raises `CheckerFailure` rather than skipping on a missing path).

Mutation-provable (Curie's own "pin the documented claim" discipline,
~/.claude/personas/Implement/Curie/Curie.md): the fold-round-5 required-coverage floor was
the confirmation review's own independently-built six-member population. Confirmation
review (T-2174) built a population independently a second time and found the fold-round-5
widening regressed on three of six mutations it had just closed and never closed two
standing gaps -- because that widening, like the one before it, was validated only against
its own author's mutations.

**Required coverage widened a third time, fold round 6 (D-SLM3601/D-SLM3602).** Per
StandardsDocument.md §4, this is the third consecutive instance of the same coverage
failure and is closed structurally, not with a fourth pattern list: the validation floor
is now `tools/ci/matmul_avx_isolation_population.json`, a committed population
(deduplicated across all three casebooks' own findings) plus a clean-tree control and a
checker-failure cell, exercised by `tests/ci/test_matmul_avx_isolation_population.py`. Any
future widening of this checker reproduces green against the *whole* committed population
before it is trusted -- never a fresh subset the widening's own author chose. The specific
mechanism repair this fold makes, read at source rather than inferred from the symptom
(design §20.4): the CLI (`-DCMAKE_CXX_FLAGS=`) and environment (`CXXFLAGS:`/`CFLAGS:`)
channels were scoped to the one `--workflow` file and are now applied, unmodified, to
every file the tree-wide walk opens -- the direct-invocation channel alone keeps its
`matmul.cpp`-substring gate, since that is the only channel for which the gate is a
correct precondition. The direct-invocation first-token match now recognizes an
`@`-prefixed batch invocation and a compiler invoked via a build-variable reference
(`$(CXX)`/`$(CC)`/`${CXX}`/`${CC}`/`%CXX%`/`%CC%`) as well as a literal basename. A new
batch/shell environment-assignment channel (`set CXXFLAGS=...`/`export CXXFLAGS=...`/a
leading `CXXFLAGS=... <command>` prefix-assignment) is applied tree-wide on the same
no-`matmul.cpp`-gate rule. A workflow `env:` value using YAML block-scalar syntax
(`|`/`>`) is a `CheckerFailure`, not a silent non-match. Every mutation in the committed
population must flip this script's exit code from 0 to 1; the unmutated real tree, and a
scratch tree missing its required workflow file, must resolve exactly as the population's
own `"green"`/`"checker-failure"` cells specify. Executed and recorded per remedy (build
log).

**Confirmation review found five gaps of the same species, fold round 7 (T-2177).** The
population self-test could not tell a genuine hit from a crashed channel (fixed: two
`capsys` assertions on the checker's own stderr tags); `ctest -R` reported success on an
empty selection (fixed: `--no-tests=error`) and the CMake test registration's own
`else()` branch degraded silently (fixed: `message(FATAL_ERROR)`); the shell environment
channel truncated at the first space (fixed: `(.*)$`, matching its batch sibling); and the
CMakeLists channel -- the widest-blast-radius spelling -- was still scoped to the one
`--cmakelists` path while the CLI and environment channels had already gone tree-wide.
That last gap is closed here the same way `--workflow`'s channels were closed at fold
round 6: `find_cmakelists_hits` now runs against every `CMakeLists.txt`/`*.cmake` file the
tree-wide walk opens, on the same no-`matmul.cpp`-gate rule as the CLI, environment, and
batch-environment channels; `--cmakelists` is a required-file existence check only, exactly
like `--workflow`.
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

# Fold round 5 (D-SLM3583): directories a tree-wide search never needs to enter --
# VCS internals and build/output trees that hold generated artifacts, never committed
# source, and can be enormous (object files, .vcxproj files, coverage profiles). This
# is an efficiency and noise exclusion only; every one of these names is also either
# git-ignored (.gitignore: build/, out/, out-clang/, .worktrees/, __pycache__/, .vs/,
# .idea/) or a VCS-internal directory (`.git`) that cannot carry committed source, so
# excluding them cannot hide a real direct-compiler-invocation site.
_TREE_SEARCH_EXCLUDE_DIRS = {
    ".git", "build", "out", "out-clang", ".vs", ".idea", "__pycache__",
    ".worktrees", "node_modules",
}

# Fold round 6 (D-SLM3601): the committed validation population itself is not a
# build-configuration file -- it is this checker's own test data, and its JSON
# necessarily quotes each mutation's own channel spelling and flag text verbatim
# (that is the whole point of the fixture -- one member's content field alone
# reproduces a full CLI-channel assignment, quotes and all). Once §20.4 item 1
# makes the CLI and environment channels tree-wide with no `matmul.cpp` gate,
# those channels would otherwise match inside the fixture's own JSON strings and
# fail the checker on itself in the real, unmutated tree -- discovered by this
# fold's own green-cell check (`test_population_member[G0]`) the moment the
# widening landed. Excluded by exact repo-root-relative path, the same reasoning
# `_TREE_SEARCH_EXCLUDE_DIRS` already uses for generated/non-source content: this
# file cannot itself be a channel through which a TU-wide AVX flag reaches a
# compiler.
_TREE_SEARCH_EXCLUDE_FILES = {
    "tools/ci/matmul_avx_isolation_population.json",
}

# Every TU-wide spelling of "enable AVX2 or AVX-512 for this whole translation unit"
# across the four toolchains design §6.4 names (GCC, Clang, MSVC, ClangCL), plus the
# `-march=`/`-mtune=` family fold round 5 adds (D-SLM3583): `native`, the x86-64-v3/v4
# psABI microarchitecture levels, and any other `-march=`/`-mtune=` value. No named
# microarchitecture is treated as safe -- a maintained list of "which named
# microarchitectures imply AVX2/AVX-512 and which do not" is exactly the kind of list
# StandardsDocument.md §4 names as narrower than the real variation and due to rot the
# same way the four-literal-spelling list already did; every `-march=`/`-mtune=`
# assignment is a hit instead, on the same reasoning the AVX-isolation claim itself
# rests on -- a false positive on a genuinely AVX-free `-march=` value costs a human one
# look at a red CI job, and a false negative here is the exact SIGILL-on-a-consumer-
# machine class this whole guard exists to prevent. Matched case-sensitively (the GCC/
# Clang spellings; case-insensitively for the MSVC `/arch:` spellings, which MSVC
# accepts in any case) against literal flag text -- these are the exact strings a real
# compiler flag would use; a false negative from a spelling outside this list is a gap
# in this list, not in the check's own logic, routed here rather than invented.
_AVX_FAMILY_FLAG_PATTERNS = [
    re.compile(r"-mavx2\b"),
    re.compile(r"-mavx512\w*"),
    re.compile(r"/arch:AVX2\b", re.IGNORECASE),
    re.compile(r"/arch:AVX512\w*", re.IGNORECASE),
    re.compile(r"-m(?:arch|tune)=[A-Za-z0-9_.\-]+"),
]


def _contains_avx_flag(text: str) -> str | None:
    """Returns the matched flag text, or None."""
    for pattern in _AVX_FAMILY_FLAG_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(0)
    return None


class CheckerFailure(Exception):
    """Fold round 5 (D-SLM3583): a file this checker's own search names -- as a
    required input (CMakeLists.txt, the workflow YAML) or as a tree-wide discovered
    direct-compiler-invocation site -- but cannot open. Distinct from "the guard's
    coverage matrix found a TU-wide AVX flag": this means the checker could not
    complete its own scan at all, which is failure, not a green result and not a
    silent skip. Replaces the `os.path.isfile`-guarded silent-skip path that used to
    print `OK` when the workflow file or `build.bat` had moved or been renamed."""


def _read_required(path: str, purpose: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as exc:
        raise CheckerFailure(
            "could not open {} ({}): {}".format(purpose, path, exc)
        ) from exc


# --- Channel A: CMakeLists.txt, six spellings -------------------------------------

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

# Spelling 6 (fold round 5, D-SLM3583): string(APPEND|PREPEND CMAKE_CXX_FLAGS ...) --
# the confirmation review's own Finding 1 named this as "a sixth CMake spelling beside
# the five", independently blind to the pre-fold-5 checker: it assigns to the same
# directory-scoped, every-target variable spelling 5 already watches, via a different
# CMake command.
_STRING_APPEND_CMAKE_CXX_FLAGS_RE = re.compile(
    r"string\s*\(\s*(APPEND|PREPEND)\s+(CMAKE_CXX_FLAGS(?:_[A-Za-z0-9_]+)?)\s+([^)]*)\)",
    re.MULTILINE,
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

    for m in _STRING_APPEND_CMAKE_CXX_FLAGS_RE.finditer(cmakelists_text):
        flag = _contains_avx_flag(m.group(3))
        if flag:
            hits.append({
                "spelling": "string({} {})".format(m.group(1), m.group(2)),
                "target": "(directory-scoped, every target)",
                "flag": flag,
                "line": _line_of(m.start()),
            })

    return hits


# --- Channel B: command line, environment, and direct compiler invocation ---------

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


# Channel B, second form (fold round 5, D-SLM3583; widened tree-wide fold round 6,
# D-SLM3602): an `env:` block's `CXXFLAGS`/`CFLAGS` entry in a workflow YAML file --
# TU-wide for every compiler invocation started under that environment, the same
# reasoning as the CLI `-DCMAKE_CXX_FLAGS=` channel above, just set via the job/step
# environment instead of a command-line argument. Line-based: YAML's own `key: value`
# shape needs no CMake-argument parsing. Fold round 6 (D-SLM3602, T-2174 #13): applied
# to every file the tree walk opens, not only the one `--workflow` path -- a second
# workflow file (e.g. a `bench.yml` this checker was never told about) carries the
# same `env:` channel and does not need to mention `matmul.cpp` anywhere in its own
# text to be TU-wide for a compile of it started under that job's environment.
_ENV_CXXFLAGS_RE = re.compile(r"^\s*(CXXFLAGS|CFLAGS)\s*:\s*(.+)$", re.MULTILINE)

# Fold round 6 (D-SLM3602, design §20.4 item 4): a YAML block-scalar value (`|` or
# `>`, optionally with a chomping/indentation indicator, spanning the following
# indented lines) is outside what a line-based regex can parse -- `_ENV_CXXFLAGS_RE`
# above would capture only the bare `|`/`>` token as `value` and silently find no AVX
# flag in it, even when the block's own content carries one. Detecting this shape and
# raising `CheckerFailure` is the honest close: general YAML block-scalar parsing is
# not owed by this guard, and a silent non-match on a shape it cannot read is the same
# failure this whole fold is about.
_YAML_BLOCK_SCALAR_RE = re.compile(r"^[|>][+\-]?\d*\s*$")


def find_env_flag_hits(text: str, source_label: str) -> list[dict]:
    hits: list[dict] = []
    for m in _ENV_CXXFLAGS_RE.finditer(text):
        raw_value = m.group(2).strip()
        if _YAML_BLOCK_SCALAR_RE.match(raw_value):
            line_no = text.count("\n", 0, m.start()) + 1
            raise CheckerFailure(
                "{}:{}: {} uses YAML block-scalar syntax ({!r}), which this "
                "checker's line-based scan cannot parse -- rewrite as a plain "
                "scalar or state explicitly that it carries no AVX-family flag "
                "before this guard can pass".format(
                    source_label, line_no, m.group(1), raw_value
                )
            )
        value = raw_value.strip("\"'")
        flag = _contains_avx_flag(value)
        if flag:
            line_no = text.count("\n", 0, m.start()) + 1
            hits.append({
                "spelling": "{} (environment)".format(m.group(1)),
                "target": "(every compiler invocation run under this env)",
                "flag": flag,
                "line": line_no,
                "source": source_label,
            })
    return hits


# Channel B, fourth form (fold round 6, D-SLM3602, design §20.4 item 3): a batch or
# shell environment assignment naming CXXFLAGS/CFLAGS -- `set CXXFLAGS=...` (`.bat`/
# `.cmd`, case-insensitive, matching batch's own case-insensitivity; T-2174 #9) and
# `export CXXFLAGS=...` or a leading `CXXFLAGS=... <command>` prefix-assignment
# (`.sh`). Applied tree-wide on the same no-`matmul.cpp`-gate rule as the CLI and
# environment channels above -- an environment assignment does not need to co-occur
# with a `matmul.cpp` mention in the same file either.
_BATCH_SET_ENV_RE = re.compile(
    r"(?im)^\s*set\s+(CXXFLAGS|CFLAGS)\s*=\s*(.*)$"
)
# T-2177 Finding 3: this used to capture the value as `(\S*)`, which stops at
# the first whitespace and misses every quoted multi-flag value (`export
# CXXFLAGS="-O2 -mavx2"` -- caught only when the AVX flag happens to be
# written first). `(.*)$` matches the batch sibling above: the value is the
# rest of the line, quotes stripped by the caller below -- same shape, same
# rule, stated in both rather than one pinned to the other's wording.
_SHELL_ENV_ASSIGN_RE = re.compile(
    r"^\s*(?:export\s+)?(CXXFLAGS|CFLAGS)=(.*)$", re.MULTILINE
)


def find_batch_env_assign_hits(text: str, source_label: str) -> list[dict]:
    hits: list[dict] = []
    for pattern, spelling in (
        (_BATCH_SET_ENV_RE, "set {} (batch environment)"),
        (_SHELL_ENV_ASSIGN_RE, "{} (shell environment)"),
    ):
        for m in pattern.finditer(text):
            value = m.group(2).strip().strip("\"'")
            flag = _contains_avx_flag(value)
            if flag:
                line_no = text.count("\n", 0, m.start()) + 1
                hits.append({
                    "spelling": spelling.format(m.group(1)),
                    "target": "(every compiler invocation run under this environment)",
                    "flag": flag,
                    "line": line_no,
                    "source": source_label,
                })
    return hits


# Channel B, fifth form: a direct compiler invocation (no CMake in between) whose
# source-file list names matmul.cpp. Fold round 5 (D-SLM3583) generalizes this from a
# single hard-coded `build.bat` scan to every file the tree-wide search finds --
# `find_direct_invocation_hits` is applied uniformly to whatever
# `scan_repo` opens.
_COMPILER_BASENAMES = {
    "cl", "cl.exe",
    "clang", "clang.exe",
    "clang++", "clang++.exe",
    "clang-cl", "clang-cl.exe",
    "gcc", "gcc.exe",
    "g++", "g++.exe",
    "cc", "cc.exe",
}

# Fold round 6 (D-SLM3602): a compiler invoked via a build-variable reference rather
# than a literal basename -- $(CXX)/$(CC) (Makefile), ${CXX}/${CC} (shell), %CXX%/%CC%
# (batch). The checker cannot know a variable's resolved value, so any of these six
# spellings as the first token is ruled a hit-candidate exactly as a literal basename
# already is: a false positive on a CXX that happens not to be a compiler costs one
# look at a red CI job; a false negative on this exact shape is the SIGILL class this
# whole guard exists to prevent (T-2174 #15, a Makefile's `$(CXX) -march=native ...`).
_COMPILER_VAR_REFS = {
    "$(cxx)", "$(cc)", "${cxx}", "${cc}", "%cxx%", "%cc%",
}


def _is_compiler_invocation_first_token(line: str) -> bool:
    """A logical statement is a candidate direct compiler invocation when its own
    first line's first whitespace-delimited token's basename (stripped of any
    directory prefix and surrounding quotes) is a known compiler command, or the
    token itself is a build-variable reference to one -- matching the shape every
    direct invocation in this repository actually uses (`cl /nologo ...`, one
    command starting the line, arguments after it), plus batch's own `@`
    echo-suppression prefix (fold round 6, D-SLM3602: T-2174 #10, `@cl ...`) and a
    variable-referenced compiler command (T-2174 #11/#15: `%CXX% ...`, `$(CXX) ...`)."""
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("@"):
        stripped = stripped[1:].lstrip()
    if not stripped:
        return False
    first = stripped.split(None, 1)[0]
    stripped_first = first.strip("\"'")
    if stripped_first.lower() in _COMPILER_VAR_REFS:
        return True
    basename = os.path.basename(stripped_first).lower()
    return basename in _COMPILER_BASENAMES


def find_direct_invocation_hits(text: str, source_label: str) -> tuple[bool, list[dict]]:
    """Scans `text` for a logical statement -- a line beginning with a known
    compiler command, joined across any `^` (batch) or trailing `\\` (shell)
    line-continuation the same way this repository's own `.bat` recipes are
    written -- that names `matmul.cpp` in its source-file list.

    Returns `(found_matmul_invocation, avx_hits)`: the first tells the tree-wide
    caller this file belongs in the scanned population at all (it carries at least
    one direct compiler invocation naming matmul.cpp, whether or not that invocation
    also carries an AVX-family flag); the second is this file's own AVX-family-flag
    hits, in the same shape `find_cli_hits`/`find_env_flag_hits` report."""
    lines = text.split("\n")
    found_invocation = False
    avx_hits: list[dict] = []
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip("\r")
        if _is_compiler_invocation_first_token(raw):
            start_line = i + 1
            parts = [raw]
            cont = raw.rstrip()
            while cont.endswith("^") or cont.endswith("\\"):
                i += 1
                if i >= len(lines):
                    break
                raw = lines[i].rstrip("\r")
                parts.append(raw)
                cont = raw.rstrip()
            statement = "\n".join(parts)
            if "matmul.cpp" in statement:
                found_invocation = True
                flag = _contains_avx_flag(statement)
                if flag:
                    avx_hits.append({
                        "spelling": "direct compiler invocation",
                        "target": "(matmul.cpp named in this recipe's source list)",
                        "flag": flag,
                        "line": start_line,
                        "source": source_label,
                    })
        i += 1
    return found_invocation, avx_hits


def _display_path(path: str) -> str:
    """Relative-to-repo-root for a clean report line; falls back to the path as
    given when it is not under the repo root at all (e.g. a scratch copy used for
    this checker's own mutation-vitality test, off the repo's drive/tree)."""
    try:
        return os.path.relpath(path, _REPO_ROOT).replace("\\", "/")
    except ValueError:
        return path.replace("\\", "/")


def scan_repo(repo_root: str) -> tuple[list[str], list[dict]]:
    """Fold round 5 (D-SLM3583) input-set axis, widened fold round 6 (D-SLM3602,
    design §20.4 item 1): walks the whole tree under `repo_root` -- excluding only
    build-output and VCS-internal directories that can never carry committed source
    (`_TREE_SEARCH_EXCLUDE_DIRS`) -- and opens every remaining file. A file this walk
    finds but cannot open is a `CheckerFailure`, not a silent skip. A file that
    cannot be decoded as UTF-8 text is not a candidate and is not an error -- this is
    the tree's own binary/generated content (images, `.pdb`, etc.), not a script this
    guard's claim is about.

    Every channel predicate applies to every file this walk opens: the CLI, workflow-
    environment, and batch/shell-environment channels (`find_cli_hits`,
    `find_env_flag_hits`, `find_batch_env_assign_hits`) carry no `matmul.cpp` gate --
    a TU-wide flag set via any of those three channels can affect a compile of
    matmul.cpp with no mention of it anywhere in the same file (T-2174 #8/#9/#13/#15's
    exact shape: fold round 5's remedy scoped these two channels to the one
    `--workflow` path, which is why it regressed on `build.bat`'s own CLI/environment
    forms and never closed the standing `bench.yml`/Makefile-shaped misses). Only the
    direct-invocation channel keeps its `matmul.cpp`-substring gate, since a compiler
    recipe not naming `matmul.cpp` cannot compile it -- that gate is a correct
    precondition for that one channel alone.

    Returns `(scanned_files, avx_hits)`: every repo-root-relative path found to carry
    at least one direct compiler invocation naming `matmul.cpp` (sorted, for a stable
    report), and every AVX-family-flag hit found across all channels in all files.
    """
    scanned_files: list[str] = []
    avx_hits: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in _TREE_SEARCH_EXCLUDE_DIRS]
        for fname in sorted(filenames):
            full_path = os.path.join(dirpath, fname)
            # Exclusion is checked relative to THIS walk's own `repo_root` (not the
            # real repo's `_REPO_ROOT`, which `_display_path` reports against) --
            # the self-test's scratch tree has a different root than the real repo,
            # and the excluded fixture must be recognized there too.
            rel_to_root = os.path.relpath(full_path, repo_root).replace("\\", "/")
            if rel_to_root in _TREE_SEARCH_EXCLUDE_FILES:
                continue
            label = _display_path(full_path)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                continue
            except OSError as exc:
                raise CheckerFailure(
                    "the tree walk found {} but could not open it: {}".format(
                        label, exc
                    )
                ) from exc
            avx_hits.extend(find_cli_hits(text, label))
            avx_hits.extend(find_env_flag_hits(text, label))
            avx_hits.extend(find_batch_env_assign_hits(text, label))
            # T-2177 Finding 4: the CMakeLists channel was still scoped to the
            # one --cmakelists path while the CLI and environment channels
            # went tree-wide at fold round 6 -- the widest-blast-radius
            # spelling (add_compile_options) was left the narrowest-input-set
            # channel of the four. Applied to every CMakeLists.txt/*.cmake
            # file this walk opens, on the same no-matmul.cpp-gate rule as
            # the CLI/environment/batch-environment channels above: a
            # directory-scoped CMake flag assignment does not need to
            # co-occur with a matmul.cpp mention in the same file either.
            if fname == "CMakeLists.txt" or fname.endswith(".cmake"):
                cmake_hits = find_cmakelists_hits(text)
                for h in cmake_hits:
                    h["source"] = label
                avx_hits.extend(cmake_hits)
            if "matmul.cpp" in text:
                found, hits = find_direct_invocation_hits(text, label)
                if found:
                    scanned_files.append(label)
                avx_hits.extend(hits)
    return sorted(scanned_files), avx_hits


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cmakelists",
        default=_DEFAULT_CMAKELISTS,
        help="Path to the root CMakeLists.txt this checker requires to exist "
        "(default: the repo root's own). Fold round 7 (T-2177 Finding 4): its "
        "six CMake spellings are no longer scanned here specifically -- they "
        "run tree-wide, against every CMakeLists.txt/*.cmake file --repo-root "
        "finds, this one included, the same widening --workflow's channels "
        "got at fold round 6.",
    )
    parser.add_argument(
        "--workflow",
        default=_DEFAULT_WORKFLOW,
        help="Path to the CI workflow YAML this checker requires to exist (default: "
        ".github/workflows/tests.yml) -- a missing/renamed file is a CheckerFailure. "
        "Fold round 6 (D-SLM3602): its CLI/environment channels are no longer scanned "
        "here specifically -- they run tree-wide, against every file --repo-root "
        "finds, this one included.",
    )
    parser.add_argument(
        "--repo-root",
        default=_REPO_ROOT,
        help="Root of the tree-wide search for files carrying a direct compiler "
        "invocation naming src/matmul.cpp (default: this repo's own root). Fold "
        "round 5 (D-SLM3583): replaces the old --build-bat single-file scan -- "
        "every file under this root is a candidate, not one hand-maintained path.",
    )
    args = parser.parse_args(argv)

    try:
        # Required-file existence checks only -- fold round 6 (D-SLM3602) moved
        # the CLI/environment/batch-environment channels into the tree-wide
        # `scan_repo` walk below, and fold round 7 (T-2177 Finding 4) moves the
        # CMakeLists channel there too (applied to every CMakeLists.txt/*.cmake
        # file it opens, this one included when it is under --repo-root, which
        # every real and scratch-test invocation has it be). These two calls'
        # only job is preserving the checker-failure invariant on a
        # missing/renamed required file -- their returned text is intentionally
        # discarded so neither file is ever scanned twice.
        _read_required(args.cmakelists, "the CMakeLists.txt to scan")
        _read_required(args.workflow, "the CI workflow YAML to scan")

        scanned_files, hits = scan_repo(args.repo_root)
    except CheckerFailure as exc:
        sys.stderr.write(
            "check_matmul_avx_isolation: CHECKER FAILURE -- {}\n".format(exc)
        )
        return 1

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
          "compile flag found in CMakeLists.txt's six spellings, the workflow's "
          "command-line/environment channels, or any of the {} file(s) under {} "
          "carrying a direct compiler invocation naming src/matmul.cpp.".format(
              len(scanned_files), _display_path(args.repo_root)
          ))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
