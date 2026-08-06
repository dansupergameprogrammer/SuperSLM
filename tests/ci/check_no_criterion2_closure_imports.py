"""CI source check: the vendored criterion-2 closure (dynamic_engine.py,
pipeline.py, silu_lut.py, constrain.py) is read only by the comparison
harness, never by the C++ engine's own implementation or by any test other
than the two the harness names (SuperSLM_S3a_WalkingSkeleton_Plan.md Sec11
S3.1c item 4; T-1523).

The property this guards: the C++ forward under test is authored from the
specification and independent test fixtures -- never by reading
dynamic_engine.py's or pipeline.py's source to backfill a value that makes a
C++ cell pass. A production file (or a fixture generator, or a future CI
test) that imports the wide package instead is a correlated oracle: the thing
being checked and the thing checking it would call the same function, which
is exactly what T-1320 excluded for the converter's emission logic and what
Sec11 S3.0 already forbids in the opposite direction (checking mlp_act
against an arbitrary-precision recomputation "written from the C10 row and
the S2.4 design rather than from the C++ source").

Mirrors tests/ci/check_no_forward_leaf_calls.py's own precedent: a text scan,
not an AST walk (over-inclusive on a comment or string literal naming a
banned module is an accepted false-positive, not a soundness gap -- the rule
is a ban, not a classifier), with a glob-derived input set rather than a
hardcoded file tuple, matching "the precedent's own weakness, not inherited"
convention that module itself names.

Two clauses, scanning two different populations for two different reasons:

  (i) No file under include/, src/, or tools/convert_model.py may import the
      wide closure -- production code acquiring a runtime dependency on the
      oracle it is checked against. Scanned via a directory glob for
      include/**/src/** plus the single named converter entry point (a text
      scan has no import-graph resolver; tools/convert_model.py is named
      explicitly rather than resolving its own transitive imports, matching
      this module's sibling's own directory-glob-not-graph-resolution
      convention).
  (ii) No file under tests/ (excluding tests/reference/superslm_spike/ itself
       -- the vendored closure's own internal imports of its sibling modules
       are the closure's own structure, not a caller to police) may import
       the wide closure either, EXCEPT an explicit allowlist naming exactly
       the producer and comparator the plan's own item 3 specifies as
       legitimate wide-package callers -- so a *future* fixture generator or
       CI test reaching for the wide package fails by construction, not only
       whichever files exist today.

Two import forms are banned, matching the codebase's own demonstrated import
idiom for this package (`from superslm_spike import X`,
tools/convert_model.py:48, dynamic_engine.py:47-49, every existing fixture
generator -- none of which uses the dotted-attribute form a naive scan
looking only for a "superslm_spike dot dynamic_engine" spelling would
require):

  - the dotted-attribute form: the package name, a literal dot, then one of
    the four banned submodule names -- e.g. package-dot-dynamic_engine.
  - the from-import form: `from superslm_spike import dynamic_engine` (and
    siblings), matched as a whole imported name so `pipeline_prob_width_
    ceiling` (a different name, the deliberately narrow excerpt's own module)
    is never matched by the `pipeline` token.

(This docstring itself deliberately never spells out the dotted form as a
literal, contiguous `superslm_spike.<module>` substring -- this file lives
under tests/ci/, which clause (ii) below scans, and is not on the allowlist;
a literal example here would trip this module's own guard the moment it
scanned itself, the same "accepted false-positive on a comment or string
literal" the text-scan convention allows for but that a checker's own source
should not gratuitously invite.)
"""

from __future__ import annotations

import glob
import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

# The wide closure's own submodule names -- banned as an import target from
# anywhere except the comparison harness itself.
BANNED_MODULES = ("dynamic_engine", "pipeline", "silu_lut", "constrain")

_DOTTED_RE = re.compile(
    r"superslm_spike\.(" + "|".join(re.escape(m) for m in BANNED_MODULES) + r")\b"
)
_FROM_IMPORT_RE = re.compile(r"from\s+superslm_spike\s+import\b(.*)$")
_FROM_IMPORT_START_RE = re.compile(r"from\s+superslm_spike\s+import\b")

# Clause (i): production code. tools/convert_model.py is named explicitly
# rather than resolved as an import graph -- see module docstring.
_DEFAULT_PRODUCTION_GLOBS = (
    "include/**/*.h",
    "include/**/*.hpp",
    "src/**/*.cpp",
    "src/**/*.h",
    "tools/convert_model.py",
)

# Clause (ii): the test tree, excluding the vendored closure's own directory
# (its internal sibling imports are its own structure, not a caller) via
# _EXCLUDED_DIR_PREFIX below.
_DEFAULT_TEST_GLOBS = (
    "tests/**/*.py",
)
_EXCLUDED_DIR_PREFIX = "tests/reference/superslm_spike/"

# The named, sized allowlist of legitimate wide-package callers under tests/.
# Item 3's own three named pieces: the producer, the comparator (the C++
# driver, tests/reference_parity_driver, has no Python import surface to
# scan). Plus two callers this build (T-1522's producer half) adds, under the
# SAME principle item 4 states -- Python-side tooling for the harness's own
# operation, never the C++ engine's implementation: the one-time, hand-run
# precompute script that renders the reference pack (not itself part of
# criterion 2's execution path -- its own output is what the producer reads,
# committed, at build/CI time), and the producer's own red-suite test file
# (which builds a small hermetic model via the same vendored `pipeline`
# module the producer itself imports, to test the producer's own
# serialization contract without a real checkpoint).
_DEFAULT_TEST_ALLOWLIST = (
    "tests/reference/run_criterion2_trace.py",
    "tests/reference/compare_criterion2_traces.py",
    "tests/reference/precompute_criterion2_prompt_pack.py",
    "tests/reference/test_run_criterion2_trace.py",
)


def _from_import_logical_lines(lines: list[str]) -> list[tuple[int, str]]:
    """Yields (1-based start line number, joined text) for every physical line
    that opens a `from superslm_spike import` statement, joining a trailing
    backslash continuation or an unbalanced opening parenthesis across
    following physical lines into one logical line.

    A single-physical-line scan never rejoins the PEP 8 / black parenthesized
    multi-line form (`from superslm_spike import (\\n    pipeline,\\n)`) or an
    explicit backslash continuation (`from superslm_spike import \\\\\\n
    dynamic_engine`) -- each splits the imported names across a line the
    from-import regex never reaches (D-SLM1058). Joining is scoped to lines
    that already open a from-superslm_spike-import statement, so an unrelated
    multi-line construct elsewhere in the file is never folded in by
    accident."""
    out: list[tuple[int, str]] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if _FROM_IMPORT_START_RE.search(line) is None:
            i += 1
            continue
        start = i + 1
        joined = line.rstrip("\n")
        while i + 1 < n and (
            joined.rstrip().endswith("\\") or joined.count("(") > joined.count(")")
        ):
            i += 1
            nxt = lines[i].rstrip("\n")
            if joined.rstrip().endswith("\\"):
                joined = joined.rstrip()[:-1].rstrip() + " " + nxt
            else:
                joined = joined + " " + nxt
        out.append((start, joined))
        i += 1
    return out


def find_banned_import_uses(path: str, modules: tuple[str, ...] = BANNED_MODULES) -> list[tuple[int, str]]:
    """Every (1-based line number, module name) hit in `path`, in file order --
    dotted-attribute and from-import forms both. A text scan, not an AST walk:
    a banned name inside a comment or a string literal is still reported; this
    is a deliberate over-approximation of a ban, not an attempt to classify
    intent (module docstring)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(lines, start=1):
        for m in _DOTTED_RE.finditer(line):
            hits.append((lineno, m.group(1)))
    for lineno, joined in _from_import_logical_lines(lines):
        m2 = _FROM_IMPORT_RE.search(joined)
        if not m2:
            continue
        names_part = m2.group(1)
        # A trailing `# comment` (e.g. `# noqa: E402`) or a `;`-separated
        # following statement is not part of the imported-names list --
        # strip either before splitting, so the last name is not glued to
        # what follows it (D-SLM1058).
        names_part = re.split(r"[#;]", names_part, maxsplit=1)[0]
        for raw_tok in re.split(r",", names_part):
            tok = raw_tok.strip().strip("()\\").strip()
            # Drop an "as alias" suffix so `from superslm_spike import
            # pipeline as p` is still recognized by its real name.
            tok = re.split(r"\s+as\s+", tok)[0].strip()
            if tok in modules:
                hits.append((lineno, tok))
    hits.sort(key=lambda h: h[0])
    return hits


def _glob_files(globs: tuple[str, ...], repo_root: str) -> list[str]:
    out: list[str] = []
    for g in globs:
        out.extend(glob.glob(os.path.join(repo_root, g), recursive=True))
    return sorted(set(out))


def scan_files(
    file_paths: list[str],
    allowlist: tuple[str, ...] = (),
    repo_root: str = _REPO_ROOT,
    modules: tuple[str, ...] = BANNED_MODULES,
) -> list[str]:
    """Scans every file in `file_paths` (absolute or repo-root-relative),
    skipping anything whose repo-root-relative path (normalized, so `/` and
    `\\` both match) is in `allowlist`. Returns one formatted failure string
    per hit, empty if clean."""
    allow_norm = {os.path.normpath(p) for p in allowlist}
    failures: list[str] = []
    for p in file_paths:
        abs_p = p if os.path.isabs(p) else os.path.join(repo_root, p)
        rel = os.path.relpath(abs_p, repo_root)
        rel_norm = os.path.normpath(rel)
        if rel_norm in allow_norm:
            continue
        if not os.path.isfile(abs_p):
            failures.append(f"{rel}: file not found at {abs_p}")
            continue
        for lineno, name in find_banned_import_uses(abs_p, modules):
            failures.append(
                f"{rel}:{lineno}: imports 'superslm_spike.{name}' -- the vendored criterion-2 "
                f"closure may only be read by the comparison harness (T-1523)"
            )
    return failures


def _is_excluded_test_path(rel_norm: str, excluded_prefix: str) -> bool:
    excluded_norm = os.path.normpath(excluded_prefix)
    return rel_norm == excluded_norm or rel_norm.startswith(excluded_norm + os.sep)


def scan_test_tree(
    file_paths: list[str],
    allowlist: tuple[str, ...] = _DEFAULT_TEST_ALLOWLIST,
    excluded_prefix: str = _EXCLUDED_DIR_PREFIX,
    repo_root: str = _REPO_ROOT,
    modules: tuple[str, ...] = BANNED_MODULES,
) -> list[str]:
    """Clause (ii): like scan_files, but additionally exempts every file
    under `excluded_prefix` -- the vendored closure's own directory, whose
    internal imports of its sibling modules are the closure's own structure,
    never a caller to police (module docstring)."""
    allow_norm = {os.path.normpath(p) for p in allowlist}
    failures: list[str] = []
    for p in file_paths:
        abs_p = p if os.path.isabs(p) else os.path.join(repo_root, p)
        rel = os.path.relpath(abs_p, repo_root)
        rel_norm = os.path.normpath(rel)
        if rel_norm in allow_norm or _is_excluded_test_path(rel_norm, excluded_prefix):
            continue
        if not os.path.isfile(abs_p):
            failures.append(f"{rel}: file not found at {abs_p}")
            continue
        for lineno, name in find_banned_import_uses(abs_p, modules):
            failures.append(
                f"{rel}:{lineno}: imports 'superslm_spike.{name}' -- only the producer/comparator "
                f"named in the allowlist may read the wide vendored closure (T-1523)"
            )
    return failures


def main(
    production_globs: tuple[str, ...] = _DEFAULT_PRODUCTION_GLOBS,
    test_globs: tuple[str, ...] = _DEFAULT_TEST_GLOBS,
    test_allowlist: tuple[str, ...] = _DEFAULT_TEST_ALLOWLIST,
    excluded_test_prefix: str = _EXCLUDED_DIR_PREFIX,
    repo_root: str = _REPO_ROOT,
) -> int:
    production_files = _glob_files(production_globs, repo_root)
    test_files = _glob_files(test_globs, repo_root)

    failures = scan_files(production_files, allowlist=(), repo_root=repo_root)
    failures += scan_test_tree(
        test_files, allowlist=test_allowlist, excluded_prefix=excluded_test_prefix, repo_root=repo_root
    )

    if failures:
        print("check_no_criterion2_closure_imports.py: FAILED", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(
        f"check_no_criterion2_closure_imports.py: OK -- {len(production_files)} production file(s) and "
        f"{len(test_files)} test file(s) scanned, zero unauthorized imports of the vendored closure"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
