#!/usr/bin/env python3
r"""T-2314 (a confirmation code review, independently verified by the conductor): the symmetric
problem to T-2139 Finding 3 (tools/ci/check_tools_have_build_recipe.py), for test SUITES rather
than tools/*.cpp files. tests/t2296-fp-free-open-red-suite/ -- this arc's entire pin, six cells
each mutation-proved to discriminate -- was referenced zero times by build.bat, CMakeLists.txt,
and .github/workflows/tests.yml: coverage that no gate runs. A hand sweep of the whole tests/
tree the same round found the population was three, not one (tests/t2018-slora-serial/,
tests/t2178-gpu-batched-prefill-red-suite/, and tests/t2296-fp-free-open-red-suite/ all had zero
references in build.bat), against four already-wired suites (t2112, t2130, t2138, t2199). This
script closes the class rather than the instance, the same way check_tools_have_build_recipe.py
already does for tools/*.cpp.

This script enumerates every SUITE DIRECTORY directly under tests/ -- a top-level directory whose
name matches the repo's own ticket-numbered suite convention, `t<digits>-<description>`, e.g.
t2018-slora-serial, t2296-fp-free-open-red-suite -- and fails, naming each one, that is referenced
by NEITHER of:
  1. build.bat                     -- the top-level MSVC quick-build script; every suite this
     repo has wired so far is wired here (a `call tests\<suite>\<entry script>.bat`, or a real,
     non-comment /I or #include reference into the suite's own header, matching this repo's own
     established convention -- see build.bat's own tests\t2138-abi-red-suite\run_green.bat and
     tests\t2199-damped-greedy-red-suite\build_green.bat call sites).
  2. CMakeLists.txt                -- the CMake build; a suite can be referenced here instead of
     or in addition to build.bat (tests/t2130-g5-red-suite/sslm_g5.h is compiled into a Gate C TU
     here).
  3. .github/workflows/tests.yml   -- the CI workflow. Checked for completeness even though, as
     measured this round, it is CMake-driven and does not currently reference any suite directory
     by name for ANY suite, wired or not: CI never invokes build.bat directly, so today this
     location contributes zero hits for both the orphaned suites and the already-wired ones. It
     stays a checked location because a suite wired directly into a CMake target CI does run
     would legitimately show up here, and the checker should recognize that wiring if it exists.

A suite named by NONE of the three locations is reported. tools/ci/tests_build_recipe_allowlist.txt
is the escape hatch for a genuine, justified exception -- same format and same discipline as
check_tools_have_build_recipe.py's own allowlist: each line is `<suite-dir-name>  # <reason>`; an
allowlisted suite is reported separately (still visible, never silently dropped) but does not fail
the check. Adding a new orphaned suite to the allowlist instead of wiring it is exactly the failure
mode this checker exists to catch, so every entry must carry its own reason, and "the suite's own
runtime cells need an artifact this environment does not have" is a reason to wire the suite's
compile/link gate and name the runtime gap in the record -- not a reason to allowlist the whole
suite out of the check.

Coverage, stated per StandardsDocument.md Sec4:
  - PATTERN coverage: the population is identified by the `^t\\d+-` directory-name convention,
    which matches every suite directory that exists in this tree today (100%) and none of the
    five fixed infrastructure directories under tests/ (ci, fixtures, manual, reference, support --
    0%, correctly excluded). A future suite directory that does NOT follow this numbering
    convention would not enter the population this checker enumerates, and so could be orphaned
    without this checker naming it. This is a real, stated narrowing, not a gap this script closes.
  - INPUT coverage: the scan is `tests/*/` -- one level deep directly under tests/. A repo-wide
    sweep this round (find across the whole tree for build_link_red.bat / build_green*.bat /
    build_red_suite.bat / build_liveness_red.bat and for any directory name containing "red" or
    "suite") found every suite-shaped thing lives directly under tests/, at depth 1, with none
    under src/, tools/, docs/, cmake/, or out/, and none nested two levels deep under tests/
    itself. This script's own scan does not itself reach a directory nested deeper than depth 1 or
    a suite living outside tests/ -- that reach was established once, this round, by the hand
    sweep above, not by this script, and a suite introduced at a different depth or location in the
    future would not be found by this checker's scan.

Non-fatal if this script cannot find its own repo root markers (mirrors check_tools_have_build_
recipe.py's own precedent: gracefully degrade rather than block machines without Python).
"""
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests"
ALLOWLIST_PATH = REPO_ROOT / "tools" / "ci" / "tests_build_recipe_allowlist.txt"

SUITE_DIR_PATTERN = re.compile(r"^t\d+-")


def load_allowlist():
    allow = {}
    if not ALLOWLIST_PATH.exists():
        return allow
    for line in ALLOWLIST_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "#" not in line:
            print(f"ALLOWLIST FORMAT ERROR: entry with no reason comment: {line!r}")
            sys.exit(1)
        name, reason = line.split("#", 1)
        name = name.strip()
        reason = reason.strip()
        if not reason:
            print(f"ALLOWLIST FORMAT ERROR: entry with an empty reason: {line!r}")
            sys.exit(1)
        allow[name] = reason
    return allow


def _strip_comment_lines(text, comment_prefixes):
    """Drop every line whose trimmed form begins with one of comment_prefixes before matching, so
    a suite name mentioned only in prose (a `rem` line, a `#` line) is not mistaken for a real
    recipe -- the same defect class check_tools_have_build_recipe.py's own comment-stripping fix
    closed (Claude/Poirot/ce5aff2-t2139-fifth-confirmation-review.md S1)."""
    kept = []
    for line in text.splitlines():
        stripped_lower = line.strip().lower()
        if any(stripped_lower.startswith(p) for p in comment_prefixes):
            continue
        kept.append(line)
    return "\n".join(kept)


def recipe_texts():
    texts = []
    build_bat = REPO_ROOT / "build.bat"
    if build_bat.exists():
        texts.append(_strip_comment_lines(
            build_bat.read_text(encoding="utf-8", errors="replace"), ("rem ", "::")))
    cmakelists = REPO_ROOT / "CMakeLists.txt"
    if cmakelists.exists():
        texts.append(_strip_comment_lines(
            cmakelists.read_text(encoding="utf-8", errors="replace"), ("#",)))
    tests_yml = REPO_ROOT / ".github" / "workflows" / "tests.yml"
    if tests_yml.exists():
        texts.append(_strip_comment_lines(
            tests_yml.read_text(encoding="utf-8", errors="replace"), ("#",)))
    return texts


def main():
    if not TESTS_DIR.is_dir():
        print("check_tests_have_build_recipe.py: tests/ not found -- skipping (non-fatal)")
        return 0

    allow = load_allowlist()
    texts = recipe_texts()

    suite_dirs = sorted(
        p.name for p in TESTS_DIR.iterdir()
        if p.is_dir() and SUITE_DIR_PATTERN.match(p.name)
    )
    missing = []
    for name in suite_dirs:
        pattern = re.escape(name)
        if any(re.search(pattern, t) for t in texts):
            continue
        missing.append(name)

    unallowlisted = [n for n in missing if n not in allow]
    allowlisted = [n for n in missing if n in allow]

    if allowlisted:
        print("check_tests_have_build_recipe.py: allowlisted (no recipe in the three checked "
              "locations, justified below):")
        for n in allowlisted:
            print(f"  {n}  -- {allow[n]}")

    if unallowlisted:
        print("check_tests_have_build_recipe.py FAILED -- tests/<suite>/ directories with NO "
              "build recipe in build.bat, CMakeLists.txt, or .github/workflows/tests.yml, and not "
              "in tools/ci/tests_build_recipe_allowlist.txt:")
        for n in unallowlisted:
            print(f"  {n}")
        print("Add a build recipe (wire the suite into build.bat, following the pattern the "
              "already-wired suites use), or add a justified allowlist entry "
              "(tools/ci/tests_build_recipe_allowlist.txt, `<suite-dir-name>  # <reason>`).")
        return 1

    print(f"check_tests_have_build_recipe.py: {len(suite_dirs)} tests/<suite>/ directories "
          f"checked, {len(allowlisted)} allowlisted, 0 unaccounted for.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
