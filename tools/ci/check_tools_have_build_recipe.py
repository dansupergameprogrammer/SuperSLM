#!/usr/bin/env python3
"""T-2139 Finding 3 (Claude/Poirot/3bcbe43-t2139-fourth-confirmation-review.md S3), closing the
class rather than the instance: the T-2045/S5 scar this repo already names once
(build.bat's own C5 comment -- "the C5 harness had no committed build recipe -- the one
load-bearing result of the whole arc was not reproducible from HEAD") recurred a second, live time
this round (tools/t2139_f2_length_error_pin.cpp and tools/t2139_f2_catchall_construction_pin.cpp,
committed at beb2355, cited by a shipped header as their own proof, with no build recipe anywhere
in HEAD until this fix round).

This script enumerates every tools/*.cpp file and fails, naming each one, that is absent from
EVERY known build-recipe location in this repo:
  1. build.bat            -- the top-level MSVC quick-build script.
  2. CMakeLists.txt        -- the CMake build (add_executable(...) referencing the file).
  3. tests/*/build_link_red.bat -- each red suite's own link-red build script.
  4. tools/build_*.bat     -- this repo's OWN convention for a standalone tool's dedicated build
     script (tools/build_inspect.bat, tools/build_layer_trace.bat, tools/build_lora_switch_demo.bat,
     ... -- checked in ADDITION to the three locations this finding's own remedy named literally,
     because these are real, working, committed build recipes under this repo's own established
     convention; excluding them would make this checker report a false positive on every one of
     them, permanently, for tools this repo can already build from HEAD).

A tool named by NEITHER location is reported. tools/ci/tools_build_recipe_allowlist.txt is the
escape hatch for a genuine, justified exception -- each line is `<basename.cpp>  # <reason>`; an
allowlisted file is reported separately (still visible, never silently dropped) but does not fail
the check. Adding a NEW file to the allowlist without a build recipe is exactly the failure mode
this checker exists to catch, so every entry must carry its own reason.

Non-fatal if this script cannot find its own repo root markers (mirrors this repo's own
python-checker precedent, build.bat's tail: gracefully degrade rather than block machines without
Python, since this IS genuinely optional tooling, unlike the interface-probe C++ gate).
"""
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
ALLOWLIST_PATH = REPO_ROOT / "tools" / "ci" / "tools_build_recipe_allowlist.txt"


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
    """Drop every line whose trimmed form begins with one of comment_prefixes before matching.
    A basename mentioned only in a rem-line, a `::`-line, or a `#`-line is not a recipe -- it is
    prose about a recipe (Claude/Poirot/ce5aff2-t2139-fifth-confirmation-review.md S1: a `rem`
    line satisfied the old bare-substring match, which is precisely the shape the checker exists
    to close). Kept lines are joined back so a real recipe line elsewhere in the file still
    matches normally."""
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        stripped_lower = stripped.lower()
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
    for p in (REPO_ROOT / "tests").glob("*/build_link_red.bat"):
        texts.append(_strip_comment_lines(
            p.read_text(encoding="utf-8", errors="replace"), ("rem ", "::")))
    for p in TOOLS_DIR.glob("build_*.bat"):
        texts.append(_strip_comment_lines(
            p.read_text(encoding="utf-8", errors="replace"), ("rem ", "::")))
    return texts


def main():
    if not TOOLS_DIR.is_dir():
        print("check_tools_have_build_recipe.py: tools/ not found -- skipping (non-fatal)")
        return 0

    allow = load_allowlist()
    texts = recipe_texts()

    cpp_files = sorted(p.name for p in TOOLS_DIR.glob("*.cpp"))
    missing = []
    for name in cpp_files:
        # Word-boundary-ish match: the basename appears literally somewhere in a recipe file
        # (a source-file argument to cl, or a CMake add_executable(... tools/<name>) line).
        pattern = re.escape(name)
        if any(re.search(pattern, t) for t in texts):
            continue
        missing.append(name)

    unallowlisted = [n for n in missing if n not in allow]
    allowlisted = [n for n in missing if n in allow]

    if allowlisted:
        print("check_tools_have_build_recipe.py: allowlisted (no recipe in the four checked "
              "locations, justified below):")
        for n in allowlisted:
            print(f"  {n}  -- {allow[n]}")

    if unallowlisted:
        print("check_tools_have_build_recipe.py FAILED -- tools/*.cpp with NO build recipe in "
              "build.bat, CMakeLists.txt, tests/*/build_link_red.bat, or tools/build_*.bat, and "
              "not in tools/ci/tools_build_recipe_allowlist.txt:")
        for n in unallowlisted:
            print(f"  {n}")
        print("Add a build recipe, or add a justified allowlist entry "
              "(tools/ci/tools_build_recipe_allowlist.txt, `<name>.cpp  # <reason>`).")
        return 1

    print(f"check_tools_have_build_recipe.py: {len(cpp_files)} tools/*.cpp checked, "
          f"{len(allowlisted)} allowlisted, 0 unaccounted for.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
