"""T-2149 fold round 6 (D-SLM3601): the committed, structural validation
population for `tools/ci/check_matmul_avx_isolation.py`.

Three consecutive review rounds (T-2170 Finding 4, T-2171 Finding 1, T-2174
Finding 3) each found the checker's required coverage narrower than the real
variation, and each remedy was validated only against its own author's
mutations -- the specific failure mode StandardsDocument.md §4 names ("a new
structure is validated against an independently-found population before it
is trusted"). This suite retires that failure mode structurally: the
population at `tools/ci/matmul_avx_isolation_population.json` is the union
of every mutation named across all three casebooks, committed once, and any
future widening of the checker is required to reproduce green against the
*whole* population -- never a fresh subset the widening's own author chose.

Each population member materializes a scratch copy of this repository's
CMakeLists.txt / .github/workflows/tests.yml / tree (via `shutil.copytree`,
excluding the same VCS/build directories the checker itself never needs to
enter), applies the member's edit or new-file content, invokes the checker
in-process against the scratch tree, and asserts the exit code matches the
member's own `expect`:

- `"red"` -- the checker MUST fail (exit 1, a real hit reported).
- `"green"` -- the unmutated tree MUST pass (exit 0) -- the checker's own
  false-positive floor, checked alongside the true-positive floor so a
  widening that starts flagging real, unmutated project files is caught by
  this same suite.
- `"checker-failure"` -- removing/renaming the scratch tree's required
  workflow file MUST exit 1 via the `CheckerFailure` path (a distinct
  stderr tag), never a silent `OK`.

Wired into CMakeLists.txt as a second `add_test`,
`check_matmul_avx_isolation_population_vitality`, picked up by the new
`matmul-avx-isolation-guard` CI job (D-SLM3604) by name prefix alongside the
checker itself.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_POPULATION_PATH = os.path.join(
    _REPO_ROOT, "tools", "ci", "matmul_avx_isolation_population.json"
)
_CI_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools", "ci")

if _CI_TOOLS_DIR not in sys.path:
    sys.path.insert(0, _CI_TOOLS_DIR)

import check_matmul_avx_isolation as checker  # noqa: E402

# Same exclusion the checker's own tree-wide scan uses (its header, fold
# round 5, D-SLM3583) -- copying a build/output/VCS-internal directory into
# a scratch tree is unnecessary and, for `.git`, can be enormous.
_COPY_EXCLUDE_DIRS = {
    ".git", "build", "out", "out-clang", ".vs", ".idea", "__pycache__",
    ".worktrees", "node_modules",
}


def _load_population() -> list[dict]:
    with open(_POPULATION_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_POPULATION = _load_population()


def _ignore_scratch_dirs(_dir: str, names: list[str]) -> set[str]:
    return {n for n in names if n in _COPY_EXCLUDE_DIRS}


def _materialize_scratch_tree(tmp_path: str) -> str:
    """Copies this repository's CMakeLists.txt, .github/, tools/, and
    build.bat -- the only paths any population member's `target` or the
    checker's own required-file arguments ever reference -- into a fresh
    scratch tree, preserving their real relative paths so `--repo-root`
    against the scratch tree walks the same shape the real repo does."""
    scratch_root = os.path.join(tmp_path, "scratch")
    os.makedirs(scratch_root, exist_ok=True)
    for rel in ("CMakeLists.txt", "build.bat"):
        src = os.path.join(_REPO_ROOT, rel)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(scratch_root, rel))
    for rel_dir in (".github", "tools"):
        src_dir = os.path.join(_REPO_ROOT, rel_dir)
        dst_dir = os.path.join(scratch_root, rel_dir)
        if os.path.isdir(src_dir):
            shutil.copytree(src_dir, dst_dir, ignore=_ignore_scratch_dirs)
    # A minimal stand-in for src/matmul.cpp's existence -- population
    # members that name it in a compiler recipe do not require the file to
    # actually exist; the checker's own direct-invocation channel matches
    # the literal substring "matmul.cpp" in the recipe text, not a real
    # source file on disk.
    return scratch_root


def _apply_member(scratch_root: str, member: dict) -> None:
    kind = member["kind"]
    if kind == "none":
        return
    if kind == "remove-workflow":
        workflow_path = os.path.join(scratch_root, member["target"])
        if os.path.isfile(workflow_path):
            os.remove(workflow_path)
        return
    if kind == "new-file":
        target_path = os.path.join(scratch_root, member["target"])
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(member["content"])
        return
    if kind in ("cmakelists-edit", "workflow-edit", "existing-file-edit"):
        target_path = os.path.join(scratch_root, member["target"])
        with open(target_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        insert_at = member["insert_after_line"]
        insert_text = member["content"]
        if not insert_text.endswith("\n"):
            insert_text += "\n"
        new_lines = insert_text.splitlines(keepends=True)
        lines[insert_at:insert_at] = new_lines
        with open(target_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return
    raise ValueError("unknown population member kind: {!r}".format(kind))


def _run_checker_against_scratch(scratch_root: str) -> int:
    cmakelists = os.path.join(scratch_root, "CMakeLists.txt")
    workflow = os.path.join(scratch_root, ".github", "workflows", "tests.yml")
    argv = [
        "--cmakelists", cmakelists,
        "--workflow", workflow,
        "--repo-root", scratch_root,
    ]
    return checker.main(argv)


def _member_ids() -> list[str]:
    return [m["id"] for m in _POPULATION]


@pytest.mark.parametrize("member", _POPULATION, ids=_member_ids())
def test_population_member(member: dict, tmp_path) -> None:
    scratch_root = _materialize_scratch_tree(str(tmp_path))
    _apply_member(scratch_root, member)
    exit_code = _run_checker_against_scratch(scratch_root)

    expect = member["expect"]
    if expect == "green":
        assert exit_code == 0, (
            "population member {!r} expects the unmutated tree to pass "
            "(exit 0) but the checker exited {}".format(member["id"], exit_code)
        )
    elif expect in ("red", "checker-failure"):
        assert exit_code == 1, (
            "population member {!r} ({}) expects the checker to fail "
            "(exit 1) but it exited {} -- this is a checker coverage gap, "
            "not a passing population member".format(
                member["id"], member["source"], exit_code
            )
        )
    else:
        raise ValueError(
            "population member {!r} has unknown expect {!r}".format(
                member["id"], expect
            )
        )


def test_population_has_twenty_two_members():
    """Twenty distinct mutation members (M1-M16, N17-N20, deduplicated
    across the three casebooks per the fold round 6 derivation table,
    Claude/Vitruvius/t2149-avx-kernel-design-2026-08-18.md §20.3) plus the
    clean-tree control (G0) and the checker-failure cell (CF0) -- 22 total.
    A count check catches a member silently dropped from the fixture file
    without anyone noticing the parametrized test count shrank."""
    assert len(_POPULATION) == 22
    red_count = sum(1 for m in _POPULATION if m["expect"] == "red")
    assert red_count == 20


def test_population_ids_are_unique():
    ids = _member_ids()
    assert len(ids) == len(set(ids))
