"""T-2368 (Curie) -- session-scoped real-build fixture for the fp-free-open
red suite (test_check_fp_free_scan.py), closing D-SLM5008's corpus-path
class.

WHY THIS EXISTS. Three cells in that file read a directory named by a
literal string, "D:/SuperSLM/.worktrees/optb-build" -- configured once by
hand by the conductor from pre-restructure sources, versioned nowhere, and
created by no CI job. Measured consequence (D-SLM5008): a cell reading that
directory grades the scanner against a corpus that predates the remedy it
exists to observe, and the cell fails rather than skips when the directory
happens to be absent on a different machine, so the suite was red-by-
construction anywhere but the one machine it was hand-configured on. A
corpus cell derives its corpus from a build it causes, never from a path it
is told -- this fixture is that cause, built once per test session and
shared by every cell that needs the real corpus.

WHAT IT RUNS. The identical two-step recipe `.github/workflows/tests.yml`'s
own `fp-free-scan-gate` job runs against this exact checkout:

    cmake -B <dir>
    cmake --build <dir> --target superslm --config Release

producing the MSBuild/Visual-Studio-generator object layout
(`<dir>/superslm.dir/<config>/*.obj`) `tests/ci/scan_build_output.py`'s own
`find_target_objects` already knows how to walk, and the same build's own
`<dir>/Release/superslm.lib` -- the real ship gate's own production entry
point since T-2381/T-2385 reads the archive, never the directory (design
Sec4.1, D-SLM5100/D-SLM5101) -- so a cell built on this fixture is grading
the scanner against a real build laid out exactly as the gate reads it,
whichever of the two surfaces a given cell needs, not a second,
hand-derived layout.

RESOLUTION ORDER:

  1. SUPERSLM_FP_SCAN_BUILD_DIR, if set in the environment and naming an
     existing directory -- the mechanism a CI job (or a developer holding a
     build from a prior run of the identical recipe above) uses to skip
     paying for a second configure+build. Setting it is the caller's own
     assertion that the directory holds a build of the checkout under test;
     this fixture does not verify that beyond checking the directory exists,
     since a CI job wires this the same way it already wires `--build-dir`
     for the production gate itself.
  2. Otherwise, a fresh configure+build into
     out/t2368_fp_scan_corpus_build under this repository's own root --
     never under the OS temp directory (D-SLM4916: a CMake build tree under
     %TEMP% throws MSB8029) and always under this worktree, to keep every
     path the build and the scan both touch under the MAX_PATH-sensitive
     limit this suite's own environment note carries (D-SLM4893).

A toolchain genuinely absent (no VsDevCmd.bat found at either documented VS
2022 install location, or no cmake.exe at that install's own bundled CMake
path) or a configure/build that fails for any reason SKIPS every dependent
test with a stated reason -- it never fails them. No cell in this suite
tests whether `superslm` builds; the `windows-x64` and `fp-free-scan-gate`
CI jobs already do, and a machine that cannot produce this build is not a
machine where the scanner's own property is violated. Building once for
the whole session, rather than once per dependent cell, holds the cost of
proving this to a single configure+build regardless of how many cells
consult the corpus.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS_ROOT = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_TESTS_ROOT)

_ENV_OVERRIDE = "SUPERSLM_FP_SCAN_BUILD_DIR"
_BUILD_DIR = os.path.join(_REPO_ROOT, "out", "t2368_fp_scan_corpus_build")

# T-2368: BuildTools first, deliberately, and NOT shared with
# tests/ci/run_fp_free_scan_real_corpus.py's own `_find_vsdevcmd()` (which
# lists Community first) -- measured this session: on a machine with both
# VS 2022 installs present, invoking `cmake -B`/`cmake --build` from a
# plain shell (no VsDevCmd) lets CMake's own Visual-Studio-generator
# auto-detection pick WHICHEVER install it finds, which was Community here;
# building the identical source and CMakeLists.txt flags under Community's
# MSVC (19.33.31629) produced a `damped_greedy_phaseD.obj` with ZERO `orps`
# instructions (611 total), where the same build run through a VsDevCmd
# environment scoped to the BuildTools install (MSVC 19.44.35214) produced
# FOUR (594 total) -- matching the two existing reference builds
# (D:/SuperSLM/.worktrees/optb-build, D:/SuperSLM/.worktrees/t2367-bld),
# both of which were themselves configured against BuildTools
# (CMakeCache.txt's own CMAKE_GENERATOR_INSTANCE, confirmed this session).
# VsDevCmd.bat sets VSINSTALLDIR/VCINSTALLDIR, which CMake's Visual Studio
# generator honors to pin the exact instance -- invoking through it is not
# optional ceremony here, it is what makes this fixture's own build
# reproduce the toolset every other measurement in this arc was taken
# against, on a machine where more than one VS 2022 install is present.
_VSDEVCMD_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
)

# T-2371 (Brunel), D-SLM5018 O2: the two hardcoded candidates above are a
# fail-skip on any machine whose VS 2022 lives at a third location (a
# GitHub-hosted `windows-latest` runner's own VS 2022 Enterprise install is
# a documented example: `C:\Program Files\Microsoft Visual Studio\2022\
# Enterprise\...`, matching neither candidate). `vswhere.exe` ships at this
# fixed path with every VS 2022 installer regardless of which edition or
# install location was chosen (Microsoft's own documented contract for the
# tool), so it is queried first and the two hardcoded paths remain the
# fallback for a machine where `vswhere.exe` itself is absent (a bare
# BuildTools-only install predating the Installer's own vswhere bundling).
_VSWHERE_PATH = (
    r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
)


def _vswhere_vsdevcmd_candidates():
    """Every `VsDevCmd.bat` belonging to a VS 2022 instance `vswhere.exe`
    reports, BuildTools-named instances first -- preserving this fixture's
    own documented preference (BuildTools first, not shared with
    `run_fp_free_scan_real_corpus.py`'s own Community-first order) across
    however many instances are actually installed, rather than only the
    two this file's own author had on hand. Returns an empty list, never
    raises, if `vswhere.exe` is absent or reports nothing usable -- this is
    a widened SEARCH, not a required dependency."""
    if not os.path.exists(_VSWHERE_PATH):
        return []
    try:
        result = subprocess.run(
            [_VSWHERE_PATH, "-products", "*", "-property", "installationPath",
             "-nologo"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    install_paths = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    candidates = [
        os.path.join(p, "Common7", "Tools", "VsDevCmd.bat") for p in install_paths
    ]
    candidates.sort(key=lambda c: 0 if "BuildTools" in c else 1)
    return candidates


def _find_vsdevcmd():
    for c in _vswhere_vsdevcmd_candidates():
        if os.path.exists(c):
            return c
    for c in _VSDEVCMD_CANDIDATES:
        if os.path.exists(c):
            return c
    return None


def _cmake_from_vsdevcmd(vsdevcmd_path: str):
    """cmake.exe ships inside each VS 2022 install's own IDE bundle, at a
    fixed offset from VsDevCmd.bat's own directory
    (Common7/Tools/VsDevCmd.bat -> Common7/IDE/CommonExtensions/Microsoft/
    CMake/CMake/bin/cmake.exe) -- derived from the SAME VsDevCmd.bat path
    `_find_vsdevcmd()` just resolved, rather than a second, independently
    maintained absolute path."""
    common7 = os.path.dirname(os.path.dirname(vsdevcmd_path))
    candidate = os.path.join(
        common7, "IDE", "CommonExtensions", "Microsoft", "CMake", "CMake",
        "bin", "cmake.exe")
    return candidate if os.path.exists(candidate) else None


def _run_ci_recipe(vsdevcmd: str, cmake_exe: str, build_dir: str):
    """The exact two-step recipe .github/workflows/tests.yml's own
    fp-free-scan-gate job runs, executed through a VsDevCmd-initialized
    environment (this suite's own environment note: cmake is absent from
    PATH but present in both VS 2022 installs) so cl.exe/MSBuild resolve
    identically to how a real CI runner's own toolchain resolves them."""
    fd, bat_path = tempfile.mkstemp(suffix=".bat", prefix="t2368_corpus_build_")
    os.close(fd)
    try:
        with open(bat_path, "w") as f:
            f.write("@echo off\r\n")
            f.write('call "{}" -arch=x64 -no_logo\r\n'.format(vsdevcmd))
            f.write('cd /d "{}"\r\n'.format(_REPO_ROOT))
            f.write('"{}" -B "{}"\r\n'.format(cmake_exe, build_dir))
            f.write("if errorlevel 1 exit /b 1\r\n")
            f.write(
                '"{}" --build "{}" --target superslm --config Release\r\n'.format(
                    cmake_exe, build_dir))
            f.write("if errorlevel 1 exit /b 1\r\n")
        r = subprocess.run(["cmd", "/c", os.path.abspath(bat_path)],
                            capture_output=True, text=True, cwd=_REPO_ROOT)
        return r.returncode == 0, r.stdout, r.stderr
    finally:
        try:
            os.remove(bat_path)
        except OSError:
            pass


@pytest.fixture(scope="session")
def real_build_dir():
    """The real, session-scoped `superslm` CMake build directory every
    corpus-dependent cell in this suite shares. Returns the build directory
    path on success; skips every dependent test, with a stated reason, when
    no usable corpus can be produced in this environment. See this module's
    own docstring for the resolution order."""
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        if not os.path.isdir(override):
            pytest.skip(
                "{} is set to {!r}, which is not a directory -- unset it or "
                "point it at a real build of this checkout".format(
                    _ENV_OVERRIDE, override))
        return override

    vsdevcmd = _find_vsdevcmd()
    if vsdevcmd is None:
        pytest.skip(
            "no VsDevCmd.bat found at either VS 2022 install location -- "
            "cannot configure or build the real corpus in this environment")
    cmake_exe = _cmake_from_vsdevcmd(vsdevcmd)
    if cmake_exe is None:
        pytest.skip(
            "cmake.exe not found under the VS 2022 install bundling {} -- "
            "cannot configure or build the real corpus in this "
            "environment".format(vsdevcmd))

    ok, stdout, stderr = _run_ci_recipe(vsdevcmd, cmake_exe, _BUILD_DIR)
    if not ok:
        pytest.skip(
            "configuring/building the real superslm corpus failed in this "
            "environment (cmake -B <dir>, then cmake --build <dir> --target "
            "superslm --config Release); not treated as a test failure "
            "since no cell in this suite asserts that the build itself "
            "succeeds -- stdout/stderr follow:\n{}\n{}".format(stdout, stderr))
    return _BUILD_DIR
