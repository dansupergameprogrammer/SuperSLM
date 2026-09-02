"""T-2371 (Brunel), D-SLM5018 O2 -- pin for `conftest.py`'s widened VS 2022
`VsDevCmd.bat` discovery.

WHY THIS FILE EXISTS. `real_build_dir` (conftest.py) used to search exactly
two hardcoded paths (BuildTools, Community) for a VS 2022 install, which
fail-skips every corpus-dependent cell in this suite on any machine whose VS
2022 lives at a third location -- a GitHub-hosted `windows-latest` runner's
own VS 2022 Enterprise install (`...\\2022\\Enterprise\\...`) is a documented
example matching neither candidate (Poirot, `c0e86be-t2369-fp-gate-build-
review.md` O2). The remedy queries `vswhere.exe`, which every VS 2022
installer bundles at a fixed, edition-independent path, for every installed
instance's own installation path, falling back to the original two
hardcoded candidates only when `vswhere.exe` itself is absent. This file
pins that widened discovery directly, independent of which VS 2022 edition
is actually installed on the machine running the suite.
"""
from __future__ import annotations

import os
import subprocess
import sys
from unittest import mock

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import conftest as _fixture_module  # noqa: E402


def test_vswhere_absent_returns_no_candidates():
    """Must-accept: a machine with no `vswhere.exe` at the documented
    Installer path gets an empty candidate list from the widened search,
    never an exception -- the widened search degrades to the original two
    hardcoded candidates, it does not replace them with a hard failure.
    """
    with mock.patch.object(_fixture_module.os.path, "exists", return_value=False):
        assert _fixture_module._vswhere_vsdevcmd_candidates() == []


def test_vswhere_reports_a_third_location_and_buildtools_sorts_first():
    """Must-accept, the exact shape O2 names: `vswhere.exe` reporting an
    installation path neither hardcoded candidate names (a stand-in for VS
    2022 Enterprise on a hosted CI runner) is surfaced as a real candidate,
    and when a BuildTools instance is also reported, it sorts first --
    preserving this fixture's own documented BuildTools-first preference
    (D-SLM5010/D-SLM5011: an unpinned toolset choice produces materially
    different object code) regardless of how many instances vswhere finds
    or in what order it reports them.
    """
    enterprise = r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise"
    buildtools = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
    fake_stdout = "{}\n{}\n".format(enterprise, buildtools)  # Enterprise reported FIRST
    fake_result = subprocess.CompletedProcess(
        args=["vswhere.exe"], returncode=0, stdout=fake_stdout, stderr="")

    def _fake_exists(path):
        return path == _fixture_module._VSWHERE_PATH

    with mock.patch.object(_fixture_module.os.path, "exists", side_effect=_fake_exists), \
         mock.patch.object(_fixture_module.subprocess, "run", return_value=fake_result) as run:
        candidates = _fixture_module._vswhere_vsdevcmd_candidates()

    assert run.called, "vswhere.exe was found to exist but subprocess.run was never invoked"
    assert candidates == [
        os.path.join(buildtools, "Common7", "Tools", "VsDevCmd.bat"),
        os.path.join(enterprise, "Common7", "Tools", "VsDevCmd.bat"),
    ], (
        "expected the BuildTools-derived VsDevCmd.bat first regardless of "
        "vswhere's own report order; got {}".format(candidates)
    )


def test_vswhere_nonzero_exit_returns_no_candidates():
    """Must-accept: a `vswhere.exe` that exists but exits non-zero (a
    corrupted install, or a future incompatible CLI) is treated the same as
    absent -- an empty list, never a crash that would take down the whole
    fixture and, with it, every corpus-dependent cell in this suite.
    """
    fake_result = subprocess.CompletedProcess(
        args=["vswhere.exe"], returncode=1, stdout="", stderr="boom")
    with mock.patch.object(_fixture_module.os.path, "exists", return_value=True), \
         mock.patch.object(_fixture_module.subprocess, "run", return_value=fake_result):
        assert _fixture_module._vswhere_vsdevcmd_candidates() == []


def test_find_vsdevcmd_prefers_a_vswhere_candidate_over_the_hardcoded_fallback():
    """Must-accept: when the widened search reports a real, existing
    `VsDevCmd.bat` that is NOT one of the two hardcoded candidates,
    `_find_vsdevcmd` returns it rather than falling through to (or past) the
    hardcoded pair -- proving the widened search is actually consulted
    first, not merely present in the module unused.
    """
    third_location = os.path.join(
        "C:\\", "fake-vs-2022-instance", "Common7", "Tools", "VsDevCmd.bat")
    with mock.patch.object(
        _fixture_module, "_vswhere_vsdevcmd_candidates", return_value=[third_location]
    ), mock.patch.object(
        _fixture_module.os.path, "exists",
        side_effect=lambda p: p == third_location
    ):
        assert _fixture_module._find_vsdevcmd() == third_location


def test_find_vsdevcmd_falls_back_to_hardcoded_candidates_when_vswhere_finds_nothing_real():
    """Must-accept: when the widened search returns paths that do not
    exist on disk (or returns nothing at all), `_find_vsdevcmd` still falls
    through to the original two hardcoded candidates -- the widened search
    is additive, it does not narrow what this fixture can find relative to
    before O2's remedy landed.
    """
    hardcoded_first = _fixture_module._VSDEVCMD_CANDIDATES[0]
    with mock.patch.object(
        _fixture_module, "_vswhere_vsdevcmd_candidates", return_value=[]
    ), mock.patch.object(
        _fixture_module.os.path, "exists",
        side_effect=lambda p: p == hardcoded_first
    ):
        assert _fixture_module._find_vsdevcmd() == hardcoded_first


def test_path_has_segment_rejects_a_substring_that_is_not_a_whole_path_component():
    """T-2533 (Poirot 4187739-t2532-superslm-ci-green-confirmation.md O-1): `_path_has_segment`
    (this file's own fix for the substring-match fragility O-1 named in the BuildTools-first sort
    key) must reject a path where the word appears only as part of a LONGER component -- e.g. a
    directory literally named `BuildToolsBackup` -- which the raw `"BuildTools" in c` test this
    file used before this round would have matched.
    """
    assert not _fixture_module._path_has_segment(
        r"C:\BuildToolsBackup\Microsoft Visual Studio\2022\Enterprise", "BuildTools"
    ), "a substring inside a longer path component must not match"
    real_path = os.path.join(
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
        "Common7", "Tools", "VsDevCmd.bat")
    assert _fixture_module._path_has_segment(real_path, "BuildTools"), (
        "the real edition segment must still match")
    upper_path = os.path.join(
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BUILDTOOLS",
        "Common7", "Tools", "VsDevCmd.bat")
    assert _fixture_module._path_has_segment(upper_path, "BuildTools"), (
        "the match must stay case-insensitive")
