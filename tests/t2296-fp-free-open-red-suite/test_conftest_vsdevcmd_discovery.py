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


def test_vswhere_version_range_is_passed_to_the_real_query():
    """T-2535 (Poirot 2945361-t2534-superslm-ci-green-confirmation2.md S-2): this file's own
    `-version` constraint (`_VSWHERE_VERSION_RANGE`, added by T-2533 S-3n -- commit `d1ee166`,
    `git log -S'"-version", _VSWHERE_VERSION_RANGE' -- tests/t2296-fp-free-open-red-suite/
    conftest.py`, confirmed by execution; T-2537 correction, Poirot 67bfcbf-t2536-superslm-
    ci-green-confirmation3.md M-3: this docstring said "T-2535 S-3n", misattributing the
    production change this cell pins to the round that added the PIN, not the round that
    added the argument) had no cell -- deleting the
    argument from `conftest.py`'s real query left this whole file at `6 passed` and the full red
    suite at `161 passed, 48 skipped, 1 xfailed`, undiscriminated, while the identical deletion in
    the sibling module `fp_scan_common.py` already failed
    `test_vswhere_version_range_is_passed_to_the_real_query` there -- the mirroring that added that
    cell ran one way. Exercised here directly, on a mocked `subprocess.run`, rather than only
    trusted by inspection -- the query passed to `vswhere.exe` must actually carry `-version` and
    this module's own range, not merely define the constant and never use it.
    """
    fake_result = subprocess.CompletedProcess(
        args=["vswhere.exe"], returncode=0, stdout="", stderr="")
    with mock.patch.object(_fixture_module.os.path, "exists", return_value=True), \
         mock.patch.object(_fixture_module.subprocess, "run", return_value=fake_result) as run:
        _fixture_module._vswhere_vsdevcmd_candidates()

    assert run.called, "vswhere.exe was found to exist but subprocess.run was never invoked"
    call_args = run.call_args[0][0]
    assert "-version" in call_args, (
        "the real vswhere invocation must pass -version -- got {}".format(call_args))
    version_idx = call_args.index("-version")
    assert call_args[version_idx + 1] == _fixture_module._VSWHERE_VERSION_RANGE, (
        "the real vswhere invocation must pass this module's own _VSWHERE_VERSION_RANGE "
        "immediately after -version -- got {}".format(call_args))


def test_buildtools_sort_key_is_adopted_at_the_real_call_site_not_only_in_the_helper():
    """T-2535 (Poirot 2945361-t2534-superslm-ci-green-confirmation2.md M-1): mirrors
    `test_fp_scan_common_vsdevcmd_discovery.py`'s own cell of the same shape -- the O-1 pin here
    (`test_path_has_segment_rejects_a_substring_that_is_not_a_whole_path_component`, above) calls
    `_path_has_segment` directly and never exercises the real call site that adopted it
    (`_vswhere_vsdevcmd_candidates`'s own sort key, line 170) -- reverting that ONE line back to
    the raw substring test `0 if "BuildTools" in c else 1` left every existing cell in this file
    green. `vswhere` reports an adversarial path whose LONGER component contains "BuildTools" as a
    mere substring (`BuildToolsBackup`) FIRST, and the real BuildTools install SECOND; under the
    raw substring test both tie at sort key 0 and the adversarial one stays first (stable sort);
    under the adopted `_path_has_segment` fix, only the real install matches the whole-segment
    test and must sort first regardless of report order.
    """
    adversarial = r"C:\BuildToolsBackup\Microsoft Visual Studio\2022\Enterprise"
    real_buildtools = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
    fake_stdout = "{}\n{}\n".format(adversarial, real_buildtools)  # adversarial reported FIRST
    fake_result = subprocess.CompletedProcess(
        args=["vswhere.exe"], returncode=0, stdout=fake_stdout, stderr="")

    def _fake_exists(path):
        return path == _fixture_module._VSWHERE_PATH

    with mock.patch.object(_fixture_module.os.path, "exists", side_effect=_fake_exists), \
         mock.patch.object(_fixture_module.subprocess, "run", return_value=fake_result):
        candidates = _fixture_module._vswhere_vsdevcmd_candidates()

    assert candidates[0] == os.path.join(real_buildtools, "Common7", "Tools", "VsDevCmd.bat"), (
        "the REAL call site (_vswhere_vsdevcmd_candidates's own sort key) must resolve the real "
        "BuildTools install first, not an adversarial path that merely contains the word as a "
        "substring of a longer component; got {}".format(candidates)
    )
