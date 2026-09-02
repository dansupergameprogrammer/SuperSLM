"""T-2533 (Poirot 4187739-t2532-superslm-ci-green-confirmation.md O-3) -- pin for
`fp_scan_common.py`'s own widened VS 2022 `VsDevCmd.bat` discovery.

WHY THIS FILE EXISTS. `fp_scan_common.py`'s own `_vswhere_vsdevcmd_candidates()`/
`find_vsdevcmd()` -- the function `compile_cl`/`compile_ml64` actually call, per this suite's own
S-1n incident (a machine's installed MSVC edition deciding a population's ACCEPT/REJECT verdict)
-- had no dedicated pin of its own: its sibling in this same directory, `conftest.py`'s
`_vswhere_vsdevcmd_candidates()`/`_find_vsdevcmd()`, already carries five cells in
`test_conftest_vsdevcmd_discovery.py`, but this module's own, separately-maintained copy (T-2529's
own comment beside `_VSWHERE_PATH` names the mirror explicitly) did not. This file closes that gap
by mirroring those same five cells against this module's own functions, adjusted for the two
places this module's own discovery genuinely differs from `conftest.py`'s: it sorts
Community-first (not BuildTools-first -- `_vswhere_vsdevcmd_candidates`'s own docstring states
the reason), and its fallback tuple is the public `VSDEVCMD_CANDIDATES` (not `_VSDEVCMD_CANDIDATES`).
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

import fp_scan_common as _fixture_module  # noqa: E402


def test_vswhere_absent_returns_no_candidates():
    """Must-accept: a machine with no `vswhere.exe` at the documented Installer path gets an
    empty candidate list from the widened search, never an exception -- the widened search
    degrades to the original hardcoded candidates, it does not replace them with a hard failure.
    """
    with mock.patch.object(_fixture_module.os.path, "exists", return_value=False):
        assert _fixture_module._vswhere_vsdevcmd_candidates() == []


def test_vswhere_reports_a_third_location_and_community_sorts_first():
    """Must-accept, this module's own version of the shape `conftest.py`'s O2 cell names:
    `vswhere.exe` reporting an installation path neither hardcoded candidate names (a stand-in
    for VS 2022 Enterprise on a hosted CI runner) is surfaced as a real candidate, and when a
    Community instance is also reported, it sorts first -- preserving THIS module's own documented
    Community-first preference (unlike `conftest.py`'s BuildTools-first order for its own,
    separate corpus-build use) regardless of how many instances vswhere finds or in what order it
    reports them.
    """
    enterprise = r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise"
    community = r"C:\Program Files\Microsoft Visual Studio\2022\Community"
    fake_stdout = "{}\n{}\n".format(enterprise, community)  # Enterprise reported FIRST
    fake_result = subprocess.CompletedProcess(
        args=["vswhere.exe"], returncode=0, stdout=fake_stdout, stderr="")

    def _fake_exists(path):
        return path == _fixture_module._VSWHERE_PATH

    with mock.patch.object(_fixture_module.os.path, "exists", side_effect=_fake_exists), \
         mock.patch.object(_fixture_module.subprocess, "run", return_value=fake_result) as run:
        candidates = _fixture_module._vswhere_vsdevcmd_candidates()

    assert run.called, "vswhere.exe was found to exist but subprocess.run was never invoked"
    assert candidates == [
        os.path.join(community, "Common7", "Tools", "VsDevCmd.bat"),
        os.path.join(enterprise, "Common7", "Tools", "VsDevCmd.bat"),
    ], (
        "expected the Community-derived VsDevCmd.bat first regardless of "
        "vswhere's own report order; got {}".format(candidates)
    )


def test_vswhere_nonzero_exit_returns_no_candidates():
    """Must-accept: a `vswhere.exe` that exists but exits non-zero (a corrupted install, or a
    future incompatible CLI) is treated the same as absent -- an empty list, never a crash that
    would take down every cell in this suite that calls `compile_cl`/`compile_ml64`.
    """
    fake_result = subprocess.CompletedProcess(
        args=["vswhere.exe"], returncode=1, stdout="", stderr="boom")
    with mock.patch.object(_fixture_module.os.path, "exists", return_value=True), \
         mock.patch.object(_fixture_module.subprocess, "run", return_value=fake_result):
        assert _fixture_module._vswhere_vsdevcmd_candidates() == []


def test_find_vsdevcmd_prefers_a_vswhere_candidate_over_the_hardcoded_fallback():
    """Must-accept: when the widened search reports a real, existing `VsDevCmd.bat` that is NOT
    one of the hardcoded `VSDEVCMD_CANDIDATES`, `find_vsdevcmd()` returns it rather than falling
    through to (or past) the hardcoded pair -- proving the widened search is actually consulted
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
        assert _fixture_module.find_vsdevcmd() == third_location


def test_find_vsdevcmd_falls_back_to_hardcoded_candidates_when_vswhere_finds_nothing_real():
    """Must-accept: when the widened search returns paths that do not exist on disk (or returns
    nothing at all), `find_vsdevcmd()` still falls through to the original hardcoded
    `VSDEVCMD_CANDIDATES` -- the widened search is additive, it does not narrow what this module
    can find relative to before its own T-2529 remedy landed.
    """
    hardcoded_first = _fixture_module.VSDEVCMD_CANDIDATES[0]
    with mock.patch.object(
        _fixture_module, "_vswhere_vsdevcmd_candidates", return_value=[]
    ), mock.patch.object(
        _fixture_module.os.path, "exists",
        side_effect=lambda p: p == hardcoded_first
    ):
        assert _fixture_module.find_vsdevcmd() == hardcoded_first


def test_vswhere_version_range_is_passed_to_the_real_query():
    """T-2533 (O-3, and matching S-1n's own root cause): this module's own `-version`
    constraint (`_VSWHERE_VERSION_RANGE`, T-2531 S-3) is exercised here directly, on a mocked
    `subprocess.run`, rather than only being trusted by inspection -- the query passed to
    `vswhere.exe` must actually carry `-version` and this module's own range, not merely define
    the constant and never use it.
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


def test_path_has_segment_rejects_a_substring_that_is_not_a_whole_path_component():
    """T-2533 (O-1): `_path_has_segment` (the fix for the substring-match fragility O-1 named
    in this module's own edition-sort key) must reject a path where the word appears only as
    part of a LONGER component -- e.g. a Windows account or a relocated directory literally
    named `CommunityUser` or `BuildToolsBackup` -- which the raw `"Community" in p` /
    `"BuildTools" in c` tests this module and `conftest.py` used before this round would both
    have matched.
    """
    assert not _fixture_module._path_has_segment(
        r"C:\Users\CommunityUser\Microsoft Visual Studio\2022\Enterprise", "Community"
    ), "a substring inside a longer path component must not match"
    assert _fixture_module._path_has_segment(
        r"C:\Program Files\Microsoft Visual Studio\2022\Community", "Community"
    ), "the real edition segment must still match"
    assert _fixture_module._path_has_segment(
        r"C:\Program Files\Microsoft Visual Studio\2022\COMMUNITY", "Community"
    ), "the match must stay case-insensitive"


def test_community_sort_key_is_adopted_at_the_real_call_site_not_only_in_the_helper():
    """T-2535 (Poirot 2945361-t2534-superslm-ci-green-confirmation2.md M-1): the O-1 pin
    (`test_path_has_segment_rejects_a_substring_that_is_not_a_whole_path_component`, above) calls
    `_path_has_segment` directly and never exercises the real call site that adopted it
    (`_vswhere_vsdevcmd_candidates`'s own sort key, line 117) -- reverting that ONE line back to
    the raw substring test `0 if "Community" in p else 1` left every existing cell in this file
    green, because none of them constructs a path where the raw substring test and
    `_path_has_segment` disagree. This cell does: `vswhere` reports an adversarial path whose
    LONGER component contains "Community" as a mere substring (`CommunityUser`, a stand-in for a
    Windows account or relocated directory carrying that word) FIRST, and the real Community
    install SECOND -- under the raw substring test both tie at sort key 0 and Python's stable sort
    keeps the adversarial one first (the wrong install resolved silently); under the adopted
    `_path_has_segment` fix, only the real install matches the whole-segment test, and it must
    sort first regardless of report order.
    """
    adversarial = r"C:\Users\CommunityUser\Microsoft Visual Studio\2022\Enterprise"
    real_community = r"C:\Program Files\Microsoft Visual Studio\2022\Community"
    fake_stdout = "{}\n{}\n".format(adversarial, real_community)  # adversarial reported FIRST
    fake_result = subprocess.CompletedProcess(
        args=["vswhere.exe"], returncode=0, stdout=fake_stdout, stderr="")

    def _fake_exists(path):
        return path == _fixture_module._VSWHERE_PATH

    with mock.patch.object(_fixture_module.os.path, "exists", side_effect=_fake_exists), \
         mock.patch.object(_fixture_module.subprocess, "run", return_value=fake_result):
        candidates = _fixture_module._vswhere_vsdevcmd_candidates()

    assert candidates[0] == os.path.join(real_community, "Common7", "Tools", "VsDevCmd.bat"), (
        "the REAL call site (_vswhere_vsdevcmd_candidates's own sort key) must resolve the real "
        "Community install first, not an adversarial path that merely contains the word as a "
        "substring of a longer component; got {}".format(candidates)
    )


def test_find_vsdevcmd_honours_the_env_var_when_set_and_it_exists():
    """T-2555: run 33648618208's own fp-free-scan-gate job errored 22 red-suite cells with "no
    VsDevCmd.bat found" -- windows-latest carries VS 2022 Enterprise only, and neither this
    module's own Community-first sort nor conftest.py's BuildTools-first one matches it. The
    job's own workflow now resolves VsDevCmd.bat itself and exports it as SUPERSLM_VSDEVCMD;
    `find_vsdevcmd` must honour it BEFORE any discovery. Mocks the environment and `os.path.exists`
    directly -- `_vswhere_vsdevcmd_candidates`/`subprocess.run` are never touched when the
    variable resolves, which this cell also proves via a patch that would raise if called.
    """
    fake_path = r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat"
    with mock.patch.dict(os.environ, {_fixture_module._VSDEVCMD_ENV_VAR: fake_path}), \
         mock.patch.object(_fixture_module.os.path, "exists", return_value=True), \
         mock.patch.object(
             _fixture_module, "_vswhere_vsdevcmd_candidates",
             side_effect=AssertionError("discovery must not run when the env var resolves")):
        assert _fixture_module.find_vsdevcmd() == fake_path, (
            "find_vsdevcmd must return the env var's own path directly when it exists, without "
            "falling back to discovery"
        )


def test_find_vsdevcmd_falls_back_to_discovery_when_env_var_is_unset():
    """T-2555: the env var is a hosted-CI-only mechanism (this job's own new workflow step) --
    a local run, or any environment that never set SUPERSLM_VSDEVCMD, must still fall back to
    discovery exactly as before this round.
    """
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(_fixture_module._VSDEVCMD_ENV_VAR, None)
        with mock.patch.object(
            _fixture_module, "_vswhere_vsdevcmd_candidates", return_value=[]
        ), mock.patch.object(
            _fixture_module.os.path, "exists",
            side_effect=lambda p: p == _fixture_module.VSDEVCMD_CANDIDATES[0]
        ):
            assert _fixture_module.find_vsdevcmd() == _fixture_module.VSDEVCMD_CANDIDATES[0], (
                "find_vsdevcmd must fall back to discovery (here, the hardcoded fallback) when "
                "the env var is unset"
            )


def test_find_vsdevcmd_falls_back_to_discovery_when_env_var_path_does_not_exist():
    """T-2555: a SET but stale/wrong env var (e.g. a future workflow edit that exports a typo'd
    path) must not be trusted blindly -- find_vsdevcmd checks the path actually exists before
    returning it, falling back to discovery exactly as an unset variable would, rather than
    handing a caller a VsDevCmd.bat path that does not resolve to a real file.
    """
    fake_path = r"C:\nonexistent\VsDevCmd.bat"
    with mock.patch.dict(os.environ, {_fixture_module._VSDEVCMD_ENV_VAR: fake_path}), \
         mock.patch.object(
             _fixture_module, "_vswhere_vsdevcmd_candidates", return_value=[]
         ), mock.patch.object(
             _fixture_module.os.path, "exists",
             side_effect=lambda p: p != fake_path and p == _fixture_module.VSDEVCMD_CANDIDATES[0]
         ):
        assert _fixture_module.find_vsdevcmd() == _fixture_module.VSDEVCMD_CANDIDATES[0], (
            "find_vsdevcmd must fall back to discovery when the env var's own path does not "
            "exist on disk, not return the nonexistent path or raise"
        )
