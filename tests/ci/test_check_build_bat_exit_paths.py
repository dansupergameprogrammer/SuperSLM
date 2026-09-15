"""Mechanism coverage for check_build_bat_exit_paths.py."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_build_bat_exit_paths as chk  # noqa: E402


def test_approved_top_level_exits_pass_and_nested_exit_fails():
    approved = ("rem padding\n" * 13) + """if not exist tool (
exit /b 1
)
pushd %~dp0
if errorlevel 1 (
	goto :hard_fail
)
popd
exit /b %ec%
:hard_fail
popd
endlocal
exit /b 1
"""
    nested_exit = approved.replace("goto :hard_fail", "popd & exit /b 1")

    assert chk.check_text(approved) == []
    assert chk.check_text(nested_exit) == ["line 19: exit /b is inside parenthesized block"]
