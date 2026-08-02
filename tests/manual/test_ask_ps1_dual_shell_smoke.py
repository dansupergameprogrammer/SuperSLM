"""Dual-shell smoke test for tools/ask.ps1 (T-1679).

Manual gate -- see tests/manual/README.md for why this directory is not
collected by CI: it needs the real multi-gigabyte model artifacts and both
Windows shells present on the same machine, neither of which the hosted
Linux runner has.

WHAT THIS GUARDS. `tools/ask.ps1`@`60e1198` fixed a defect that was invisible
under PowerShell 7 (`pwsh`) and reproduced on every healthy run under Windows
PowerShell 5.1 (`powershell.exe`): the driver writes an informational
preflight line to stderr on every successful run, and the script used to
merge that stderr into its success stream with `2>&1` while
`$ErrorActionPreference = 'Stop'` was in effect. Under 5.1 -- but not 7 -- a
native executable writing to stderr while `$ErrorActionPreference` is
`'Stop'` raises a terminating `NativeCommandError`, so a perfectly healthy
run looked like a crash on 5.1 only. The fix (already on `tools/ask.ps1`) is
to redirect stderr to its own temp file instead of merging it. No test
exercised the two shells before this ticket, so the defect shipped
unnoticed; this test runs the exact scenario -- a healthy `ask.ps1`
invocation, asked to run under both shells -- and would have caught it.

Each test function invokes `tools/ask.ps1` as a real subprocess under a real
shell binary and asserts: exit code 0, and the captured output shows a
completed run (a `wall_time_seconds:` line, which `ask.ps1` prints only after
the driver returns and the answer is decoded) with no
`NativeCommandError` -- the literal exception type the merged-stderr defect
raised. This is a coarse behavioral smoke check, not a content assertion on
what the model says; it is deliberately silent about the *quality* of the
answer, only that the pipeline ran to completion and exited clean.

PROOF THIS TEST CAN FAIL (StandardsDocument Sec4 -- a check shown able to
fail on a fault it exists to catch, not only shown to pass on unchanged
input). Reintroducing the merge (`2> $errFile` -> `2>&1`, dropping the
try/finally that raises `$ErrorActionPreference` around the call) into a
scratch copy of `tools/ask.ps1` and pointing `test_powershell_5_1` at that
scratch copy reproduces `NativeCommandError` and a nonzero exit under
`powershell.exe`, while the same scratch copy still passes under `pwsh` --
exactly the 5.1-only failure signature this test exists to catch. See the
build log for the transcript of that run; it is not reproduced as an
automated cell here because doing so would require shipping a second, bad
copy of the driver-invoking script in the tree, which is worse than the risk
it would guard against.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ASK_PS1 = REPO_ROOT / "tools" / "ask.ps1"
DRIVER_EXE = REPO_ROOT / "out" / "sslm_generate.exe"
MODEL_ARTIFACT = Path(r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct.sslm")
TOKENIZER_ARTIFACT = REPO_ROOT / "tests" / "fixtures" / "qwen2.5-1.5b.tok.sslm"

POWERSHELL_5_1 = shutil.which("powershell.exe") or shutil.which("powershell")
POWERSHELL_7 = shutil.which("pwsh") or shutil.which("pwsh.exe")

_missing = []
if POWERSHELL_5_1 is None:
    _missing.append("powershell.exe (Windows PowerShell 5.1) not found on PATH")
if POWERSHELL_7 is None:
    _missing.append("pwsh (PowerShell 7) not found on PATH")
if not MODEL_ARTIFACT.exists():
    _missing.append(f"model artifact not found: {MODEL_ARTIFACT}")
if not TOKENIZER_ARTIFACT.exists():
    _missing.append(f"tokenizer artifact not found: {TOKENIZER_ARTIFACT}")
if os.name != "nt":
    _missing.append("this gate only runs on Windows")

SKIP_REASON = "; ".join(_missing) if _missing else None


def _run_ask(shell_exe: str, question: str, script: Path = ASK_PS1) -> subprocess.CompletedProcess:
    """Invoke `script` under `shell_exe` exactly the way a person on the
    command line would, and return the completed process. A real subprocess
    call, not a mock -- the whole point is to exercise the actual shell's
    stderr-merging behavior, which cannot be observed by calling PowerShell
    functions in-process."""
    return subprocess.run(
        [
            shell_exe,
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(script),
            question,
            "-MaxNew",
            "24",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )


def _assert_healthy_run(result: subprocess.CompletedProcess, shell_label: str) -> None:
    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"{shell_label}: ask.ps1 exited {result.returncode}, expected 0.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "NativeCommandError" not in combined, (
        f"{shell_label}: NativeCommandError present in output -- this is exactly "
        f"the merged-stderr defect (fixed at tools/ask.ps1@60e1198) reproducing.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "wall_time_seconds:" in combined, (
        f"{shell_label}: no wall_time_seconds line -- the driver did not "
        f"complete a run.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.skipif(SKIP_REASON is not None, reason=str(SKIP_REASON))
def test_powershell_5_1() -> None:
    """ask.ps1 returns exit code 0 and a decoded answer under Windows
    PowerShell 5.1 -- the shell the fixed defect was specific to."""
    result = _run_ask(POWERSHELL_5_1, "What is the capital of France?")
    _assert_healthy_run(result, "powershell.exe (5.1)")


@pytest.mark.skipif(SKIP_REASON is not None, reason=str(SKIP_REASON))
def test_powershell_7() -> None:
    """ask.ps1 returns exit code 0 and a decoded answer under PowerShell 7 --
    the shell the defect never reproduced under, kept as the sibling cell so
    a regression that broke both shells is still caught rather than read as
    '5.1-specific' by default."""
    result = _run_ask(POWERSHELL_7, "What is the capital of France?")
    _assert_healthy_run(result, "pwsh (7)")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
