# Manual gates

Tests in this directory are **not** collected by CI (`.github/workflows/tests.yml`
only runs `pytest tests/ci/`, `pytest tools/ tests/reference/`, and the C++
suite via `build.bat`/`ctest`). They require things a hosted Linux runner does
not have: the real, multi-gigabyte model artifacts under `D:\hf_cache\`, a
built `out\sslm_generate.exe`, and — for the test below specifically — both
Windows shells (`powershell.exe` and `pwsh`) on the same machine.

**Run these by hand, locally, on Windows, after any change to `tools/ask.ps1`
or the driver it wraps.** A test that would have caught a real shipped defect
belongs in the tree even when it cannot run unattended; hiding it because it
does not fit the CI runner's constraints is how the defect ships again the
same way. `python -m pytest tests/manual/ -v` runs everything here.

| Test | What it guards | Requires |
|---|---|---|
| `test_ask_ps1_dual_shell_smoke.py` | `tools\ask.ps1` returns exit code 0 and a decoded answer under **both** Windows PowerShell 5.1 (`powershell.exe`) and PowerShell 7 (`pwsh`) | `powershell.exe`, `pwsh`, `out\sslm_generate.exe` (built on demand), `D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct.sslm`, `tests\fixtures\qwen2.5-1.5b.tok.sslm` |

Each test skips (does not fail) when a required binary or artifact is absent,
so a checkout on a machine without the model still collects cleanly.
