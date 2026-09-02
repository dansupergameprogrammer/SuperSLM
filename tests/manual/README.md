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

## The QK-norm real-candidate acceptance gate (T-2564, S4, ruled D-SLM6199)

`tests/t2551_qk_norm_harness.cpp` is not a `pytest` test — it is a standalone C++ harness,
compiled by hand (`build.bat`'s own `t2432_geometry_harness` recipe, source list swapped for this
file; see that recipe for the exact `cl` invocation and the shader-staging step it depends on) and
run against a real, multi-gigabyte QK-norm-bearing artifact under `D:\hf_cache\` — the same reason
this directory's own tests cannot run in hosted CI. It is the width>1, 28-layer, N=100-repeated-GPU-
dispatch acceptance for any checkpoint carrying `q_norm`/`k_norm` tensors (Ask 5 Track B), and it is
the ONLY construction in this repository that exercises `ApplyQkNormSite` at real production scale;
the small synthetic cells in `superslm_tests.exe` (`TestT2564_S4_ApplyQkNormSitePerHeadQScaleNotCollapsed`,
`TestT2564_S4_ApplyQkNormSiteKLandsOnNewPostNormScaleNotJustNorm`, `tests/test_main.cpp`) catch a
regression of the per-head Q carry or the K re-landing on every local `ctest` and in hosted CI, but
neither they nor anything else here proves the real 633 MB candidate still passes at production
scale and width.

**Run this by hand, on Windows, with the RTX GPU this repo builds against: after any change to
`ApplyQkNormSite`, `qk_norm_site.hlsl`, `q_proj_site.hlsl`, `softmax_site.hlsl`, or
`LayerWeights`'s q_norm/k_norm fields; at every pin bump of the calibrated Qwen3-Embedding-0.6B
candidate; and before every release tag that ships Qwen3-architecture (QK-norm) support.** Usage:
`out\t2551_qk_norm_harness.exe <model.sslm> [token_id]`; a `RESULT: PASS` line with `CELL 1` and
`CELL 4` both `PASS` and `DETERMINISM CROWN: PASS` is the acceptance. Never a per-push hosted CI
leg (D-SLM6161) — hosted runners have neither the artifact nor the GPU this gate needs.
