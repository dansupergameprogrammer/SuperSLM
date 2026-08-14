@echo off
setlocal
rem T-2046 (design §24.2 D-SLM3162, closing Poirot c81e48c review Significant 6): builds
rem sslm_adapter_dump directly via cl.exe, matching this repo's own established convention for
rem standalone tools (tests/t2018-slora-serial/build*.bat) rather than depending on a configured
rem CMake build directory (this worktree has none). Self-referential ENG, matching build.bat's
rem own T-2041 fix -- always links whichever checkout this script is run from.
set HEREDIR=%~dp0
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj mkdir obj
set ENG=%HEREDIR%..
cl /nologo /std:c++20 /O2 /fp:precise /EHsc /I%ENG%\include ^
  "%HEREDIR%sslm_adapter_dump.cpp" %ENG%\src\intmath.cpp %ENG%\src\forward\forward_sites.cpp ^
  %ENG%\src\forward\checked_chain_funnel.cpp %ENG%\src\matmul.cpp %ENG%\src\silu_lut.cpp ^
  %ENG%\src\trace_hook.cpp %ENG%\src\model.cpp %ENG%\src\artifact.cpp %ENG%\src\sha256.cpp ^
  %ENG%\src\tokenizer.cpp %ENG%\src\proof_manifest.cpp %ENG%\src\decode_digest.cpp ^
  /Fo:"%HEREDIR%obj/" /Fe:"%HEREDIR%sslm_adapter_dump.exe"
