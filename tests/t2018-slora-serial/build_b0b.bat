@echo off
setlocal
rem T-2041 (Poirot c81e48c review, Significant 4): t2029_b0b_red.cpp had no committed build path
rem anywhere in the tree -- its reported 38/0 was reproducible only from a committed .exe and an
rem uncommitted ad-hoc `cl` invocation. This script is that committed path, mirroring build.bat's
rem own self-referential ENG resolution (Significant 5) so it always links whichever engine
rem checkout it is run from.
set HEREDIR=%~dp0
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj mkdir obj
set ENG=%HEREDIR%..\..
cl /nologo /std:c++20 /O2 /fp:precise /EHsc /I%ENG%\include ^
  "%HEREDIR%t2029_b0b_red.cpp" %ENG%\src\intmath.cpp %ENG%\src\forward\forward_sites.cpp ^
  %ENG%\src\forward\checked_chain_funnel.cpp %ENG%\src\matmul.cpp %ENG%\src\silu_lut.cpp ^
  %ENG%\src\trace_hook.cpp %ENG%\src\model.cpp %ENG%\src\artifact.cpp %ENG%\src\sha256.cpp ^
  %ENG%\src\tokenizer.cpp %ENG%\src\proof_manifest.cpp %ENG%\src\decode_digest.cpp ^
  /Fo:"%HEREDIR%obj/" /Fe:"%HEREDIR%t2029_b0b_red.exe"
