@echo off
setlocal
set HEREDIR=%~dp0
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj mkdir obj
set ENG=D:\SuperSLM\.worktrees\run
cl /nologo /std:c++20 /O2 /fp:precise /EHsc /I%ENG%\include ^
  "%HEREDIR%t2018_offline_red.cpp" %ENG%\src\intmath.cpp %ENG%\src\forward\forward_sites.cpp ^
  %ENG%\src\forward\checked_chain_funnel.cpp %ENG%\src\matmul.cpp %ENG%\src\silu_lut.cpp ^
  %ENG%\src\trace_hook.cpp %ENG%\src\model.cpp %ENG%\src\artifact.cpp %ENG%\src\sha256.cpp ^
  %ENG%\src\tokenizer.cpp %ENG%\src\proof_manifest.cpp %ENG%\src\decode_digest.cpp ^
  /Fo:"%HEREDIR%obj/" /Fe:"%HEREDIR%t2018_offline_red.exe"
