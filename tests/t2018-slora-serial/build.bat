@echo off
setlocal
set HEREDIR=%~dp0
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj mkdir obj
rem T-2041 (Poirot c81e48c review, Significant 5): ENG used to be a hardcoded absolute path to a
rem SIBLING worktree (D:\SuperSLM\.worktrees\run), which drifts out of sync with whichever engine
rem is actually under review -- the review's own Phase 6 prediction was that this script "will
rem keep printing 75/0 whatever the engine under review does, because it is not linking the
rem engine under review." ENG now resolves relative to this script's own location, so this
rem binary always links the SAME checkout it is run from -- self-referential, not pinned to any
rem one worktree's name.
set ENG=%HEREDIR%..\..
cl /nologo /std:c++20 /O2 /fp:precise /EHsc /I%ENG%\include ^
  "%HEREDIR%t2018_offline_red.cpp" %ENG%\src\intmath.cpp %ENG%\src\forward\forward_sites.cpp ^
  %ENG%\src\forward\checked_chain_funnel.cpp %ENG%\src\matmul.cpp %ENG%\src\silu_lut.cpp ^
  %ENG%\src\trace_hook.cpp %ENG%\src\model.cpp %ENG%\src\artifact.cpp %ENG%\src\sha256.cpp ^
  %ENG%\src\tokenizer.cpp %ENG%\src\proof_manifest.cpp %ENG%\src\decode_digest.cpp ^
  /Fo:"%HEREDIR%obj/" /Fe:"%HEREDIR%t2018_offline_red.exe"
