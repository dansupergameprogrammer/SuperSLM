@echo off
rem T-2296 (Curie): builds and runs Sec4.2/Sec4.3's own two named liveness-control cells,
rem tests/fp_free_open_red.cpp and tests/fp_free_sticky_readback.cpp -- both filed at the exact
rem top-level tests/ path the design's own text names, not inside this subdirectory (which holds
rem this campaign's Coverage-Model-dimension cells instead). This script lives in this
rem subdirectory so `--child-mode=` re-exec's own #include "t2296-fp-free-open-red-suite/
rem fixture_common.h" path resolves the same way whether the .exe is invoked from here or from
rem the repo's tests/ directory.
rem
rem CPU-ONLY SOURCE LIST, matching tests/t2138-abi-red-suite/build_link_red.bat's own convention
rem exactly (same reasoning: neither cell references a GPU symbol, and superslm_gpu.cpp pulls in
rem D3D12/DXGI link libraries this build does not wire in).
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
rem Normalized (no literal "..\.." segments), not %HEREDIR%..\.. verbatim -- an unnormalized
rem /I path plus a two-level-nested quoted #include (sslm_abi.h -> sslm_abi_functions.inc ->
rem sslm_abi_functions_g5_comparable.inc) pushed the LITERAL resolved path past MAX_PATH under
rem this checkout's own deeply-nested worktree location (a Claude Code session-scratchpad path),
rem which cl.exe silently truncates rather than erroring on cleanly -- C1083 "cannot open
rem include file" on the second-level .inc specifically, reproduced and isolated in this
rem session (both t2138-abi-red-suite's own working build and a %%~fi-normalized rebuild of the
rem identical compile line succeed; the literal-".." form does not). %%~fi is FOR's own
rem path-canonicalization, the standard batch idiom -- shorter than pushd/popd, no cwd change.
for %%i in ("%HEREDIR%..\..") do set ENG=%%~fi
set TESTSDIR=%HEREDIR%..
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo >nul 2>&1
cd /d "%TESTSDIR%"
if not exist "%HEREDIR%obj" mkdir "%HEREDIR%obj"

set ANY_COMPILE_ERROR=0
set ANY_RED=0

for %%f in (fp_free_open_red.cpp fp_free_sticky_readback.cpp) do (
    echo ===== %%f =====
    cl /nologo /std:c++20 /O2 /W4 /EHsc /I%ENG%\include /I. ^
        %ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp ^
        %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp ^
        %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
        %ENG%\src\forward\forward_sites.cpp %ENG%\src\decode_digest.cpp %ENG%\src\sslm_abi.cpp ^
        %ENG%\src\damped_greedy_antilm.cpp %ENG%\src\damped_greedy_topk.cpp ^
        %ENG%\src\damped_greedy_phaseD.cpp %ENG%\src\damped_greedy_phaseD_loop.cpp ^
        "%%f" /Fo:"%HEREDIR%obj\\" /Fe:"%HEREDIR%obj\%%~nf.exe" ^
        /link > "%HEREDIR%obj\%%~nf.log" 2>&1
    findstr /C:"error C" "%HEREDIR%obj\%%~nf.log" >nul
    if not errorlevel 1 (
        echo    COMPILE ERROR:
        type "%HEREDIR%obj\%%~nf.log"
        set ANY_COMPILE_ERROR=1
    ) else (
        findstr /C:"LNK" "%HEREDIR%obj\%%~nf.log" >nul
        if not errorlevel 1 (
            echo    LINK FAILURE:
            type "%HEREDIR%obj\%%~nf.log"
            set ANY_COMPILE_ERROR=1
        ) else (
            "%HEREDIR%obj\%%~nf.exe"
            set CELLEXIT=!ERRORLEVEL!
            if not "!CELLEXIT!"=="0" (
                echo    RED ^(exit !CELLEXIT!^) -- see the FAIL/SKIP lines printed above.
                set ANY_RED=1
            ) else (
                echo    GREEN.
            )
        )
    )
)

echo.
if "%ANY_COMPILE_ERROR%"=="1" (
    echo SUITE STATUS: COMPILE/LINK ERROR -- see logs above.
    exit /b 2
) else if "%ANY_RED%"=="1" (
    echo SUITE STATUS: RED -- see the FAIL/SKIP lines above.
    exit /b 1
) else (
    echo SUITE STATUS: GREEN.
    exit /b 0
)
