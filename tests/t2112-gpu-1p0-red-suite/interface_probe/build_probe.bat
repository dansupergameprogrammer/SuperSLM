@echo off
rem T-2112 (Curie): dim-7 standing suite fixture, promoted from the T-2111
rem strike's own probe (design Sec10 B9 / Sec11 dim 7). Compiles every cell
rem in this directory against the CANONICAL declared header one level up
rem (..\sslm_gpu_1p0.h -- there is exactly one header, never a per-probe copy)
rem and reports pass/fail per cell. Compile-only (/c) -- this fixture checks
rem that the DECLARED SURFACE is internally consistent and buildable, never
rem that an implementation exists (that is every other file in this suite's
rem job, and their own red mode is red-by-link, not red-by-compile).
rem
rem T-2113 (B9): /I..\..\..\include added -- cell_dim7_real_engine_headers.c
rem (design Sec11 dim 7's own real-engine-headers leg, added at the T-2112
rem mini-fold) includes superslm/model.h alongside the suite's own canonical
rem header; every other cell is unaffected by the extra include path.
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj mkdir obj
set FAIL=0
set /a COUNT=0
for %%f in (cell_*.c) do (
    set /a COUNT+=1
    cl /nologo /c /TP /std:c++20 /W3 /WX /I.. /I..\..\..\include "%%f" /Fo:"obj\%%~nf.obj" 1>obj\%%~nf.log 2>&1
    if errorlevel 1 (
        echo === %%f ===   FAIL
        type obj\%%~nf.log
        set FAIL=1
    ) else (
        echo === %%f ===   exit: 0
    )
)
if "%FAIL%"=="1" (
    echo.
    echo INTERFACE PROBE: RED -- the declared surface does not compile as claimed.
    exit /b 1
) else (
    echo.
    echo INTERFACE PROBE: GREEN -- %COUNT%/%COUNT% cells compile clean against the declared surface.
    exit /b 0
)
