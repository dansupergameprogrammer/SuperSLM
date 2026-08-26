@echo off
rem T-2296 (Curie): builds every cell file in this directory and RUNS it, reporting the exit
rem code as the primary red/green signal: the not-yet-built src/detail/int_hash.h is gated
rem behind __has_include("detail/int_hash.h") in each cell file, so a missing header routes
rem that cell's own post-remedy checks to an explicit, counted SKIP + FAIL rather than a build
rem error. 0 = every CHECK in that cell passed; nonzero = at least one CHECK failed (including
rem the "whole cell-group unattemptable" sentinel failure dim6/dim7 add when their
rem __has_include gate is closed).
rem
rem LINK FAILURE is its own class, distinct from both a compile error and a red CHECK (T-2301,
rem 2026-08-26, mirroring this directory's own build_liveness_red.bat findstr /C:"LNK" branch):
rem dim7_contract_red.cpp calls BucketCountFor, which calls the engine's out-of-line
rem superslm::Clz64 (see below) -- an omitted or reverted compile line produces
rem LNK2019/LNK1120, no .exe is produced, and the loop body's own attempt to run a nonexistent
rem .exe (errorlevel 9009) would otherwise print as an indistinguishable red CHECK failure --
rem the exact detection gap T-2297 fixed the symptom of but left open (S4,
rem Claude/Poirot/t2299-fp-free-open-review-2026-08-26.md). The LINK FAILURE branch below
rem catches this before the run-and-check-exit-code step is ever reached.
rem
rem These cells exercise src/detail/{int_hash,context_hash}.h. FixedIntMap/FixedIntSet/
rem GrowableIntSet/GrowableIntMap/GrowableContextMap are templates defined entirely in those
rem two headers (design Sec3.1/Sec3.5/Sec3.6) -- but BucketCountFor calls the engine's
rem out-of-line superslm::Clz64, so the #include alone is NOT sufficient: intmath.cpp must
rem also be on the compile line, or both cells fail to LINK and the suite reports a false RED.
rem (T-2297, 2026-08-26: the original source list omitted it; the conductor reproduced 18/18
rem passing once linked, and added it below.) Kept deliberately minimal rather than matching
rem tests/t2138-abi-red-suite's own full CPU-core source list: the two headers plus the one
rem .cpp their templates call out to is everything these cells need.
rem
rem dim7_contract_red.cpp is compiled WITH /DNDEBUG (this file's own header comment: the
rem probe-exhaustion std::abort() is documented release-safe/NDEBUG-independent, and building
rem NDEBUG is what actually exercises that specific guard rather than the earlier debug-only
rem assert()). Every other cell is built without /DNDEBUG (the ordinary debug/assert-enabled
rem config the rest of this repo's suites use). Loop shape (for %%f in (...), no call/goto
rem subroutine) matches tests/t2138-abi-red-suite/build_link_red.bat's own working convention.
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
rem Normalized via %%~fi (FOR's own path canonicalization), not %HEREDIR%..\.. verbatim -- see
rem build_liveness_red.bat's own header comment, this directory: an unnormalized /I path
rem combined with a nested nested quoted #include chain can push the literal resolved path past
rem MAX_PATH under a deeply-nested checkout location, which cl.exe silently mishandles (C1083)
rem rather than erroring cleanly. These cells do not hit that specific chain today (they never
rem include sslm_abi.h), but normalizing here too costs nothing and avoids the class recurring
rem if a future cell does.
for %%i in ("%HEREDIR%..\..") do set ENG=%%~fi
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo >nul 2>&1
cd /d "%HEREDIR%"
if not exist obj mkdir obj

set ANY_COMPILE_ERROR=0
set ANY_RED=0

for %%f in (dim4_shape_red.cpp dim6_determinism_red.cpp dim7_contract_red.cpp dim11_guard_red.cpp) do (
    echo ===== %%f =====
    set EXTRAFLAGS=
    if "%%f"=="dim7_contract_red.cpp" set EXTRAFLAGS=/DNDEBUG
    cl /nologo /std:c++20 /O2 /W4 /EHsc !EXTRAFLAGS! /I%ENG%\src /I%ENG%\include -I. ^
        "%%f" "%ENG%\src\intmath.cpp" /Fo:"obj\\" /Fe:"obj\%%~nf.exe" ^
        /link > "obj\%%~nf.log" 2>&1
    findstr /C:"error C" "obj\%%~nf.log" >nul
    if not errorlevel 1 (
        echo    COMPILE ERROR:
        type "obj\%%~nf.log"
        set ANY_COMPILE_ERROR=1
    ) else (
        findstr /C:"LNK" "obj\%%~nf.log" >nul
        if not errorlevel 1 (
            echo    LINK FAILURE:
            type "obj\%%~nf.log"
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
    echo SUITE STATUS: RED -- one or more cells reported a CHECK failure ^(expected pre-build:
    echo the post-remedy legs gated behind __has_include^("detail/int_hash.h"^) cannot pass until
    echo src/detail/int_hash.h and src/detail/context_hash.h exist^).
    exit /b 1
) else (
    echo SUITE STATUS: GREEN -- every cell's every CHECK passed.
    exit /b 0
)
