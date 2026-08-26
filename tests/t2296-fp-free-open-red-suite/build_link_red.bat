@echo off
rem T-2296 (Curie): builds every cell file in this directory and RUNS it, reporting the exit
rem code as the red/green signal -- unlike tests/t2138-abi-red-suite's own build_link_red.bat
rem (which distinguishes COMPILE/LINK failure classes because that suite's cells reference a
rem declared-but-unimplemented C ABI), every cell here compiles clean today: the not-yet-built
rem src/detail/int_hash.h is gated behind __has_include("detail/int_hash.h") in each cell file,
rem so a missing header routes that cell's own post-remedy checks to an explicit, counted SKIP +
rem FAIL rather than a build error. The red/green signal is therefore each .exe's own exit code:
rem 0 = every CHECK in that cell passed; nonzero = at least one CHECK failed (including the
rem "whole cell-group unattemptable" sentinel failure dim6/dim7 add when their __has_include
rem gate is closed).
rem
rem These cells are header-only against src/detail/{int_hash,context_hash}.h -- no engine .cpp
rem source list is linked (contrast tests/t2138-abi-red-suite's own full CPU-core source list):
rem FixedIntMap/FixedIntSet/GrowableIntSet/GrowableIntMap/GrowableContextMap are templates
rem defined entirely in those two headers (design Sec3.1/Sec3.5/Sec3.6), so once they exist,
rem #include "detail/int_hash.h" with -I<repo>/src is everything a cell needs to compile and
rem link against them.
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
        "%%f" /Fo:"obj\\" /Fe:"obj\%%~nf.exe" ^
        /link > "obj\%%~nf.log" 2>&1
    findstr /C:"error C" "obj\%%~nf.log" >nul
    if not errorlevel 1 (
        echo    COMPILE ERROR:
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

echo.
if "%ANY_COMPILE_ERROR%"=="1" (
    echo SUITE STATUS: COMPILE ERROR -- see logs above.
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
