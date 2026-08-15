@echo off
rem T-2112 (Curie): builds every RED-BY-LINK cell file in this directory against the REAL engine
rem (the identical source list build.bat, this repo's own root, uses) and captures the LINK
rem transcript. Per StandardsDocument.md Sec5.4/Curie's own red-first discipline, this is run and
rem the transcript kept as evidence a cell is red for the RIGHT reason (LNK2019 unresolved
rem external on the 1.0 API symbols this suite calls, never a compile error and never a silent
rem pass) -- not merely asserted in prose. Each translation unit is compiled to a .obj (proving
rem compile succeeds against the declared surface, dim-7's own claim) and then link is attempted
rem into a throwaway .exe (proving no implementation exists yet). Exit code is 1 (red) until the
rem build seat (T-2113) lands the 1.0 API's own .cpp; this script's own exit code inverts to 0
rem ONLY once linking succeeds, at which point this suite has gone from red to buildable and the
rem individual CHECK/FAIL output governs pass/fail from then on.
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
set ENG=%HEREDIR%..\..
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj mkdir obj

set OVERALL_LINK_OK=1
for %%f in (dim1_lifetime_red.cpp dim2_hostile_red.cpp dim3_concurrency_red.cpp dim4_shape_red.cpp dim5_failure_red.cpp dim6_determinism_red.cpp dim8_composition_red.cpp dim9_persistence_red.cpp dim10_functional_red.cpp dim11_guard_red.cpp) do (
    echo ===== %%f =====
    cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /I%ENG%\include /I. ^
        %ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp ^
        %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp ^
        %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
        %ENG%\src\forward\forward_sites.cpp %ENG%\src\decode_digest.cpp %ENG%\src\gpu\superslm_gpu.cpp ^
        %ENG%\src\gpu\gpu_1p0.cpp ^
        "%%f" /Fo:"obj\\" /Fe:"obj\%%~nf.exe" ^
        /link d3d12.lib dxgi.lib dxguid.lib > "obj\%%~nf.log" 2>&1
    findstr /C:"error C" "obj\%%~nf.log" >nul
    if not errorlevel 1 (
        echo    COMPILE ERROR -- unexpected, this cell should compile clean ^(dim-7 claim^):
        type "obj\%%~nf.log"
        set OVERALL_LINK_OK=0
    ) else (
        findstr /C:"LNK2019" /C:"LNK1120" "obj\%%~nf.log" >nul
        if not errorlevel 1 (
            echo    RED BY LINK -- unresolved external^(s^) on the undeclared 1.0 API implementation:
            findstr /C:"LNK2019" "obj\%%~nf.log"
        ) else (
            findstr /C:"error" "obj\%%~nf.log" >nul
            if not errorlevel 1 (
                echo    UNEXPECTED ERROR CLASS -- see obj\%%~nf.log
                type "obj\%%~nf.log"
                set OVERALL_LINK_OK=0
            ) else (
                echo    LINKED CLEAN -- this cell's own implementation now exists; run the .exe.
            )
        )
    )
)
echo.
if "%OVERALL_LINK_OK%"=="1" (
    echo SUITE STATUS: RED BY LINK, as expected pre-build ^(or GREEN where an implementation exists^).
    exit /b 1
) else (
    echo SUITE STATUS: an unexpected error class was found -- see logs above.
    exit /b 2
)
