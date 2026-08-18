@echo off
rem T-2138 (Curie): builds every RED-BY-LINK C++ cell file in this directory against the REAL
rem engine (the identical source list build.bat, this repo's own root, uses) and captures the
rem LINK transcript. Per StandardsDocument.md Sec5.4/Curie's own red-first discipline, this is
rem run and the transcript kept as evidence a cell is red for the RIGHT reason (LNK2019
rem unresolved external on the sslm_* CPU ABI symbols this suite calls, never a compile error
rem and never a silent pass) -- not merely asserted in prose. Mirrors
rem tests/t2112-gpu-1p0-red-suite/build_link_red.bat and tests/t2130-g5-red-suite/
rem build_link_red.bat's own convention exactly. Each translation unit is compiled to a .obj
rem (proving compile succeeds against the declared surface, sslm_abi.h) and then link is
rem attempted into a throwaway .exe (proving no implementation exists yet, D-SLM3450). Exit code
rem is 1 (red) until the build seat lands this ABI's own .cpp; this script's own exit code
rem inverts to 0 ONLY once linking succeeds, at which point this suite has gone from red to
rem buildable and the individual CHECK/FAIL output governs pass/fail from then on.
rem
rem CPU-ONLY SOURCE LIST (deliberate, distinct from tests/t2112-gpu-1p0-red-suite and
rem tests/t2130-g5-red-suite's own copies of this script): this suite targets the CPU-side
rem sslm_* ABI only (design Sec1) and NEVER links src/gpu/superslm_gpu.cpp or
rem src/gpu/gpu_1p0.cpp -- neither this suite's own sslm_abi.h surface nor any file in this
rem directory references a GPU symbol, and superslm_gpu.cpp pulls in D3D12CreateDevice/
rem CreateDXGIFactory2/D3D12SerializeRootSignature, which are unresolved on a machine without
rem the d3d12.lib/dxgi.lib/dxguid.lib link libraries wired in -- noise this suite's own red
rem signal does not need. D-SLM3388 (a live hazard on this machine, reddening real-content GPU
rem decode) is therefore structurally inapplicable to this suite: it never builds against or
rem exercises a GPU surface.
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
set ENG=%HEREDIR%..\..
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj mkdir obj

set OVERALL_LINK_OK=1
for %%f in (dim1_lifetime_red.cpp dim2_hostile_red.cpp dim3_concurrency_red.cpp dim4_shape_red.cpp dim5_failure_red.cpp dim6_determinism_red.cpp dim7_contract_red.cpp dim8_composition_red.cpp dim9_persistence_red.cpp dim10_functional_red.cpp dim11_guard_red.cpp) do (
    echo ===== %%f =====
    rem S6 fix round (Claude/Poirot/2c18dab-t2139-abi-build-review.md): src\sslm_abi.cpp was
    rem absent from this source list -- the whole reason this suite reported RED BY LINK forever,
    rem even after the ABI implementation existed, and the reason not one of these 51 cells had
    rem ever actually executed against it. Added here, matching build.bat's own root source list.
    cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /I%ENG%\include /I. ^
        %ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp ^
        %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp ^
        %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
        %ENG%\src\forward\forward_sites.cpp %ENG%\src\decode_digest.cpp %ENG%\src\sslm_abi.cpp ^
        "%%f" /Fo:"obj\\" /Fe:"obj\%%~nf.exe" ^
        /link > "obj\%%~nf.log" 2>&1
    findstr /C:"error C" "obj\%%~nf.log" >nul
    if not errorlevel 1 (
        echo    COMPILE ERROR -- unexpected, this cell should compile clean against sslm_abi.h:
        type "obj\%%~nf.log"
        set OVERALL_LINK_OK=0
    ) else (
        findstr /C:"LNK2019" /C:"LNK1120" "obj\%%~nf.log" >nul
        if not errorlevel 1 (
            echo    RED BY LINK -- unresolved external^(s^) on the undeclared CPU ABI implementation:
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
