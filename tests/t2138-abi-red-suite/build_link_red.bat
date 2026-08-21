@echo off
rem T-2138 (Curie): builds every C++ cell file in this directory against the REAL engine (the
rem identical source list build.bat, this repo's own root, uses) and captures the LINK
rem transcript. Historical purpose (D-SLM3450 era): prove a cell is RED BY LINK for the right
rem reason (LNK2019 on the undeclared CPU ABI, never a compile error) before that ABI existed.
rem
rem N1 STRUCTURAL FIX (Claude/Poirot/a12bbdd-t2199-phaseD-closing.md): that era ended long ago
rem -- src/sslm_abi.cpp has shipped a real implementation since long before T-2199, and this
rem script's own header comment already SAID its exit code "inverts to 0 ONLY once linking
rem succeeds" -- but the tail logic never implemented that inversion, so a LINKED-CLEAN run and
rem a genuinely-broken-link-list run were BOTH reported as "SUITE STATUS: RED BY LINK, as
rem expected pre-build," exit /b 1, indistinguishable. That is exactly how this suite's own
rem source list rotted silently once (S6, missing sslm_abi.cpp itself) and rotted silently a
rem SECOND time (N1, missing the four damped_greedy_*.cpp files T-2199 added as sslm_abi.cpp's
rem own new dependencies) -- both times the script kept printing the same reassuring line while
rem all eleven cells failed to link and zero .exe files existed. Per Curie's own build_green
rem gating precedent (build_green.bat/build_green_phaseD.bat, this repo's own T-2199 suite): a
rem link failure is now NEVER reported as this suite's expected/red state. This script's own
rem contract, going forward:
rem   - every cell links clean (the ABI is real, shipped, and this script's own source list is
rem     current) -> SUITE STATUS: LINKED CLEAN, exit /b 0.
rem   - ANY cell hits LNK2019/LNK1120 -> SUITE STATUS: LINK FAILURE (loud, per-cell, per-symbol),
rem     exit /b 1 -- a real defect (a rotted source list, same as N1/S6, or a genuine missing
rem     production symbol), never "expected."
rem   - a compile error or any other unexpected error class -> exit /b 2, unchanged.
rem This suite is a link-and-execute check, not a red-first pre-implementation gate, from this
rem fold forward -- there is no implementation left to wait for.
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

set ANY_COMPILE_ERROR=0
set ANY_LINK_FAILURE=0
for %%f in (dim1_lifetime_red.cpp dim2_hostile_red.cpp dim3_concurrency_red.cpp dim4_shape_red.cpp dim5_failure_red.cpp dim6_determinism_red.cpp dim7_contract_red.cpp dim8_composition_red.cpp dim9_persistence_red.cpp dim10_functional_red.cpp dim11_guard_red.cpp) do (
    echo ===== %%f =====
    rem S6 fix round (Claude/Poirot/2c18dab-t2139-abi-build-review.md): src\sslm_abi.cpp was
    rem absent from this source list -- the whole reason this suite reported RED BY LINK forever,
    rem even after the ABI implementation existed, and the reason not one of these 51 cells had
    rem ever actually executed against it. Added here, matching build.bat's own root source list.
    rem
    rem N1 FIX (Claude/Poirot/a12bbdd-t2199-phaseD-closing.md): the SAME class recurred --
    rem src\sslm_abi.cpp has called superslm::AntiLmCreate/AntiLmDestroy/AntiLmUpdate/
    rem DampedGreedyScoreAndArgmax/ValidateDampedGreedyParams since T-2199 Phase D landed
    rem (git log -S, first reference at 5ced7a6), and this script's source list was never
    rem updated -- every cell LNK2019'd on all five symbols, zero .exe produced, and this
    rem script's own findstr-based branch below reported that as the suite's ordinary
    rem pre-implementation RED BY LINK state, indistinguishable from a real link-list defect.
    rem Fixed: the four damped_greedy_*.cpp translation units appended, matching
    rem build_green_phaseD.bat's own source list exactly.
    cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /I%ENG%\include /I. ^
        %ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp ^
        %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp ^
        %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
        %ENG%\src\forward\forward_sites.cpp %ENG%\src\decode_digest.cpp %ENG%\src\sslm_abi.cpp ^
        %ENG%\src\damped_greedy_antilm.cpp %ENG%\src\damped_greedy_topk.cpp ^
        %ENG%\src\damped_greedy_phaseD.cpp %ENG%\src\damped_greedy_phaseD_loop.cpp ^
        "%%f" /Fo:"obj\\" /Fe:"obj\%%~nf.exe" ^
        /link > "obj\%%~nf.log" 2>&1
    findstr /C:"error C" "obj\%%~nf.log" >nul
    if not errorlevel 1 (
        echo    COMPILE ERROR -- unexpected, this cell should compile clean against sslm_abi.h:
        type "obj\%%~nf.log"
        set ANY_COMPILE_ERROR=1
    ) else (
        findstr /C:"LNK2019" /C:"LNK1120" "obj\%%~nf.log" >nul
        if not errorlevel 1 (
            rem N1 FIX: this is no longer "expected pre-build" -- the ABI this cell links
            rem against is real and shipped. Any LNK2019/LNK1120 here is a STRUCTURAL FAILURE
            rem (a rotted source list, same class as this file's own history) and is reported
            rem and counted as one, loudly, per-symbol -- never silently absorbed into a
            rem "SUITE STATUS: RED BY LINK, as expected" line.
            echo    *** LINK FAILURE *** unresolved external^(s^) -- this is a DEFECT, not an
            echo    expected pre-build state ^(the CPU ABI this cell links against is real and
            echo    shipped^). Missing symbols:
            findstr /C:"LNK2019" "obj\%%~nf.log"
            set ANY_LINK_FAILURE=1
        ) else (
            findstr /C:"error" "obj\%%~nf.log" >nul
            if not errorlevel 1 (
                echo    UNEXPECTED ERROR CLASS -- see obj\%%~nf.log
                type "obj\%%~nf.log"
                set ANY_COMPILE_ERROR=1
            ) else (
                echo    LINKED CLEAN.
            )
        )
    )
)
echo.
if "%ANY_COMPILE_ERROR%"=="1" (
    echo SUITE STATUS: COMPILE ERROR -- an unexpected error class was found, see logs above.
    exit /b 2
) else if "%ANY_LINK_FAILURE%"=="1" (
    echo SUITE STATUS: LINK FAILURE -- one or more cells did not link against the real,
    echo shipped CPU ABI. This is a real defect ^(a rotted source list, or a genuine missing
    echo production symbol^), never this suite's expected state -- see the per-cell LNK2019
    echo lines above.
    exit /b 1
) else (
    echo SUITE STATUS: LINKED CLEAN -- every cell links against the real, shipped CPU ABI.
    exit /b 0
)
