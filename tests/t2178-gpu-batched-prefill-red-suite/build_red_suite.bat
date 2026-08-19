@echo off
rem T-2178 (Curie): builds every cell file in this directory against the REAL engine (the
rem identical source list build.bat's own GPU line uses) and captures the compile/link
rem transcript. Per StandardsDocument.md Sec5.4/Curie's own red-first discipline, this is run and
rem the transcript kept as evidence each cell is red for the RIGHT reason -- LNK2019/LNK1120
rem unresolved-external on the g_gpu_chunk_submit_count_probe/g_gpu_chunk_dispatch_count_probe
rem instrumentation counters (tests/support/gpu_chunk_dispatch_instrument.h) this suite's own
rem cells reference and Rung 2 has not yet defined -- never a compile error and never a silent
rem pass. cell_ceiling_boundary.cpp additionally references
rem superslm_gpu::kT2169TdrSafeMaxChunkTokens (Rung 2's own owed TDR-bound figure) and is
rem expected red by the SAME mechanism until that symbol exists too.
rem
rem T-2180 (Brunel, Rung 2b): /DSUPERSLM_ENABLE_GPU_CHUNK_DISPATCH_INSTRUMENT added to the cl
rem invocation below -- superslm_gpu.cpp now DEFINES the two counters under this exact macro
rem (SubmitChunkToFullDepthForG5Bridge, its own header comment), resolving the counter half of
rem the LNK2019s this suite was gated on. kT2169TdrSafeMaxChunkTokens remains undefined until
rem its own TDR-measurement sub-step lands -- cell_ceiling_boundary.cpp is expected to stay red
rem on that one symbol alone until then.
rem
rem T-2180 (Brunel, T-2180 addendum, D-SLM3660): /DSUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION
rem added -- the tenth-failure-origin fault-injection seam (gpu_port.h) this suite's own
rem cell_exit_census.cpp now arms; not defined by build.bat's own superslm_tests.exe line, which
rem never calls the Arm/Clear functions this macro gates -- matching SUPERSLM_O11_ALLOC_INJECTION's
rem own established precedent of being defined only where a cell actually calls its Arm/Clear pair.
rem
rem cell_exit_census.cpp is the one file in this suite NOT expected red today: its five cells
rem (P1, P4, S1, S2, S5, Claude/Curie/t2178-t2169-gpu-batched-prefill-red-suite-2026-08-18.md)
rem assert PRESERVED baseline behavior the design's own census marks "Yes -- unchanged" -- a
rem standing regression suite, green today by construction and required to stay green post-build,
rem mirroring Claude/Curie/t2158-t2149-avx-red-suite-2026-08-18.md Sec3 dimension 7(d)'s own
rem "authored, green today, standing" disposition.
rem
rem This suite is documented-local (D-SLM3432: no GPU CI runner) -- run directly:
rem tests\t2178-gpu-batched-prefill-red-suite\build_red_suite.bat [--model1p5b=PATH]
rem [--g5fixture=PATH]
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
set ENG=%HEREDIR%..\..
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj mkdir obj
if exist ..\..\out\shaders xcopy /Y /I /Q ..\..\out\shaders obj\shaders >nul

set OVERALL_LINK_OK=1
for %%f in (cell_bitidentity.cpp cell_ceiling_boundary.cpp cell_trust_and_guard.cpp cell_exit_census.cpp cell_embedding_range.cpp) do (
    echo ===== %%f =====
    cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /DSUPERSLM_ENABLE_GPU_CHUNK_DISPATCH_INSTRUMENT /DSUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION /I%ENG%\include /I%ENG%\tests /I. ^
        %ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp ^
        %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp ^
        %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
        %ENG%\src\forward\forward_sites.cpp %ENG%\src\decode_digest.cpp %ENG%\src\gpu\superslm_gpu.cpp ^
        %ENG%\src\gpu\gpu_1p0.cpp ^
        "%%f" /Fo:"obj\\" /Fe:"obj\%%~nf.exe" ^
        /link d3d12.lib dxgi.lib dxguid.lib > "obj\%%~nf.log" 2>&1
    findstr /C:"error C" "obj\%%~nf.log" >nul
    if not errorlevel 1 (
        echo    COMPILE ERROR -- unexpected, every cell here should compile clean against the
        echo    declared surface:
        type "obj\%%~nf.log"
        set OVERALL_LINK_OK=0
    ) else (
        findstr /C:"LNK2019" /C:"LNK1120" "obj\%%~nf.log" >nul
        if not errorlevel 1 (
            echo    RED BY LINK -- unresolved external^(s^) on the not-yet-wired probe/bound
            echo    symbols:
            findstr /C:"LNK2019" "obj\%%~nf.log"
        ) else (
            findstr /C:"error" "obj\%%~nf.log" >nul
            if not errorlevel 1 (
                echo    UNEXPECTED ERROR CLASS -- see obj\%%~nf.log
                type "obj\%%~nf.log"
                set OVERALL_LINK_OK=0
            ) else (
                echo    LINKED CLEAN -- run: obj\%%~nf.exe %*
            )
        )
    )
)
echo.
if "%OVERALL_LINK_OK%"=="1" (
    echo SUITE STATUS: RED BY LINK where a probe/bound symbol is not yet wired ^(expected,
    echo pre-build^), or LINKED CLEAN where a cell needs none ^(cell_exit_census.cpp -- run its
    echo own .exe to see genuine pass/fail counts^).
    exit /b 1
) else (
    echo SUITE STATUS: an unexpected error class was found -- see logs above.
    exit /b 2
)
