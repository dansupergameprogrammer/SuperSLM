@echo off
rem T-2932: red-first public API witness.  Exit 0 means the expected red API absence was
rem observed and named; exit nonzero means the instrument itself did not establish that state.
setlocal enabledelayedexpansion
set HERE=%~dp0
set ENG=%HERE%..\..
set OBJ=%HERE%obj_t2922
if not exist "%OBJ%" mkdir "%OBJ%"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cl /nologo /std:c++20 /W4 /EHsc /I"%ENG%\include" /c "%HERE%t2922_gpu_schema_accepting_red.cpp" /Fo"%OBJ%\red.obj" > "%OBJ%\red.compile.log" 2>&1
if errorlevel 1 (
    findstr /C:"SslmGpuSeqSchemaAcceptingForG5Bridge" "%OBJ%\red.compile.log" >nul || goto :unexpected
    findstr /C:"SslmGpuSeqSchemaBoundForG5Bridge" "%OBJ%\red.compile.log" >nul || goto :unexpected
    findstr /C:"sslm_seq_schema_bound" "%OBJ%\red.compile.log" >nul || goto :unexpected
    for %%C in (UNBOUND_APPLICATION_FINAL BOUND_MID_WALK BOUND_ACCEPTING DEAD_END_PREDECESSOR SLM5_RESTORE_BOUND SLM4_RESTORE_UNBOUND RESET_PRESERVES_BINDING CPU_ADOPT_PREFIX_PROGRESS CPU_ADOPT_PREFIX_PROMPT_ONLY_RESET SUBMITTED_BUSY_UNCHANGED_OUT MALFORMED_AND_FOREIGN_UNCHANGED_OUT DRAIN_THEN_FINALIZE_QUERY DISTINCT_SCHEMA_RESTORE_REBIND LATE_LIVE_BIND_REJECT LATE_SLM5_BIND_REJECT FRESH_BIND_ACCEPT RESET_BIND_ACCEPT GPU_IDLE_RESET_REUSE GPU_SUBMITTED_RESET_BUSY ADAPTER_SWAP_MIDTOKEN_THEN_RESET_ACCEPT NO_DEVICE_WORK FOUR_SURFACE_PUBLIC_WORDING REAL_MODEL_BUDGET_DECODE REAL_MODEL_LATE_BIND) do echo CELL %%C RED: API_SURFACE_ABSENT
    echo checks=24 failures=24 skips=0 red_reason=API_SURFACE_ABSENT
    exit /b 0
)
echo API declarations now compile. The red witness must be replaced by the live fixture runner before this invocation is accepted.
exit /b 1
:unexpected
type "%OBJ%\red.compile.log"
echo checks=0 failures=1 skips=0 red_reason=UNEXPECTED_COMPILER_FAILURE
exit /b 2
