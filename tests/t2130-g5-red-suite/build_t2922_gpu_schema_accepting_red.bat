@echo off
rem T-2933: three independent compile/link/run cells.  A missing declaration is
rem an expected RED result, but any unrelated compiler failure is infrastructure.
setlocal enabledelayedexpansion
set HERE=%~dp0
set ENG=%HERE%..\..
set OBJ=%HERE%obj_t2922
if not exist "%OBJ%" mkdir "%OBJ%"
if not exist "%OBJ%\accepting" mkdir "%OBJ%\accepting"
if not exist "%OBJ%\gpu_bound" mkdir "%OBJ%\gpu_bound"
if not exist "%OBJ%\cpu_bound" mkdir "%OBJ%\cpu_bound"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
set COMMON=%ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp %ENG%\src\forward\forward_sites.cpp %ENG%\src\decode_digest.cpp %ENG%\src\damped_greedy_antilm.cpp %ENG%\src\damped_greedy_topk.cpp %ENG%\src\damped_greedy_phaseD.cpp %ENG%\src\damped_greedy_phaseD_loop.cpp
set GPU=%ENG%\src\gpu\gpu_1p0.cpp %ENG%\src\gpu\superslm_gpu.cpp
set RED=0
set GREEN=0
set BAD=0

cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /I"%ENG%\include" /I"%ENG%\src" /I"%ENG%\src\gpu" %COMMON% %GPU% "%HERE%t2922_gpu_schema_accepting_red.cpp" /Fo"%OBJ%\accepting\\" /Fe"%OBJ%\accepting.exe" /link d3d12.lib dxgi.lib dxguid.lib >"%OBJ%\accepting.log" 2>&1
if errorlevel 1 (
  findstr /C:"SslmGpuSeqSchemaAcceptingForG5Bridge" "%OBJ%\accepting.log" >nul && (echo RED gpu_accepting_api reason=API_ABSENT& set /a RED+=1) || (type "%OBJ%\accepting.log"& set /a BAD+=1)
) else (
  "%OBJ%\accepting.exe" && (echo GREEN gpu_accepting_api& set /a GREEN+=1) || (echo RED gpu_accepting_api reason=CONTRACT_MISMATCH& set /a RED+=1)
)

cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /I"%ENG%\include" /I"%ENG%\src" /I"%ENG%\src\gpu" %COMMON% %GPU% "%HERE%t2922_gpu_schema_bound_red.cpp" /Fo"%OBJ%\gpu_bound\\" /Fe"%OBJ%\gpu_bound.exe" /link d3d12.lib dxgi.lib dxguid.lib >"%OBJ%\gpu_bound.log" 2>&1
if errorlevel 1 (
  findstr /C:"SslmGpuSeqSchemaBoundForG5Bridge" "%OBJ%\gpu_bound.log" >nul && (echo RED gpu_bound_api reason=API_ABSENT& set /a RED+=1) || (type "%OBJ%\gpu_bound.log"& set /a BAD+=1)
) else (
  "%OBJ%\gpu_bound.exe" && (echo GREEN gpu_bound_api& set /a GREEN+=1) || (echo RED gpu_bound_api reason=CONTRACT_MISMATCH& set /a RED+=1)
)

cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /I"%ENG%\include" /I"%ENG%\src" %COMMON% "%ENG%\src\sslm_abi.cpp" "%HERE%t2922_cpu_schema_bound_red.cpp" /Fo"%OBJ%\cpu_bound\\" /Fe"%OBJ%\cpu_bound.exe" >"%OBJ%\cpu_bound.log" 2>&1
if errorlevel 1 (
  findstr /C:"sslm_seq_schema_bound" "%OBJ%\cpu_bound.log" >nul && (echo RED cpu_bound_api reason=API_ABSENT& set /a RED+=1) || (type "%OBJ%\cpu_bound.log"& set /a BAD+=1)
) else (
  "%OBJ%\cpu_bound.exe" && (echo GREEN cpu_bound_api& set /a GREEN+=1) || (echo RED cpu_bound_api reason=CONTRACT_MISMATCH& set /a RED+=1)
)

if not "!BAD!"=="0" exit /b 2
if "!RED!"=="0" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%run_t2922_query_runtime.ps1"
  if errorlevel 1 (set /a BAD+=1) else (set /a GREEN+=1)
)
echo SUMMARY checks=3 red=!RED! green=!GREEN! infrastructure_failures=!BAD!
if not "!BAD!"=="0" exit /b 2
if "!RED!"=="0" exit /b 0
exit /b 1
