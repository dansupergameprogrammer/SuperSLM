param(
    [string]$ScratchRoot = '',
    [string]$ShaderDir = '',
    [string]$GpuSource = '',
    [switch]$ExpectResetRed
)
$ErrorActionPreference = 'Stop'
$Here = $PSScriptRoot
$Engine = (Resolve-Path (Join-Path $Here '..\..')).Path
if (-not $ScratchRoot) { $ScratchRoot = Join-Path $Engine 'build\t2933-lifecycle' }
if (-not $ShaderDir) { $ShaderDir = Join-Path $Engine 'build\gpu-shaders-staged' }
if (-not $GpuSource) { $GpuSource = Join-Path $Engine 'src\gpu\gpu_1p0.cpp' }
$Vs = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1'
$Obj = Join-Path $ScratchRoot 'obj'; $Bin = Join-Path $ScratchRoot 'bin'
New-Item -ItemType Directory -Force $ScratchRoot,$Obj,$Bin,(Join-Path $Bin 'shaders'),
    (Join-Path $Obj 'common'),(Join-Path $Obj 'gpu'),(Join-Path $Obj 'cells'),(Join-Path $Obj 'cpu') | Out-Null
Remove-Item "$Obj\common\*.obj","$Obj\gpu\*.obj","$Obj\cells\*.obj","$Obj\cpu\*.obj" -Force -ErrorAction SilentlyContinue
& $Vs -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null

$Common = @('artifact.cpp','sha256.cpp','tokenizer.cpp','model.cpp','intmath.cpp','silu_lut.cpp',
 'matmul.cpp','proof_manifest.cpp','trace_hook.cpp','forward\checked_chain_funnel.cpp',
 'forward\forward_sites.cpp','decode_digest.cpp','damped_greedy_antilm.cpp',
 'damped_greedy_topk.cpp','damped_greedy_phaseD.cpp','damped_greedy_phaseD_loop.cpp') |
 ForEach-Object { Join-Path "$Engine\src" $_ }
& cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /I"$Engine\include" /I"$Engine\src" /c $Common /Fo"$Obj\common\\"
if ($LASTEXITCODE) { throw "common compile failed: $LASTEXITCODE" }
& cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /I"$Engine\include" /I"$Engine\src\gpu" /c `
  $GpuSource "$Engine\src\gpu\superslm_gpu.cpp" /Fo"$Obj\gpu\\"
if ($LASTEXITCODE) { throw "GPU compile failed: $LASTEXITCODE" }
$CellIncludes = @("/I$Engine\include","/I$Engine\src","/I$Engine\tests\t2791-gpu-prefill-read-red-suite")
& cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc $CellIncludes /c `
  "$Here\t2922_gpu_late_bind_red.cpp" "$Here\t2922_gpu_idle_reset.cpp" /Fo"$Obj\cells\\"
if ($LASTEXITCODE) { throw "GPU cell compile failed: $LASTEXITCODE" }
& cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /I"$Engine\include" /I"$Engine\src" /c `
  "$Engine\src\sslm_abi.cpp" "$Here\t2922_cpu_terminal_reset_red.cpp" /Fo"$Obj\cpu\\"
if ($LASTEXITCODE) { throw "CPU cell compile failed: $LASTEXITCODE" }
$CommonObj = Get-ChildItem "$Obj\common" -Filter *.obj | Select-Object -Expand FullName
$GpuObj = Get-ChildItem "$Obj\gpu" -Filter *.obj | Select-Object -Expand FullName
& link /nologo /OUT:"$Bin\gpu_late_bind.exe" $CommonObj $GpuObj "$Obj\cells\t2922_gpu_late_bind_red.obj" d3d12.lib dxgi.lib dxguid.lib
if ($LASTEXITCODE) { throw "late-bind link failed: $LASTEXITCODE" }
& link /nologo /OUT:"$Bin\gpu_idle_reset.exe" $CommonObj $GpuObj "$Obj\cells\t2922_gpu_idle_reset.obj" d3d12.lib dxgi.lib dxguid.lib
if ($LASTEXITCODE) { throw "GPU reset link failed: $LASTEXITCODE" }
& link /nologo /OUT:"$Bin\cpu_terminal_reset.exe" $CommonObj "$Obj\cpu\sslm_abi.obj" "$Obj\cpu\t2922_cpu_terminal_reset_red.obj"
if ($LASTEXITCODE) { throw "CPU reset link failed: $LASTEXITCODE" }
if (-not (Test-Path $ShaderDir)) { throw "shader directory missing: $ShaderDir" }
Copy-Item (Join-Path $ShaderDir '*') (Join-Path $Bin 'shaders') -Force

$Base = Join-Path $ScratchRoot 'base.sslm'; $CpuBase = Join-Path $ScratchRoot 'cpu-base.sslm'
$Dual = Join-Path $ScratchRoot 'dual.sslm'
& python "$Engine\tools\_t2199_s8_synthetic_full_model_fixture.py" $Base
if ($LASTEXITCODE) { throw "base fixture failed" }
& python "$Here\make_t2922_cpu_reset_base.py" $CpuBase
if ($LASTEXITCODE) { throw "CPU base fixture failed" }
& python "$Here\make_t2922_dual_schema_fixture.py" $Dual
if ($LASTEXITCODE) { throw "dual fixture failed" }
$red = 0; $green = 0; $bad = 0
Push-Location $Bin
try {
  & "$Bin\gpu_late_bind.exe" $Dual; $late = $LASTEXITCODE
  if ($late -eq 1) { ++$red } elseif ($late -eq 0) { ++$green } else { ++$bad }
  foreach ($layer in 1..7) {
    $Guard = Join-Path $ScratchRoot "gpu-guard-layer$layer.sslm"
    $CpuGuard = Join-Path $ScratchRoot "cpu-guard-layer$layer.sslm"
    & python "$Here\make_t2922_reset_guard_fixture.py" $Base $layer $Guard
    if ($LASTEXITCODE) { ++$bad; continue }
    & "$Bin\gpu_idle_reset.exe" $Guard $layer
    if ($ExpectResetRed) {
      if ($LASTEXITCODE -eq 1) { ++$red } elseif ($LASTEXITCODE -eq 0) { ++$green } else { ++$bad }
    } else {
      if ($LASTEXITCODE -eq 0) { ++$green } else { ++$bad }
    }
    & python "$Here\make_t2922_reset_guard_fixture.py" $CpuBase $layer $CpuGuard
    if ($LASTEXITCODE) { ++$bad; continue }
    & "$Bin\cpu_terminal_reset.exe" $CpuGuard $layer
    if ($LASTEXITCODE -eq 1) { ++$red } elseif ($LASTEXITCODE -eq 0) { ++$green } else { ++$bad }
  }
} finally { Pop-Location }
Write-Output "SUMMARY checks=15 red=$red green=$green infrastructure_failures=$bad"
if ($bad) { exit 2 }
if ($red) { exit 1 }
exit 0
