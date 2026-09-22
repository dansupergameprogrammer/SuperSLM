param(
    [string]$ScratchRoot = '',
    [string]$ShaderDir = '',
    [string]$GpuSource = '',
    [string]$CpuSource = ''
)
$ErrorActionPreference = 'Stop'
$Here = $PSScriptRoot
. (Join-Path $Here 'resolve_toolchain.ps1')
$Engine = (Resolve-Path (Join-Path $Here '..\..')).Path
if (-not $ScratchRoot) { $ScratchRoot = Join-Path $Engine 'build\t2933-lifecycle' }
if (-not $ShaderDir) { $ShaderDir = Join-Path $Engine 'build\gpu-shaders-staged' }
if (-not $GpuSource) { $GpuSource = Join-Path $Engine 'src\gpu\gpu_1p0.cpp' }
if (-not $CpuSource) { $CpuSource = Join-Path $Engine 'src\sslm_abi.cpp' }
$Python = Resolve-SuperSlmPython
$Obj = Join-Path $ScratchRoot 'obj'; $Bin = Join-Path $ScratchRoot 'bin'
New-Item -ItemType Directory -Force $ScratchRoot,$Obj,$Bin,(Join-Path $Bin 'shaders') | Out-Null
Get-ChildItem -LiteralPath $Obj -File -ErrorAction SilentlyContinue | Remove-Item -Force
Enter-SuperSlmVsDevShell

$Common = @('artifact.cpp','sha256.cpp','tokenizer.cpp','model.cpp','intmath.cpp','silu_lut.cpp',
 'matmul.cpp','proof_manifest.cpp','trace_hook.cpp','forward\checked_chain_funnel.cpp',
 'forward\forward_sites.cpp','decode_digest.cpp','damped_greedy_antilm.cpp',
 'damped_greedy_topk.cpp','damped_greedy_phaseD.cpp','damped_greedy_phaseD_loop.cpp') |
 ForEach-Object { Join-Path "$Engine\src" $_ }
& cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /I"$Engine\include" /I"$Engine\src" `
  /I"$Engine\src\gpu" /I"$Engine\tests" /I"$Engine\tests\t2791-gpu-prefill-read-red-suite" `
  /c $Common $GpuSource "$Engine\src\gpu\superslm_gpu.cpp" $CpuSource `
  "$Here\t2941_simple_bind_red.cpp" /Fo"$Obj\\"
if ($LASTEXITCODE) { throw "lifecycle compile failed: $LASTEXITCODE" }
& link /nologo /OUT:"$Bin\t2941_simple_bind_red.exe" `
  (Get-ChildItem -LiteralPath $Obj -Filter *.obj | Select-Object -Expand FullName) `
  d3d12.lib dxgi.lib dxguid.lib
if ($LASTEXITCODE) { throw "lifecycle link failed: $LASTEXITCODE" }
if (-not (Test-Path $ShaderDir)) { throw "shader directory missing: $ShaderDir" }
Copy-Item (Join-Path $ShaderDir '*') (Join-Path $Bin 'shaders') -Force

$Dual = Join-Path $ScratchRoot 'dual.sslm'
$Guard = Join-Path $ScratchRoot 'guard-layer4.sslm'
$Adapter = Join-Path $ScratchRoot 'zero-adapter.sslm'
& $Python "$Here\make_t2922_dual_schema_fixture.py" $Dual
if ($LASTEXITCODE) { throw "dual fixture failed" }
& $Python "$Here\make_t2922_reset_guard_fixture.py" $Dual 4 $Guard
if ($LASTEXITCODE) { throw "guard fixture failed" }
& $Python "$Here\make_t2941_zero_adapter_fixture.py" $Dual $Adapter
if ($LASTEXITCODE) { throw "adapter fixture failed" }
Push-Location $Bin
try {
  & "$Bin\t2941_simple_bind_red.exe" $Dual $Guard $Adapter
  exit $LASTEXITCODE
} finally { Pop-Location }
