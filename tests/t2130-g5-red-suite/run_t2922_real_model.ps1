param(
 [string]$Artifact = 'D:\hf_cache\superslm_artifacts\example\qwen2.5-0.5b-instruct-cap4096-aex.sslm',
 [string]$ScratchRoot = 'D:\_t2933\real-model', [string]$ShaderDir = ''
)
if (-not $env:SUPERSLM_G5_REAL_MODEL_TESTS) { Write-Output 'SKIP real_model gate unset'; exit 0 }
if (-not (Test-Path $Artifact)) { Write-Error "required real artifact missing: $Artifact"; exit 2 }
$Here = $PSScriptRoot; $Engine = (Resolve-Path (Join-Path $Here '..\..')).Path
if (-not $ShaderDir) { $ShaderDir = Join-Path $Engine 'build\gpu-shaders-staged' }
$Vs = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1'
$Obj = Join-Path $ScratchRoot 'obj'; $Bin = Join-Path $ScratchRoot 'bin'
New-Item -ItemType Directory -Force $Obj,$Bin,(Join-Path $Bin 'shaders') | Out-Null
Get-ChildItem $Obj -File -ErrorAction SilentlyContinue | Remove-Item -Force
& $Vs -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null
$Common = @('artifact.cpp','sha256.cpp','tokenizer.cpp','model.cpp','intmath.cpp','silu_lut.cpp',
 'matmul.cpp','proof_manifest.cpp','trace_hook.cpp','forward\checked_chain_funnel.cpp',
 'forward\forward_sites.cpp','decode_digest.cpp','damped_greedy_antilm.cpp',
 'damped_greedy_topk.cpp','damped_greedy_phaseD.cpp','damped_greedy_phaseD_loop.cpp') |
 ForEach-Object { Join-Path "$Engine\src" $_ }
$Sources = @($Common) + @("$Engine\src\gpu\gpu_1p0.cpp", "$Engine\src\gpu\superslm_gpu.cpp",
 "$Engine\src\sslm_abi.cpp", "$Here\t2922_real_model_red.cpp")
& cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /I"$Engine\include" /I"$Engine\src" `
 /I"$Engine\src\gpu" /I"$Engine\tests\t2791-gpu-prefill-read-red-suite" /c $Sources /Fo"$Obj\\"
if ($LASTEXITCODE) { exit 2 }
$Exe = Join-Path $Bin 't2922_real_model.exe'
& link /nologo /OUT:"$Exe" (Get-ChildItem $Obj -Filter *.obj | Select-Object -Expand FullName) d3d12.lib dxgi.lib dxguid.lib
if ($LASTEXITCODE) { exit 2 }
if (-not (Test-Path $ShaderDir)) { Write-Error "shader directory missing: $ShaderDir"; exit 2 }
Copy-Item (Join-Path $ShaderDir '*') (Join-Path $Bin 'shaders') -Force
Push-Location $Bin
try { & $Exe $Artifact (Join-Path $Here 't2922_real_prompt.ids'); exit $LASTEXITCODE } finally { Pop-Location }
