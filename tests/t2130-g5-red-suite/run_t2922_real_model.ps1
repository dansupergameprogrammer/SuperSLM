param(
 [string]$Artifact = '',
 [string]$ScratchRoot = '', [string]$ShaderDir = ''
)
$Here = $PSScriptRoot; $Engine = (Resolve-Path (Join-Path $Here '..\..')).Path
$ManifestPath = Join-Path $Here 't2922_real_model_manifest.json'
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
if (-not $Artifact) { $Artifact = [string]$Manifest.artifact }
if (-not $env:SUPERSLM_G5_REAL_MODEL_TESTS) { Write-Output 'SKIP real_model gate unset'; exit 0 }
if (-not (Test-Path $Artifact)) { Write-Error "required real artifact missing: $Artifact"; exit 2 }
$ArtifactItem = Get-Item -LiteralPath $Artifact
$ArtifactHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Artifact).Hash.ToLowerInvariant()
$Prompt = Join-Path $Here 't2922_real_prompt.ids'
$PromptHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Prompt).Hash.ToLowerInvariant()
if ($ArtifactItem.Length -ne [int64]$Manifest.artifact_bytes -or
    $ArtifactHash -ne [string]$Manifest.artifact_sha256 -or
    $PromptHash -ne [string]$Manifest.prompt_ids_sha256 -or
    [string]$Manifest.schema_name -ne 'prompt_result' -or
    [int]$Manifest.decode_budget -ne 300 -or [string]$Manifest.expected_stop -ne 'budget') {
    Write-Error 'real-model manifest/input mismatch'; exit 2
}
Write-Output "MANIFEST VERIFIED artifact_sha256=$ArtifactHash prompt_sha256=$PromptHash schema=$($Manifest.schema_name)"
if (-not $ScratchRoot) { $ScratchRoot = Join-Path $Engine 'build\t2933-real-model' }
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
& cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /DSUPERSLM_ENABLE_GPU_CHUNK_DISPATCH_INSTRUMENT /I"$Engine\include" /I"$Engine\src" /I"$Engine\tests" `
 /I"$Engine\src\gpu" /I"$Engine\tests\t2791-gpu-prefill-read-red-suite" /c $Sources /Fo"$Obj\\"
if ($LASTEXITCODE) { exit 2 }
$Exe = Join-Path $Bin 't2922_real_model.exe'
& link /nologo /OUT:"$Exe" (Get-ChildItem $Obj -Filter *.obj | Select-Object -Expand FullName) d3d12.lib dxgi.lib dxguid.lib
if ($LASTEXITCODE) { exit 2 }
if (-not (Test-Path $ShaderDir)) { Write-Error "shader directory missing: $ShaderDir"; exit 2 }
Copy-Item (Join-Path $ShaderDir '*') (Join-Path $Bin 'shaders') -Force
Push-Location $Bin
try { & $Exe $Artifact $Prompt; exit $LASTEXITCODE } finally { Pop-Location }
