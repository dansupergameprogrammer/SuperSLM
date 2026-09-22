# T-2948 F1: stage an isolated developer tree under tests/, then run both no-GPU arms.
param([string]$CompiledShaderDir = 'D:\SuperSLM\build\Release\shaders')
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$engine = (Resolve-Path (Join-Path $here '..\..')).Path
$root = Join-Path $here ('obj\r2-fixture-' + [char]0x00E9)
$sourceDir = Join-Path $root 'src\gpu\shaders'
$binaryDir = Join-Path $root 'compiled'
$exeDir = Join-Path $root 'out\probe'
foreach ($dir in @($sourceDir, $binaryDir, $exeDir)) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
}
$source = Join-Path $sourceDir 'dyn_recip.hlsl'
$binary = Join-Path $binaryDir 'dyn_recip.cso'
$exe = Join-Path $exeDir 'cell_shader_stale_unicode.exe'
Copy-Item -LiteralPath (Join-Path $engine 'src\gpu\shaders\dyn_recip.hlsl') -Destination $source -Force
Copy-Item -LiteralPath (Join-Path $CompiledShaderDir 'dyn_recip.cso') -Destination $binary -Force
[System.IO.File]::SetLastWriteTimeUtc($binary, [datetime]::UtcNow.AddMinutes(-5))
[System.IO.File]::SetLastWriteTimeUtc($source, [datetime]::UtcNow)
Copy-Item -LiteralPath (Join-Path $here 'obj\r2\cell_shader_stale_unicode.exe') -Destination $exe -Force
Write-Host "source_sha256=$((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant())"
Write-Host "cso_sha256=$((Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant())"
& $exe baseline
$baseline = $LASTEXITCODE
& $exe override
$override = $LASTEXITCODE
Write-Host "F1 baseline_exit=$baseline override_exit=$override"
if ($baseline -ne 0) { exit 2 }
exit $override
