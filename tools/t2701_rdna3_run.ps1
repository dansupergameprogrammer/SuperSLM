param(
    [Parameter(Mandatory = $true)] [string] $Artifact,
    [string] $PackageRoot = $PSScriptRoot,
    [int[]] $TokenIds = @(42, 7, 151)
)

$ErrorActionPreference = 'Stop'
$probe = Join-Path $PackageRoot 't2100_gpu_throughput.exe'
if (-not (Test-Path -LiteralPath $probe)) { throw "Missing probe: $probe" }
if (-not (Test-Path -LiteralPath $Artifact)) { throw "Missing artifact: $Artifact" }

Write-Host 'T-2701 RDNA3 run-only package'
foreach ($token in $TokenIds) {
    Write-Host "--- token $token ---"
    & $probe $Artifact 1 $token
    if ($LASTEXITCODE -ne 0) { throw "probe failed for token $token" }
}
Write-Host 'Record each IDENTICAL verdict as the gpu-rdna3 product row; this package does not mutate an artifact.'
