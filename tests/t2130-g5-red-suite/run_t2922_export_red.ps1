param([string]$Out = '')
$Engine = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not $Out) { $Out = Join-Path $Engine 'build\t2933-x2' }
& (Join-Path $Engine 'tests\t2807-api-slot\build_x2.ps1') -Engine $Engine -Out $Out
exit $LASTEXITCODE
