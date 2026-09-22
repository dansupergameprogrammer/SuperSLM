param([string]$Out = 'D:\_t2933\x2-committed')
$Engine = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$IncludeFirst = Join-Path $PSScriptRoot 't2922_x2_include'
& (Join-Path $Engine 'tests\t2807-api-slot\build_x2.ps1') `
    -Engine $Engine -Out $Out -IncludeFirst $IncludeFirst
exit $LASTEXITCODE
