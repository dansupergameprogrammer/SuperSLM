param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('stream-accept','stream-reject','byte-accept','byte-reject')]
    [string]$Cell,
    [switch]$Reuse
)

$pythonCell = Join-Path $PSScriptRoot 't3071_pre_cache_cells.py'
if ($Reuse) {
    & python $pythonCell --cell $Cell --reuse
} else {
    & python $pythonCell --cell $Cell
}
$cellExit = $LASTEXITCODE

if ($Cell.EndsWith('-reject')) {
    # A crash or failed expectation must read as acceptance by the registry tool,
    # making the instrument DEAD. Only a clean, observed refusal exits nonzero.
    if ($cellExit -eq 0) {
        Write-Host "INSTRUMENT REJECTED $Cell through its own observation"
        exit 1
    }
    Write-Host "CONSTRUCTION FAILED $Cell, exit $cellExit; no valid must-reject"
    exit 0
}

if ($cellExit -eq 0) {
    Write-Host "INSTRUMENT ACCEPTED $Cell correct restore"
    exit 0
}
Write-Host "CONSTRUCTION FAILED $Cell, exit $cellExit"
exit 1
