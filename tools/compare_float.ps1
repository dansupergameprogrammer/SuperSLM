<#
.SYNOPSIS
compare_float.ps1 -- ask the same question of the .sslm int8 engine and the
float reference checkpoint, greedy, and print both answers side by side.

.DESCRIPTION
WHAT THIS COMPARISON ESTABLISHES AND DOES NOT ESTABLISH. READ THIS BEFORE
ACTING ON ITS OUTPUT.

This tool answers one question: when the engine's output looks mediocre, is
that the 1.5B model being a 1.5B model, or a numeric defect in the int8 path?
It runs the SAME prompt, SAME chat template, SAME system prompt, SAME token
budget, and SAME stop ids through both paths under GREEDY decoding
(do_sample=False on the float side; the .sslm side has no other decode mode)
and prints both decoded answers together with the first token position at
which the two output-id streams diverge.

Divergence in token choice between an int8-quantized path and a float path is
EXPECTED and is NOT by itself a defect. The two paths are not required to
agree token-for-token, and this is not bit-equality against the reference --
that is plan section 12 criterion 2, and it is a separate, unbuilt thing. The
committed C++ suite cannot help answer this question either: no oracle in it
discriminates the layer weights (DecisionLog D-SLM493), so a green suite says
nothing about whether these two outputs should agree.

What this comparison IS good for: the gross case -- the float answer is
coherent while the .sslm answer is mush; the two answers agree closely; or
the divergence point comes suspiciously early (e.g. within the first few
generated tokens) rather than late, drifting apart the way two otherwise-sane
decoders naturally do over a long generation. The first-divergence index is
the single most informative number this tool reports and is reported whether
or not either output is separately judged coherent.

.PARAMETER Question
The question to ask, in plain text. Identical on both paths.

.PARAMETER MaxNew
Token budget, applied identically to both paths.

.PARAMETER System
The system prompt, applied identically to both paths.

.PARAMETER Model
Path to the .sslm model artifact (the int8 side).

.PARAMETER Tokenizer
Path to the .sslm tokenizer artifact used by the C++ driver.

.PARAMETER RefModel
Path to the local HuggingFace float checkpoint directory (or its hub-cache
repo directory, which is resolved to its one snapshot automatically).

.EXAMPLE
.\tools\compare_float.ps1 "Write a two-line limerick about a cat."

.EXAMPLE
.\tools\compare_float.ps1 "Name three colours and say which is warmest." -MaxNew 120
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Question,

    [int]$MaxNew = 128,

    [string]$System = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",

    [string]$Model = $env:SUPERSLM_MODEL_PATH,

    [string]$Tokenizer = "tests\fixtures\qwen2.5-1.5b.tok.sslm",

    [string]$RefModel = $env:SUPERSLM_FLOAT_REF_MODEL
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    Write-Host "=====================================================================" -ForegroundColor DarkGray
    Write-Host "float side-by-side -- NOT a correctness check against the reference." -ForegroundColor Yellow
    Write-Host "Divergence between the two paths is EXPECTED; this is a coarse" -ForegroundColor Yellow
    Write-Host "behavioral read (gross mush vs. coherent, or a suspiciously early" -ForegroundColor Yellow
    Write-Host "split), not bit-equality (plan section 12 criterion 2, unbuilt)." -ForegroundColor Yellow
    Write-Host "=====================================================================" -ForegroundColor DarkGray
    Write-Host ""

    $exe = Join-Path $root "out\sslm_generate.exe"
    if (-not (Test-Path $exe)) {
        Write-Host "building the decode driver (one time)..." -ForegroundColor DarkGray
        & (Join-Path $root "tools\build_generate.bat") 2>&1 | Out-Null
        if (-not (Test-Path $exe)) { throw "driver build failed; run tools\build_generate.bat directly to see why" }
    }
    if ([string]::IsNullOrEmpty($Model)) {
        throw "no model artifact path given -- pass -Model <path> or set SUPERSLM_MODEL_PATH"
    }
    if ([string]::IsNullOrEmpty($RefModel)) {
        throw "no float reference model path given -- pass -RefModel <path> or set SUPERSLM_FLOAT_REF_MODEL"
    }
    if (-not (Test-Path $Model)) { throw "model artifact not found: $Model" }

    $prompt = "<|im_start|>system`n$System<|im_end|>`n<|im_start|>user`n$Question<|im_end|>`n<|im_start|>assistant`n"

    # --- ours: the .sslm int8 engine (same invocation and stderr handling as tools\ask.ps1) ---
    $errFile = [System.IO.Path]::GetTempFileName()
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $raw = & $exe $Model $Tokenizer $prompt --max-new $MaxNew --stop 151645,151643 2> $errFile
    }
    finally {
        $ErrorActionPreference = $prevEap
    }
    $driverExit = $LASTEXITCODE
    $stderrText = if (Test-Path $errFile) { Get-Content $errFile -Raw } else { "" }
    Remove-Item $errFile -ErrorAction SilentlyContinue

    $idMatch = $raw | Select-String -Pattern '^output_tokens \(\d+\):'
    if (-not $idMatch) {
        Write-Host "OURS (.sslm int8): the driver did not produce output tokens (exit $driverExit):" -ForegroundColor Red
        Write-Host ($raw | Out-String)
        Write-Host $stderrText
        exit 1
    }
    $oursIdsRaw = (($idMatch.ToString()) -replace '^output_tokens \(\d+\):\s*', '').Trim()
    $stopSet = @('151645', '151643')
    $oursIds = ($oursIdsRaw -split '\s+') | Where-Object { $_ }
    $oursIdsFiltered = ($oursIds | Where-Object { $stopSet -notcontains $_ }) -join ' '
    $oursAnswer = & python (Join-Path $root "tools\tokenizer_bridge.py") decode $oursIdsFiltered
    $oursTimeMatch = $raw | Select-String -Pattern '^wall_time_seconds:'
    $oursTime = if ($oursTimeMatch) { ($oursTimeMatch.ToString() -replace '^wall_time_seconds:\s*', '') } else { "?" }

    # --- reference: the float base checkpoint via transformers, greedy (do_sample=False) ---
    $refRaw = & python (Join-Path $root "tools\float_reference_generate.py") $Question --system $System --max-new $MaxNew --model $RefModel 2>&1
    $refExit = $LASTEXITCODE
    if ($refExit -ne 0) {
        Write-Host "FLOAT REFERENCE: failed (exit $refExit):" -ForegroundColor Red
        Write-Host ($refRaw | Out-String)
        exit 1
    }
    $refIdsLine = ($refRaw | Select-String -Pattern '^output_ids:').ToString()
    $refIds = (($refIdsLine -replace '^output_ids:\s*', '').Trim() -split '\s+') | Where-Object { $_ }
    $refTimeMatch = $refRaw | Select-String -Pattern '^wall_time_seconds:'
    $refTime = if ($refTimeMatch) { ($refTimeMatch.ToString() -replace '^wall_time_seconds:\s*', '') } else { "?" }
    $decodedMarkerIdx = ($refRaw | Select-String -Pattern '^---DECODED---').LineNumber
    $refAnswer = if ($decodedMarkerIdx) { ($refRaw | Select-Object -Skip $decodedMarkerIdx) -join "`n" } else { "" }

    # --- first-divergence index: compare generated id streams position by position,
    # including any stop id each path emitted (both paths' ids are aligned from the
    # same prompt, so index i on both sides is decode step i). ---
    $maxLen = [Math]::Max($oursIds.Count, $refIds.Count)
    $divergeAt = -1
    for ($i = 0; $i -lt $maxLen; $i++) {
        $o = if ($i -lt $oursIds.Count) { $oursIds[$i] } else { $null }
        $r = if ($i -lt $refIds.Count) { $refIds[$i] } else { $null }
        if ($o -ne $r) { $divergeAt = $i; break }
    }
    if ($divergeAt -eq -1 -and $oursIds.Count -eq $refIds.Count) {
        $divergenceReport = "no divergence -- both paths produced token-for-token identical output ($($oursIds.Count) tokens)"
    }
    elseif ($divergeAt -eq -1) {
        $divergenceReport = "no divergence within the shorter stream, but lengths differ (ours=$($oursIds.Count), float=$($refIds.Count)) -- one path stopped earlier"
    }
    else {
        $oursTok = if ($divergeAt -lt $oursIds.Count) { $oursIds[$divergeAt] } else { "<none, ours ended>" }
        $refTok = if ($divergeAt -lt $refIds.Count) { $refIds[$divergeAt] } else { "<none, float ended>" }
        $divergenceReport = "first divergence at generated-token index $divergeAt (ours=$oursTok, float=$refTok)"
    }

    Write-Host "Question: $Question" -ForegroundColor Cyan
    Write-Host "System:   $System" -ForegroundColor DarkGray
    Write-Host "MaxNew:   $MaxNew"
    Write-Host ""
    Write-Host "--- OURS (.sslm int8 engine, greedy) [$oursTime s] ---" -ForegroundColor Green
    Write-Host $oursAnswer
    Write-Host ""
    Write-Host "--- FLOAT REFERENCE (transformers, greedy, do_sample=False) [$refTime s] ---" -ForegroundColor Green
    Write-Host $refAnswer
    Write-Host ""
    Write-Host "--- FIRST-DIVERGENCE INDEX ---" -ForegroundColor Magenta
    Write-Host $divergenceReport
    Write-Host ""
    Write-Host "Reminder: divergence is expected and is not by itself a defect. See this" -ForegroundColor DarkGray
    Write-Host "script's own header (Get-Help .\tools\compare_float.ps1 -Full) for what" -ForegroundColor DarkGray
    Write-Host "this comparison does and does not establish." -ForegroundColor DarkGray
}
finally {
    Pop-Location
}
