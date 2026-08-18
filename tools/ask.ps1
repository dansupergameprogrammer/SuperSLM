<#
.SYNOPSIS
ask.ps1 -- ask the .sslm engine a question and read its answer.

.DESCRIPTION
Wraps the three steps a human otherwise runs by hand: apply the checkpoint's own
chat template to the question, run the greedy decode driver on the real model
artifact, and decode the returned token ids back to text.

The chat template and the vocabulary both come from the checkpoint's own files --
nothing about the model is hardcoded here except the default artifact paths.

WHAT THIS DOES AND DOES NOT ESTABLISH. A readable answer shows the engine
computes a coherent forward pass over real weights. It is NOT evidence that the
output matches what the float reference model would have produced; that is
bit-equality against the reference (plan section 12 criterion 2) and it is not
built. Nor does the C++ suite speak to it: no committed oracle discriminates the
layer weights (DecisionLog D-SLM493), so a green suite says nothing here.

.PARAMETER Question
The question to ask, in plain text.

.PARAMETER MaxNew
Token budget. The model stops on its own end-of-turn token before this in most
cases; raise it if an answer is being cut off mid-sentence.

.PARAMETER System
The system prompt. Defaults to the checkpoint's own conventional one.

.PARAMETER Model
Path to the .sslm model artifact. Required, either as this parameter or via the
SUPERSLM_ASK_MODEL environment variable (T-2152 outside strike item 6: no default
private filesystem path is baked in).

.PARAMETER ShowIds
Also print the raw prompt and output token ids.

.EXAMPLE
.\tools\ask.ps1 "What's on the menu today?"

.EXAMPLE
.\tools\ask.ps1 "Name three colours and say which is warmest." -MaxNew 400
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Question,

    [int]$MaxNew = 256,

    [string]$System = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",

    [string]$Model = $env:SUPERSLM_ASK_MODEL,

    [string]$Tokenizer = "tests\fixtures\qwen2.5-1.5b.tok.sslm",

    [switch]$ShowIds
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    $exe = Join-Path $root "out\sslm_generate.exe"
    if (-not (Test-Path $exe)) {
        Write-Host "building the decode driver (one time)..." -ForegroundColor DarkGray
        & (Join-Path $root "tools\build_generate.bat") 2>&1 | Out-Null
        if (-not (Test-Path $exe)) { throw "driver build failed; run tools\build_generate.bat directly to see why" }
    }
    if ([string]::IsNullOrEmpty($Model)) {
        throw "no model artifact path given -- pass -Model <path> or set SUPERSLM_ASK_MODEL"
    }
    if (-not (Test-Path $Model)) { throw "model artifact not found: $Model" }

    # The template markers must reach the engine as real control tokens, so the
    # prompt is assembled in the checkpoint's own chat format. Written to a file
    # with newline="" so no CRLF conversion changes the line-break token ids.
    $prompt = "<|im_start|>system`n$System<|im_end|>`n<|im_start|>user`n$Question<|im_end|>`n<|im_start|>assistant`n"

    # 151645 = <|im_end|>, 151643 = <|endoftext|>. The .sslm format carries no
    # end-of-sequence id and the tokenizer view exposes none by design, so the
    # stop set is supplied by the caller (DecisionLog D-SLM684).
    #
    # stderr goes to its own file rather than being merged into the success stream.
    # The driver writes an informational preflight line to stderr on every healthy
    # run, and under Windows PowerShell 5.1 a native program writing to stderr while
    # $ErrorActionPreference is 'Stop' raises a terminating NativeCommandError -- so
    # merging with 2>&1 makes a normal run look like a crash on 5.1 while working
    # fine on PowerShell 7. The stderr text is kept and shown if the run fails.
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
    $idLine = if ($idMatch) { $idMatch.ToString() } else { $null }
    if (-not $idLine) {
        Write-Host "the driver did not produce output tokens (exit $driverExit):" -ForegroundColor Red
        Write-Host ($raw | Out-String)
        Write-Host $stderrText
        exit 1
    }
    $ids = ($idLine -replace '^output_tokens \(\d+\):\s*', '').Trim()

    # The stop id the model emitted is part of the returned stream and decodes to
    # a literal "<|im_end|>". It is a turn boundary rather than something the model
    # said, so it is dropped before decoding. Only the stop set is filtered -- every
    # other id the model produced is shown, including anything unexpected.
    $stopSet = @('151645', '151643')
    $ids = (($ids -split '\s+') | Where-Object { $_ -and ($stopSet -notcontains $_) }) -join ' '
    $stopMatch = $raw | Select-String -Pattern '^stop_reason:'
    $stopLine = if ($stopMatch) { $stopMatch.ToString() } else { "" }
    $timeMatch = $raw | Select-String -Pattern '^wall_time_seconds:'
    $timeLine = if ($timeMatch) { $timeMatch.ToString() } else { "" }

    if ($ShowIds) {
        ($raw | Select-String -Pattern '^prompt_tokens')
        Write-Host $idLine
        Write-Host ""
    }

    $answer = & python (Join-Path $root "tools\tokenizer_bridge.py") decode $ids
    Write-Host ""
    Write-Host $answer
    Write-Host ""

    # stop_reason 1 means the model emitted a stop id and finished its turn.
    # stop_reason 0 means it ran out of budget mid-thought -- raise -MaxNew.
    $stopped = $stopLine -replace '^stop_reason:\s*', ''
    if ($stopped.Trim() -eq '0') {
        Write-Host "[cut off at the $MaxNew-token budget, not finished -- re-run with a larger -MaxNew]" -ForegroundColor Yellow
    }
    Write-Host "[$timeLine]" -ForegroundColor DarkGray
}
finally {
    Pop-Location
}
