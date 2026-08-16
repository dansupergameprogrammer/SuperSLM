<#
.SYNOPSIS
verify_clean_checkout.ps1 -- T-2123/T-2137 design SS5's nine-step clean-checkout acceptance
gate, as a release-verification procedure. NOT a CI job (SS5's own stated constraint: step 5
needs a real, locally-present Qwen2.5-1.5B-Instruct checkpoint snapshot and network access for
its one-time download, and the full corpus/adapter run costs real wall-clock time). Run before
each release that claims "convert your own model" works.

.DESCRIPTION
Extracts a clean `git archive` copy of HEAD (or -Commit) into a scratch directory with no
D:\Wizard/D:\SuperSLM ancestor at any path, then runs every one of design SS5's nine steps
against it using every default implementation (no injected test doubles): configure+build the
CMake targets the CLI/decode steps depend on (sslm_verify, sslm_generate -- build item B9),
calibrate the checkpoint (tools/calibrate_checkpoint.py -- build item B1), convert the
tokenizer, convert the model, decode through the compiled engine, and convert the LoRA adapter
(tools/sslm_convert_adapter.py -- build item B8). Fails loudly (non-zero exit, printed reason)
at whichever step breaks first.

.PARAMETER Checkpoint
Path to a local Qwen2.5-1.5B-Instruct snapshot directory (config.json + safetensors +
tokenizer files). Required.

.PARAMETER Adapter
Path to a local PEFT LoRA adapter directory (adapter_config.json + adapter weights).
Required.

.PARAMETER ScratchDir
Where to extract the clean checkout and write artifacts. Defaults to a temp directory.

.PARAMETER Commit
The commit to archive. Defaults to HEAD.

.PARAMETER CalibrationRecordLimit
Reduce calibration to the first N corpus records (design SS6 B9 confirmation note's own
2-record cell). 0 (default) runs the full 600-record corpus at its real ~105-minute cost --
the actual release-gate obligation. Set to a small number for a fast smoke pass.

.PARAMETER AdapterLayerLimit
Reduce the adapter conversion to the first N of the real adapter's 28 layers (design SS6 B9
confirmation note's own 2-of-28-layer cell). 0 (default) runs the full 196-pair population at
its real ~40-90 minute cost -- the actual release-gate obligation. Set to a small number for a
fast smoke pass.

.PARAMETER Prompt
The prompt sslm_generate decodes at step 8. Defaults to "Hello".

.PARAMETER MaxNewTokens
Tokens sslm_generate decodes at step 8. Defaults to 8.

.EXAMPLE
pwsh -File tools/verify_clean_checkout.ps1 -Checkpoint D:\hf_cache\Qwen2.5-1.5B-Instruct `
    -Adapter D:\hf_cache\qwen2.5-1.5b-shopkeeper-lora-v2\final
Full release-gate run: the real 600-record corpus, the real 196-pair adapter population.

.EXAMPLE
pwsh -File tools/verify_clean_checkout.ps1 -Checkpoint D:\hf_cache\Qwen2.5-1.5B-Instruct `
    -Adapter D:\hf_cache\qwen2.5-1.5b-shopkeeper-lora-v2\final -CalibrationRecordLimit 2 -AdapterLayerLimit 2
Fast smoke pass (minutes, not the full release-gate cost) -- proves the gate's SHAPE and every
step's real code path, at reduced scale.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [Parameter(Mandatory = $true)][string]$Adapter,
    [string]$ScratchDir = (Join-Path ([System.IO.Path]::GetTempPath()) "sslm-clean-checkout-$([guid]::NewGuid().ToString('N').Substring(0,8))"),
    [string]$Commit = "HEAD",
    [int]$CalibrationRecordLimit = 0,
    [int]$AdapterLayerLimit = 0,
    [string]$Prompt = "Hello",
    [int]$MaxNewTokens = 8
)

$ErrorActionPreference = "Stop"
$RepoRoot = (git rev-parse --show-toplevel).Trim()

function Step($n, $msg) {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] step ${n}: $msg" -ForegroundColor Cyan
}

function Fail($n, $msg) {
    Write-Host "verify_clean_checkout.ps1: FAILED at step ${n}: $msg" -ForegroundColor Red
    exit 1
}

$Clean = Join-Path $ScratchDir "clean"
$Artifacts = Join-Path $ScratchDir "artifacts"
New-Item -ItemType Directory -Force -Path $Clean, $Artifacts | Out-Null

# --- step 1: clean git archive extraction, no D:\Wizard/D:\SuperSLM ancestor ----------------
Step 1 "git archive extract ($Commit)"
Push-Location $RepoRoot
try {
    $tarPath = Join-Path $ScratchDir "archive.tar"
    git archive --format=tar $Commit -o $tarPath
    if ($LASTEXITCODE -ne 0) { Fail 1 "git archive failed" }
} finally {
    Pop-Location
}
tar -xf (Join-Path $ScratchDir "archive.tar") -C $Clean
if (-not (Test-Path (Join-Path $Clean "CMakeLists.txt"))) { Fail 1 "extraction did not produce CMakeLists.txt" }

# --- step 2: pip install -r requirements.txt ------------------------------------------------
Step 2 "pip install -r requirements.txt"
$reqPath = Join-Path $Clean "requirements.txt"
if (-not (Test-Path $reqPath)) { Fail 2 "requirements.txt missing from the clean checkout" }
python -m pip install -q -r $reqPath
if ($LASTEXITCODE -ne 0) { Fail 2 "pip install failed" }

# --- steps 3-4: cmake configure + build (sslm_verify, sslm_generate) -----------------------
Step "3-4" "cmake configure + build"
Push-Location $Clean
try {
    cmake -B build -DCMAKE_BUILD_TYPE=Release *> (Join-Path $Artifacts "01_cmake_configure.log")
    if ($LASTEXITCODE -ne 0) { Fail "3-4" "cmake configure failed" }
    cmake --build build --config Release --target sslm_verify --target sslm_generate `
        *> (Join-Path $Artifacts "02_cmake_build.log")
    if ($LASTEXITCODE -ne 0) { Fail "3-4" "cmake build failed" }
} finally {
    Pop-Location
}
$Verify = Join-Path $Clean "build\Release\sslm_verify.exe"
$Generate = Join-Path $Clean "build\Release\sslm_generate.exe"
if (-not (Test-Path $Verify)) { $Verify = Join-Path $Clean "build\sslm_verify" }
if (-not (Test-Path $Generate)) { $Generate = Join-Path $Clean "build\sslm_generate" }
if (-not (Test-Path $Verify)) { Fail "3-4" "sslm_verify binary not produced" }
if (-not (Test-Path $Generate)) { Fail "3-4" "sslm_generate binary not produced" }

# --- step 5: calibrate the checkpoint (build item B1) ----------------------------------------
Step 5 "calibrate_checkpoint.py (the dominant real-cost step -- ~105 min at full scale)"
$ArtifactDir = Join-Path $Artifacts "clean-checkout-artifact"
Push-Location (Join-Path $Clean "tools")
try {
    if ($CalibrationRecordLimit -gt 0) {
        $env:SSLM_CALIBRATION_RECORD_LIMIT = "$CalibrationRecordLimit"
        python -u -c @"
import os, sys
sys.path.insert(0, '.')
from reference_pipeline import pipeline as pl
orig = pl.calibration_records
n = int(os.environ['SSLM_CALIBRATION_RECORD_LIMIT'])
pl.calibration_records = lambda: orig()[:n]
import calibrate_checkpoint as CLI
raise SystemExit(CLI.main(['--checkpoint', r'$Checkpoint', '--out', r'$ArtifactDir']))
"@ *> (Join-Path $Artifacts "03_step5_calibrate.log")
        Remove-Item Env:\SSLM_CALIBRATION_RECORD_LIMIT
    } else {
        python -u calibrate_checkpoint.py --checkpoint "$Checkpoint" --out "$ArtifactDir" `
            *> (Join-Path $Artifacts "03_step5_calibrate.log")
    }
    if ($LASTEXITCODE -ne 0) { Fail 5 "calibrate_checkpoint.py failed" }
} finally {
    Pop-Location
}
if (-not (Select-String -Path (Join-Path $Artifacts "03_step5_calibrate.log") -Pattern "written fingerprint" -Quiet)) {
    Fail 5 "no 'written fingerprint' line in the calibration log"
}

# --- step 6: convert_tokenizer.py -------------------------------------------------------------
Step 6 "convert_tokenizer.py"
$TokenizerArtifact = Join-Path $Artifacts "clean-checkout-tokenizer.sslm"
Push-Location (Join-Path $Clean "tools")
try {
    python -u convert_tokenizer.py --ckpt "$Checkpoint" --emit "$TokenizerArtifact" `
        *> (Join-Path $Artifacts "05_step6_convert_tokenizer.log")
    if ($LASTEXITCODE -ne 0) { Fail 6 "convert_tokenizer.py failed" }
} finally {
    Pop-Location
}
if (-not (Select-String -Path (Join-Path $Artifacts "05_step6_convert_tokenizer.log") -Pattern "^wrote " -Quiet)) {
    Fail 6 "no 'wrote ...' line in the tokenizer conversion log"
}

# --- step 7: convert_model.py (invokes the real compiled sslm_verify.exe) ------------------
Step 7 "convert_model.py"
$ModelArtifact = Join-Path $Artifacts "clean-checkout-model.sslm"
Push-Location (Join-Path $Clean "tools")
try {
    python -u convert_model.py --artifact "$ArtifactDir" --out "$ModelArtifact" `
        *> (Join-Path $Artifacts "04_step7_convert_model.log")
    if ($LASTEXITCODE -ne 0) { Fail 7 "convert_model.py failed" }
} finally {
    Pop-Location
}
if (-not (Select-String -Path (Join-Path $Artifacts "04_step7_convert_model.log") -Pattern "verified: independent loader accepted the artifact" -Quiet)) {
    Fail 7 "no verifier success line in the convert_model.py log"
}

# --- step 8: sslm_generate decode through the compiled engine ------------------------------
Step 8 "sslm_generate decode"
Push-Location $Clean
try {
    & $Generate "$ModelArtifact" "$TokenizerArtifact" "$Prompt" --max-new $MaxNewTokens `
        *> (Join-Path $Artifacts "06_step8_sslm_generate.log")
    if ($LASTEXITCODE -ne 0) { Fail 8 "sslm_generate failed" }
} finally {
    Pop-Location
}
if (-not (Select-String -Path (Join-Path $Artifacts "06_step8_sslm_generate.log") -Pattern "^output_tokens \(" -Quiet)) {
    Fail 8 "no 'output_tokens (...)' line in the decode log"
}

# --- step 9: sslm_convert_adapter.py (build item B8, D-SLM3449) ----------------------------
Step 9 "sslm_convert_adapter.py"
$AdapterArtifact = Join-Path $Artifacts "clean-checkout-adapter.sslm"
Push-Location (Join-Path $Clean "tools")
try {
    if ($AdapterLayerLimit -gt 0) {
        # T-2137 fix round (Poirot casebook f83afe0-t2137-vendoring-review.md, S3): this
        # branch previously hand-reassembled sslm_convert_adapter.py's own main() body
        # (build_runtime_additive_sections + dispatch_conversion_outcome + write_artifact +
        # verify_and_merge) instead of invoking the CLI -- which meant the substitute dropped
        # main()'s own stale-artifact unlink and its rejection-diagnostics printing, and every
        # recorded gate run at this cell exercised the substitute, never the real CLI. The
        # layer-count reduction is the SAME mechanism either way (patch pl.load_config before
        # the call) -- the fix is to patch it in front of the real A.main() invocation, not to
        # avoid calling main() at all. Written to a temp .py file rather than piped through
        # `python -c` -- a PowerShell here-string does not reliably round-trip embedded quotes
        # to the child process.
        $reducedScript = @'
import dataclasses
import os
import sys

sys.path.insert(0, '.')
import sslm_convert_adapter as A
from reference_pipeline import pipeline as pl

n = int(os.environ['SSLM_ADAPTER_LAYER_LIMIT'])
orig_load_config = pl.load_config
pl.load_config = lambda path: dataclasses.replace(orig_load_config(path), num_hidden_layers=n)

sys.argv = [
    'sslm_convert_adapter.py',
    '--adapter', os.environ['SSLM_ADAPTER_DIR'],
    '--base', os.environ['SSLM_BASE_MODEL_PATH'],
    '--out', os.environ['SSLM_ADAPTER_OUT_PATH'],
]
raise SystemExit(A.main())
'@
        $reducedScriptPath = Join-Path $Artifacts "_step9_reduced_cell.py"
        Set-Content -Path $reducedScriptPath -Value $reducedScript -Encoding UTF8

        $env:SSLM_ADAPTER_LAYER_LIMIT = "$AdapterLayerLimit"
        $env:SSLM_ADAPTER_DIR = "$Adapter"
        $env:SSLM_BASE_MODEL_PATH = "$ModelArtifact"
        $env:SSLM_ADAPTER_OUT_PATH = "$AdapterArtifact"
        python -u $reducedScriptPath *> (Join-Path $Artifacts "07_step9_convert_adapter.log")
        Remove-Item Env:\SSLM_ADAPTER_LAYER_LIMIT, Env:\SSLM_ADAPTER_DIR, Env:\SSLM_BASE_MODEL_PATH, Env:\SSLM_ADAPTER_OUT_PATH
    } else {
        python -u sslm_convert_adapter.py --adapter "$Adapter" --base "$ModelArtifact" `
            --out "$AdapterArtifact" *> (Join-Path $Artifacts "07_step9_convert_adapter.log")
    }
    if ($LASTEXITCODE -ne 0) { Fail 9 "sslm_convert_adapter.py failed" }
} finally {
    Pop-Location
}
if (-not (Select-String -Path (Join-Path $Artifacts "07_step9_convert_adapter.log") -Pattern "verified: independent loader accepted the artifact" -Quiet)) {
    Fail 9 "no verifier success line in the adapter conversion log"
}

Write-Host "verify_clean_checkout.ps1: all nine gate steps executed for real; evidence in $Artifacts" -ForegroundColor Green
exit 0
