<#
.SYNOPSIS
  Assembles the T-1872 portable probe bundle: a self-contained folder that runs
  on a Windows machine with no internet access and no pre-existing Python or
  ROCm install, to answer whether a 7900 XTX (gfx1100) can run this project's
  model under native-Windows ROCm PyTorch at all.

.DESCRIPTION
  This script runs on a machine WITH internet access (the assembly machine) and
  writes the finished bundle to -OutDir. Nothing this script downloads or builds
  is committed to git -- see tools/t1872_probe/README.txt in this repo for the
  probe SOURCE, which this script copies into the bundle; the wheels, the
  portable Python, and the model checkpoint are fetched/copied fresh each run
  and never enter version control.

  Seven steps, each idempotent (safe to re-run; skips work already done):
    1. Download the Python 3.12 embeddable distribution and bootstrap pip.
    2. Download AMD's official Windows ROCm 7.2.1 SDK wheels + the matching
       torch 2.9.1+rocm7.2.1 wheel for Python 3.12 (from repo.radeon.com,
       per https://rocm.docs.amd.com/projects/radeon-ryzen/.../install-pytorch.html).
    3. Install those wheels, plus transformers/tokenizers/safetensors/numpy,
       into the portable Python's own site-packages -- so nothing needs
       installing on the target machine, ever.
    4. Copy this project's own model checkpoint (Qwen2.5-1.5B-Instruct, the
       identical checkpoint the RTX 2080 Super harness uses) into the bundle,
       DEREFERENCING the HuggingFace hub cache's symlinks -- exFAT/FAT32 drives
       do not support symlinks, so the files must be real copies.
    5. Copy the probe source (this directory's probe\ subfolder) into the
       bundle and patch the portable Python's python312._pth file so
       "import common" works from a stage script -- confirmed necessary by
       execution: the embeddable distribution's isolated path mode does not
       add a script's own directory to sys.path, and does not honor
       PYTHONPATH either.
    6. Sanity checks: no symlinks anywhere, no file over 4 GiB.
    7. Pack the whole source tree (model/probe/python) into ONE plain tar
       archive at <OutDir>-drive\T1872_Probe.tar, alongside run_probe.bat,
       run_probe.ps1, and README.txt -- THAT staging folder, not -OutDir
       itself, is what gets copied to the target drive. A directory tree of
       this many small files was measured copying to a budget USB flash drive
       at 0.04 MB/s (the controller's random-write floor); one sequential
       archive file avoids that floor entirely. Plain tar, not gzip: measured
       on this bundle's own components, compression bought 20-27% smaller
       output at roughly 1/20th of plain tar's throughput -- not worth it
       unless the drive's SEQUENTIAL write speed is also unusually slow (see
       the comment at Step 7 below for the arithmetic).

.PARAMETER OutDir
  Where to write the assembled SOURCE TREE (not what goes on the drive --
  see Step 7). Must be outside any git repository.

.PARAMETER ModelDir
  Local path to the already-downloaded Qwen2.5-1.5B-Instruct HF snapshot
  (the directory containing config.json, model.safetensors, tokenizer.json,
  etc -- symlinks are fine here, this script dereferences them on copy).

.PARAMETER CorpusFile
  Local path to this project's shopkeeper_corpus_v1.jsonl (or any JSONL with
  an "utterance" field) -- a small subset is copied in as the probe's real
  prompts.

.EXAMPLE
  pwsh -File assemble_bundle.ps1 -OutDir D:\T1872-PortableProbe `
    -ModelDir "D:\hf_cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306" `
    -CorpusFile "D:\Wizard\Claude\Docs\spike\shopkeeper_corpus_v1.jsonl"
#>
param(
    [Parameter(Mandatory=$true)][string]$OutDir,
    [Parameter(Mandatory=$true)][string]$ModelDir,
    [Parameter(Mandatory=$true)][string]$CorpusFile,
    [int]$NumPrompts = 12
)

$ErrorActionPreference = "Stop"
$ProbeSrc = $PSScriptRoot

Write-Host "=== T-1872 bundle assembly -> $OutDir ==="

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path "$OutDir\wheels" | Out-Null
New-Item -ItemType Directory -Force -Path "$OutDir\model" | Out-Null

# --- Step 1: portable Python 3.12 embeddable + pip ---
$PyDir = "$OutDir\python"
if (-not (Test-Path "$PyDir\python.exe")) {
    Write-Host "[1/7] Downloading Python 3.12.10 embeddable distribution..."
    $zipPath = "$OutDir\python-3.12.10-embed-amd64.zip"
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip" -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $PyDir -Force
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile "$OutDir\get-pip.py"
    & "$PyDir\python.exe" "$OutDir\get-pip.py" --no-warn-script-location
} else {
    Write-Host "[1/7] Portable Python already present, skipping."
}

# --- Step 2: AMD ROCm 7.2.1 Windows SDK + torch wheels ---
$RocmBase = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
$RocmFiles = @(
    "rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
    "rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
    "rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
    "rocm-7.2.1.tar.gz"
)
Write-Host "[2/7] Downloading AMD ROCm 7.2.1 Windows SDK wheels + torch 2.9.1+rocm7.2.1..."
foreach ($f in $RocmFiles) {
    $dest = "$OutDir\wheels\$f"
    if (-not (Test-Path $dest)) {
        Invoke-WebRequest -Uri "$RocmBase/$f" -OutFile $dest
    }
}
$torchWheel = "$OutDir\wheels\torch-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl"
if (-not (Test-Path $torchWheel)) {
    Invoke-WebRequest -Uri "$RocmBase/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" -OutFile $torchWheel
}

# --- Step 3: install everything into the portable Python ---
Write-Host "[3/7] Installing ROCm SDK + torch + transformers stack (offline-capable after this step)..."
$pipExe = "$PyDir\python.exe"

# setuptools/wheel/mpmath must be present BEFORE torch, so its "rocm" sdist
# dependency can be built with --no-build-isolation instead of needing network
# access inside an isolated build environment (confirmed necessary by
# execution -- the default isolated build tries to fetch setuptools from PyPI
# and fails offline).
& $pipExe -m pip download --no-deps -d "$OutDir\wheels" wheel setuptools "mpmath<1.4,>=1.1.0" sympy networkx jinja2 MarkupSafe 2>&1 | Out-Null
& $pipExe -m pip install --no-warn-script-location --no-index --find-links="$OutDir\wheels" setuptools wheel

& $pipExe -m pip install --no-warn-script-location --no-index --find-links="$OutDir\wheels" `
    "$OutDir\wheels\rocm_sdk_core-7.2.1-py3-none-win_amd64.whl" `
    "$OutDir\wheels\rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl" `
    "$OutDir\wheels\rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl"

& $pipExe -m pip install --no-warn-script-location --no-build-isolation --no-index --find-links="$OutDir\wheels" `
    $torchWheel

& $pipExe -m pip install --no-warn-script-location `
    "transformers==4.57.1" safetensors tokenizers huggingface_hub regex pyyaml numpy

# --- Step 4: model checkpoint, symlinks dereferenced ---
Write-Host "[4/7] Copying model checkpoint (dereferencing any hub-cache symlinks)..."
$modelFiles = @(
    "config.json", "generation_config.json", "merges.txt", "model.safetensors",
    "tokenizer.json", "tokenizer_config.json", "vocab.json"
)
foreach ($f in $modelFiles) {
    $src = Join-Path $ModelDir $f
    $dst = "$OutDir\model\$f"
    if (-not (Test-Path $src)) {
        Write-Warning "Expected model file not found: $src"
        continue
    }
    $item = Get-Item $src -Force
    if ($item.LinkType) {
        # HuggingFace hub-cache snapshots are symlinks into ../../blobs/<hash>, stored as a
        # RELATIVE target. .Target resolves relative to the symlink's OWN directory, never to
        # $PWD -- Copy-Item -Path <relative target> resolves against $PWD instead and silently
        # copies the wrong file (or nothing) if $PWD happens to differ. Confirmed necessary by
        # execution against this project's real hub cache, not assumed.
        $target = $item.Target
        if ($target -is [array]) { $target = $target[0] }
        if (-not [System.IO.Path]::IsPathRooted($target)) {
            $target = Join-Path (Split-Path $src -Parent) $target
        }
        $resolved = [System.IO.Path]::GetFullPath($target)
        Copy-Item -Path $resolved -Destination $dst -Force
    } else {
        Copy-Item -Path $src -Destination $dst -Force
    }
}
Get-ChildItem "$OutDir\model" | ForEach-Object {
    if ($_.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
        throw "REFUSING to ship a symlink in the bundle: $($_.FullName) -- exFAT/FAT32 do not support them."
    }
}

# --- Step 5: probe source + _pth patch ---
Write-Host "[5/7] Copying probe source and patching python312._pth..."
New-Item -ItemType Directory -Force -Path "$OutDir\probe" | Out-Null
Copy-Item -Path "$ProbeSrc\probe\*.py" -Destination "$OutDir\probe\" -Force

$pthPath = "$PyDir\python312._pth"
$pthContent = Get-Content $pthPath -Raw
if ($pthContent -notmatch [regex]::Escape("..\probe")) {
    $pthContent = $pthContent -replace "(?m)^Lib\\site-packages\s*$", "Lib\site-packages`r`n..\probe"
    Set-Content -Path $pthPath -Value $pthContent -NoNewline
}
if ($pthContent -notmatch "(?m)^import site\s*$") {
    Add-Content -Path $pthPath -Value "`r`nimport site`r`n"
}

# Sample real corpus documents for the probe's forward/throughput stages.
Write-Host "  extracting $NumPrompts real corpus documents..."
$corpusLines = Get-Content $CorpusFile -TotalCount $NumPrompts
$promptLines = foreach ($line in $corpusLines) {
    $rec = $line | ConvertFrom-Json
    @{ id = $rec.id; text = $rec.utterance } | ConvertTo-Json -Compress
}
$promptLines -join "`n" | Set-Content -Path "$OutDir\probe\prompts.jsonl" -NoNewline -Encoding utf8

# --- Step 6: sanity ---
Write-Host "[6/7] Verifying no symlinks anywhere in the source tree, and no file exceeds 4 GiB..."
$anyLinks = Get-ChildItem -Recurse -Force $OutDir | Where-Object { $_.Attributes -band [System.IO.FileAttributes]::ReparsePoint }
if ($anyLinks) {
    Write-Warning "Symlinks/reparse points found in bundle -- these will NOT survive a copy to exFAT/FAT32:"
    $anyLinks | ForEach-Object { Write-Warning "  $($_.FullName)" }
}
$oversize = Get-ChildItem -Recurse -Force $OutDir -File | Where-Object { $_.Length -gt 4GB }
if ($oversize) {
    Write-Warning "File(s) exceed 4 GiB -- would not fit on a FAT32 target, though exFAT has no such limit:"
    $oversize | ForEach-Object { Write-Warning ("  {0} ({1:N2} GB)" -f $_.FullName, ($_.Length / 1GB)) }
}

# --- Step 7: pack into ONE archive file, not a directory tree ---
#
# WHY: a directory tree of ~25,000 small files copies to a budget USB flash
# drive at the controller's RANDOM-write floor, not its sequential one --
# measured on this exact bundle at 0.04 MB/s (would have taken ~17 hours for
# the remaining data). A single sequential archive file copies at the drive's
# actual rated throughput instead. Plain tar (no compression) is used rather
# than gzip: measured on this bundle's own two largest components,
# model.safetensors compressed at 32.4 MB/s (20.5% smaller) and torch's DLL
# tree at 42.7 MB/s (27.0% smaller) -- against plain tar's 771.7 MB/s (disk
# read speed, negligible tar overhead). The compression time this would cost
# on an 8.69 GB bundle (~4-5 minutes) is not recovered by the write-time saved
# UNLESS the drive's SEQUENTIAL write speed is below roughly 8 MB/s, which
# would itself be unusually slow even for a "budget" drive's sequential path
# (the failure mode measured here was specifically the RANDOM-write floor).
Write-Host "[7/7] Packing the source tree into one archive (plain tar, no compression -- see comment above)..."
$DriveStagingDir = "$OutDir-drive"
New-Item -ItemType Directory -Force -Path $DriveStagingDir | Out-Null
$ArchivePath = "$DriveStagingDir\T1872_Probe.tar"
& tar.exe -cf $ArchivePath -C $OutDir model probe python
if ($LASTEXITCODE -ne 0) { throw "tar archive creation failed with exit code $LASTEXITCODE" }

Copy-Item -Path "$ProbeSrc\run_probe.bat" -Destination "$DriveStagingDir\run_probe.bat" -Force
Copy-Item -Path "$ProbeSrc\run_probe.ps1" -Destination "$DriveStagingDir\run_probe.ps1" -Force
Copy-Item -Path "$ProbeSrc\README.txt" -Destination "$DriveStagingDir\README.txt" -Force

$archiveSize = (Get-Item $ArchivePath).Length
$totalSize = (Get-ChildItem -Recurse -Force $DriveStagingDir | Measure-Object -Property Length -Sum).Sum
Write-Host ("=== Assembly complete. Archive: {0:N2} GB. Drive-staging folder total: {1:N2} GB ===" -f `
    ($archiveSize / 1GB), ($totalSize / 1GB))
Write-Host "Copy the entire '$DriveStagingDir' folder (NOT '$OutDir') to the target drive, then run run_probe.bat there."
