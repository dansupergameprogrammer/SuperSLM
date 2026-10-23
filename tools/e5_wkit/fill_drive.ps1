# Fills a removable drive with the E5 W1-W4 kit (cell 9.10's RX 7900 XTX half), on the home box.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\e5_wkit\fill_drive.ps1 -Drive F:
#
# Writes <drive>\e5-w\:
#   START.cmd              the one command the work machine runs
#   kit\                   run.ps1 and README.md, from this folder
#   inputs\e5w.bundle      full history to ce82c898 (the release-step tree) and d697927f (v1.13.1), under refs/e5w/*
#   inputs\fixtures\       pkv_def.sslm (994b58b4...), the fixture 9.10 captures and grades on
#   results\               left as it is (a rerun keeps finished pieces)
# Windows PowerShell 5.1.
param([Parameter(Mandatory = $true)][string]$Drive, [string]$Repo = 'D:\SuperSLM',
      [string]$Fixtures = 'D:\_artifacts\SuperSLM\t3035-precache-refs\fixtures')

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$Release = 'ce82c89831ac880a5dd58b0a76599332343fb6ea'
$V1131 = 'd697927f6d5ce5fa73ecbb8660020a425518b923'
$FixtureSha = '994b58b49d3660bccfdd59dc6315028ea0203f37f4705d70eb7ce4dbecdb1db3'

function Write-Line([string]$Text) { Write-Host ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $Text) }
function Invoke-Git([string[]]$Arguments) {
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    $out = & git -C $Repo @Arguments 2>&1
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($code -ne 0) { throw "git $($Arguments -join ' ') failed: $out" }
    return $out
}

$root = Join-Path ($Drive.TrimEnd('\') + '\') 'e5-w'
$inputs = Join-Path $root 'inputs'
$kit = Join-Path $root 'kit'
foreach ($d in @($root, $inputs, (Join-Path $inputs 'fixtures'), $kit, (Join-Path $root 'results'))) { New-Item -ItemType Directory -Force -Path $d | Out-Null }

Write-Line "fetching claude/precache-1p14-post-m1 into $Repo"
Invoke-Git @('fetch', '-q', 'origin', 'claude/precache-1p14-post-m1') | Out-Null
foreach ($c in @($Release, $V1131)) { Invoke-Git @('rev-parse', '-q', '--verify', "$c^{commit}") | Out-Null }

# A bundle carries refs, not bare commits: two refs of the kit's own, removed again afterwards.
$bundle = Join-Path $inputs 'e5w.bundle'
Write-Line "writing the source bundle ($Release and $V1131, full history)"
Invoke-Git @('update-ref', 'refs/e5w/release', $Release) | Out-Null
Invoke-Git @('update-ref', 'refs/e5w/v1131', $V1131) | Out-Null
try {
    Invoke-Git @('bundle', 'create', $bundle, 'refs/e5w/release', 'refs/e5w/v1131') | Out-Null
    Invoke-Git @('bundle', 'verify', '-q', $bundle) | Out-Null
} finally {
    Invoke-Git @('update-ref', '-d', 'refs/e5w/release') | Out-Null
    Invoke-Git @('update-ref', '-d', 'refs/e5w/v1131') | Out-Null
}

$fx = Join-Path $Fixtures 'pkv_def.sslm'
if ((Get-FileHash -LiteralPath $fx -Algorithm SHA256).Hash.ToLowerInvariant() -ne $FixtureSha) { throw "$fx does not hash to $FixtureSha" }
Copy-Item -LiteralPath $fx -Destination (Join-Path $inputs 'fixtures\pkv_def.sslm') -Force

foreach ($f in @('run.ps1', 'README.md')) { Copy-Item -LiteralPath (Join-Path $PSScriptRoot $f) -Destination (Join-Path $kit $f) -Force }
$start = "@echo off`r`nrem SuperSLM 1.14.0 E5 W1-W4 (9.10 on the RX 7900 XTX). Safe to run again; finished pieces are skipped.`r`n" +
         "powershell -NoProfile -ExecutionPolicy Bypass -File `"%~dp0kit\run.ps1`" %*`r`npause`r`n"
[System.IO.File]::WriteAllText((Join-Path $root 'START.cmd'), $start, (New-Object System.Text.ASCIIEncoding))
$commit = (Invoke-Git @('-C', $PSScriptRoot, 'rev-parse', 'HEAD')) | Select-Object -Last 1
[System.IO.File]::WriteAllText((Join-Path $root 'KIT.txt'), "kit_commit=$commit`r`nsource=$Release`r`nv1131=$V1131`r`nfilled=$((Get-Date).ToString('s'))`r`n", (New-Object System.Text.UTF8Encoding($false)))
$size = (Get-ChildItem -LiteralPath $root -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Line ("done: {0:N0} MB in {1}. On the work machine: {1}\START.cmd" -f ($size / 1MB), $root)
