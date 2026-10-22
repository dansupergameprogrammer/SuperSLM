# E4 on the home box: the environment every piece of the E4 run sheet dot-sources first (plan §9 E4).
#
#   . D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
#
# It sets no state outside this PowerShell session: MSVC 14.34 (vcvarsall x64, Windows SDK 10.0.22621.0), the paths,
# and the SUPERSLM_PRECACHE_* inputs of the E4 cells. Windows PowerShell 5.1 and PowerShell 7 both run it. Each piece
# runs in a fresh window, so a crash costs only that piece; every piece writes its log under $E4Logs as it goes.

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$global:E4Work   = 'D:\_scratch\e4'
$global:E4Src    = Join-Path $E4Work 'src'               # a detached worktree of D:\SuperSLM at the pinned commit
$global:E4Build  = Join-Path $E4Work 'build-msvc'
$global:E4Logs   = Join-Path $E4Work 'logs'
$global:E4T3035  = Join-Path $E4Work 't3035'             # a detached worktree at origin/curie/t3035-precache-refs
$global:E4Art    = 'D:\_artifacts\SuperSLM'
$global:E4Model05 = Join-Path $E4Art 'te442-qwen2.5-0.5b-instruct-dg\qwen2.5-0.5b-instruct-dg.sslm'
$global:E4Model15 = Join-Path $E4Art 't2901-qwen2.5-1.5b-dg\qwen2.5-1.5b-instruct-dg-g5.sslm'
$global:E4Prompts = Join-Path $E4T3035 'tests\t3035-precache-refs\prompts'
$global:E4Disc    = Join-Path $E4T3035 'tests\t3035-precache-refs\discrimination'
$global:E4Refs    = Join-Path $E4Art 't3035-precache-refs-post-m1'   # the regenerated population (P1, P2)
$global:E4Timing  = Join-Path $E4Work 'timing'
$global:E4Runner  = Join-Path $E4Build 'superslm_precache_e1.exe'
$global:E4Gen     = Join-Path $E4Build 'superslm_precache_reference_gen.exe'
foreach ($d in @($E4Work, $E4Logs, $E4Timing)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }

function global:E4-Log([string]$Name) { return (Join-Path $E4Logs ($Name + '.log')) }

# Runs a program with its output appended to a piece's log as it is written, and returns its exit code.
function global:E4-Run([string]$Log, [string]$Exe, [string[]]$Arguments) {
    ('[{0}] {1} {2}' -f (Get-Date -Format 's'), $Exe, ($Arguments -join ' ')) | Out-File -Append -Encoding utf8 $Log
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $Exe @Arguments 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $Log -Append
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    ('[{0}] exit {1}' -f (Get-Date -Format 's'), $code) | Out-File -Append -Encoding utf8 $Log
    return $code
}

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $vs = & $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.14.34.17.4.x86.x64 -property installationPath |
          Select-Object -First 1
    if (-not $vs) { throw 'no Visual Studio install carries MSVC 14.34 (Microsoft.VisualStudio.Component.VC.14.34.17.4.x86.x64)' }
    $vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvarsall.bat'
    $lines = & cmd.exe /d /c "call `"$vcvars`" x64 10.0.22621.0 -vcvars_ver=14.34 >nul 2>&1 && set"
    if ($LASTEXITCODE -ne 0) { throw "vcvarsall x64 -vcvars_ver=14.34 failed ($vcvars)" }
    foreach ($line in $lines) {
        $at = $line.IndexOf('=')
        if ($at -gt 0) { Set-Item -Path ('env:' + $line.Substring(0, $at)) -Value $line.Substring($at + 1) }
    }
}
if ($env:VCToolsVersion -notlike '14.34.*') { throw "MSVC is $env:VCToolsVersion, not 14.34; open a fresh window" }

# The E4 cells' inputs (e4_real.cpp, precache_reference.h, run_real_two_process.py).
$env:SUPERSLM_PAGED_KV_FIXTURE_DIR = Join-Path $E4Art 't3035-precache-refs\fixtures'   # pkv_def for 4.3 and 6.6
$env:SUPERSLM_PRECACHE_REAL_MODEL = $E4Model05
$env:SUPERSLM_PRECACHE_REAL_1P5_MODEL = $E4Model15
$env:SUPERSLM_PRECACHE_PROMPT_DIR = $E4Prompts
$env:SUPERSLM_PRECACHE_BLOB_DIR = Join-Path $E4Work 'blobs'
$env:SUPERSLM_PRECACHE_REFERENCE_DIR = $E4Refs
$env:SUPERSLM_PRECACHE_REFERENCE_MANIFEST = Join-Path $E4Refs 'manifest.json'          # the debunker's licence names it
$env:SUPERSLM_PRECACHE_DISCRIMINATION_RECEIPT = Join-Path $E4Refs 'license.txt'        # PC-DISCRIMINATION-1, the debunker's
$env:SUPERSLM_PRECACHE_DISCRIMINATION_INSTRUMENT = Join-Path $E4Src 'tests\precache\precache_reference.h'
$env:SUPERSLM_PRECACHE_TIMING_RECEIPT = Join-Path $E4Timing 'PC-TIMING-1.txt'          # the debunker's, after commissioning
$env:SUPERSLM_PRECACHE_TIMING_INSTRUMENT = Join-Path $E4Src 'tests\precache\e4_real.cpp'
$env:SUPERSLM_PRECACHE_TIMING_PROMPT_MANIFEST = Join-Path $E4Timing 'prompt-manifest.txt'
$env:SUPERSLM_PRECACHE_BOX_ID = 'home-3950x'
$env:SUPERSLM_PRECACHE_THREADS = '1'
$env:SUPERSLM_PRECACHE_WORST_BLOB = Join-Path $E4Work 'worst\cap1p5.spx1'
if (Test-Path -LiteralPath (Join-Path $E4Src '.git')) {
    $env:SUPERSLM_PRECACHE_TIMING_ANCHOR = (& git -C $E4Src merge-base HEAD origin/main).Trim()
}
Write-Host ("E4 env: MSVC {0}, src {1}, build {2}" -f $env:VCToolsVersion, $E4Src, $E4Build)
