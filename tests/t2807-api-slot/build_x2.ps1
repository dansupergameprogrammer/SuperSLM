# T-2814 (Curie) -- cell X2, the modular-build cell of plan Sec3.7 item 5 (TE-266 GPU-path plan;
# D-SLM7313, the SUPERSLM_API export slot). Batch-built on the dev box, like t2130/t2178: MSVC x64, no
# CMake ctest, no hosted CI minutes.
#
#  (a) the core sources (CMakeLists.txt's SUPERSLM_CORE_SOURCES, read from the file) are compiled with
#      /DSUPERSLM_API=__declspec(dllexport) and linked as sslm_engine.dll;
#  (b) x2_consumer.cpp is compiled with /DSUPERSLM_API=__declspec(dllimport) and linked as
#      sslm_api_consumer.dll against sslm_engine.lib ONLY; it calls every listed symbol with a known
#      answer and exports x2_run;
#  (c) x2_loader.exe links the consumer DLL, calls x2_run and exits 0 only on ALL=PASS;
#  (d) two exact structural checks, each printing the differing names:
#      - the engine DLL's export names (dumpbin /exports) EQUAL consumed_symbols.txt;
#      - the consumer DLL's imports from sslm_engine.dll (dumpbin /imports) EQUAL consumed_symbols.txt.
# X2 passes only when all four verdicts pass: EXPORTS, CONSUMER-LINK, IMPORTS, RUN.
#
# Usage: build_x2.ps1 [-Engine <repo root>] [-Out <dir>] [-IncludeFirst <dir>] [-ConsumerDefine <NAME>]
#                     [-List <consumed_symbols.txt>] [-Quiet]
#   -Engine         the tree whose include/ and src/ are built (default: this checkout)
#   -IncludeFirst   an include directory searched before <Engine>\include (how X3's mutants are built)
#   -ConsumerDefine one extra definition for the consumer only (X3's harness mutant: X2_DROP_CHUNK_BATCHED)
# Exit code: 0 when X2 passes, 1 when it fails, 2 on an environment error. The verdict lines
# ("X2 EXPORTS: ...", etc.) are the machine-readable output run_x3_mutants.ps1 reads.
param(
    [string]$Engine = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
    [string]$Out = '',
    [string]$IncludeFirst = '',
    [string]$ConsumerDefine = '',
    [string]$List = (Join-Path $PSScriptRoot 'consumed_symbols.txt'),
    [switch]$Quiet
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vsenv.ps1')
if (-not $Out) { $Out = Join-Path $Engine 'build\t2807-x2' }
foreach ($d in 'engine', 'consumer', 'bin') { New-Item -ItemType Directory -Force (Join-Path $Out $d) | Out-Null }
Get-ChildItem (Join-Path $Out 'engine'), (Join-Path $Out 'consumer'), (Join-Path $Out 'bin') -File | Remove-Item -Force

function Say($s) { if (-not $Quiet) { Write-Output $s } }
$want = @(Get-Content $List | Where-Object { $_ -and $_ -notmatch '^#' })
$wantSet = New-Object 'System.Collections.Generic.HashSet[string]' ([string[]]$want), ([System.StringComparer]::Ordinal)

# The core sources, from CMakeLists.txt.
$cm = Get-Content (Join-Path $Engine 'CMakeLists.txt') -Raw
if ($cm -notmatch '(?s)set\(SUPERSLM_CORE_SOURCES\s+(.*?)\)') { Write-Output 'X2 ENV: SUPERSLM_CORE_SOURCES not found'; exit 2 }
$sources = @($Matches[1] -split '\s+' | Where-Object { $_ } | ForEach-Object { Join-Path $Engine $_ })
$inc = @()
if ($IncludeFirst) { $inc += "/I$IncludeFirst" }
$inc += "/I$(Join-Path $Engine 'include')"
$common = @('/nologo', '/O2', '/Ob2', '/DNDEBUG', '/MD', '/EHsc', '/std:c++20', '/W4', '/fp:precise')

# (a) engine DLL
$elog = Join-Path $Out 'engine\build.log'
& cl.exe /c @common /MP4 @inc '/DSUPERSLM_API=__declspec(dllexport)' "/Fo$(Join-Path $Out 'engine')\" @sources *> $elog
if ($LASTEXITCODE -ne 0) { Get-Content $elog | Select-String 'error' | Select-Object -First 20 | ForEach-Object Line; Write-Output 'X2 ENV: engine compile failed'; exit 2 }
$attrWarn = @(Select-String -Path $elog -Pattern 'warning C4(251|273|275|297|190)' | ForEach-Object Line)
$allWarn = @(Select-String -Path $elog -Pattern 'warning C' | ForEach-Object Line)
Say "X2 engine: $($sources.Count) sources compiled; $($allWarn.Count) warnings at /W4, $($attrWarn.Count) of them linkage-attribute warnings (C4251/C4273/C4275)"
$attrWarn | ForEach-Object { Say "  $_" }
$dll = Join-Path $Out 'bin\sslm_engine.dll'; $implib = Join-Path $Out 'engine\sslm_engine.lib'
& link.exe /nologo /DLL "/OUT:$dll" "/IMPLIB:$implib" (Get-ChildItem (Join-Path $Out 'engine') -Filter *.obj | ForEach-Object FullName) *> (Join-Path $Out 'engine\link.log')
if ($LASTEXITCODE -ne 0) { Get-Content (Join-Path $Out 'engine\link.log') | Select-Object -First 20; Write-Output 'X2 ENV: engine link failed'; exit 2 }

# (d.1) exports
$exports = @(& dumpbin.exe /nologo /exports $dll | ForEach-Object {
    if ($_ -match '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)') { $Matches[1] } })
$exSet = New-Object 'System.Collections.Generic.HashSet[string]' ([string[]]$exports), ([System.StringComparer]::Ordinal)
$missingEx = @($want | Where-Object { -not $exSet.Contains($_) })
$extraEx = @($exports | Where-Object { -not $wantSet.Contains($_) })
$exOk = $missingEx.Count -eq 0 -and $extraEx.Count -eq 0
Write-Output ("X2 EXPORTS: {0} -- {1} exported, {2} listed, {3} missing, {4} extra" -f ($(if ($exOk) { 'EQUAL' } else { 'DIFFER' })), $exports.Count, $want.Count, $missingEx.Count, $extraEx.Count)
$missingEx | ForEach-Object { Write-Output "  X2 EXPORT MISSING $_" }
$extraEx | ForEach-Object { Write-Output "  X2 EXPORT EXTRA   $_" }

# (b) consumer DLL
$clog = Join-Path $Out 'consumer\build.log'
$cdef = @('/DSUPERSLM_API=__declspec(dllimport)')
if ($ConsumerDefine) { $cdef += "/D$ConsumerDefine" }
$cdll = Join-Path $Out 'bin\sslm_api_consumer.dll'; $clib = Join-Path $Out 'consumer\sslm_api_consumer.lib'
$linkOk = $false; $unresolved = @()
& cl.exe /c @common @inc @cdef "/Fo$(Join-Path $Out 'consumer')\x2_consumer.obj" (Join-Path $PSScriptRoot 'x2_consumer.cpp') *> $clog
if ($LASTEXITCODE -ne 0) { Get-Content $clog | Select-String 'error' | Select-Object -First 20 | ForEach-Object Line; Write-Output 'X2 ENV: consumer compile failed'; exit 2 }
if (Test-Path $implib) {
    & link.exe /nologo /DLL "/OUT:$cdll" "/IMPLIB:$clib" (Join-Path $Out 'consumer\x2_consumer.obj') $implib *>> $clog
    $linkOk = $LASTEXITCODE -eq 0
    # The unresolved symbol is the FIRST decorated name, right after its quoted undecorated form; a
    # trailing "referenced in function ... (name)" names the referencer, not the missing symbol. One
    # symbol can be reported once per referencing function, so the names are de-duplicated.
    $unresolved = @(Select-String -Path $clog -Pattern 'LNK2019: unresolved external symbol "[^"]*" \((\S+?)\)' |
        ForEach-Object { $_.Matches[0].Groups[1].Value } | Sort-Object -Unique -CaseSensitive)
} else {
    Add-Content $clog 'no sslm_engine.lib: the engine DLL exports nothing, so the linker wrote no import library'
    $unresolved = $want
}
Write-Output ("X2 CONSUMER-LINK: {0}{1}" -f ($(if ($linkOk) { 'OK' } else { 'FAILED' })), $(if (-not (Test-Path $implib)) { ' (no import library: the engine DLL exports nothing)' } else { " ($($unresolved.Count) LNK2019)" }))
if (Test-Path $implib) { $unresolved | ForEach-Object { Write-Output "  X2 LNK2019 $_" } }

$imOk = $false; $runOk = $false
if ($linkOk) {
    # (d.2) the consumer's imports from sslm_engine.dll
    $imports = @(); $inEngine = $false
    foreach ($l in (& dumpbin.exe /nologo /imports $cdll)) {
        if ($l -match '^\s{4}(\S+\.dll)\s*$') { $inEngine = ($Matches[1] -ieq 'sslm_engine.dll'); continue }
        if ($l -match '^\s*Summary\s*$') { $inEngine = $false; continue }
        if ($inEngine -and $l -match '^\s+[0-9A-F]+\s+(\S+)\s*$') { $imports += $Matches[1] }
    }
    $imSet = New-Object 'System.Collections.Generic.HashSet[string]' ([string[]]$imports), ([System.StringComparer]::Ordinal)
    $missingIm = @($want | Where-Object { -not $imSet.Contains($_) })
    $extraIm = @($imports | Where-Object { -not $wantSet.Contains($_) })
    $imOk = $missingIm.Count -eq 0 -and $extraIm.Count -eq 0
    Write-Output ("X2 IMPORTS: {0} -- the consumer imports {1} names from sslm_engine.dll, {2} listed, {3} missing, {4} extra" -f ($(if ($imOk) { 'EQUAL' } else { 'DIFFER' })), $imports.Count, $want.Count, $missingIm.Count, $extraIm.Count)
    $missingIm | ForEach-Object { Write-Output "  X2 IMPORT MISSING $_" }
    $extraIm | ForEach-Object { Write-Output "  X2 IMPORT EXTRA   $_" }

    # (c) loader
    $lexe = Join-Path $Out 'bin\x2_loader.exe'
    & cl.exe @common "/Fo$(Join-Path $Out 'consumer')\x2_loader.obj" "/Fe$lexe" (Join-Path $PSScriptRoot 'x2_loader.cpp') /link $clib *>> $clog
    if ($LASTEXITCODE -ne 0) { Write-Output 'X2 ENV: loader build failed'; exit 2 }
    $runOut = & $lexe 2>&1
    $runOk = $LASTEXITCODE -eq 0 -and ($runOut -contains 'ALL=PASS')
    $runOut | ForEach-Object { Say "  $_" }
    Write-Output ("X2 RUN: {0}" -f ($(if ($runOk) { 'ALL=PASS' } else { "FAILED (exit $LASTEXITCODE)" })))
} else {
    Write-Output 'X2 IMPORTS: NOT CHECKED (the consumer did not link)'
    Write-Output 'X2 RUN: NOT RUN (the consumer did not link)'
}
$pass = $exOk -and $linkOk -and $imOk -and $runOk
Write-Output ("X2: {0}" -f ($(if ($pass) { 'PASS' } else { 'FAIL' })))
exit ($(if ($pass) { 0 } else { 1 }))
