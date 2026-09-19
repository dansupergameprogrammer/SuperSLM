# T-2814, T-2825 (Curie) -- cell X2, the modular-build cell of plan Sec3.7 item 5 (TE-266 GPU-path plan;
# D-SLM7313, the SUPERSLM_API export slot, widened to the whole C ABI by the T-2823 fold). Batch-built on the
# dev box, like t2130/t2178: MSVC x64, no CMake ctest, no hosted CI minutes.
#
# THE EXPECTED SET is consumed_symbols.txt's C++ entries (every line not starting `sslm_`) united with the C
# ABI's declared verbs: 25 + 36 = 61 at v1.5.0. The C verbs are re-derived here from the headers under test
# (c_abi_verbs.ps1) and must EQUAL the committed c_abi_verbs.txt ("X2 CVERBS"), so a verb appended to the ABI
# fails X2 whether or not it was slotted and whether or not the list was regenerated.
#
# THE READER IS CROSS-CHECKED ("X2 CVERBS-MENTION", T-2835 for plan Sec3.7 item 3, T-2824 G6, T-2825 F-3).
# CVERBS re-reads with the same reader that wrote c_abi_verbs.txt, so a declaration form the reader misses
# hides from both. X2 therefore also requires c_abi_verbs.txt to EQUAL an independent enumeration: the name
# set of tools/t2139_count_abi_verbs.sh's OWN pipeline (its grep -oE and sed, read out of that script, then
# sort -u, without its final wc -l), run by Git's bash over the two .inc files under test. That pipeline
# matches every `sslm_x(`, so its set is a MENTION set: it sees a declaration split across lines and one
# behind an unknown prefix macro, and it also fails loudly on a comment that writes `sslm_x(` -- the remedy
# is to reword the comment. A counter script whose pipeline cannot be read is an environment error (exit 2).
#
#  (a) the core sources (CMakeLists.txt's SUPERSLM_CORE_SOURCES, read from the file) are compiled with
#      /DSUPERSLM_API=__declspec(dllexport) and linked as sslm_engine.dll;
#  (b) x2_consumer.cpp is compiled with /DSUPERSLM_API=__declspec(dllimport) and linked as
#      sslm_api_consumer.dll against sslm_engine.lib ONLY; it calls every listed symbol with a known
#      answer and exports x2_run;
#  (c) x2_loader.exe links the consumer DLL, calls x2_run and exits 0 only on ALL=PASS;
#  (d) two exact structural checks, each printing the differing names:
#      - the engine DLL's export names (dumpbin /exports) EQUAL the expected set;
#      - the consumer DLL's imports from sslm_engine.dll (dumpbin /imports) EQUAL the expected set.
# X2 passes only when all six verdicts pass: CVERBS, CVERBS-MENTION, EXPORTS, CONSUMER-LINK, IMPORTS, RUN.
#
# Usage: build_x2.ps1 [-Engine <repo root>] [-Out <dir>] [-IncludeFirst <dir>] [-ConsumerDefine <NAME>]
#                     [-List <consumed_symbols.txt>] [-CVerbs <c_abi_verbs.txt>] [-Reader <c_abi_verbs.ps1>]
#                     [-Quiet]
#   -Engine         the tree whose include/ and src/ are built (default: this checkout)
#   -IncludeFirst   an include directory searched before <Engine>\include (how X3's mutants are built)
#   -ConsumerDefine one extra definition for the consumer only (X3's harness mutant: X2_DROP_CHUNK_BATCHED)
#   -Reader         a different C verb reader to dot-source (the cross-check's must-reject grades a reader that
#                   misses a declaration form; see Claude/Curie/t2835-s3p8-cells-2026-09-19.md)
# Exit code: 0 when X2 passes, 1 when it fails, 2 on an environment error. The verdict lines
# ("X2 EXPORTS: ...", etc.) are the machine-readable output run_x3_mutants.ps1 reads.
param(
    [string]$Engine = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
    [string]$Out = '',
    [string]$IncludeFirst = '',
    [string]$ConsumerDefine = '',
    [string]$List = (Join-Path $PSScriptRoot 'consumed_symbols.txt'),
    [string]$CVerbs = (Join-Path $PSScriptRoot 'c_abi_verbs.txt'),
    [string]$Reader = (Join-Path $PSScriptRoot 'c_abi_verbs.ps1'),
    [switch]$Quiet
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vsenv.ps1')
. $Reader
if (-not $Out) { $Out = Join-Path $Engine 'build\t2807-x2' }
foreach ($d in 'engine', 'consumer', 'bin') { New-Item -ItemType Directory -Force (Join-Path $Out $d) | Out-Null }
Get-ChildItem (Join-Path $Out 'engine'), (Join-Path $Out 'consumer'), (Join-Path $Out 'bin') -File | Remove-Item -Force

function Say($s) { if (-not $Quiet) { Write-Output $s } }
# The expected set: the list's C++ entries united with the C verbs the headers under test declare.
$cpp = @(Get-Content $List | Where-Object { $_ -and $_ -notmatch '^#' -and $_ -cnotmatch '^sslm_' })
$hdrDir = Join-Path $Engine 'include\superslm'
if ($IncludeFirst -and (Test-Path (Join-Path $IncludeFirst 'superslm\sslm_abi_functions.inc'))) { $hdrDir = Join-Path $IncludeFirst 'superslm' }
$derived = @(Get-CAbiVerbs $hdrDir | ForEach-Object Name)
$committed = @(Get-Content $CVerbs | Where-Object { $_ -and $_ -notmatch '^#' })
$cvDiff = @(Compare-Object -CaseSensitive $committed $derived | ForEach-Object { "$($_.SideIndicator) $($_.InputObject)" })
$cvOk = $cvDiff.Count -eq 0
Write-Output ("X2 CVERBS: {0} -- the headers declare {1} C verbs, c_abi_verbs.txt holds {2}" -f ($(if ($cvOk) { 'EQUAL' } else { 'DIFFER (regenerate c_abi_verbs.txt with gen_consumed_symbols.ps1)' })), $derived.Count, $committed.Count)
$cvDiff | ForEach-Object { Write-Output "  X2 CVERB $_  (=> only in the headers, <= only in c_abi_verbs.txt)" }

# CVERBS-MENTION: the counter script's own pipeline, read out of the script, run by Git's bash.
# The counter is this checkout's tool; the .inc files it reads are the headers under test (-Engine or
# -IncludeFirst), so a scratch or mutant tree without tools/ is still cross-checked.
$counter = Join-Path $PSScriptRoot '..\..\tools\t2139_count_abi_verbs.sh'
if (-not (Test-Path $counter)) { Write-Output "X2 ENV: $counter not found"; exit 2 }
$pipeLine = @(Get-Content $counter | Where-Object { $_ -match "^\s*grep -oE '[^']+' `"\`$TARGET`" \| sed -E '[^']+' \| sort -u \| wc -l\s*$" })
if ($pipeLine.Count -ne 1) { Write-Output "X2 ENV: the verb counter's pipeline (grep -oE '...' `"`$TARGET`" | sed -E '...' | sort -u | wc -l) is not found once in $counter"; exit 2 }
$null = $pipeLine[0] -match "grep -oE '([^']+)'.*sed -E '([^']+)'"
$grepRe = $Matches[1]; $sedRe = $Matches[2]
$gitExe = (Get-Command git -ErrorAction SilentlyContinue).Source
# git.exe sits in <Git>\cmd or <Git>\mingw64\bin; bash.exe is <Git>\bin\bash.exe. Never a bare `bash` (WSL's may win on PATH).
$gitBash = ''
if ($gitExe) { $d = Split-Path $gitExe; for ($i = 0; $i -lt 3 -and $d; $i++) { $c = Join-Path $d 'bin\bash.exe'; if (Test-Path $c) { $gitBash = $c; break }; $d = Split-Path $d } }
if (-not $gitBash -or -not (Test-Path $gitBash)) { Write-Output 'X2 ENV: Git''s bash.exe not found beside git.exe (the mention set needs its grep, sed and sort)'; exit 2 }
$incA = (Join-Path $hdrDir 'sslm_abi_functions.inc') -replace '\\', '/'
$incB = (Join-Path $hdrDir 'sslm_abi_functions_g5_comparable.inc') -replace '\\', '/'
$bashCmd = "grep -ohE '$grepRe' '$incA' '$incB' | sed -E '$sedRe' | LC_ALL=C sort -u"
$mention = @(& $gitBash -c $bashCmd | ForEach-Object { "$_".Trim() } | Where-Object { $_ })
if ($LASTEXITCODE -ne 0) { Write-Output "X2 ENV: the mention pipeline failed ($bashCmd)"; exit 2 }
$mnDiff = @(Compare-Object -CaseSensitive $committed $mention | ForEach-Object { "$($_.SideIndicator) $($_.InputObject)" })
$mnOk = $mnDiff.Count -eq 0
Write-Output ("X2 CVERBS-MENTION: {0} -- the verb counter's pipeline mentions {1} names in the two .inc files, c_abi_verbs.txt holds {2}" -f ($(if ($mnOk) { 'EQUAL' } else { 'DIFFER (a declaration form the reader misses, or a comment that writes sslm_x( -- reword it)' })), $mention.Count, $committed.Count)
$mnDiff | ForEach-Object { Write-Output "  X2 CVERB-MENTION $_  (=> mentioned in the .inc files, <= only in c_abi_verbs.txt)" }
$want = @($cpp + $derived)
$wantSet = New-Object 'System.Collections.Generic.HashSet[string]' ([string[]]$want), ([System.StringComparer]::Ordinal)
Say ("X2 expected set: {0} C++ (consumed_symbols.txt) + {1} C (the headers) = {2}" -f $cpp.Count, $derived.Count, $want.Count)

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
& cl.exe /c @common /MP @inc '/DSUPERSLM_API=__declspec(dllexport)' "/Fo$(Join-Path $Out 'engine')\" @sources *> $elog
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
    # A C verb has no quoted undecorated form: the line names it bare ("unresolved external symbol
    # sslm_prefill referenced in function x2_run", T-2825). Both shapes are parsed; an __imp_ prefix is
    # stripped so the name compares with the list.
    $unresolved = @(Select-String -Path $clog -Pattern 'LNK2019: unresolved external symbol (?:"[^"]*" \((\S+?)\)|(\S+) referenced in function)' |
        ForEach-Object { $g = $_.Matches[0].Groups; $n = if ($g[1].Success) { $g[1].Value } else { $g[2].Value }; $n -replace '^__imp_', '' } |
        Sort-Object -Unique -CaseSensitive)
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
$pass = $cvOk -and $mnOk -and $exOk -and $linkOk -and $imOk -and $runOk
Write-Output ("X2: {0}" -f ($(if ($pass) { 'PASS' } else { 'FAIL' })))
exit ($(if ($pass) { 0 } else { 1 }))
