# T-2814, T-2825 (Curie) -- X3, the export slot's guard-vitality cell (plan Sec3.7 item 5 and item 6
# dimension 11; TE-266, D-SLM7313, widened to the whole C ABI by the T-2823 fold).
#
# SINGLE-DROP MUTANTS. The runner enumerates every SUPERSLM_API slot in <Engine>\include\superslm\*.h and
# *.inc -- the C++ declarations and the C ABI's verb declarations (api.h's own definition, preprocessor lines
# and comment lines are not slots). The enumerated count must equal X2's expected set (the C++ entries of
# consumed_symbols.txt plus the header's C verbs, 61 at v1.5.0); a mismatch fails X3 before any mutant runs,
# so a slot the enumeration cannot see is reported rather than silently never dropped. Mutant k copies the
# headers into
# <Out>\mut_<k>\include\superslm with slot k's `SUPERSLM_API ` token removed, and runs build_x2.ps1 with
# that directory first on the include path. A mutant is KILLED AS SPECIFIED only when X2 fails at BOTH
# checks the plan names, each naming the dropped symbol:
#   - X2 EXPORTS differs by exactly one missing name and no extra one;
#   - X2 CONSUMER-LINK fails with LNK2019 on exactly that name (the consumer calls every listed symbol).
# Anything else is reported: SURVIVED (X2 passed) or KILLED-OTHERWISE (X2 failed some other way).
#
# HARNESS MUTANT (-Harness): the unmutated headers with the consumer built /DX2_DROP_CHUNK_BATCHED, which
# drops its one RunLayerLoopChunkBatched call. KILLED AS SPECIFIED only when EXPORTS are EQUAL, the consumer
# LINKS and RUNS, and X2 IMPORTS differs by exactly that one missing name -- the import-equality check is
# then the only thing standing between a harness that skips a symbol and a green X2.
#
# Usage: run_x3_mutants.ps1 [-Engine <root>] [-Out <dir>] [-Indices 0,5,...] [-Harness | -HarnessOnly] [-List]
#   -List        print the enumerated slots with their indices and exit
#   -Indices     run only these slots. Without it EVERY slot runs (61 mutants, about 6 minutes on the dev box),
#                whether or not -Harness is given.
#   -Harness     also run the harness mutant. -HarnessOnly runs the harness mutant and no single-drop mutant.
# The acceptance invocation is `run_x3_mutants.ps1 -Harness` with no -Indices: 61 single-drop mutants, the
# harness mutant, and the reconciliation (X3 RECONCILE: EQUAL). A run narrowed by -Indices or -HarnessOnly
# prints X3 RECONCILE: PARTIAL RUN and is never an acceptance run.
# Exit code: 0 when every mutant run was killed as specified and the reconciliation holds, 1 otherwise.
param(
    [string]$Engine = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
    [string]$Out = '',
    [int[]]$Indices = @(),
    [switch]$Harness,
    [switch]$HarnessOnly,
    [switch]$List
)
$ErrorActionPreference = 'Stop'
if (-not $Out) { $Out = Join-Path $Engine 'build\t2807-x3' }
$x2 = Join-Path $PSScriptRoot 'build_x2.ps1'
$incDir = Join-Path $Engine 'include\superslm'

# Enumerate the slots: every SUPERSLM_API token outside api.h, preprocessor lines and comment lines, in the
# headers and in the two C ABI .inc files.
$slots = @()
foreach ($h in (Get-ChildItem $incDir -File | Where-Object { $_.Extension -in '.h', '.inc' } | Sort-Object Name)) {
    if ($h.Name -eq 'api.h') { continue }
    $n = 0
    foreach ($l in (Get-Content $h.FullName)) {
        $n++
        if ($l -match '^\s*#' -or $l -match '^\s*(//|/\*|\*)') { continue }
        $c = ([regex]::Matches($l, '\bSUPERSLM_API\b')).Count
        for ($j = 0; $j -lt $c; $j++) { $slots += [pscustomobject]@{ File = $h.Name; Line = $n; Nth = $j; Text = $l.Trim() } }
    }
}
if ($List) {
    for ($i = 0; $i -lt $slots.Count; $i++) { '{0,2} {1}:{2} {3}' -f $i, $slots[$i].File, $slots[$i].Line, $slots[$i].Text }
    exit 0
}
if ($slots.Count -eq 0) { Write-Output 'X3: no SUPERSLM_API slots in this tree (red: the slot is not built)'; exit 1 }
. (Join-Path $PSScriptRoot 'c_abi_verbs.ps1')
$cppNames = @(Get-Content (Join-Path $PSScriptRoot 'consumed_symbols.txt') | Where-Object { $_ -and $_ -notmatch '^#' -and $_ -cnotmatch '^sslm_' })
# The C part comes from the COMMITTED c_abi_verbs.txt, never from re-reading the tree under test: a declaration
# form that hides a slot from this enumerator could hide the verb from a re-read too, and the two counts would
# then agree over a partial population. X2 separately requires the committed list to equal the headers.
$cNames = @(Get-Content (Join-Path $PSScriptRoot 'c_abi_verbs.txt') | Where-Object { $_ -and $_ -notmatch '^#' })
$cppWant = $cppNames.Count
$cWant = $cNames.Count
$expected = New-Object 'System.Collections.Generic.HashSet[string]' ([string[]]@($cppNames + $cNames)), ([System.StringComparer]::Ordinal)
if ($slots.Count -ne $cppWant + $cWant) {
    Write-Output ("X3: {0} SUPERSLM_API slots enumerated, but X2 expects {1} ({2} C++ + {3} C) -- AT LEAST ONE MUTANT WAS NOT KILLED AS SPECIFIED" -f $slots.Count, ($cppWant + $cWant), $cppWant, $cWant)
    exit 1
}
Write-Output ("X3: {0} slots enumerated ({1} C++ + {2} C), equal to X2's expected set" -f $slots.Count, $cppWant, $cWant)
# T-2825: every slot runs unless -Indices narrows the run. (At T-2814 an unnarrowed -Harness run skipped every
# single-drop mutant, while the handoff described it as the full run.)
if ($HarnessOnly) { $Harness = [switch]$true; $Indices = @() }
elseif ($Indices.Count -eq 0) { $Indices = 0..($slots.Count - 1) }
$fullRun = @($Indices | Sort-Object -Unique).Count -eq $slots.Count

function Verdicts($lines) {
    $v = @{}
    foreach ($l in $lines) { if ($l -match '^(X2 [A-Z-]+): (.*)$') { $v[$Matches[1]] = $Matches[2] } }
    $v['missing'] = @($lines | Where-Object { $_ -match '^\s+X2 EXPORT MISSING (\S+)' } | ForEach-Object { ($_ -split '\s+')[-1] })
    $v['extra'] = @($lines | Where-Object { $_ -match '^\s+X2 EXPORT EXTRA' })
    $v['lnk'] = @($lines | Where-Object { $_ -match '^\s+X2 LNK2019 (\S+)' } | ForEach-Object { ($_ -split '\s+')[-1] })
    $v['immissing'] = @($lines | Where-Object { $_ -match '^\s+X2 IMPORT MISSING (\S+)' } | ForEach-Object { ($_ -split '\s+')[-1] })
    $v['imextra'] = @($lines | Where-Object { $_ -match '^\s+X2 IMPORT EXTRA' })
    return $v
}

$allOk = $true
$dropped = @{}   # name -> the mutant indices whose single drop removed it
foreach ($k in $Indices) {
    $s = $slots[$k]
    $mdir = Join-Path $Out "mut_$k"
    $minc = Join-Path $mdir 'include\superslm'
    if (Test-Path $mdir) { Remove-Item -Recurse -Force $mdir }
    New-Item -ItemType Directory -Force $minc | Out-Null
    Copy-Item (Join-Path $incDir '*') $minc -Recurse
    $path = Join-Path $minc $s.File
    $text = [System.IO.File]::ReadAllText($path)
    $lines = $text -split "`n"
    $li = $s.Line - 1
    $ms = [regex]::Matches($lines[$li], '\bSUPERSLM_API\b ?')
    $m = $ms[$s.Nth]
    $lines[$li] = $lines[$li].Remove($m.Index, $m.Length)
    [System.IO.File]::WriteAllText($path, ($lines -join "`n"))
    $sw = [Diagnostics.Stopwatch]::StartNew(); $res = @(& $x2 -Engine $Engine -Out (Join-Path $mdir 'x2') -IncludeFirst (Join-Path $mdir 'include') -Quiet 2>&1 | ForEach-Object { "$_" }); $sw.Stop(); $t = $sw.Elapsed
    $v = Verdicts $res
    $name = if ($v.missing.Count -eq 1) { $v.missing[0] } else { '?' }
    $spec = $v['X2 EXPORTS'] -like 'DIFFER*' -and $v.missing.Count -eq 1 -and $v.extra.Count -eq 0 -and
            $v['X2 CONSUMER-LINK'] -like 'FAILED*' -and $v.lnk.Count -eq 1 -and $v.lnk[0] -ceq $name
    $verdict = if ($spec) { 'KILLED AS SPECIFIED' } elseif ($v['X2'] -eq 'PASS') { 'SURVIVED' } else { 'KILLED-OTHERWISE' }
    if (-not $spec) { $allOk = $false }
    if ($spec) { if (-not $dropped.ContainsKey($name)) { $dropped[$name] = @() }; $dropped[$name] += $k }
    Write-Output ("X3 mutant {0,2} {1}:{2} [{3}] -> {4} ({5} s)" -f $k, $s.File, $s.Line, $s.Text, $verdict, [int]$t.TotalSeconds)
    Write-Output ("    exports: {0}; missing {1}; link: {2}; LNK2019 {3}" -f $v['X2 EXPORTS'], $name, $v['X2 CONSUMER-LINK'], ($v.lnk -join ' '))
}
if ($Harness) {
    $hdir = Join-Path $Out 'harness'
    $sw = [Diagnostics.Stopwatch]::StartNew(); $res = @(& $x2 -Engine $Engine -Out $hdir -ConsumerDefine 'X2_DROP_CHUNK_BATCHED' -Quiet 2>&1 | ForEach-Object { "$_" }); $sw.Stop(); $t = $sw.Elapsed
    $v = Verdicts $res
    $spec = $v['X2 EXPORTS'] -like 'EQUAL*' -and $v['X2 CONSUMER-LINK'] -like 'OK*' -and $v['X2 RUN'] -eq 'ALL=PASS' -and
            $v['X2 IMPORTS'] -like 'DIFFER*' -and $v.immissing.Count -eq 1 -and $v.imextra.Count -eq 0 -and
            $v.immissing[0] -clike '?RunLayerLoopChunkBatched@*'
    if (-not $spec) { $allOk = $false }
    Write-Output ("X3 harness mutant (X2_DROP_CHUNK_BATCHED) -> {0} ({1} s)" -f ($(if ($spec) { 'KILLED AS SPECIFIED' } elseif ($v['X2'] -eq 'PASS') { 'SURVIVED' } else { 'KILLED-OTHERWISE' })), [int]$t.TotalSeconds)
    Write-Output ("    exports: {0}; link: {1}; run: {2}; imports: {3}; import missing {4}" -f $v['X2 EXPORTS'], $v['X2 CONSUMER-LINK'], $v['X2 RUN'], $v['X2 IMPORTS'], ($v.immissing -join ' '))
}
# G7 (T-2824): each slot's mutant removes a DISTINCT expected name, and a full run removes every one of them.
$recon = @()
foreach ($n in $dropped.Keys) {
    if ($dropped[$n].Count -gt 1) { $recon += "the name $n was removed by more than one mutant ($($dropped[$n] -join ', '))" }
    if (-not $expected.Contains($n)) { $recon += "mutant $($dropped[$n] -join ', ') removed $n, which X2's expected set does not hold" }
}
if ($fullRun) {
    $never = @($expected | Where-Object { -not $dropped.ContainsKey($_) })
    if ($never.Count -gt 0) { $recon += "$($never.Count) expected name(s) removed by no mutant: $($never -join ', ')" }
    Write-Output ("X3 RECONCILE: {0} -- {1} mutants run, {2} distinct names removed, {3} expected" -f ($(if ($recon.Count -eq 0) { 'EQUAL' } else { 'DIFFER' })), $slots.Count, $dropped.Count, $expected.Count)
} else {
    Write-Output ("X3 RECONCILE: PARTIAL RUN ({0} of {1} slots) -- distinctness checked; coverage of the expected set is checked only on a full run" -f @($Indices).Count, $slots.Count)
}
$recon | ForEach-Object { Write-Output "  X3 RECONCILE $_" }
if ($recon.Count -gt 0) { $allOk = $false }
Write-Output ("X3: {0}" -f ($(if ($allOk) { 'every mutant run was killed as specified' } else { 'AT LEAST ONE MUTANT WAS NOT KILLED AS SPECIFIED' })))
exit ($(if ($allOk) { 0 } else { 1 }))
