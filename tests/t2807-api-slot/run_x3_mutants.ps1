# T-2814 (Curie) -- X3, the export slot's guard-vitality cell (plan Sec3.7 item 5 and item 6 dimension 11;
# TE-266, D-SLM7313).
#
# SINGLE-DROP MUTANTS. The runner enumerates every SUPERSLM_API slot in <Engine>\include\superslm\*.h
# (api.h's own definition and the include lines are not slots). Mutant k copies the headers into
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
# Usage: run_x3_mutants.ps1 [-Engine <root>] [-Out <dir>] [-Indices 0,5,...] [-Harness] [-List]
#   -List     print the enumerated slots with their indices and exit
#   -Indices  run only these slots (default: all). The full set is 24 mutants.
# Exit code: 0 when every mutant run was killed as specified, 1 otherwise.
param(
    [string]$Engine = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
    [string]$Out = '',
    [int[]]$Indices = @(),
    [switch]$Harness,
    [switch]$List
)
$ErrorActionPreference = 'Stop'
if (-not $Out) { $Out = Join-Path $Engine 'build\t2807-x3' }
$x2 = Join-Path $PSScriptRoot 'build_x2.ps1'
$incDir = Join-Path $Engine 'include\superslm'

# Enumerate the slots: every SUPERSLM_API token outside api.h and outside preprocessor lines.
$slots = @()
foreach ($h in (Get-ChildItem $incDir -Filter *.h | Sort-Object Name)) {
    if ($h.Name -eq 'api.h') { continue }
    $n = 0
    foreach ($l in (Get-Content $h.FullName)) {
        $n++
        if ($l -match '^\s*#') { continue }
        $c = ([regex]::Matches($l, '\bSUPERSLM_API\b')).Count
        for ($j = 0; $j -lt $c; $j++) { $slots += [pscustomobject]@{ File = $h.Name; Line = $n; Nth = $j; Text = $l.Trim() } }
    }
}
if ($List) {
    for ($i = 0; $i -lt $slots.Count; $i++) { '{0,2} {1}:{2} {3}' -f $i, $slots[$i].File, $slots[$i].Line, $slots[$i].Text }
    exit 0
}
if ($slots.Count -eq 0) { Write-Output 'X3: no SUPERSLM_API slots in this tree (red: the slot is not built)'; exit 1 }
if ($Indices.Count -eq 0 -and -not $Harness) { $Indices = 0..($slots.Count - 1) }

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
Write-Output ("X3: {0}" -f ($(if ($allOk) { 'every mutant run was killed as specified' } else { 'AT LEAST ONE MUTANT WAS NOT KILLED AS SPECIFIED' })))
exit ($(if ($allOk) { 0 } else { 1 }))
