# T-2843 (Curie) -- the full-copy mutant runner (TE-266 plan Sec3.5 step 2; the coverage audit's T-2834 F2 and F3).
#
# WHY A FULL COPY. src\gpu\gpu_1p0.cpp and superslm_gpu.cpp include the harness as `#include "d3d12_harness.h"`,
# and MSVC resolves a quoted include from the including file's own directory before any /I path. A mutated
# header placed first on the include path is therefore never read, and compiling one .cpp from a copy and the
# other from src\gpu puts two definitions of the header's inline functions into one link. So every mutant is a
# copy of the WHOLE src\gpu directory with one change, built through build_red_suite.bat --gpu-dir, which
# compiles both GPU translation units from that directory and compiles the cell against that directory's header.
#
# THE APPLIED CHECK. A mutant's edit adds `#pragma message("SSLM MUTANT <id> APPLIED")` in the file it changes.
# The runner compares the mutant directory with the candidate's, file by file, and requires that line in the
# compiler output of every translation unit that includes a changed file:
#   d3d12_harness.h    both GPU units (gpu_fi\gpu_1p0.log, gpu_fi\superslm_gpu.log), and the cell when its
#                      source includes "d3d12_harness.h" (cells\<cell>.log)
#   gpu_1p0.cpp        gpu_fi\gpu_1p0.log
#   superslm_gpu.cpp   gpu_fi\superslm_gpu.log
# A mutant whose build lacks the line is NOT-APPLIED and is never scored: an unapplied mutant would read as
# survived, a red that blames correct code. A mutant directory identical to the candidate, or differing in any
# other file, is REFUSED.
#
# SCORING. The cell is a self-relaunching parent (one child process per leg) that prints one line per child:
#   LEG <leg> run=<i> exit=<n> summary=<present|absent> -> <PASS|FAIL|CRASH>
# where CRASH is an abnormal exit status or a missing summary line. The table (-Table) names, per mutant, the
# legs that must kill it, how many times each runs, and what the kill must look like:
#   <id> TAB <leg> TAB <runs> TAB <expect>
#   expect = CRASH   the mutant's CRASH rows are pooled: it is killed when any of their processes is CRASH
#                    (T-2834 F2: a crash is how a race mutant dies, and none crashes every time, so it runs
#                    several times; a crash is never a skip or a pass)
#   expect = <text>  every run of the leg is FAIL or CRASH, and its output has a line containing both "FAIL"
#                    and <text> (the check the plan names as the killer)
# A mutant is KILLED when every one of its rows holds; KILLED-OTHERWISE when every row's leg is red but not as
# specified; SURVIVED when some row's leg passed. Lines starting with '#' in the table are comments.
#
# THE CANDIDATE RUNS FIRST. The candidate directory is built the same way and the cell is run over every leg;
# unless it passes, no mutant is scored (a kill against a red candidate means nothing).
#
# Usage:
#   run_fullcopy_mutants.ps1 -Mutants <dir> -Table <tsv> -Cell <name> [-Candidate <gpu dir>]
#       [-CellArgs '--synthetic=...'] [-IncludeFirst <dir>] [-Shaders <dir>] [-Out <dir>] [-Only r,s]
#   -Mutants    a directory holding one full src\gpu copy per mutant, named by the table's ids
#   -Candidate  the unmutated GPU source directory (default <repo>\src\gpu)
# Exit: 0 only when every mutant in the table ran and was KILLED; 3 when -Only narrowed the run and every mutant
# that ran was KILLED (a PARTIAL RUN, never acceptance); 1 otherwise.
param(
    [Parameter(Mandatory = $true)][string]$Mutants,
    [Parameter(Mandatory = $true)][string]$Table,
    [string]$Candidate = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path 'src\gpu'),
    [Parameter(Mandatory = $true)][string]$Cell,
    [string[]]$CellArgs = @(),
    [string]$IncludeFirst = '',
    [string]$Shaders = '',
    [string]$Out = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path 'build\t2791-fullcopy-mutants'),
    [string[]]$Only = @()
)
$ErrorActionPreference = 'Stop'
# `powershell -File` hands a comma list over as one string.
$Only = @($Only | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$bat = Join-Path $PSScriptRoot 'build_red_suite.bat'
$cellSrc = Join-Path $PSScriptRoot "$Cell.cpp"
$watched = @('d3d12_harness.h', 'gpu_1p0.cpp', 'superslm_gpu.cpp')
$cellIncludesHarness = [bool](Select-String -LiteralPath $cellSrc -Pattern '^\s*#\s*include\s+"d3d12_harness\.h"' -Quiet)

# --- the table ---
$rows = @()
foreach ($l in (Get-Content -LiteralPath $Table)) {
    if ($l -match '^\s*#' -or -not $l.Trim()) { continue }
    $f = $l -split "`t"
    if ($f.Count -ne 4) { throw "bad table row (want id TAB leg TAB runs TAB expect): $l" }
    $rows += [pscustomobject]@{ Id = $f[0].Trim(); Leg = $f[1].Trim(); Runs = [int]$f[2]; Expect = $f[3].Trim() }
}
$ids = @($rows | ForEach-Object Id | Select-Object -Unique)
$runIds = if ($Only.Count -gt 0) { @($ids | Where-Object { $Only -contains $_ }) } else { $ids }
if ($Only.Count -gt 0) { foreach ($o in $Only) { if ($ids -notcontains $o) { throw "-Only names $o, which the table does not hold" } } }

function Build([string]$name, [string]$gpuDir, [string]$cpuLib) {
    $dir = Join-Path $Out $name
    if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
    $a = @("`"--out=$dir`"", "`"--gpu-dir=$gpuDir`"", "`"--only=$Cell`"")
    if ($cpuLib) { $a += "`"--cpu-lib=$cpuLib`"" }
    if ($IncludeFirst) { $a += "`"--include-first=$IncludeFirst`"" }
    if ($Shaders) { $a += "`"--shaders=$Shaders`"" }
    $log = & cmd /c ("`"$bat`" " + ($a -join ' ') + ' 2>&1')
    $exe = Join-Path $dir "bin\$Cell.exe"
    return [pscustomobject]@{ Dir = $dir; Exe = $exe; Built = (Test-Path $exe); Log = $log }
}
function Run-Cell([string]$exe, [string]$legs, [int]$runs) {
    $o = @(& $exe @CellArgs "--cells=$legs" "--repeat=$runs" 2>&1 | ForEach-Object { "$_" })
    Add-Content -LiteralPath (Join-Path (Split-Path (Split-Path $exe)) 'mutant-run.txt') -Value $o
    $verdicts = @($o | Where-Object { $_ -match '^LEG \S+ run=\d+ exit=\S+ summary=\S+ -> (PASS|FAIL|CRASH)$' } | ForEach-Object { $Matches[1] })
    return [pscustomobject]@{ Text = $o; Verdicts = $verdicts }
}

New-Item -ItemType Directory -Force $Out | Out-Null
# --- the candidate ---
Write-Output "== candidate: $Candidate"
$cand = Build 'candidate' $Candidate ''
if (-not $cand.Built) { $cand.Log | Select-Object -Last 15; Write-Output 'FULLCOPY MUTANTS: the candidate did not build; no mutant is scored'; exit 1 }
$candRun = @(& $cand.Exe @CellArgs 2>&1 | ForEach-Object { "$_" })
$candOk = $LASTEXITCODE -eq 0
[IO.File]::WriteAllLines((Join-Path $Out 'candidate-run.txt'), [string[]]$candRun)
$candRun | Where-Object { $_ -match '^LEG |^cell_|^frames|(^|\| |^  )FAIL ' } | ForEach-Object { "   $_" }
if (-not $candOk) { Write-Output "FULLCOPY MUTANTS: the candidate is red; no mutant is scored (transcript: $(Join-Path $Out 'candidate-run.txt'))"; exit 1 }
$cpuLib = Join-Path $cand.Dir 'cpu.lib'

# --- the mutants ---
$results = @()
foreach ($id in $runIds) {
    $mdir = Join-Path $Mutants $id
    $verdict = $null; $detail = ''
    if (-not (Test-Path $mdir)) { $verdict = 'REFUSED'; $detail = "no mutant directory $mdir" }
    if (-not $verdict) {
        $cf = @{}; $mf = @{}
        Get-ChildItem -LiteralPath $Candidate -Recurse -File | ForEach-Object { $cf[$_.FullName.Substring($Candidate.TrimEnd('\').Length + 1)] = (Get-FileHash -Algorithm SHA256 $_.FullName).Hash }
        Get-ChildItem -LiteralPath $mdir -Recurse -File | ForEach-Object { $mf[$_.FullName.Substring($mdir.TrimEnd('\').Length + 1)] = (Get-FileHash -Algorithm SHA256 $_.FullName).Hash }
        $changed = @(@($cf.Keys) + @($mf.Keys) | Select-Object -Unique | Where-Object { $cf[$_] -ne $mf[$_] })
        if ($changed.Count -eq 0) { $verdict = 'REFUSED'; $detail = 'identical to the candidate' }
        elseif (@($changed | Where-Object { $watched -notcontains $_ }).Count -gt 0) { $verdict = 'REFUSED'; $detail = "differs outside $($watched -join ', '): $((@($changed | Where-Object { $watched -notcontains $_ })) -join ', ')" }
    }
    if (-not $verdict) {
        $b = Build $id $mdir $cpuLib
        if (-not $b.Built) { $verdict = 'BUILD-FAILED'; $detail = (($b.Log | Select-Object -Last 5) -join ' | ') }
    }
    if (-not $verdict) {
        $need = @()
        if ($changed -contains 'd3d12_harness.h') { $need += 'gpu_fi\gpu_1p0.log', 'gpu_fi\superslm_gpu.log'; if ($cellIncludesHarness) { $need += "cells\$Cell.log" } }
        if ($changed -contains 'gpu_1p0.cpp') { $need += 'gpu_fi\gpu_1p0.log' }
        if ($changed -contains 'superslm_gpu.cpp') { $need += 'gpu_fi\superslm_gpu.log' }
        $need = @($need | Select-Object -Unique)
        $tag = "SSLM MUTANT $id APPLIED"
        $lacking = @($need | Where-Object { -not (Select-String -LiteralPath (Join-Path $b.Dir $_) -SimpleMatch $tag -Quiet) })
        if ($lacking.Count -gt 0) { $verdict = 'NOT-APPLIED'; $detail = "no '$tag' in $($lacking -join ', ')" }
        else { $detail = "applied in $($need -join ', ')" }
    }
    if (-not $verdict) {
        $crashRows = @($rows | Where-Object { $_.Id -eq $id -and $_.Expect -eq 'CRASH' })
        $labelRows = @($rows | Where-Object { $_.Id -eq $id -and $_.Expect -ne 'CRASH' })
        $allRed = $true; $asSpecified = $true; $notes = @()
        if ($crashRows.Count -gt 0) {
            $all = @()
            foreach ($r in $crashRows) { $res = Run-Cell $b.Exe $r.Leg $r.Runs; $all += $res.Verdicts; $notes += "$($r.Leg): $(@($res.Verdicts | Where-Object { $_ -eq 'CRASH' }).Count) of $($res.Verdicts.Count) crashed" }
            $crashes = @($all | Where-Object { $_ -eq 'CRASH' }).Count
            if ($crashes -eq 0) { $asSpecified = $false; if (@($all | Where-Object { $_ -eq 'PASS' }).Count -eq $all.Count) { $allRed = $false } }
        }
        foreach ($r in $labelRows) {
            $res = Run-Cell $b.Exe $r.Leg $r.Runs
            $red = $res.Verdicts.Count -eq $r.Runs -and @($res.Verdicts | Where-Object { $_ -eq 'PASS' }).Count -eq 0
            $hit = [bool]($res.Text | Where-Object { $_ -like '*FAIL*' -and $_.Contains($r.Expect) })
            $notes += "$($r.Leg): $($res.Verdicts -join ',')$(if ($hit) { ", '$($r.Expect)' failed" } else { ", '$($r.Expect)' NOT among the failures" })"
            if (-not $red) { $allRed = $false }
            if (-not ($red -and $hit)) { $asSpecified = $false }
        }
        $verdict = if ($asSpecified) { 'KILLED' } elseif ($allRed) { 'KILLED-OTHERWISE' } else { 'SURVIVED' }
        $detail += '; ' + ($notes -join '; ')
    }
    Write-Output "MUTANT $id : $verdict -- $detail"
    $results += [pscustomobject]@{ Id = $id; Verdict = $verdict }
}
$killed = @($results | Where-Object { $_.Verdict -eq 'KILLED' }).Count
$partial = $Only.Count -gt 0
Write-Output ("FULLCOPY MUTANTS: {0} of {1} killed as specified{2}" -f $killed, $results.Count, $(if ($partial) { " -- PARTIAL RUN ($($results.Count) of $($ids.Count) mutants; never acceptance)" } else { '' }))
if ($killed -ne $results.Count) { exit 1 }
if ($partial) { exit 3 }
exit 0
