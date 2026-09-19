# T-2835 (Curie) -- commissioning of gen_consumed_symbols.ps1's refusals (plan Sec3.7 item 3, T-2830).
#
# The CONSTRUCTIONS are the planner's (plan Sec3.7 item 3, "The refusals' commissioning constructions"),
# authored independently of the generator's builder (this seat) under StandardsDocument Sec5.4; this script
# only executes them, each through the generator's real command line, and grades the exit status AND the
# criterion the refusal names. T-2825's real-input cases MR-1 to MR-4 and MA-1 stand beside these.
#
# The scratch consumer: one committed .cpp, compiled to one .obj and archived into one .lib after its commit.
# It calls superslm::Sha256Hash and the C verbs sslm_model_unmap and sslm_kv_block_size, so its object imports
# one C++ and two C engine names. Every case builds its own scratch repository from nothing.
#
#   MR-A committed after the build  a change is committed after the .lib was built        criterion 4
#   MR-B dirty                      the source is edited and not committed                criterion 2
#   MR-C mirrored then restored     the .obj is compiled from a copy that differs from     criterion 3, names
#                                   the named checkout by one line; the read names the      consumer.cpp
#                                   checkout with the copy as the build source tree
#   MR-D partial build              the .obj is rebuilt after the .lib                     criterion 4
#   MR-E recorded at release        -Release with any -Recorded input                      -Release
#   MR-F undeclared import          a -Recorded TSV with a row naming sslm_not_a_verb      undeclared, names it
#   MA-1 must-accept                the untouched consumer, built after its commit, build   exit 0; the header
#                                   source tree = the checkout; also under -Release          names the commit
#   MA-2 must-accept (MR-F's pair)  the same TSV without the sslm_not_a_verb row           exit 0
#
# Usage: commission_gen_consumed_symbols.ps1 [-Scratch <dir>] [-Engine <root>]
# Exit 0 only when every case gives its expected result.
param(
    [string]$Scratch = (Join-Path ([IO.Path]::GetTempPath()) 't2835-gen-commission'),
    [string]$Engine = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vsenv.ps1')
$gen = Join-Path $PSScriptRoot 'gen_consumed_symbols.ps1'
if (Test-Path $Scratch) { Remove-Item -Recurse -Force $Scratch }
New-Item -ItemType Directory -Force $Scratch | Out-Null

$src = @'
// Scratch consumer for the export-slot generator's commissioning (T-2835). Not engine code.
#include <cstddef>
#include <cstdint>
#include "superslm/sha256.h"
#include "superslm/sslm_abi.h"
extern "C" int scratch_consumer(const uint8_t* p, size_t n, uint8_t* out) {
	superslm::Sha256Hash(p, n, out);
	(void)sslm_model_unmap(nullptr);
	return static_cast<int>(sslm_kv_block_size(nullptr));
}
'@

function Invoke-Git([string]$repo) { & git -C $repo -c user.name=t2835 -c user.email=t2835@invalid @args 2>&1 | Out-Null; if ($LASTEXITCODE -ne 0) { throw "git $args failed in $repo" } }
function Compile([string]$cpp, [string]$obj) {
    $o = & cl.exe /nologo /c /std:c++20 /EHsc /O2 "/I$Engine\include" "/Fo$obj" $cpp 2>&1
    if ($LASTEXITCODE -ne 0) { $o | Select-Object -First 10; throw "compile of $cpp failed" }
}
function Archive([string]$obj, [string]$lib) { & lib.exe /nologo "/OUT:$lib" $obj | Out-Null; if ($LASTEXITCODE -ne 0) { throw 'lib failed' } }
# A fresh scratch consumer: repo with src\consumer.cpp committed, built to obj\consumer.obj, archived to
# out\consumer.lib after the commit. Returns its paths.
function New-Consumer([string]$name) {
    $r = Join-Path $Scratch $name
    New-Item -ItemType Directory -Force (Join-Path $r 'repo\src'), (Join-Path $r 'obj'), (Join-Path $r 'out') | Out-Null
    $repo = Join-Path $r 'repo'
    [IO.File]::WriteAllText((Join-Path $repo 'src\consumer.cpp'), $src)
    Invoke-Git $repo init -q
    Invoke-Git $repo add -A
    Invoke-Git $repo commit -q -m 'scratch consumer'
    Start-Sleep -Seconds 2   # the build strictly after the commit's whole-second timestamp
    $obj = Join-Path $r 'obj\consumer.obj'; $lib = Join-Path $r 'out\consumer.lib'
    Compile (Join-Path $repo 'src\consumer.cpp') $obj
    Archive $obj $lib
    return [pscustomobject]@{ Root = $r; Repo = $repo; Obj = $obj; Lib = $lib; ObjDir = (Join-Path $r 'obj') }
}
function Head([string]$repo) { (& git -C $repo rev-parse HEAD).Trim() }
function Spec($c, [string]$buildTree = '') { "scratch consumer|$($c.Repo)|$(Head $c.Repo)|src|$($c.Lib)|$buildTree|$($c.ObjDir)" }
function Run-Gen([string]$case, [hashtable]$a) {
    $out = Join-Path $Scratch "$case.list.txt"; $cv = Join-Path $Scratch "$case.cverbs.txt"
    $a.Out = $out; $a.CVerbsOut = $cv; $a.Engine = $Engine
    # A child process per case: the generator's refusal calls `exit`, which must end the generator, not this script.
    $q = { param($v) "'" + ("$v" -replace "'", "''") + "'" }
    $argText = foreach ($k in $a.Keys) {
        $v = $a[$k]
        if ($v -is [bool]) { if ($v) { "-$k" } }
        elseif ($v -is [array]) { "-$k @(" + (($v | ForEach-Object { & $q $_ }) -join ',') + ")" }
        else { "-$k " + (& $q $v) }
    }
    $res = @(& pwsh -NoProfile -Command ("& " + (& $q $gen) + " " + ($argText -join ' ') + "; exit `$LASTEXITCODE") 2>&1 | ForEach-Object { "$_" })
    return [pscustomobject]@{ Exit = $LASTEXITCODE; Text = $res; Out = $out; Written = (Test-Path $out) }
}

$recordedTsv = Join-Path $PSScriptRoot 'recorded\superslmunreal-upgrade-census-imports.tsv'
$badTsv = Join-Path $Scratch 'undeclared.tsv'
$goodTsv = Join-Path $Scratch 'declared.tsv'
$rows = @("# T-2835 commissioning TSV (constructed; plan Sec3.7 item 3): object<TAB>name",
          "scratch.obj`tsslm_model_unmap",
          "scratch.obj`t?Sha256Hash@superslm@@YAXPEBE_KQEAE@Z")
[IO.File]::WriteAllLines($goodTsv, [string[]]$rows)
[IO.File]::WriteAllLines($badTsv, [string[]]($rows + "scratch.obj`tsslm_not_a_verb"))

$results = @()
function Grade([string]$case, $r, [int]$wantExit, [string]$wantText) {
    $hit = [bool]($r.Text | Where-Object { $_ -like "*$wantText*" })
    $ok = $r.Exit -eq $wantExit -and $hit -and ($wantExit -eq 0 -or -not $r.Written)
    $script:results += [pscustomobject]@{ Case = $case; Ok = $ok }
    Write-Output ("== {0}: {1} (exit {2}, want {3}; expected text '{4}' {5}; list written {6})" -f $case, $(if ($ok) { 'AS EXPECTED' } else { 'NOT AS EXPECTED' }), $r.Exit, $wantExit, $wantText, $(if ($hit) { 'found' } else { 'NOT found' }), $r.Written)
    $r.Text | Where-Object { $_ -match 'REFUSED|wrote ' } | ForEach-Object { "   $_" }
}

# MA-1: the untouched consumer.
$c = New-Consumer 'ma1'
$r = Run-Gen 'MA-1' @{ Consumer = @(Spec $c) }
Grade 'MA-1 must-accept, untouched scratch consumer' $r 0 'wrote '
if ($r.Written) {
    $named = [bool](Get-Content $r.Out | Where-Object { $_ -like "# consumer: scratch consumer @ $(Head $c.Repo) -- LIVE read*" })
    Write-Output "   list header names the commit $(Head $c.Repo): $named"
    if (-not $named) { $results += [pscustomobject]@{ Case = 'MA-1 header'; Ok = $false } }
}
$r = Run-Gen 'MA-1r' @{ Consumer = @(Spec $c); Release = $true }
Grade 'MA-1 must-accept under -Release (every input live)' $r 0 'wrote '

# MR-A: committed after the build.
$c = New-Consumer 'mra'
Start-Sleep -Seconds 2
Add-Content (Join-Path $c.Repo 'src\consumer.cpp') '// a change committed after the build'
Invoke-Git $c.Repo commit -q -am 'change after the build'
$r = Run-Gen 'MR-A' @{ Consumer = @(Spec $c) }
Grade 'MR-A committed after the build' $r 1 'criterion 4 -- the link output consumer.lib'

# MR-B: dirty.
$c = New-Consumer 'mrb'
Add-Content (Join-Path $c.Repo 'src\consumer.cpp') '// an uncommitted edit'
$r = Run-Gen 'MR-B' @{ Consumer = @(Spec $c) }
Grade 'MR-B dirty' $r 1 'criterion 2'

# MR-C: mirrored then restored -- the object was compiled from a copy one line different from the checkout.
$c = New-Consumer 'mrc'
$mirror = Join-Path $c.Root 'mirror'
New-Item -ItemType Directory -Force (Join-Path $mirror 'src') | Out-Null
[IO.File]::WriteAllText((Join-Path $mirror 'src\consumer.cpp'), $src.Replace('(void)sslm_model_unmap(nullptr);', '(void)sslm_model_unmap(nullptr);  // the mirrored copy'))
Start-Sleep -Seconds 1
Compile (Join-Path $mirror 'src\consumer.cpp') $c.Obj
Archive $c.Obj $c.Lib
$r = Run-Gen 'MR-C' @{ Consumer = @(Spec $c $mirror) }
Grade 'MR-C mirrored then restored' $r 1 'criterion 3 -- the build source tree is not the named checkout''s files: src/consumer.cpp differs'

# MR-D: partial build -- the object rebuilt after the library.
$c = New-Consumer 'mrd'
Start-Sleep -Seconds 1
Compile (Join-Path $c.Repo 'src\consumer.cpp') $c.Obj
$r = Run-Gen 'MR-D' @{ Consumer = @(Spec $c) }
Grade 'MR-D partial build' $r 1 'criterion 4 -- the link output consumer.lib'

# MR-E: recorded at release.
$c = New-Consumer 'mre'
$rec = "SuperSLMUnreal upgrade runtime module|census|$recordedTsv|commissioning run (T-2835)"
$r = Run-Gen 'MR-E' @{ Consumer = @(Spec $c); Recorded = @($rec); Release = $true }
Grade 'MR-E recorded at release' $r 1 '-Release refuses every -Recorded input'

# MR-F / MA-2: an undeclared import in a recorded TSV, and the same TSV without it.
$r = Run-Gen 'MR-F' @{ Recorded = @("constructed|none|$badTsv|commissioning run (T-2835): a constructed TSV") }
Grade 'MR-F undeclared import' $r 1 'imports sslm_not_a_verb'
$r = Run-Gen 'MA-2' @{ Recorded = @("constructed|none|$goodTsv|commissioning run (T-2835): a constructed TSV") }
Grade 'MA-2 must-accept, the same TSV without sslm_not_a_verb' $r 0 'wrote '

$bad = @($results | Where-Object { -not $_.Ok })
Write-Output ("GENERATOR COMMISSIONING: {0} -- {1} of {2} cases as expected" -f $(if ($bad.Count -eq 0) { 'PASS' } else { 'FAIL' }), ($results.Count - $bad.Count), $results.Count)
exit ($(if ($bad.Count -eq 0) { 0 } else { 1 }))
