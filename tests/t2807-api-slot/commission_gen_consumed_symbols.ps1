# T-2835, T-2843 (Curie) -- commissioning of gen_consumed_symbols.ps1's refusals (plan Sec3.7 item 3, T-2830 and
# T-2837).
#
# The CONSTRUCTIONS are the planner's (plan Sec3.7 item 3, "The refusals' commissioning constructions"),
# authored independently of the generator's builder (this seat) under StandardsDocument Sec5.4; this script
# only executes them, each through the generator's real command line, and grades the exit status AND the
# check the refusal names, and that a refused run wrote no list. T-2825's real-input cases MR-1 to MR-4 and MA-1
# stand beside these.
#
# The scratch consumer: one committed .cpp, compiled to one .obj and archived into one .lib after its commit.
# It calls superslm::Sha256Hash and the C verbs sslm_model_unmap and sslm_kv_block_size, so its object imports
# one C++ and two C engine names. Every case builds its own scratch repository from nothing.
#
#   MA-1 must-accept          the untouched consumer, built after its commit, build        exit 0; the header
#                             source tree = the checkout; also under -Release (T-2834 N1)   names the commit
#   MR-A committed after      a change is committed after the .lib was built               criterion 4
#   MR-B dirty                the source is edited and not committed                       criterion 2
#   MR-D partial build        the .obj is rebuilt after the .lib                           criterion 4
#   THE MIRROR, ON THE REAL IN-PLACE PATH (T-2837, T-2834 F1): a scratch repository's develop checkout and a
#   worktree whose consumer .cpp differs by one line; the worktree's src is mirrored into the checkout with
#   robocopy /MIR and built there; the checkout is then restored in place with git checkout and git clean.
#   MC-A must-accept          mirror in place: the read names the worktree and its commit, exit 0
#                             with the checkout as the build source tree
#   MC-3 restored, worktree   after the restore, the same read                             criterion 3, names
#                                                                                           src/consumer.cpp
#   MC-5 restored, checkout   after the restore, the read names the checkout at its HEAD   criterion 5, names
#                             with the default build source tree (criteria 1-4 pass it)     src/consumer.cpp
#   THE RECORDED-READ ADMISSION (T-2837, T-2834 F4):
#   MA-3 must-accept          the 5ffed4c conversion as filed: the upgrade census TSV,      exit 0
#                             its census, a disposition with all three parts
#   MR-H census hash          the same TSV, its census altered by one appended line         names the hash
#   MR-D0..MR-D3 disposition  the sentence missing; each of its three parts missing          names what is missing
#   MR-T hand-typed row       the TSV with one row appended that the census does not hold    not the conversion
#                             (an addition of this seat's: the generator re-runs the conversion; not a plan case)
#   MR-E recorded at release  -Release with any -Recorded input                            -Release
#   MR-F undeclared import    a recorded read, converted from a constructed census, with a   undeclared, names it
#                             row naming sslm_not_a_verb (pinned through a scratch -CensusPins)
#   MA-2 must-accept          the same without the sslm_not_a_verb row                     exit 0
#
# Usage: commission_gen_consumed_symbols.ps1 [-Scratch <dir>] [-Engine <root>] [-UpgradeCensus <census-output.txt>]
#   -UpgradeCensus  the planner's census the committed upgrade TSV converts (records tree,
#                   Claude/Vitruvius/t2823-probe/census-output.txt, sha256 6d75b502...). MA-3, MR-H and MR-T need it.
# Exit 0 only when every case gives its expected result.
param(
    [string]$Scratch = (Join-Path ([IO.Path]::GetTempPath()) 't2835-gen-commission'),
    [string]$Engine = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
    [string]$UpgradeCensus = 'D:\Wizard\Claude\Vitruvius\t2823-probe\census-output.txt'
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vsenv.ps1')
$gen = Join-Path $PSScriptRoot 'gen_consumed_symbols.ps1'
$conv = Join-Path $PSScriptRoot 'census_to_tsv.ps1'
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

$results = @()
# $wantText is a -like pattern matched against the generator's output lines.
function Grade([string]$case, $r, [int]$wantExit, [string]$wantText) {
    $hit = [bool]($r.Text | Where-Object { $_ -like "*$wantText*" })
    $ok = $r.Exit -eq $wantExit -and $hit -and ($wantExit -eq 0 -or -not $r.Written)
    $script:results += [pscustomobject]@{ Case = $case; Ok = $ok }
    Write-Output ("== {0}: {1} (exit {2}, want {3}; expected text '{4}' {5}; list written {6})" -f $case, $(if ($ok) { 'AS EXPECTED' } else { 'NOT AS EXPECTED' }), $r.Exit, $wantExit, $wantText, $(if ($hit) { 'found' } else { 'NOT found' }), $r.Written)
    $r.Text | Where-Object { $_ -match 'REFUSED|wrote |Exception|error' } | ForEach-Object { "   $_" }
}

# ---------------------------------------------------------------------------------------------------------------
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

# MR-D: partial build -- the object rebuilt after the library.
$c = New-Consumer 'mrd'
Start-Sleep -Seconds 1
Compile (Join-Path $c.Repo 'src\consumer.cpp') $c.Obj
$r = Run-Gen 'MR-D' @{ Consumer = @(Spec $c) }
Grade 'MR-D partial build' $r 1 'criterion 4 -- the link output consumer.lib'

# ---------------------------------------------------------------------------------------------------------------
# The mirror, on the real in-place path.
$m = Join-Path $Scratch 'mirror'
$main = Join-Path $m 'main'; $wt = Join-Path $m 'wt'
New-Item -ItemType Directory -Force (Join-Path $main 'src'), (Join-Path $m 'obj'), (Join-Path $m 'out') | Out-Null
[IO.File]::WriteAllText((Join-Path $main 'src\consumer.cpp'), $src)
Invoke-Git $main init -q -b develop
Invoke-Git $main add -A
Invoke-Git $main commit -q -m 'scratch consumer on develop'
Invoke-Git $main worktree add -q -b feature $wt
[IO.File]::WriteAllText((Join-Path $wt 'src\consumer.cpp'), $src.Replace('(void)sslm_model_unmap(nullptr);', '(void)sslm_model_unmap(nullptr);  // the worktree''s line'))
Invoke-Git $wt commit -q -am 'the worktree differs by one line'
Start-Sleep -Seconds 2
& robocopy (Join-Path $wt 'src') (Join-Path $main 'src') /MIR /NJH /NJS /NP /NFL /NDL | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy /MIR failed ($LASTEXITCODE)" }
$mObj = Join-Path $m 'obj\consumer.obj'; $mLib = Join-Path $m 'out\consumer.lib'
Compile (Join-Path $main 'src\consumer.cpp') $mObj
Archive $mObj $mLib
$wtSpec = "scratch consumer|$wt|$(Head $wt)|src|$mLib|$main|$(Join-Path $m 'obj')"
$r = Run-Gen 'MC-A' @{ Consumer = @($wtSpec) }
Grade 'MC-A must-accept, mirror in place, read names the worktree' $r 0 'wrote '
Start-Sleep -Seconds 2
Invoke-Git $main checkout -q -- .
Invoke-Git $main clean -q -fd
$r = Run-Gen 'MC-3' @{ Consumer = @($wtSpec) }
Grade 'MC-3 restored in place, read names the worktree' $r 1 'criterion 3 -- *src/consumer.cpp differs'
$mainSpec = "scratch consumer|$main|$(Head $main)|src|$mLib||$(Join-Path $m 'obj')"
$r = Run-Gen 'MC-5' @{ Consumer = @($mainSpec) }
Grade 'MC-5 restored in place, read names the checkout (criterion 5''s own must-reject)' $r 1 'criterion 5 -- *src/consumer.cpp (modified'

# ---------------------------------------------------------------------------------------------------------------
# The recorded-read admission.
$recordedTsv = Join-Path $PSScriptRoot 'recorded\superslmunreal-upgrade-census-imports.tsv'
$disp = "why no live read: the upgrade was in flight, and its objects (2026-09-18 23:15-23:17) predate the branch's later sources (tip cabc702bef at 23:43 plus uncommitted edits), so no commit names them; " +
        "provenance: the planner's object census Claude/Vitruvius/t2823-probe/census-output.txt (T-2823 record Sec2), converted mechanically by census_to_tsv.ps1; " +
        "replaced by: a live read of the upgrade's re-pinning commit at the release reading (plan Sec3.5 step 5)"
function RecSpec([string]$tsv, [string]$d, [string]$census) { "SuperSLMUnreal upgrade runtime module|census|$tsv|$d|$census" }
if (-not (Test-Path -LiteralPath $UpgradeCensus)) {
    Write-Output "== SETUP: the upgrade census $UpgradeCensus is not readable; MA-3, MR-H and MR-T cannot run"
    $results += [pscustomobject]@{ Case = 'SETUP upgrade census'; Ok = $false }
} else {
    $r = Run-Gen 'MA-3' @{ Recorded = @(RecSpec $recordedTsv $disp $UpgradeCensus) }
    Grade 'MA-3 must-accept, the 5ffed4c conversion as filed' $r 0 'wrote '
    $tampered = Join-Path $Scratch 'census-tampered.txt'
    Copy-Item -LiteralPath $UpgradeCensus $tampered
    Add-Content -LiteralPath $tampered '# one appended line'
    $r = Run-Gen 'MR-H' @{ Recorded = @(RecSpec $recordedTsv $disp $tampered) }
    Grade 'MR-H census does not hash to the pin' $r 1 "hashes to $((Get-FileHash -Algorithm SHA256 $tampered).Hash.ToLower()), not the pinned 6d75b502*"
    $handDir = Join-Path $Scratch 'handtyped'
    New-Item -ItemType Directory -Force $handDir | Out-Null
    $handTsv = Join-Path $handDir (Split-Path -Leaf $recordedTsv)
    Copy-Item -LiteralPath $recordedTsv $handTsv
    Add-Content -LiteralPath $handTsv "SuperSLMUnreal`tSuperSLMModelImport.cpp.obj`tsslm_prefill"
    $r = Run-Gen 'MR-T' @{ Recorded = @(RecSpec $handTsv $disp $UpgradeCensus) }
    Grade 'MR-T a hand-typed row (addition)' $r 1 'is not the mechanical conversion of its census*sslm_prefill'
}
$r = Run-Gen 'MR-D0' @{ Recorded = @(RecSpec $recordedTsv '' $UpgradeCensus) }
Grade 'MR-D0 disposition missing' $r 1 'the disposition sentence is missing'
$parts = [ordered]@{ 'MR-D1' = 'why no live read:'; 'MR-D2' = 'provenance:'; 'MR-D3' = 'replaced by:' }
foreach ($k in $parts.Keys) {
    $d = $disp.Replace($parts[$k], 'and')
    $r = Run-Gen $k @{ Recorded = @(RecSpec $recordedTsv $d $UpgradeCensus) }
    Grade "$k disposition lacks '$($parts[$k])'" $r 1 "the disposition lacks '$($parts[$k])'"
}

# MR-E: recorded at release.
$c = New-Consumer 'mre'
$r = Run-Gen 'MR-E' @{ Consumer = @(Spec $c); Recorded = @(RecSpec $recordedTsv $disp $UpgradeCensus); Release = $true }
Grade 'MR-E recorded at release' $r 1 '-Release refuses every -Recorded input'

# MR-F / MA-2: an undeclared import in a recorded read converted from a constructed census, and the same without it.
$cdir = Join-Path $Scratch 'constructed'
New-Item -ItemType Directory -Force $cdir | Out-Null
$censusLines = @('vendored objects: 1; engine-defined externals: 3 (T-2843 commissioning, constructed)',
                 '  [Scratch] scratch.cpp.obj (2026-09-19 00:00:00): 2',
                 '      sslm_model_unmap',
                 '      ?Sha256Hash@superslm@@YAXPEBE_KQEAE@Z')
$pinsFile = Join-Path $cdir 'census_pins.txt'
$pinRows = @('# constructed pins for the commissioning run')
foreach ($case in @(@{ Name = 'declared'; Extra = @() }, @{ Name = 'undeclared'; Extra = @('      sslm_not_a_verb') })) {
    $cen = Join-Path $cdir "$($case.Name).census.txt"
    [IO.File]::WriteAllLines($cen, [string[]]($censusLines + $case.Extra))
    $tsv = Join-Path $cdir "$($case.Name).tsv"
    & pwsh -NoProfile -File $conv -FromCensus $cen -ToTsv $tsv | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "census_to_tsv failed on $cen" }
    $pinRows += "$($case.Name).tsv`t$((Get-FileHash -Algorithm SHA256 $cen).Hash.ToLower())`tconstructed"
}
[IO.File]::WriteAllLines($pinsFile, [string[]]$pinRows)
$cd = 'why no live read: a constructed census; provenance: this commissioning script; replaced by: none, a commissioning construction'
$r = Run-Gen 'MR-F' @{ Recorded = @("constructed|none|$(Join-Path $cdir 'undeclared.tsv')|$cd|$(Join-Path $cdir 'undeclared.census.txt')"); CensusPins = $pinsFile }
Grade 'MR-F undeclared import' $r 1 'imports sslm_not_a_verb'
$r = Run-Gen 'MA-2' @{ Recorded = @("constructed|none|$(Join-Path $cdir 'declared.tsv')|$cd|$(Join-Path $cdir 'declared.census.txt')"); CensusPins = $pinsFile }
Grade 'MA-2 must-accept, the same without sslm_not_a_verb' $r 0 'wrote '

$bad = @($results | Where-Object { -not $_.Ok })
Write-Output ("GENERATOR COMMISSIONING: {0} -- {1} of {2} cases as expected" -f $(if ($bad.Count -eq 0) { 'PASS' } else { 'FAIL' }), ($results.Count - $bad.Count), $results.Count)
exit ($(if ($bad.Count -eq 0) { 0 } else { 1 }))
