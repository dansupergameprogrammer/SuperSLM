# T-2814, T-2825 (Curie) -- the export-slot list generator of plan Sec3.7 item 3 (TE-266 GPU-path plan;
# D-SLM7313; widened to the whole C ABI by the T-2823 fold).
#
# Writes two lists; neither is ever typed by hand.
#
#   c_abi_verbs.txt       THE C PART, from the header. Every verb declared in include/superslm/
#                         sslm_abi_functions.inc and sslm_abi_functions_g5_comparable.inc of -Engine, read by
#                         c_abi_verbs.ps1 (a declaration, never a comment mention). 36 at v1.5.0.
#   consumed_symbols.txt  THE C++ PART'S SOURCE OF TRUTH, from the consumers' compiled objects: every name a
#                         sibling consumer's objects leave UNDEF and External that matches `@superslm@@` (a C++
#                         name in namespace superslm) or `^sslm_` (a C verb), CASE-SENSITIVELY (-cmatch;
#                         PowerShell's -match also admits a consumer's own `SuperSLM::` facade). Both parts are
#                         kept, for provenance. The header names each consumer, its commit and its objects.
#
# X2's expected export and import set (build_x2.ps1) is consumed_symbols.txt's C++ entries united with
# c_abi_verbs.txt: 61 at v1.5.0 (25 C++ + 36 C).
#
# REFUSALS (the generator writes nothing and exits non-zero):
#   - a consumer imports an `sslm_*` name that c_abi_verbs.txt does not hold. The C part of the slot is the
#     header's whole C surface, so an import outside it is a verb this engine does not declare, a stale
#     consumer, or a consumer of another engine -- never something to add to the list;
#   - a live consumer read fails any of the five criteria below (-Consumer);
#   - a -Recorded input fails its admission (below);
#   - -Release is given together with any -Recorded input.
#
# Usage:
#   gen_consumed_symbols.ps1 [-Engine <tree>] -Consumer '<spec>', ... [-Recorded '<spec>', ...] [-Release]
#                            [-Exclude 'SuperSLMVendored_*'] [-Out consumed_symbols.txt] [-CVerbsOut c_abi_verbs.txt]
#   (-Consumer and -Recorded take comma-separated arrays; PowerShell rejects a flag given twice.)
#
# -Consumer '<label>|<checkout>|<commit>|<source roots>|<link output>|<build source tree>|<objdir>[|<objdir>...]'
#   a LIVE read (plan Sec3.7 item 3, T-2830: T-2824 G5 states the criterion, T-2825 F-4 adds the mirrored build).
#   <checkout>          the NAMED checkout: the git working tree whose files the compiler read, and
#   <commit>            the commit named for it in the list's header;
#   <source roots>      paths relative to <checkout>, separated by ';'. An entry starting with '!' is an
#                       exclusion glob over the same relative paths ('/' separators, `**` any depth), for
#                       example 'Plugins/SuperSLMUnreal/Source;!Plugins/SuperSLMUnreal/Source/**/Private/Vendored/**'
#                       (the census reads only non-vendored objects, so the vendored engine is not a root);
#   <link output>       the module DLL or static library the build linked from those objects;
#   <build source tree> the directory the compiler actually read, standing for <checkout>'s root: each root is
#                       compared as <build source tree>\<root> against <checkout>\<root>. Empty means <checkout>
#                       itself. A UE plugin built from a worktree is mirrored into the main checkout and compiled
#                       there: name the WORKTREE and its commit as <checkout>, the MAIN checkout as <build source
#                       tree>, and take the read while the mirror is still in place;
#   <objdir>            searched recursively for *.obj; -Exclude drops objects by file name (SuperSLMUnreal's
#                       vendored SuperSLMVendored_*.cpp).
#   The read is REFUSED, naming the file or commit, when any of these holds:
#     1. `git -C <checkout> rev-parse HEAD` is not <commit>;
#     2. `git -C <checkout> status --porcelain -- <source roots>` is not empty;
#     3. <build source tree> differs from <checkout> over the source roots in any relative path, or in any
#        file's SHA-256 -- so naming the main checkout after a mirror was restored fails (its files are
#        develop's, not the ones compiled), and so does naming the worktree after the restore;
#     4. the link output is older than the last commit touching the source roots
#        (`git log -1 --format=%ct <commit> -- <source roots>`), or older than the newest object: a build of
#        sources committed later, or a build that failed part-way (fresh objects, no new link output);
#     5. any file in <build source tree> within the source roots (exclusions applied, as for criterion 3) was
#        modified after the link output (T-2837, the coverage audit's T-2834 F1). The documented restore of a
#        mirrored UE build is `git checkout` and `git clean` in place, which rewrites every file that differs with
#        the current time: naming the main checkout after that restore passes criteria 1-4 (the build source tree
#        is then the checkout's own files, and the restore makes no commit) and fails criterion 5 alone. Not
#        closed: a restore that keeps older modification times (robocopy /MIR back from a pristine copy).
#   False refusals are loud and safe: a build of uncommitted edits committed afterwards trips criterion 4, a
#   checkout, pull or branch switch after the build trips criterion 5, and the remedy for both is a rebuild.
#   Nothing is repaired by hand.
#
# -Recorded '<label>|<commit>|<tsv>|<disposition>|<census>'   BETWEEN RELEASES ONLY: an earlier per-object dumpbin
#   read, for a consumer the generator cannot read live, admitted only as a mechanical, hash-pinned conversion of a
#   recorded census (census_to_tsv.ps1). The TSV's LAST column is an imported name and its second-to-last the
#   object it came from; lines starting with '#' are provenance comments. The same case-sensitive filter and the
#   same undeclared-verb refusal apply. ADMISSION: the input is REFUSED, naming the check, when
#     - <disposition> is missing, or lacks any of its three labelled parts: "why no live read:" (why no live
#       read was possible), "provenance:" (the census's provenance) and "replaced by:" (which live read replaces
#       it); the sentence is copied into the list's header;
#     - the TSV has no pin in -CensusPins (default recorded\census_pins.txt), <census> does not hash to the
#       pinned SHA-256 (the refusal names both hashes), or the TSV's header names another census hash;
#     - the TSV's rows are not exactly the rows census_to_tsv.ps1 produces from <census> (a hand-typed row).
#   These are the checks that keep a hand-typed list out of the build between releases; -Release keeps a
#   recorded read out of the tag.
#
# -Release   the release reading (plan Sec3.5 step 5): every input must be live, and ANY -Recorded input is
#   refused. A recorded read can reach the tag only through a run without -Release.
param(
    [string]$Engine = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
    [string[]]$Consumer = @(),
    [string[]]$Recorded = @(),
    [string[]]$Exclude = @('SuperSLMVendored_*'),
    [string]$Out = (Join-Path $PSScriptRoot 'consumed_symbols.txt'),
    [string]$CVerbsOut = (Join-Path $PSScriptRoot 'c_abi_verbs.txt'),
    [string]$CensusPins = (Join-Path $PSScriptRoot 'recorded\census_pins.txt'),
    [switch]$Release
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vsenv.ps1')
. (Join-Path $PSScriptRoot 'c_abi_verbs.ps1')
. (Join-Path $PSScriptRoot 'census_to_tsv.ps1')

function Refuse([string]$why) { Write-Output "gen_consumed_symbols: REFUSED -- $why"; exit 1 }
if ($Release -and $Recorded.Count -gt 0) {
    Refuse ("-Release refuses every -Recorded input; at the release reading every consumer is read live (plan Sec3.5 step 5). Recorded: " + (($Recorded | ForEach-Object { ($_ -split '\|')[0] }) -join '; '))
}

# The files under <base>\<root> for every root, keyed by '/'-separated path relative to <base>, valued by
# SHA-256; exclusion globs ('!' entries) drop paths. A root that does not exist contributes nothing (so a
# build tree missing a whole root differs from a checkout that has it).
function Get-RootFiles([string]$base, [string[]]$roots, [string[]]$excl) {
    $h = [ordered]@{}
    $baseFull = (Resolve-Path $base).Path.TrimEnd('\')
    foreach ($r in $roots) {
        $p = Join-Path $base $r
        if (-not (Test-Path $p)) { continue }
        $items = if ((Get-Item $p).PSIsContainer) { Get-ChildItem -Path $p -Recurse -File } else { @(Get-Item $p) }
        foreach ($f in $items) {
            $rel = $f.FullName.Substring($baseFull.Length).TrimStart('\').Replace('\', '/')
            if ($excl | Where-Object { $rel -like ($_ -replace '\*\*', '*') }) { continue }
            $h[$rel] = (Get-FileHash -Algorithm SHA256 $f.FullName).Hash.ToLower()
        }
    }
    return $h
}

# --- the C part, from the header ------------------------------------------------------------------
$incDir = Join-Path $Engine 'include\superslm'
$verbs = @(Get-CAbiVerbs $incDir)
if ($verbs.Count -eq 0) { Refuse "no C verb declarations under $incDir" }
$verbSet = New-Object 'System.Collections.Generic.HashSet[string]' ([string[]]@($verbs | ForEach-Object Name)), ([System.StringComparer]::Ordinal)
# The engine tree's revision is provenance only; a tree outside git (a scratch copy) is named as such.
$engineRev = 'a tree outside git'; $engineDirty = $false
$eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
$rev = & git -C $Engine rev-parse --short HEAD 2>$null
if ($LASTEXITCODE -eq 0 -and $rev) {
    $engineRev = "$rev".Trim()
    $engineDirty = [bool](& git -C $Engine status --porcelain -- include/superslm/sslm_abi_functions.inc include/superslm/sslm_abi_functions_g5_comparable.inc 2>$null)
}
$ErrorActionPreference = $eap
$byFile = $verbs | Group-Object File | ForEach-Object { "$($_.Name) $($_.Count)" }
$cHeader = @(
    '# The C ABI''s declared verbs: every declaration in include/superslm/sslm_abi_functions.inc and',
    '# sslm_abi_functions_g5_comparable.inc. GENERATED by tests/t2807-api-slot/gen_consumed_symbols.ps1 (reader:',
    '# c_abi_verbs.ps1) -- do not edit by hand. Plan: TE-266 Sec3.7 item 3 (T-2823). X2 requires the engine DLL to',
    '# export every name below, with the C++ entries of consumed_symbols.txt, and nothing else.',
    "# read from: include/superslm of the engine tree at $engineRev$(if ($engineDirty) { ', .inc files modified in the working tree' })",
    "# per file: $($byFile -join '; ')",
    "# verbs: $($verbs.Count)"
)

# --- the consumers ---------------------------------------------------------------------------------
$union = New-Object 'System.Collections.Generic.SortedSet[string]' ([System.StringComparer]::Ordinal)
$header = @(
    '# SuperSLM engine symbols consumed across a shared-library boundary by sibling UE modules.',
    '# GENERATED by tests/t2807-api-slot/gen_consumed_symbols.ps1 -- do not edit by hand. Plan: TE-266 Sec3.7.',
    '# Filter: dumpbin /symbols, UNDEF External, -cmatch ''@superslm@@'' or ''^sslm_''. Both parts are kept for',
    '# provenance; X2''s expected set is the C++ entries (not ^sslm_) united with c_abi_verbs.txt.'
)
$undeclared = @()
function Add-Consumer([string]$label, $names) {
    foreach ($n in $names) {
        if ($n -cmatch '^sslm_' -and -not $verbSet.Contains($n)) { $script:undeclared += "$label imports $n" }
        [void]$union.Add($n)
    }
}

foreach ($spec in $Consumer) {
    $parts = $spec -split '\|'
    if ($parts.Count -lt 7) { throw "bad -Consumer '$spec' (want <label>|<checkout>|<commit>|<source roots>|<link output>|<build source tree>|<objdir>[|<objdir>...])" }
    $label = $parts[0]; $repo = $parts[1]; $commit = $parts[2]
    $rootSpec = @($parts[3] -split ';' | Where-Object { $_ })
    $srcs = @($rootSpec | Where-Object { -not $_.StartsWith('!') })
    $excl = @($rootSpec | Where-Object { $_.StartsWith('!') } | ForEach-Object { $_.Substring(1) })
    $linkOut = $parts[4]; $buildTree = if ($parts[5]) { $parts[5] } else { $repo }
    $dirs = $parts[6..($parts.Count - 1)]
    if ($srcs.Count -eq 0) { throw "consumer '$label' names no source root" }
    $pathspec = @($srcs) + @($excl | ForEach-Object { ":(exclude,glob)$_" })
    # Criterion 1: HEAD is the named commit.
    $eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    $head = "$(& git -C $repo rev-parse HEAD 2>$null)".Trim()
    $want = "$(& git -C $repo rev-parse --verify "$commit^{commit}" 2>$null)".Trim()
    $ErrorActionPreference = $eap
    if (-not $head) { Refuse "$label : $repo is not a git working tree" }
    if (-not $want) { Refuse "$label : commit '$commit' does not resolve in $repo" }
    if ($head -ne $want) { Refuse "$label : criterion 1 -- $repo is at $head, not the named commit $want" }
    # Criterion 2: the source roots are clean.
    $dirty = @(& git -C $repo status --porcelain -- @pathspec)
    if ($dirty.Count -gt 0) { Refuse "$label : criterion 2 -- source roots modified in $repo, so its objects are not attributable to $commit : $($dirty -join '; ')" }
    # Criterion 3: the build source tree the compiler read equals the named checkout, file by file.
    if (-not (Test-Path $buildTree)) { Refuse "$label : criterion 3 -- the build source tree $buildTree does not exist" }
    $mineFiles = Get-RootFiles $repo $srcs $excl
    $sameTree = (Resolve-Path $buildTree).Path.TrimEnd('\') -ieq (Resolve-Path $repo).Path.TrimEnd('\')
    $buildFiles = if ($sameTree) { $mineFiles } else { Get-RootFiles $buildTree $srcs $excl }
    if ($mineFiles.Count -eq 0) { Refuse "$label : criterion 3 -- no files under the source roots $($srcs -join ', ') in $repo" }
    $c3 = @()
    foreach ($k in $mineFiles.Keys) {
        if (-not $buildFiles.Contains($k)) { $c3 += "$k is in $repo but not in the build source tree $buildTree" }
        elseif ($buildFiles[$k] -ne $mineFiles[$k]) { $c3 += "$k differs (SHA-256 $($mineFiles[$k].Substring(0,12)) in $repo, $($buildFiles[$k].Substring(0,12)) in $buildTree)" }
    }
    foreach ($k in $buildFiles.Keys) { if (-not $mineFiles.Contains($k)) { $c3 += "$k is in the build source tree $buildTree but not in $repo" } }
    if ($c3.Count -gt 0) { Refuse ("$label : criterion 3 -- the build source tree is not the named checkout's files: " + (($c3 | Select-Object -First 10) -join '; ') + $(if ($c3.Count -gt 10) { " (and $($c3.Count - 10) more)" } else { '' })) }
    # Criterion 4: the link output is no older than the last commit touching the roots, nor than the newest object.
    if (-not (Test-Path $linkOut)) { Refuse "$label : criterion 4 -- the link output $linkOut does not exist" }
    $linkItem = Get-Item $linkOut
    $linkT = [DateTimeOffset]::new($linkItem.LastWriteTimeUtc).ToUnixTimeSeconds()
    $srcTime = [int64](& git -C $repo log -1 --format=%ct $want -- @pathspec)
    $objs = foreach ($d in $dirs) {
        if (-not (Test-Path $d)) { throw "object directory not found: $d" }
        Get-ChildItem -Path $d -Recurse -Filter *.obj -File |
            Where-Object { $n = $_.Name; -not ($Exclude | Where-Object { $n -like $_ }) }
    }
    $objs = @($objs | Sort-Object FullName)
    if ($objs.Count -eq 0) { throw "no objects for consumer '$label'" }
    $newestObj = ($objs | Sort-Object LastWriteTimeUtc | Select-Object -Last 1)
    $srcTimeText = [DateTimeOffset]::FromUnixTimeSeconds($srcTime).LocalDateTime.ToString('yyyy-MM-dd HH:mm:ss')
    if ($linkT -lt $srcTime) {
        Refuse ("$label : criterion 4 -- the link output $($linkItem.Name) ($($linkItem.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))) is older than the last commit touching " +
                "$($srcs -join ', ') at $commit ($srcTimeText): the build did not compile the committed sources")
    }
    if ($linkItem.LastWriteTimeUtc -lt $newestObj.LastWriteTimeUtc) {
        Refuse ("$label : criterion 4 -- the link output $($linkItem.Name) ($($linkItem.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))) is older than the newest object " +
                "$($newestObj.Name) ($($newestObj.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))): the build did not finish linking what it compiled")
    }
    # Criterion 5: no file the compiler could have read was modified after the link output.
    $buildBase = (Resolve-Path $buildTree).Path.TrimEnd('\')
    $c5 = @()
    foreach ($k in $buildFiles.Keys) {
        $fi = Get-Item -LiteralPath (Join-Path $buildBase ($k -replace '/', '\'))
        if ($fi.LastWriteTimeUtc -gt $linkItem.LastWriteTimeUtc) {
            $c5 += "$k (modified $($fi.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss.fff')))"
        }
    }
    if ($c5.Count -gt 0) {
        Refuse ("$label : criterion 5 -- files in the build source tree $buildTree were modified after the link output $($linkItem.Name) " +
                "($($linkItem.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss.fff'))), so the objects were not compiled from them: " +
                (($c5 | Select-Object -First 10) -join '; ') + $(if ($c5.Count -gt 10) { " (and $($c5.Count - 10) more)" } else { '' }))
    }
    $mine = New-Object 'System.Collections.Generic.SortedSet[string]' ([System.StringComparer]::Ordinal)
    foreach ($o in $objs) {
        $lines = & dumpbin.exe /nologo /symbols $o.FullName
        if ($LASTEXITCODE -ne 0) { throw "dumpbin failed on $($o.FullName)" }
        foreach ($l in $lines) {
            if ($l -cmatch '\bUNDEF\b.*\bExternal\s+\|\s+(\S+)') {
                $name = $Matches[1]
                if ($name -cmatch '@superslm@@' -or $name -cmatch '^sslm_') { [void]$mine.Add($name) }
            }
        }
    }
    $oldest = ($objs | Sort-Object LastWriteTime | Select-Object -First 1).LastWriteTime.ToString('yyyy-MM-dd HH:mm')
    $nC = @($mine | Where-Object { $_ -cmatch '^sslm_' }).Count
    $header += "# consumer: $label @ $want -- LIVE read of $($objs.Count) objects (written $oldest .. $($newestObj.LastWriteTime.ToString('yyyy-MM-dd HH:mm'))), $($mine.Count - $nC) C++ and $nC C engine symbols"
    $header += "#   live-read criteria (plan Sec3.7 item 3): 1 HEAD = the named commit; 2 roots clean ($($rootSpec -join '; ')); 3 build source tree $buildTree equals the checkout over $($mineFiles.Count) files by SHA-256; 4 link output $linkOut ($($linkItem.LastWriteTime.ToString('yyyy-MM-dd HH:mm'))) not older than the roots' last commit ($srcTimeText) nor the newest object; 5 no build-source-tree file modified after the link output"
    foreach ($d in $dirs) { $header += "#   objects: $d (excluding $($Exclude -join ', '))" }
    Add-Consumer $label $mine
}
# The census pins: <tsv file name> TAB <census sha256> TAB <description>.
$pins = @{}
if (Test-Path $CensusPins) {
    foreach ($l in (Get-Content -LiteralPath $CensusPins)) {
        if ($l -match '^\s*#' -or -not $l.Trim()) { continue }
        $f = $l -split "`t"
        if ($f.Count -ge 2) { $pins[$f[0].Trim()] = $f[1].Trim().ToLower() }
    }
}
$dispLabels = @('why no live read:', 'provenance:', 'replaced by:')
foreach ($spec in $Recorded) {
    $parts = $spec -split '\|'
    if ($parts.Count -ne 5) { Refuse "bad -Recorded '$spec' (want <label>|<commit>|<tsv>|<disposition>|<census>)" }
    $label = $parts[0]; $commit = $parts[1]; $tsv = $parts[2]; $disp = $parts[3].Trim(); $census = $parts[4]
    # Admission 1: the disposition sentence and its three labelled parts.
    if (-not $disp) { Refuse "$label : recorded-read admission -- the disposition sentence is missing" }
    $at = @($dispLabels | ForEach-Object { $disp.IndexOf($_, [StringComparison]::Ordinal) })
    $missing = @()
    for ($i = 0; $i -lt $dispLabels.Count; $i++) {
        if ($at[$i] -lt 0) { $missing += "'$($dispLabels[$i])'"; continue }
        $start = $at[$i] + $dispLabels[$i].Length
        $ends = @($at | Where-Object { $_ -gt $at[$i] }) + $disp.Length
        $text = $disp.Substring($start, ($ends | Measure-Object -Minimum).Minimum - $start).Trim(" ;.`t")
        if ($text -notmatch '[A-Za-z0-9]') { $missing += "'$($dispLabels[$i])' (empty)" }
    }
    if ($missing.Count -gt 0) { Refuse "$label : recorded-read admission -- the disposition lacks $($missing -join ', ') (it states why no live read was possible, the census's provenance, and which live read replaces it)" }
    if (-not (Test-Path $tsv)) { Refuse "$label : recorded TSV not found: $tsv" }
    # Admission 2: the census is the pinned one, and the TSV names it.
    $leaf = Split-Path -Leaf $tsv
    if (-not $pins.ContainsKey($leaf)) { Refuse "$label : recorded-read admission -- $leaf has no census pin in $CensusPins" }
    if (-not $census -or -not (Test-Path -LiteralPath $census)) { Refuse "$label : recorded-read admission -- the census '$census' is not readable, so the conversion cannot be checked" }
    $censusSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $census).Hash.ToLower()
    if ($censusSha -ne $pins[$leaf]) { Refuse "$label : recorded-read admission -- the census $census hashes to $censusSha, not the pinned $($pins[$leaf])" }
    $stated = @(Get-Content -LiteralPath $tsv | Where-Object { $_ -match '^#' } | Select-String -Pattern 'sha256 ([0-9a-f]{64})' -AllMatches |
                ForEach-Object { $_.Matches } | ForEach-Object { $_.Groups[1].Value })
    if ($stated -notcontains $pins[$leaf]) { Refuse "$label : recorded-read admission -- the header of $leaf does not name the pinned census hash $($pins[$leaf])" }
    # Admission 3: the rows are exactly the mechanical conversion of that census.
    $conv = @(ConvertFrom-ObjectCensus $census)
    $have = @(Get-Content -LiteralPath $tsv | Where-Object { $_ -notmatch '^#' -and $_ })
    if (($conv -join "`n") -cne ($have -join "`n")) {
        $extra = @($have | Where-Object { $conv -cnotcontains $_ }); $lost = @($conv | Where-Object { $have -cnotcontains $_ })
        Refuse "$label : recorded-read admission -- $leaf is not the mechanical conversion of its census ($($have.Count) rows, conversion $($conv.Count); not produced by the census: $((($extra | Select-Object -First 3) -join ' / ')); missing: $((($lost | Select-Object -First 3) -join ' / ')))"
    }
    $mine = New-Object 'System.Collections.Generic.SortedSet[string]' ([System.StringComparer]::Ordinal)
    $objs = New-Object 'System.Collections.Generic.HashSet[string]'
    $rows = 0; $dropped = 0
    foreach ($l in (Get-Content $tsv)) {
        if ($l -match '^#') { continue }
        $f = $l -split "`t"
        if ($f.Count -lt 2) { continue }
        $rows++
        $name = $f[-1]
        if ($name -cmatch '@superslm@@' -or $name -cmatch '^sslm_') { [void]$mine.Add($name); [void]$objs.Add($f[-2]) }
        else { $dropped++ }
    }
    $sha = (Get-FileHash -Algorithm SHA256 $tsv).Hash.ToLower()
    $nC = @($mine | Where-Object { $_ -cmatch '^sslm_' }).Count
    $rel = $tsv
    if ($tsv.StartsWith($PSScriptRoot, [StringComparison]::OrdinalIgnoreCase)) { $rel = $tsv.Substring($PSScriptRoot.Length).TrimStart('\', '/') }
    $header += "# consumer: $label @ $commit -- RECORDED per-object dumpbin read $rel (sha256 $sha): $rows rows, $($objs.Count) importing objects, $($mine.Count - $nC) C++ and $nC C engine symbols, $dropped rows dropped by the case-sensitive filter"
    $header += "#   admission: census $census sha256 $censusSha = the pin; the TSV's $($have.Count) rows = its mechanical conversion"
    $header += "#   disposition (no commit check is possible on a recorded read): $disp"
    Add-Consumer $label $mine
}
if ($Consumer.Count + $Recorded.Count -eq 0) { throw 'give at least one -Consumer or -Recorded' }
if ($Release) { $header += '# -Release: the release reading; every input above is a live read (no -Recorded input is admitted)' }
if ($undeclared.Count -gt 0) {
    Refuse ("a consumer imports an sslm_* name the header does not declare (c_abi_verbs.txt holds $($verbs.Count)): " + ($undeclared -join '; '))
}
$cpp = @($union | Where-Object { $_ -cnotmatch '^sslm_' })
$cUsed = @($union | Where-Object { $_ -cmatch '^sslm_' })
$header += "# union: $($union.Count) symbols ($($cpp.Count) C++, $($cUsed.Count) C)"
$header += "# X2 expected set: $($cpp.Count) C++ + $($verbs.Count) C (c_abi_verbs.txt) = $($cpp.Count + $verbs.Count)"
[System.IO.File]::WriteAllLines($CVerbsOut, [string[]]($cHeader + @($verbs | ForEach-Object Name)))
[System.IO.File]::WriteAllLines($Out, [string[]]($header + @($union)))
Write-Output "wrote $CVerbsOut : $($verbs.Count) C verbs"
Write-Output "wrote $Out : $($union.Count) symbols ($($cpp.Count) C++, $($cUsed.Count) C); X2 expects $($cpp.Count + $verbs.Count)"
