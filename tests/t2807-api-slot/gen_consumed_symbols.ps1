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
#   - a live consumer's objects are not attributable to the commit named for them (see -Consumer).
#
# Usage:
#   gen_consumed_symbols.ps1 [-Engine <tree>] -Consumer '<spec>', ... [-Recorded '<spec>', ...]
#                            [-Exclude 'SuperSLMVendored_*'] [-Out consumed_symbols.txt] [-CVerbsOut c_abi_verbs.txt]
#   (-Consumer and -Recorded take comma-separated arrays; PowerShell rejects a flag given twice.)
#
# -Consumer '<label>|<repo>|<commit>|<source paths>|<objdir>[|<objdir>...]'   a LIVE read.
#   <repo> is the git worktree the objects were built from, <commit> the commit they are attributed to,
#   <source paths> the consumer's compiled sources relative to <repo>, separated by ';' (for example
#   'src;include;CMakeLists.txt'). Each <objdir> is searched recursively for *.obj; -Exclude drops objects by
#   file name (a consumer that compiles the engine's own sources into its module, SuperSLMUnreal's vendored
#   SuperSLMVendored_*.cpp, must exclude them). The read is REFUSED unless all three hold:
#     1. <repo>'s HEAD is <commit>;
#     2. `git status --porcelain -- <source paths>` is empty (the objects cannot have been built from
#        uncommitted sources, and no source changed after the commit);
#     3. the newest object is not older than the last commit that touched <source paths> at <commit>:
#        the build ran after the sources last changed.
#   What this does not detect, stated so it is not mistaken for detected: (a) a build that ran after the
#   last source change and failed part-way, leaving some objects stale -- the consumer's build log is the
#   evidence for that; (b) objects that were not built from <repo> at all. A UE plugin built from a
#   worktree by mirroring it into the main checkout writes its objects under the main checkout from the
#   worktree's sources; naming the main checkout and its commit then passes all three checks while the
#   objects belong to another tree. Name the checkout the build actually compiled.
#
# -Recorded '<label>|<commit>|<tsv>|<disposition>'   an earlier per-object dumpbin read, for a consumer whose
#   objects at the named commit no longer exist on disk. The TSV's LAST column is an imported name and its
#   second-to-last the object it came from; lines starting with '#' are provenance comments. The same
#   case-sensitive filter and the same undeclared-verb refusal apply. The commit check above CANNOT be run on
#   a recorded read, so <disposition> is mandatory: one sentence stating when the objects were built against
#   their sources, copied into the list's header, so the reader sees what the read does and does not stand on.
param(
    [string]$Engine = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
    [string[]]$Consumer = @(),
    [string[]]$Recorded = @(),
    [string[]]$Exclude = @('SuperSLMVendored_*'),
    [string]$Out = (Join-Path $PSScriptRoot 'consumed_symbols.txt'),
    [string]$CVerbsOut = (Join-Path $PSScriptRoot 'c_abi_verbs.txt')
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vsenv.ps1')
. (Join-Path $PSScriptRoot 'c_abi_verbs.ps1')

function Refuse([string]$why) { Write-Output "gen_consumed_symbols: REFUSED -- $why"; exit 1 }

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
    if ($parts.Count -lt 5) { throw "bad -Consumer '$spec' (want <label>|<repo>|<commit>|<source paths>|<objdir>[|<objdir>...])" }
    $label = $parts[0]; $repo = $parts[1]; $commit = $parts[2]; $srcs = @($parts[3] -split ';' | Where-Object { $_ })
    $dirs = $parts[4..($parts.Count - 1)]
    $head = (& git -C $repo rev-parse HEAD).Trim()
    $want = (& git -C $repo rev-parse --verify "$commit^{commit}").Trim()
    if ($LASTEXITCODE -ne 0 -or -not $want) { Refuse "$label : commit '$commit' does not resolve in $repo" }
    if ($head -ne $want) { Refuse "$label : $repo is at $head, not the named commit $want" }
    $dirty = @(& git -C $repo status --porcelain -- @srcs)
    if ($dirty.Count -gt 0) { Refuse "$label : sources modified in $repo, so its objects are not attributable to $commit : $($dirty -join '; ')" }
    $srcTime = [int64](& git -C $repo log -1 --format=%ct $want -- @srcs)
    $objs = foreach ($d in $dirs) {
        if (-not (Test-Path $d)) { throw "object directory not found: $d" }
        Get-ChildItem -Path $d -Recurse -Filter *.obj -File |
            Where-Object { $n = $_.Name; -not ($Exclude | Where-Object { $n -like $_ }) }
    }
    $objs = @($objs | Sort-Object FullName)
    if ($objs.Count -eq 0) { throw "no objects for consumer '$label'" }
    $newestObj = ($objs | Sort-Object LastWriteTimeUtc | Select-Object -Last 1)
    $newestT = [DateTimeOffset]::new($newestObj.LastWriteTimeUtc).ToUnixTimeSeconds()
    if ($newestT -lt $srcTime) {
        Refuse ("$label : the newest object ($($newestObj.Name), $($newestObj.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))) is older than " +
                "the last commit touching $($srcs -join ', ') at $commit ($([DateTimeOffset]::FromUnixTimeSeconds($srcTime).LocalDateTime.ToString('yyyy-MM-dd HH:mm:ss')))")
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
    $header += "#   commit check: HEAD = the named commit; $($srcs -join ', ') clean; newest object not older than their last commit ($([DateTimeOffset]::FromUnixTimeSeconds($srcTime).LocalDateTime.ToString('yyyy-MM-dd HH:mm')))"
    foreach ($d in $dirs) { $header += "#   objects: $d (excluding $($Exclude -join ', '))" }
    Add-Consumer $label $mine
}
foreach ($spec in $Recorded) {
    $parts = $spec -split '\|'
    if ($parts.Count -ne 4 -or -not $parts[3].Trim()) { throw "bad -Recorded '$spec' (want <label>|<commit>|<tsv>|<disposition>; the disposition is mandatory)" }
    $label = $parts[0]; $commit = $parts[1]; $tsv = $parts[2]; $disp = $parts[3].Trim()
    if (-not (Test-Path $tsv)) { throw "recorded TSV not found: $tsv" }
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
    $header += "#   disposition (no commit check is possible on a recorded read): $disp"
    Add-Consumer $label $mine
}
if ($Consumer.Count + $Recorded.Count -eq 0) { throw 'give at least one -Consumer or -Recorded' }
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
