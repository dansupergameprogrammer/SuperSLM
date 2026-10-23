<#
.SYNOPSIS
Fills a plug-in drive for longleg.ps1, on the home box. Run it once per drive.

.DESCRIPTION
    powershell -NoProfile -ExecutionPolicy Bypass -File D:\SuperSLM\tools\longleg_kit\fill_drive.ps1 -Drive F:

Writes <Drive>\longleg\:
  inputs\synthetic\  synthetic0p5.sslm, synthetic1p5.sslm and their -three-schema twins (SUPERSLM_PRECACHE_SYNTHETIC_DIR)
  inputs\pkv\        pkv_def, pkv_odd, pkv_qk, pkv_32k .sslm (SUPERSLM_PAGED_KV_FIXTURE_DIR)
  inputs\e0\         the E0 debunker's three artifacts (SUPERSLM_PRECACHE_E0_ARTIFACT_DIR)
  inputs\models\     the converted real 0.5B and 1.5B models (for E4 once it is unblocked; no leg reads them yet)
  inputs\source\     superslm.bundle (full history of the branch and main) and SOURCE.txt (commit=, branch=);
                     e4w1.bundle (full history to E4's pinned commit e8c029ed, which longleg.ps1 -E4Worst builds for W1)
  inputs\MANIFEST.tsv  path, bytes, sha256 of every input; longleg.ps1 verifies against it once
  kit\               this folder, as committed at the bundled commit
  START.cmd, LONGLEG_DRIVE.txt

Fixtures and models are checked against the SHA-256 values in their records before they are copied. A rerun
re-copies only files whose size or hash differs. It never touches results\ on the drive.

.PARAMETER Drive
The drive to fill, such as F: (NTFS or exFAT; FAT32 is refused if any file is over 4 GB).

.PARAMETER Artifacts
The artifact root. Default D:\_artifacts\SuperSLM.

.PARAMETER Repo
A full (not shallow) clone of SuperSLM. Default D:\SuperSLM.

.PARAMETER Branch
The branch the kit builds. Default claude/precache-1p14-post-m1.

.PARAMETER SkipModels
Leave the real models off the drive (only -E4Worst's W1 reads one, the 1.5B). Saves about 4 GB.

.PARAMETER KitOnly
Refresh a drive that is part-way through its long legs, for longleg.ps1 -E4Worst: copy the kit as committed at the
branch's head, add e4w1.bundle and the 1.5B model if it is not there, and keep everything else as it is. SOURCE.txt and
superslm.bundle are not touched, so the long-leg pieces keep building the commit they were started at.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Drive,
    [string]$Artifacts = 'D:\_artifacts\SuperSLM',
    [string]$Repo = 'D:\SuperSLM',
    [string]$Branch = 'claude/precache-1p14-post-m1',
    [switch]$SkipModels,
    [switch]$KitOnly
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

function Write-Line([string]$Text) { Write-Output ('[{0}] {1}' -f (Get-Date).ToString('HH:mm:ss'), $Text) }
function Get-Sha256([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Write-Utf8File([string]$Path, [string]$Text) {
    [System.IO.File]::WriteAllText($Path, $Text, (New-Object System.Text.UTF8Encoding($false)))
}
function Invoke-Git([string[]]$Arguments) {
    $out = & git -C $Repo @Arguments
    if ($LASTEXITCODE -ne 0) { throw ('git ' + ($Arguments -join ' ') + " failed ($LASTEXITCODE)") }
    return $out
}

# What goes on the drive. Sha is the recorded SHA-256 (fixtures: the T-3035 record; models: their conversion
# records); an empty Sha means none is recorded, and the file's own hash goes into the manifest as found.
$fixtures = Join-Path $Artifacts 't3035-precache-refs\fixtures'
$copies = @(
    @{ From = "$fixtures\synthetic0p5.sslm"; To = 'synthetic\synthetic0p5.sslm'; Sha = 'd19e7a3676793dd18f14e73a738ae34d89a81a5331b3af0f8c62c266746f6951' }
    @{ From = "$fixtures\synthetic1p5.sslm"; To = 'synthetic\synthetic1p5.sslm'; Sha = '93fa76fcee1e9a919bc35a2a55440cb3f769302882353f2d5c29240164abf6fc' }
    @{ From = "$fixtures\synthetic0p5-three-schema.sslm"; To = 'synthetic\synthetic0p5-three-schema.sslm'; Sha = '4810e7dbf1d78556cada06e7f18ec9d4e9fef6708e368181eaca4f899b3ea199' }
    @{ From = "$fixtures\synthetic1p5-three-schema.sslm"; To = 'synthetic\synthetic1p5-three-schema.sslm'; Sha = '56f1432393105ad4fb10547b4bcf50e60b8e5181c8d4132749ee97c4a6a09670' }
    @{ From = "$fixtures\pkv_def.sslm"; To = 'pkv\pkv_def.sslm'; Sha = '994b58b49d3660bccfdd59dc6315028ea0203f37f4705d70eb7ce4dbecdb1db3' }
    @{ From = "$fixtures\pkv_qk.sslm"; To = 'pkv\pkv_qk.sslm'; Sha = '1d04133d3e3563617584dbb37bdfbdb1b6c347a96fdea1b558fbd1f8c0e5540e' }
    @{ From = "$fixtures\pkv_odd.sslm"; To = 'pkv\pkv_odd.sslm'; Sha = '2a43398dd253c97e5b3da244e80faeae0a7ab305ff09116a6bbc02808ed76d2d' }
    @{ From = "$fixtures\pkv_32k.sslm"; To = 'pkv\pkv_32k.sslm'; Sha = 'd45cabb1da261194a01d6b30b21374f025448e8b4c2e8c0289bc76fb9c045fbc' }
    @{ From = (Join-Path $Artifacts 't3030-commission\must-reject-two-name.sslm'); To = 'e0\must-reject-two-name.sslm'; Sha = '' }
    @{ From = (Join-Path $Artifacts 't3030-commission\must-reject-zero-hash.sslm'); To = 'e0\must-reject-zero-hash.sslm'; Sha = '' }
    @{ From = (Join-Path $Artifacts 't3030-commission\must-accept-renamed-pair.sslm'); To = 'e0\must-accept-renamed-pair.sslm'; Sha = '' }
)
if (-not $SkipModels) {
    $copies += @(
        @{ From = (Join-Path $Artifacts 'te442-qwen2.5-0.5b-instruct-dg\qwen2.5-0.5b-instruct-dg.sslm'); To = 'models\qwen2.5-0.5b-instruct-dg.sslm'; Sha = '4b27e2d6e132d710a4b174d5ffc15e7705691ca37abf15eca65503803d0dba35' }
        @{ From = (Join-Path $Artifacts 't2901-qwen2.5-1.5b-dg\qwen2.5-1.5b-instruct-dg-g5.sslm'); To = 'models\qwen2.5-1.5b-instruct-dg-g5.sslm'; Sha = '93c8149d8a40a9700f5f783b497c5b35b55c0eb3c9b4f26ba7198220479b6790' }
    )
}

# E4's W1 (longleg.ps1 -E4Worst) builds the E4 run sheet's pinned commit, so B9's restore on the home box accepts its blob.
$e4Commit = 'e8c029ed68f6d3fb0a728673c9a41d16d433916d'
$e4Model = 'models\qwen2.5-1.5b-instruct-dg-g5.sslm'
if ($KitOnly) {
    if ($SkipModels) { throw '-KitOnly adds the 1.5B model for W1; do not pass -SkipModels with it' }
    $copies = @($copies | Where-Object { $_.To -eq $e4Model })
}

# ---------------------------------------------------------------------------------------------------------------
# Check everything before writing anything
# ---------------------------------------------------------------------------------------------------------------
$Drive = $Drive.TrimEnd('\', ':') + ':'
if ($Drive -notmatch '^[A-Za-z]:$') { throw "-Drive must be a drive letter such as F: (got $Drive)" }
$root = "$Drive\"
if (-not (Test-Path -LiteralPath $root)) { throw "drive $Drive is not there" }
if ($Drive -ieq (Split-Path -Qualifier $Artifacts) -or $Drive -ieq (Split-Path -Qualifier $Repo)) { throw "drive $Drive holds the artifacts or the repo; pick the plug-in drive" }

$missing = @($copies | Where-Object { -not (Test-Path -LiteralPath $_.From) } | ForEach-Object { $_.From })
if ($missing.Count -gt 0) { throw ("these source files are missing:`n  " + ($missing -join "`n  ")) }

if (-not (Test-Path -LiteralPath (Join-Path $Repo '.git'))) { throw "$Repo is not a git clone" }
if ((Invoke-Git @('rev-parse', '--is-shallow-repository')).Trim() -ne 'false') { throw "$Repo is a shallow clone; run git fetch --unshallow there first" }
Write-Line "fetching $Branch and main into $Repo"
Invoke-Git @('fetch', '-q', 'origin', "+refs/heads/${Branch}:refs/remotes/origin/$Branch", '+refs/heads/main:refs/remotes/origin/main') | Out-Null
$commit = (Invoke-Git @('rev-parse', "refs/remotes/origin/$Branch^{commit}")).Trim()
Invoke-Git @('cat-file', '-e', "${commit}:tools/longleg_kit/longleg.ps1") | Out-Null
Invoke-Git @('merge-base', '--is-ancestor', $e4Commit, $commit) | Out-Null
if ($KitOnly) { Write-Line "refreshing the kit to $Branch at $commit; the long legs' source is kept" }
else { Write-Line "the kit builds $Branch at $commit" }

$volume = Get-Volume -DriveLetter $Drive.Substring(0, 1)
$fs = [string]$volume.FileSystem
$largest = ($copies | ForEach-Object { (Get-Item -LiteralPath $_.From).Length } | Measure-Object -Maximum).Maximum
if ($fs -match 'FAT32' -and $largest -ge 4GB) { throw "drive $Drive is FAT32 and a file is over 4 GB; reformat it as exFAT or NTFS" }
$inputBytes = ($copies | ForEach-Object { (Get-Item -LiteralPath $_.From).Length } | Measure-Object -Sum).Sum
$gitBytes = (Get-ChildItem -LiteralPath (Join-Path $Repo '.git\objects') -Recurse -File | Measure-Object -Property Length -Sum).Sum
$need = $inputBytes + $gitBytes + 1GB
Write-Line ('drive {0}: {1}, {2:N1} GB free; inputs {3:N1} GB, bundle at most {4:N1} GB' -f $Drive, $fs, ($volume.SizeRemaining / 1GB), ($inputBytes / 1GB), ($gitBytes / 1GB))
Write-Line ('the run needs about 7 GB more on the drive if C: has under 30 GB free (builds and journals go there then)')
if ($volume.SizeRemaining -lt $need) { throw ('drive {0} has {1:N1} GB free; it needs {2:N1} GB' -f $Drive, ($volume.SizeRemaining / 1GB), ($need / 1GB)) }

# ---------------------------------------------------------------------------------------------------------------
# Copy, verifying the recorded hashes on the way
# ---------------------------------------------------------------------------------------------------------------
$longLeg = Join-Path $root 'longleg'
$inputs = Join-Path $longLeg 'inputs'
foreach ($d in @('synthetic', 'pkv', 'e0', 'models', 'source')) { New-Item -ItemType Directory -Force -Path (Join-Path $inputs $d) | Out-Null }
New-Item -ItemType Directory -Force -Path (Join-Path $longLeg 'results') | Out-Null

$manifest = New-Object System.Collections.Generic.List[string]
$rewritten = @($copies | ForEach-Object { $_.To }) + @('source\e4w1.bundle')
if ($KitOnly) {
    $oldManifest = Join-Path $inputs 'MANIFEST.tsv'
    if (-not (Test-Path -LiteralPath $oldManifest)) { throw "$oldManifest is missing: fill the drive in full first (no -KitOnly)" }
    foreach ($line in [System.IO.File]::ReadAllLines($oldManifest)) {
        if ($line -and -not $line.StartsWith('#') -and $rewritten -notcontains $line.Split("`t")[0]) { $manifest.Add($line) }
    }
    $manifest.Insert(0, "# path`tbytes`tsha256 (kit refreshed by fill_drive.ps1 -KitOnly at $commit)")
} else {
    $manifest.Add("# path`tbytes`tsha256 (written by fill_drive.ps1 for $commit)")
}
foreach ($c in $copies) {
    $src = Get-Item -LiteralPath $c.From
    $srcSha = Get-Sha256 $src.FullName
    if ($c.Sha -and $srcSha -ne $c.Sha) { throw "$($c.From) has SHA-256 $srcSha, not the recorded $($c.Sha)" }
    $dest = Join-Path $inputs $c.To
    $same = (Test-Path -LiteralPath $dest) -and ((Get-Item -LiteralPath $dest).Length -eq $src.Length) -and ((Get-Sha256 $dest) -eq $srcSha)
    if ($same) {
        Write-Line "already on the drive: $($c.To)"
    } else {
        Write-Line ('copying {0} ({1:N2} GB)' -f $c.To, ($src.Length / 1GB))
        Copy-Item -LiteralPath $src.FullName -Destination $dest -Force
        if ((Get-Sha256 $dest) -ne $srcSha) { throw "the copy of $($c.To) does not match its source; the drive may be failing" }
    }
    $manifest.Add($c.To + "`t" + $src.Length + "`t" + $srcSha)
}

# Source: a bundle with the full history of the branch and main, so the PC clones with no network or credentials.
$sourceFiles = @('source\e4w1.bundle')
if (-not $KitOnly) {
    $bundle = Join-Path $inputs 'source\superslm.bundle'
    Write-Line 'writing the source bundle (full history)'
    Invoke-Git @('bundle', 'create', $bundle, "refs/remotes/origin/$Branch", 'refs/remotes/origin/main') | Out-Null
    Invoke-Git @('bundle', 'verify', '-q', $bundle) | Out-Null
    $sourceTxt = Join-Path $inputs 'source\SOURCE.txt'
    Write-Utf8File $sourceTxt ("commit=$commit`nbranch=$Branch`nfilled=" + (Get-Date).ToString('s') + "`n")
    $sourceFiles = @('source\superslm.bundle', 'source\SOURCE.txt') + $sourceFiles
}
# W1's source: E4's pinned commit, under a ref of its own (a bundle carries refs, not bare commits).
$e4Bundle = Join-Path $inputs 'source\e4w1.bundle'
Write-Line "writing the W1 source bundle ($e4Commit, full history)"
Invoke-Git @('update-ref', 'refs/longleg/e4w1', $e4Commit) | Out-Null
try {
    Invoke-Git @('bundle', 'create', $e4Bundle, 'refs/longleg/e4w1') | Out-Null
} finally {
    Invoke-Git @('update-ref', '-d', 'refs/longleg/e4w1') | Out-Null
}
Invoke-Git @('bundle', 'verify', '-q', $e4Bundle) | Out-Null
foreach ($f in $sourceFiles) {
    $p = Join-Path $inputs $f
    $manifest.Add($f + "`t" + (Get-Item -LiteralPath $p).Length + "`t" + (Get-Sha256 $p))
}
Write-Utf8File (Join-Path $inputs 'MANIFEST.tsv') (($manifest -join "`n") + "`n")

# The kit, as committed at that commit (not the working tree), so the drive runs what was reviewed.
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ('longleg-kit-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
try {
    $zip = Join-Path $tmp 'kit.zip'
    Invoke-Git @('archive', '--format=zip', '-o', $zip, $commit, 'tools/longleg_kit') | Out-Null
    Expand-Archive -LiteralPath $zip -DestinationPath (Join-Path $tmp 'x') -Force
    $kit = Join-Path $longLeg 'kit'
    if (Test-Path -LiteralPath $kit) { Remove-Item -LiteralPath $kit -Recurse -Force }
    Copy-Item -LiteralPath (Join-Path $tmp 'x\tools\longleg_kit') -Destination $kit -Recurse
} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

if (-not $KitOnly) { Write-Utf8File (Join-Path $longLeg 'LONGLEG_DRIVE.txt') ("SuperSLM 1.14.0 long-leg drive`ncommit=$commit`n") }
$start = "@echo off`r`n" +
    "rem Runs the SuperSLM long legs from this drive. Safe to run again at any time; finished pieces are skipped.`r`n" +
    "powershell -NoProfile -ExecutionPolicy Bypass -File `"%~d0\longleg\kit\longleg.ps1`" %*`r`n" +
    "pause`r`n"
[System.IO.File]::WriteAllText((Join-Path $root 'START.cmd'), $start, (New-Object System.Text.ASCIIEncoding))

$total = (Get-ChildItem -LiteralPath $longLeg -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Line ('done: {0:N1} GB on {1}. Plug it into the PC and double-click {1}\START.cmd.' -f ($total / 1GB), $Drive)
