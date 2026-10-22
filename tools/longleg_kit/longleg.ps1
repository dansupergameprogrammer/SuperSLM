<#
.SYNOPSIS
Runs SuperSLM 1.14.0's long CPU legs from a plug-in drive filled by fill_drive.ps1 (plan section 9: the 4.1 grid, 4.2/E3,
and the unpinned synthetic 1.5B legs of 4.7 and 9.8), each as its own resumable piece.

.DESCRIPTION
Run it from the drive, in Windows PowerShell 5.1 or PowerShell 7:

    powershell -NoProfile -ExecutionPolicy Bypass -File F:\longleg\kit\longleg.ps1

(any drive letter; START.cmd at the drive's root does the same). It:
 1. checks the prerequisites and prints exactly what is missing, with the line that installs it;
 2. clones the source from the drive's bundle (a full history, never shallow), checks out the commit fill_drive.ps1
    recorded, and builds superslm_precache_e1 and superslm_precache_unpinned once per toolchain;
 3. runs every piece in its own process. A finished piece writes results\<piece>.done and is skipped on any rerun.
    The 4.1 and 4.2 grids also keep a per-case checkpoint journal, so an interrupted grid resumes at its next case;
 4. adds a resume line to the user's Startup folder, so after a crash or reboot the next logon carries on, and keeps
    the PC awake while it runs. The line is removed when every piece is done;
 5. rewrites results\SUMMARY.md after every piece. That one file goes back for the record.

Pieces never rerun once done. To repeat one, delete its results\<piece>.done.

.PARAMETER Resume
Unattended run (the Startup line passes it): never prompts, never installs.

.PARAMETER CheckOnly
Check the prerequisites and inputs, print what is missing, and stop.

.PARAMETER Install
Install missing prerequisites without asking (winget; Visual Studio asks for elevation).

.PARAMETER WorkRoot
Where the clone, the builds and the checkpoint journals go. Default: C:\_scratch\longleg when C: has 30 GB free,
otherwise <drive>\longleg\work. Fixed on the first run; later runs reuse it.

.PARAMETER Toolchains
Any of msvc, clangcl, msvc-asan. Default: all three, msvc first.

.PARAMETER Only
Run only the pieces whose id starts with one of these strings (for example msvc__4.2).

.PARAMETER Parallel
Pieces at once. Each piece is one single-threaded process. Default 4 (the i5-14400F has six performance cores).

.PARAMETER SelfTest
Exercise the piece scheduler, resume and summary with stand-in pieces in a temporary folder (any OS). Builds nothing.
#>
[CmdletBinding()]
param(
    [switch]$Resume,
    [switch]$CheckOnly,
    [switch]$Install,
    [string]$WorkRoot = '',
    [string[]]$Toolchains = @('msvc', 'clangcl', 'msvc-asan'),
    [string[]]$Only = @(),
    [ValidateRange(1, 16)][int]$Parallel = 4,
    [switch]$SelfTest
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------------------------------------------
# What runs. Hours are wall-clock per piece on one core, from the box and cloud records (merge-path-2026-10-20 section 7,
# t3051's 4.1 grid at 6.6 h on the 3950X); msvc-asan is taken as 2.5 times as long. They order the queue (longest
# first) and size the estimate; nothing depends on them being exact.
# ---------------------------------------------------------------------------------------------------------------
$script:CellTable = @(
    @{ Cell = '4.1/E2:synthetic-shapes'; Binary = 'e1';       Checkpointed = $true;  Hours = 6.6;  Toolchains = @('msvc') },
    @{ Cell = '4.2/E3';                  Binary = 'e1';       Checkpointed = $true;  Hours = 7.0;  Toolchains = @('msvc', 'clangcl', 'msvc-asan') },
    @{ Cell = '4.7/E2:synthetic1p5';     Binary = 'unpinned'; Checkpointed = $false; Hours = 2.3;  Toolchains = @('msvc', 'clangcl', 'msvc-asan') },
    @{ Cell = '9.8/E2:synthetic1p5';     Binary = 'unpinned'; Checkpointed = $false; Hours = 2.0;  Toolchains = @('msvc', 'clangcl', 'msvc-asan') },
    @{ Cell = '4.7/E3:synthetic1p5';     Binary = 'unpinned'; Checkpointed = $false; Hours = 1.0;  Toolchains = @('msvc', 'clangcl', 'msvc-asan') },
    @{ Cell = '9.8/E3:synthetic1p5';     Binary = 'unpinned'; Checkpointed = $false; Hours = 0.75; Toolchains = @('msvc', 'clangcl', 'msvc-asan') }
)
$script:ToolchainOrder = @('msvc', 'clangcl', 'msvc-asan')
$script:HourScale = @{ 'msvc' = 1.0; 'clangcl' = 1.0; 'msvc-asan' = 2.5 }
$script:NotRun = @(
    @{ What = 'TSan 4.7/E2:pkv_32k and 9.8/E2:pkv_32k'
       Why  = 'Linux only. ThreadSanitizer has no Windows runtime for MSVC or clang-cl. Each cell needs more than 2 hours under TSan; run them on Linux (GCC 13, -fsanitize=thread) with no time limit.' },
    @{ What = 'E4: 4.3, 6.6/E4, 6.6/E4:rule-b, 10.1-10.3/E4, real_two_process'
       Why  = 'No GPU needed, but not ready: the E4 re-derivation to the rule B (10.1, 10.2) is not written, the real-artifact references at the post-M1 anchor are not regenerated or licensed (box B3), the timing instrument (10.3) is not commissioned, and 4.3 needs the three compilers'' blobs. The drive carries the real 0.5B and 1.5B models for when they are.' }
)
$script:MaxCrashes = 2            # a piece that crashes twice is recorded CRASHED and the queue moves on
$script:PollSeconds = 20
$script:Msvc = '14.34'            # the box's MSVC 14.34.31933 (vcvarsall -vcvars_ver=14.34)
$script:Sdk = '10.0.22621.0'
$script:LlvmBin = 'C:\Program Files\LLVM\bin'

# ---------------------------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------------------------
function Write-Line([string]$Text) { Write-Host ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $Text) }

function Write-Utf8File([string]$Path, [string]$Text) {
    # No BOM, LF kept as written; the same bytes on PowerShell 5.1 and 7.
    [System.IO.File]::WriteAllText($Path, $Text, (New-Object System.Text.UTF8Encoding($false)))
}

function Read-KeyValue([string]$Path) {
    $map = @{}
    if (Test-Path -LiteralPath $Path) {
        foreach ($line in [System.IO.File]::ReadAllLines($Path)) {
            $at = $line.IndexOf('=')
            if ($at -gt 0) { $map[$line.Substring(0, $at)] = $line.Substring($at + 1) }
        }
    }
    return $map
}

function Write-KeyValue([string]$Path, [hashtable]$Map) {
    $lines = foreach ($k in ($Map.Keys | Sort-Object)) { "$k=$($Map[$k])" }
    Write-Utf8File $Path (($lines -join "`n") + "`n")
}

function Get-Sha256([string]$Path) { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }

function Test-IsWindows {
    if (Test-Path variable:IsWindows) { return [bool]$IsWindows }
    return $true   # Windows PowerShell 5.1 has no $IsWindows and runs only on Windows
}

function Get-PieceId([string]$Toolchain, [string]$Cell) {
    return $Toolchain + '__' + ($Cell -replace '[/:]', '-')
}

function Get-Pieces([string[]]$Wanted, [string[]]$Prefixes) {
    $pieces = @()
    foreach ($row in $script:CellTable) {
        foreach ($tc in $row.Toolchains) {
            if ($Wanted -notcontains $tc) { continue }
            $id = Get-PieceId $tc $row.Cell
            if ($Prefixes.Count -gt 0) {
                $hit = $false
                foreach ($p in $Prefixes) { if ($id.StartsWith($p)) { $hit = $true } }
                if (-not $hit) { continue }
            }
            $pieces += [pscustomobject]@{
                Id = $id; Toolchain = $tc; Cell = $row.Cell; Binary = $row.Binary
                Checkpointed = $row.Checkpointed; Hours = [double]$row.Hours * $script:HourScale[$tc]
                Phase = [array]::IndexOf($script:ToolchainOrder, $tc)
            }
        }
    }
    return $pieces
}

# The queue: msvc first (the plan's own list), then clang-cl and msvc-asan together; longest first inside each
# phase, so the wall clock is set by the longest piece rather than by the order.
function Get-QueueOrder($Pieces) {
    return @($Pieces | Sort-Object -Property @{ Expression = { if ($_.Phase -eq 0) { 0 } else { 1 } } }, @{ Expression = 'Hours'; Descending = $true }, 'Id')
}

# ---------------------------------------------------------------------------------------------------------------
# Prerequisites (Windows)
# ---------------------------------------------------------------------------------------------------------------
function Update-PathFromRegistry {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = (@($machine, $user) | Where-Object { $_ }) -join ';'
}

function Find-App([string]$Name) {
    $cmd = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue |
        Where-Object { $_.Source -notlike '*\WindowsApps\*' } | Select-Object -First 1
    if ($cmd) { return $cmd.Source }
    return $null
}

function Get-VsInfo {
    $info = [ordered]@{ VsWhere = $null; Install = $null; Vcvars = $null; Msvc = $null; AsanLib = $false; Dxc = $null; SdkOk = $false }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path -LiteralPath $vswhere) {
        $info.VsWhere = $vswhere
        $install = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null |
            Select-Object -First 1
        if ($install) {
            $info.Install = $install
            $vcvars = Join-Path $install 'VC\Auxiliary\Build\vcvarsall.bat'
            if (Test-Path -LiteralPath $vcvars) { $info.Vcvars = $vcvars }
            $tools = Join-Path $install 'VC\Tools\MSVC'
            if (Test-Path -LiteralPath $tools) {
                $msvc = Get-ChildItem -LiteralPath $tools -Directory | Where-Object { $_.Name -like "$script:Msvc.*" } |
                    Sort-Object Name -Descending | Select-Object -First 1
                if ($msvc) {
                    $info.Msvc = $msvc.Name
                    $info.AsanLib = Test-Path -LiteralPath (Join-Path $msvc.FullName 'lib\x64\clang_rt.asan_dynamic-x86_64.lib')
                }
            }
        }
    }
    $kits = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10'
    $info.SdkOk = Test-Path -LiteralPath (Join-Path $kits "Include\$script:Sdk\um\windows.h")
    $dxc = Join-Path $kits "bin\$script:Sdk\x64\dxc.exe"
    if (Test-Path -LiteralPath $dxc) { $info.Dxc = $dxc }
    return $info
}

function Get-Prerequisites([string[]]$Wanted) {
    Update-PathFromRegistry
    $rows = @()
    $vsComponents = @('Microsoft.VisualStudio.Workload.VCTools', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
                      'Microsoft.VisualStudio.Component.VC.14.34.17.4.x86.x64', 'Microsoft.VisualStudio.Component.Windows11SDK.22621')
    if ($Wanted -contains 'msvc-asan') { $vsComponents += 'Microsoft.VisualStudio.Component.VC.ASAN' }

    $git = Find-App 'git'
    $rows += [pscustomobject]@{ Name = 'Git'; Ok = [bool]$git; Have = $git; Fix = 'winget install --id Git.Git -e --source winget' }

    $py = Find-App 'python'
    $pyOk = $false
    if ($py) { $pyOk = ((& $py --version 2>&1) -join ' ') -match 'Python 3\.' }
    $rows += [pscustomobject]@{ Name = 'Python 3'; Ok = $pyOk; Have = $py; Fix = 'winget install --id Python.Python.3.12 -e --source winget' }

    $cmake = Find-App 'cmake'
    $cmakeOk = $false
    if ($cmake) {
        $v = ((& $cmake --version 2>&1) | Select-Object -First 1) -replace '[^0-9.]', ''
        try { $cmakeOk = ([version]$v) -ge ([version]'3.20') } catch { $cmakeOk = $false }
    }
    $rows += [pscustomobject]@{ Name = 'CMake 3.20 or later'; Ok = $cmakeOk; Have = $cmake; Fix = 'winget install --id Kitware.CMake -e --source winget' }

    $ninja = Find-App 'ninja'
    $rows += [pscustomobject]@{ Name = 'Ninja'; Ok = [bool]$ninja; Have = $ninja; Fix = 'winget install --id Ninja-build.Ninja -e --source winget' }

    $vs = Get-VsInfo
    $addArgs = ($vsComponents | ForEach-Object { "--add $_" }) -join ' '
    if (-not $vs.Install) {
        $vsFix = 'winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --override "--passive --wait --norestart ' + $addArgs + '"'
    } else {
        $setup = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\setup.exe'
        $vsFix = '& "' + $setup + '" modify --installPath "' + $vs.Install + '" --passive --norestart ' + $addArgs
    }
    $rows += [pscustomobject]@{ Name = 'VS 2022 Build Tools (C++)'; Ok = [bool]$vs.Vcvars; Have = $vs.Install; Fix = $vsFix }
    $rows += [pscustomobject]@{ Name = "MSVC $script:Msvc toolset"; Ok = [bool]$vs.Msvc; Have = $vs.Msvc; Fix = $vsFix }
    $rows += [pscustomobject]@{ Name = "Windows SDK $script:Sdk (with dxc.exe)"; Ok = ($vs.SdkOk -and [bool]$vs.Dxc); Have = $vs.Dxc; Fix = $vsFix }
    if ($Wanted -contains 'msvc-asan') {
        $rows += [pscustomobject]@{ Name = "MSVC $script:Msvc AddressSanitizer libraries"; Ok = $vs.AsanLib; Have = $null; Fix = $vsFix }
    }
    if ($Wanted -contains 'clangcl') {
        $clang = Join-Path $script:LlvmBin 'clang-cl.exe'
        $clangOk = $false
        $have = $null
        if (Test-Path -LiteralPath $clang) {
            $have = ((& $clang --version 2>&1) | Select-Object -First 1)
            $clangOk = $have -match 'clang version 15\.'
        }
        $rows += [pscustomobject]@{ Name = 'clang-cl 15 (LLVM 15.0.1)'; Ok = $clangOk; Have = $have; Fix = 'winget install --id LLVM.LLVM -e --version 15.0.1 --source winget' }
    }
    return $rows
}

function Show-Prerequisites($Rows) {
    $missing = @($Rows | Where-Object { -not $_.Ok })
    foreach ($r in $Rows) {
        if ($r.Ok) { Write-Host ("  ok       {0}  ({1})" -f $r.Name, $r.Have) }
        else { Write-Host ("  MISSING  {0}" -f $r.Name) -ForegroundColor Yellow }
    }
    if ($missing.Count -gt 0) {
        Write-Host ''
        Write-Host 'To install what is missing, run these lines in this window (Visual Studio asks for permission):'
        foreach ($fix in ($missing | ForEach-Object { $_.Fix } | Select-Object -Unique)) { Write-Host "  $fix" -ForegroundColor Cyan }
        Write-Host ''
        Write-Host 'Visual Studio Build Tools takes 10-20 minutes. Then run the kit again; it picks up where it is.'
    }
    return $missing
}

function Install-Missing($Missing) {
    foreach ($fix in ($Missing | ForEach-Object { $_.Fix } | Select-Object -Unique)) {
        Write-Line "installing: $fix"
        Invoke-Expression $fix
    }
    Update-PathFromRegistry
}

# ---------------------------------------------------------------------------------------------------------------
# Inputs on the drive (fill_drive.ps1 writes inputs\MANIFEST.tsv: path, bytes, sha256)
# ---------------------------------------------------------------------------------------------------------------
function Test-Inputs([string]$LongLeg, [string]$Results) {
    $inputs = Join-Path $LongLeg 'inputs'
    $manifest = Join-Path $inputs 'MANIFEST.tsv'
    $problems = @()
    if (-not (Test-Path -LiteralPath $manifest)) { return @("inputs\MANIFEST.tsv is missing: fill the drive with fill_drive.ps1 on the home box") }
    $entries = @()
    foreach ($line in [System.IO.File]::ReadAllLines($manifest)) {
        if (-not $line -or $line.StartsWith('#')) { continue }
        $f = $line.Split("`t")
        $entries += [pscustomobject]@{ Path = $f[0]; Bytes = [int64]$f[1]; Sha = $f[2] }
    }
    foreach ($required in @('synthetic\synthetic0p5.sslm', 'synthetic\synthetic1p5.sslm', 'source\superslm.bundle', 'source\SOURCE.txt')) {
        if (-not ($entries | Where-Object { $_.Path -eq $required })) { $problems += "inputs\$required is not in the manifest" }
    }
    $stamp = Join-Path $Results 'inputs.verified'
    $manifestSha = Get-Sha256 $manifest
    $verified = (Test-Path -LiteralPath $stamp) -and (([System.IO.File]::ReadAllText($stamp)).Trim() -eq $manifestSha)
    foreach ($e in $entries) {
        $p = Join-Path $inputs $e.Path
        if (-not (Test-Path -LiteralPath $p)) { $problems += "inputs\$($e.Path) is missing"; continue }
        if ((Get-Item -LiteralPath $p).Length -ne $e.Bytes) { $problems += "inputs\$($e.Path) has the wrong size"; continue }
        if (-not $verified) {
            Write-Line "verifying inputs\$($e.Path)"
            if ((Get-Sha256 $p) -ne $e.Sha) { $problems += "inputs\$($e.Path) has the wrong SHA-256" }
        }
    }
    if ($problems.Count -eq 0 -and -not $verified) { Write-Utf8File $stamp ($manifestSha + "`n") }
    return $problems
}

# ---------------------------------------------------------------------------------------------------------------
# Source and builds
# ---------------------------------------------------------------------------------------------------------------
function Invoke-Logged([string]$Log, [string]$Exe, [string[]]$Arguments) {
    Add-Content -LiteralPath $Log -Value ("> " + $Exe + ' ' + ($Arguments -join ' '))
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Exe @Arguments *>> $Log
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
    return $code
}

function Initialize-Source([string]$LongLeg, [string]$Work) {
    $source = Read-KeyValue (Join-Path $LongLeg 'inputs\source\SOURCE.txt')
    $commit = $source['commit']
    if (-not $commit -or $commit -notmatch '^[0-9a-f]{40}$') { throw 'inputs\source\SOURCE.txt names no 40-hex commit' }
    $src = Join-Path $Work 'src'
    $log = Join-Path $Work 'source.log'
    $bundle = Join-Path $LongLeg 'inputs\source\superslm.bundle'
    if (-not (Test-Path -LiteralPath (Join-Path $src '.git'))) {
        New-Item -ItemType Directory -Force -Path $src | Out-Null
        if ((Invoke-Logged $log 'git' @('-C', $src, 'init', '-q')) -ne 0) { throw "git init failed; see $log" }
    }
    $have = (& git -C $src rev-parse -q --verify "$commit^{commit}" 2>$null)
    if (-not $have) {
        Write-Line 'unpacking the source from the drive''s bundle (full history)'
        if ((Invoke-Logged $log 'git' @('-C', $src, 'fetch', '-q', $bundle, '+refs/*:refs/bundle/*')) -ne 0) { throw "git fetch from the bundle failed; see $log" }
    }
    if ((Invoke-Logged $log 'git' @('-C', $src, 'checkout', '-q', '--detach', '--force', $commit)) -ne 0) { throw "git checkout $commit failed; see $log" }
    $shallow = (& git -C $src rev-parse --is-shallow-repository).Trim()
    if ($shallow -ne 'false') { throw "the clone at $src is shallow ($shallow); the kit needs a full history" }
    $head = (& git -C $src rev-parse HEAD).Trim()
    if ($head -ne $commit) { throw "HEAD is $head, not $commit" }
    return @{ Src = $src; Commit = $commit; Branch = $source['branch'] }
}

function Import-VcVars([string]$Vcvars) {
    $lines = & cmd.exe /d /c "call `"$Vcvars`" x64 $script:Sdk -vcvars_ver=$script:Msvc >nul 2>&1 && set"
    if ($LASTEXITCODE -ne 0) { throw "vcvarsall x64 $script:Sdk -vcvars_ver=$script:Msvc failed" }
    foreach ($line in $lines) {
        $at = $line.IndexOf('=')
        if ($at -gt 0) { Set-Item -Path ("env:" + $line.Substring(0, $at)) -Value $line.Substring($at + 1) }
    }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { throw 'cl.exe is not on PATH after vcvarsall' }
}

function Get-BuildDir([string]$Work, [string]$Toolchain) { return (Join-Path $Work ("build-" + $Toolchain)) }

function Get-BinaryPath([string]$Work, [string]$Toolchain, [string]$Binary) {
    return (Join-Path (Get-BuildDir $Work $Toolchain) ("superslm_precache_" + $Binary + '.exe'))
}

function Invoke-Build([string]$Work, [string]$Src, [string]$Commit, [string]$Toolchain, [string]$Results) {
    $dir = Get-BuildDir $Work $Toolchain
    $stamp = Join-Path $Results ("build-" + $Toolchain + '.done')
    $done = Read-KeyValue $stamp
    if ($done.Count -gt 0) {
        if ($done['commit'] -eq $Commit -and $done['status'] -eq 'ok') { return $done }
        if ($done['commit'] -eq $Commit -and $done['status'] -eq 'failed') { return $done }
    }
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $log = Join-Path $Results ("build-" + $Toolchain + '.log')
    Write-Utf8File $log ''
    Write-Line "building $Toolchain (about 10 minutes); log: $log"
    $common = @('-S', $Src, '-B', $dir, '-G', 'Ninja', '-DBUILD_TESTING=ON')
    switch ($Toolchain) {
        'msvc' { $cfg = @('-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl') }
        'clangcl' {
            $cc = Join-Path $script:LlvmBin 'clang-cl.exe'
            $ld = Join-Path $script:LlvmBin 'lld-link.exe'
            $cfg = @('-DCMAKE_BUILD_TYPE=Release', "-DCMAKE_C_COMPILER=$cc", "-DCMAKE_CXX_COMPILER=$cc", "-DCMAKE_LINKER=$ld")
        }
        'msvc-asan' {
            $cfg = @('-DCMAKE_BUILD_TYPE=RelWithDebInfo', '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl',
                     '-DCMAKE_C_FLAGS=/DWIN32 /D_WINDOWS /fsanitize=address',
                     '-DCMAKE_CXX_FLAGS=/DWIN32 /D_WINDOWS /EHsc /GR /fsanitize=address',
                     '-DCMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO=/debug /INCREMENTAL:NO')
        }
        default { throw "unknown toolchain $Toolchain" }
    }
    $code = Invoke-Logged $log 'cmake' ($common + $cfg)
    if ($code -eq 0) {
        $code = Invoke-Logged $log 'cmake' @('--build', $dir, '--target', 'superslm_precache_e1', 'superslm_precache_unpinned', '--parallel')
    }
    $result = @{ commit = $Commit; toolchain = $Toolchain; finished = (Get-Date).ToString('s') }
    if ($code -ne 0) {
        $result['status'] = 'failed'
        $result['detail'] = "exit $code; see build-$Toolchain.log"
        Write-Line "BUILD FAILED for $Toolchain (its pieces are skipped); see $log"
    } else {
        $result['status'] = 'ok'
        foreach ($b in @('e1', 'unpinned')) { $result["sha256_$b"] = Get-Sha256 (Get-BinaryPath $Work $Toolchain $b) }
        $result['cmake'] = ((& cmake --version) | Select-Object -First 1)
        $clLine = ((& cmd.exe /d /c 'cl 2>&1') | Select-Object -First 1)
        $result['cl'] = $clLine
        if ($Toolchain -eq 'clangcl') { $result['clang'] = ((& (Join-Path $script:LlvmBin 'clang-cl.exe') --version) | Select-Object -First 1) }
        $result['vcvars'] = "x64 $script:Sdk -vcvars_ver=$script:Msvc (VCToolsVersion $env:VCToolsVersion)"
    }
    Write-KeyValue $stamp $result
    return $result
}

# ---------------------------------------------------------------------------------------------------------------
# Pieces: start, watch, finish. A piece is done when results\<id>.done exists.
# ---------------------------------------------------------------------------------------------------------------
function Get-PieceFile([string]$Results, [string]$Id, [string]$Suffix) { return (Join-Path $Results ($Id + $Suffix)) }

function Get-CrashCount([string]$Results, [string]$Id) {
    $f = Get-PieceFile $Results $Id '.attempts'
    if (-not (Test-Path -LiteralPath $f)) { return 0 }
    return @([System.IO.File]::ReadAllLines($f) | Where-Object { $_ -like 'crash*' }).Count
}

function Get-AttemptCount([string]$Results, [string]$Id) {
    $f = Get-PieceFile $Results $Id '.attempts'
    if (-not (Test-Path -LiteralPath $f)) { return 0 }
    return @([System.IO.File]::ReadAllLines($f) | Where-Object { $_ -like 'start*' }).Count
}

function Start-Piece($Piece, [hashtable]$Launch, [string]$Results) {
    $attempt = (Get-AttemptCount $Results $Piece.Id) + 1
    $running = Get-PieceFile $Results $Piece.Id '.running'
    if (Test-Path -LiteralPath $running) {
        Add-Content -LiteralPath (Get-PieceFile $Results $Piece.Id '.attempts') -Value ("interrupted " + (Get-Date).ToString('s'))
    }
    $out = Get-PieceFile $Results $Piece.Id (".attempt$attempt.out.log")
    $err = Get-PieceFile $Results $Piece.Id (".attempt$attempt.err.log")
    foreach ($k in $Launch.Env.Keys) {
        if ($null -eq $Launch.Env[$k]) { Remove-Item -Path ("env:" + $k) -ErrorAction SilentlyContinue }
        else { Set-Item -Path ("env:" + $k) -Value $Launch.Env[$k] }
    }
    $p = Start-Process -FilePath $Launch.Exe -ArgumentList $Launch.Arguments -WorkingDirectory $Launch.Dir `
        -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -PassThru
    $null = $p.Handle   # keeps the exit code readable after the process ends
    try { $p.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal } catch { Write-Verbose 'priority unchanged' }
    Add-Content -LiteralPath (Get-PieceFile $Results $Piece.Id '.attempts') -Value ("start $attempt " + (Get-Date).ToString('s') + " pid " + $p.Id)
    Write-KeyValue $running @{ attempt = $attempt; pid = $p.Id; started = (Get-Date).ToString('s') }
    Write-Line ("started  {0}  (attempt {1}, about {2:N1} h)" -f $Piece.Id, $attempt, $Piece.Hours)
    return [pscustomobject]@{ Piece = $Piece; Process = $p; Out = $out; Err = $err; Attempt = $attempt; Started = Get-Date }
}

function Complete-Piece($Run, [string]$Results) {
    $piece = $Run.Piece
    $code = $Run.Process.ExitCode
    $text = ''
    if (Test-Path -LiteralPath $Run.Out) { $text = [System.IO.File]::ReadAllText($Run.Out) }
    $pattern = '\[E\d\] ' + [regex]::Escape($piece.Cell) + ' (green|RED) \((\d+) checks, (\d+) failed\)'
    $m = [regex]::Matches($text, $pattern)
    $verdict = $null; $checks = ''; $failures = ''
    if ($m.Count -gt 0) {
        $last = $m[$m.Count - 1]
        $checks = $last.Groups[2].Value; $failures = $last.Groups[3].Value
        if ($last.Groups[1].Value -eq 'green' -and $code -eq 0) { $verdict = 'GREEN' }
        elseif ($last.Groups[1].Value -eq 'RED' -and ($code -eq 1 -or $code -eq 0)) { $verdict = 'RED' }
    }
    Remove-Item -LiteralPath (Get-PieceFile $Results $piece.Id '.running') -ErrorAction SilentlyContinue
    $hours = ((Get-Date) - $Run.Started).TotalHours
    if (-not $verdict) {
        Add-Content -LiteralPath (Get-PieceFile $Results $piece.Id '.attempts') -Value ("crash " + (Get-Date).ToString('s') + " exit $code")
        if ((Get-CrashCount $Results $piece.Id) -lt $script:MaxCrashes) {
            Write-Line ("crashed  {0}  (exit {1}); it runs again" -f $piece.Id, $code)
            return $false
        }
        $verdict = 'CRASHED'
    }
    $fails = @()
    foreach ($f in @($Run.Err, $Run.Out)) {
        if (Test-Path -LiteralPath $f) { $fails += @([System.IO.File]::ReadAllLines($f) | Where-Object { $_ -match '^FAIL |ERROR: AddressSanitizer|^SUMMARY: AddressSanitizer' }) }
    }
    $resultLines = @("piece: $($piece.Id)", "cell: $($piece.Cell)", "verdict: $verdict", "exit: $code", "checks: $checks", "failures: $failures")
    if ($m.Count -gt 0) { $resultLines += ('line: ' + $m[$m.Count - 1].Value) }
    $resultLines += ($fails | Select-Object -First 200)
    Write-Utf8File (Get-PieceFile $Results $piece.Id '.result.txt') (($resultLines -join "`n") + "`n")
    Write-KeyValue (Get-PieceFile $Results $piece.Id '.done') @{
        verdict = $verdict; exit = $code; checks = $checks; failures = $failures; cell = $piece.Cell; toolchain = $piece.Toolchain
        attempts = (Get-AttemptCount $Results $piece.Id); last_attempt_hours = ('{0:N2}' -f $hours); finished = (Get-Date).ToString('s')
        log = (Split-Path -Leaf $Run.Out)
    }
    if ($m.Count -gt 0) { Write-Line ("finished {0}: {1} ({2} checks, {3} failed, {4:N1} h)" -f $piece.Id, $verdict, $checks, $failures, $hours) }
    else { Write-Line ("finished {0}: {1} (exit {2}, no verdict line, {3:N1} h)" -f $piece.Id, $verdict, $code, $hours) }
    return $true
}

function Invoke-Queue($Pieces, [scriptblock]$LaunchFor, [string]$Results, [int]$Slots, [scriptblock]$OnChange) {
    $queue = New-Object System.Collections.ArrayList
    foreach ($p in (Get-QueueOrder $Pieces)) {
        if (-not (Test-Path -LiteralPath (Get-PieceFile $Results $p.Id '.done'))) { [void]$queue.Add($p) }
    }
    $running = New-Object System.Collections.ArrayList
    $lastStatus = Get-Date
    while ($queue.Count -gt 0 -or $running.Count -gt 0) {
        while ($running.Count -lt $Slots -and $queue.Count -gt 0) {
            $next = $queue[0]
            $queue.RemoveAt(0)
            [void]$running.Add((Start-Piece $next (& $LaunchFor $next) $Results))
        }
        Start-Sleep -Seconds $script:PollSeconds
        foreach ($run in @($running)) {
            if ($run.Process.HasExited) {
                $run.Process.WaitForExit()
                $running.Remove($run)
                if (Complete-Piece $run $Results) { & $OnChange }
                else { [void]$queue.Insert(0, $run.Piece) }
            }
        }
        if (((Get-Date) - $lastStatus).TotalMinutes -ge 10) {
            $lastStatus = Get-Date
            Write-Line ("running: {0}; waiting: {1}" -f (($running | ForEach-Object { $_.Piece.Id }) -join ', '), $queue.Count)
        }
    }
}

# ---------------------------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------------------------
function Get-MachineLine {
    if (-not (Test-IsWindows)) { return "self-test on $([System.Environment]::OSVersion.VersionString)" }
    try {
        $cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name.Trim()
        $ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
        $os = Get-CimInstance Win32_OperatingSystem
        return "$cpu, $ram GB, $($os.Caption) build $($os.BuildNumber)"
    } catch { return 'machine details unavailable' }
}

function Write-Summary([string]$LongLeg, [string]$Results, $Pieces, [hashtable]$Source, [string]$Work) {
    $L = New-Object System.Collections.Generic.List[string]
    $L.Add('# SuperSLM 1.14.0 long CPU legs: drive kit summary')
    $L.Add('')
    $L.Add("Written $((Get-Date).ToString('yyyy-MM-dd HH:mm')) on $(Get-MachineLine).")
    $L.Add("Source: ``$($Source.Commit)`` ($($Source.Branch)), cloned from the drive's bundle; ``--is-shallow-repository`` = false. Work root: ``$Work``.")
    $L.Add('')
    $L.Add('## Builds')
    $L.Add('')
    $L.Add('| toolchain | status | compiler | superslm_precache_e1 SHA-256 | superslm_precache_unpinned SHA-256 |')
    $L.Add('|---|---|---|---|---|')
    foreach ($tc in $script:ToolchainOrder) {
        $b = Read-KeyValue (Join-Path $Results ("build-" + $tc + '.done'))
        if ($b.Count -eq 0) { continue }
        $compiler = $b['cl']
        if ($b.ContainsKey('clang')) { $compiler = $b['clang'] + '; ' + $b['cl'] }
        $L.Add("| $tc | $($b['status']) | $compiler | $($b['sha256_e1']) | $($b['sha256_unpinned']) |")
    }
    $L.Add('')
    $L.Add('## Pieces')
    $L.Add('')
    $L.Add('| piece | cell | toolchain | verdict | checks | failed | hours (last attempt) | attempts | finished |')
    $L.Add('|---|---|---|---|---|---|---|---|---|')
    $open = 0
    foreach ($p in (Get-QueueOrder $Pieces)) {
        $d = Read-KeyValue (Get-PieceFile $Results $p.Id '.done')
        if ($d.Count -gt 0) {
            $L.Add("| $($p.Id) | $($p.Cell) | $($p.Toolchain) | **$($d['verdict'])** | $($d['checks']) | $($d['failures']) | $($d['last_attempt_hours']) | $($d['attempts']) | $($d['finished']) |")
        } else {
            $open++
            $state = 'not started'
            if (Test-Path -LiteralPath (Get-PieceFile $Results $p.Id '.running')) { $state = 'running' }
            $b = Read-KeyValue (Join-Path $Results ("build-" + $p.Toolchain + '.done'))
            if ($b.Count -gt 0 -and $b['status'] -eq 'failed') { $state = 'build failed' }
            $L.Add("| $($p.Id) | $($p.Cell) | $($p.Toolchain) | $state |  |  |  | $(Get-AttemptCount $Results $p.Id) |  |")
        }
    }
    $L.Add('')
    $reds = @($Pieces | Where-Object { $d = Read-KeyValue (Get-PieceFile $Results $_.Id '.done'); $d.Count -gt 0 -and $d['verdict'] -ne 'GREEN' })
    if ($reds.Count -gt 0) {
        $L.Add('## Failures (first lines; the full logs are in results\)')
        $L.Add('')
        foreach ($p in $reds) {
            $L.Add("### $($p.Id)")
            $L.Add('')
            $L.Add('```')
            foreach ($line in ([System.IO.File]::ReadAllLines((Get-PieceFile $Results $p.Id '.result.txt')) | Select-Object -First 40)) { $L.Add($line) }
            $L.Add('```')
            $L.Add('')
        }
    }
    $L.Add('## Not run by this kit')
    $L.Add('')
    foreach ($n in $script:NotRun) { $L.Add("- **$($n.What)**: $($n.Why)") }
    $L.Add('')
    if ($open -eq 0) { $L.Add('**All pieces are done.** Bring this file back.') }
    else { $L.Add("**$open piece(s) not done yet.**") }
    Write-Utf8File (Join-Path $Results 'SUMMARY.md') (($L -join "`n") + "`n")
    return $open
}

# ---------------------------------------------------------------------------------------------------------------
# Windows conveniences: stay awake, resume at logon, one instance at a time
# ---------------------------------------------------------------------------------------------------------------
function Set-StayAwake {
    if (-not (Test-IsWindows)) { return }
    if (-not ('LongLeg.Power' -as [type])) {
        Add-Type -Namespace LongLeg -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
    }
    [void][LongLeg.Power]::SetThreadExecutionState([uint32]'0x80000001')   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
}

function Get-ResumeHookPath { return (Join-Path ([Environment]::GetFolderPath('Startup')) 'superslm-longleg-resume.cmd') }

function Install-ResumeHook {
    if (-not (Test-IsWindows)) { return }
    $cmd = @(
        '@echo off',
        'rem SuperSLM long legs: resumes the kit after a restart. The kit deletes this file when every piece is done.',
        'for %%D in (D E F G H I J K L M N O P Q R S T U V W X Y Z) do (',
        '  if exist "%%D:\longleg\LONGLEG_DRIVE.txt" (',
        '    start "SuperSLM long legs" powershell -NoProfile -ExecutionPolicy Bypass -File "%%D:\longleg\kit\longleg.ps1" -Resume',
        '    goto :eof',
        '  )',
        ')'
    ) -join "`r`n"
    Write-Utf8File (Get-ResumeHookPath) ($cmd + "`r`n")
}

function Remove-ResumeHook {
    if (-not (Test-IsWindows)) { return }
    Remove-Item -LiteralPath (Get-ResumeHookPath) -ErrorAction SilentlyContinue
}

function Enter-Lock([string]$Results) {
    $lock = Join-Path $Results 'kit.lock'
    if (Test-Path -LiteralPath $lock) {
        $other = Read-KeyValue $lock
        if ($other['pid']) {
            $proc = Get-Process -Id ([int]$other['pid']) -ErrorAction SilentlyContinue
            if ($proc -and $proc.StartTime.ToString('s') -eq $other['started']) {
                throw "the kit is already running (process $($other['pid'])); close that window first"
            }
        }
    }
    $me = Get-Process -Id $PID
    Write-KeyValue $lock @{ pid = $PID; started = $me.StartTime.ToString('s') }
}

# ---------------------------------------------------------------------------------------------------------------
# Self-test: the scheduler, resume and summary with stand-in pieces (any OS, nothing built)
# ---------------------------------------------------------------------------------------------------------------
function Invoke-SelfTest {
    $root = Join-Path ([System.IO.Path]::GetTempPath()) ("longleg-selftest-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
    $results = Join-Path $root 'results'
    New-Item -ItemType Directory -Force -Path $results | Out-Null
    $script:PollSeconds = 1
    $shell = (Get-Process -Id $PID).Path
    $fake = Join-Path $root 'fake_cell.ps1'
    Write-Utf8File $fake @'
param([string]$Arg, [string]$Mode, [string]$Flag)
$cell = $Arg.TrimStart('=')
Start-Sleep -Milliseconds 300
switch ($Mode) {
  'green' { Write-Output "[E1] $cell ..."; Write-Output "[E1] $cell green (12 checks, 0 failed)"; exit 0 }
  'red'   { [Console]::Error.WriteLine("FAIL [$cell] x.cpp:1: stand-in failure"); Write-Output "[E1] $cell RED (12 checks, 1 failed)"; exit 1 }
  'crash-once' { if (-not (Test-Path $Flag)) { Set-Content $Flag 'x'; exit 3 }; Write-Output "[E1] $cell green (7 checks, 0 failed)"; exit 0 }
  'crash' { exit 5 }
}
'@
    $script:CellTable = @(
        @{ Cell = 'T.1/E2:green'; Binary = 'green'; Checkpointed = $true; Hours = 3.0; Toolchains = @('msvc', 'clangcl') },
        @{ Cell = 'T.2/E3:red'; Binary = 'red'; Checkpointed = $false; Hours = 2.0; Toolchains = @('msvc') },
        @{ Cell = 'T.3/E2:crash-once'; Binary = 'crash-once'; Checkpointed = $false; Hours = 1.0; Toolchains = @('msvc') },
        @{ Cell = 'T.4/E2:crash'; Binary = 'crash'; Checkpointed = $false; Hours = 0.5; Toolchains = @('msvc-asan') }
    )
    $pieces = Get-Pieces @('msvc', 'clangcl', 'msvc-asan') @()
    $launch = {
        param($p)
        @{ Exe = $shell; Dir = $root; Env = @{ LONGLEG_SELFTEST = $p.Id }
           Arguments = @('-NoProfile', '-File', $fake, ('=' + $p.Cell), $p.Binary, (Join-Path $root ($p.Id + '.flag'))) }
    }.GetNewClosure()
    $src = @{ Commit = ('0' * 40); Branch = 'self-test' }
    $fails = New-Object System.Collections.Generic.List[string]
    function Assert([bool]$Cond, [string]$What) { if ($Cond) { Write-Host "ok   $What" } else { Write-Host "FAIL $What"; $fails.Add($What) } }

    $order = @(Get-QueueOrder $pieces | ForEach-Object { $_.Id })
    Assert ($order[0] -eq 'msvc__T.1-E2-green' -and $order[-1] -eq 'msvc-asan__T.4-E2-crash') "queue: msvc first, longest first ($($order -join ', '))"

    # A piece left running by an interrupted run (no .done) must start again and record the interruption.
    Write-KeyValue (Get-PieceFile $results 'msvc__T.2-E3-red' '.running') @{ attempt = 1; pid = 1 }
    Add-Content -LiteralPath (Get-PieceFile $results 'msvc__T.2-E3-red' '.attempts') -Value 'start 1 earlier pid 1'

    Invoke-Queue $pieces $launch $results 2 { [void](Write-Summary $root $results $pieces $src $root) }
    $open = Write-Summary $root $results $pieces $src $root
    $v = @{}
    foreach ($p in $pieces) { $v[$p.Id] = (Read-KeyValue (Get-PieceFile $results $p.Id '.done'))['verdict'] }
    Assert ($v['msvc__T.1-E2-green'] -eq 'GREEN') 'green piece recorded GREEN'
    Assert ($v['clangcl__T.1-E2-green'] -eq 'GREEN') 'second toolchain of a cell is its own piece'
    Assert ($v['msvc__T.2-E3-red'] -eq 'RED') 'red piece recorded RED'
    Assert ($v['msvc__T.3-E2-crash-once'] -eq 'GREEN') 'a crash reruns the piece, which then finishes'
    Assert ($v['msvc-asan__T.4-E2-crash'] -eq 'CRASHED') "a piece that crashes $script:MaxCrashes times is recorded CRASHED"
    Assert ((Get-AttemptCount $results 'msvc__T.2-E3-red') -eq 2) 'the interrupted attempt is counted and the piece ran again'
    Assert (([System.IO.File]::ReadAllText((Get-PieceFile $results 'msvc__T.2-E3-red' '.attempts'))) -match 'interrupted') 'the interruption is recorded'
    Assert ($open -eq 0) 'summary: nothing open'
    $summary = [System.IO.File]::ReadAllText((Join-Path $results 'SUMMARY.md'))
    Assert ($summary -match 'All pieces are done' -and $summary -match 'stand-in failure' -and $summary -match 'Linux only') 'summary carries verdicts, failure lines and the not-run list'

    # A rerun starts nothing: every piece is done.
    $before = @(Get-ChildItem -LiteralPath $results -Filter '*.out.log').Count
    Invoke-Queue $pieces $launch $results 2 { }
    Assert (@(Get-ChildItem -LiteralPath $results -Filter '*.out.log').Count -eq $before) 'a rerun skips every done piece'

    # Deleting a .done reruns exactly that piece.
    Remove-Item -LiteralPath (Get-PieceFile $results 'clangcl__T.1-E2-green' '.done')
    Invoke-Queue $pieces $launch $results 2 { }
    Assert (@(Get-ChildItem -LiteralPath $results -Filter '*.out.log').Count -eq $before + 1) 'deleting one .done reruns that piece only'

    Assert ((Get-Pieces @('msvc') @('msvc__T.2')).Count -eq 1) '-Only selects by id prefix'
    Remove-Item -LiteralPath $root -Recurse -Force
    if ($fails.Count -gt 0) { Write-Host "self-test: $($fails.Count) failure(s)"; return 1 }
    Write-Host 'self-test: 0 failures'
    return 0
}

# ---------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------
if ($SelfTest) { exit (Invoke-SelfTest) }

$bad = @($Toolchains | Where-Object { $script:ToolchainOrder -notcontains $_ })
if ($bad.Count -gt 0) { throw "unknown toolchain(s): $($bad -join ', ') (use msvc, clangcl, msvc-asan)" }
if (-not (Test-IsWindows)) { throw 'the kit runs on Windows; use -SelfTest elsewhere' }

$kitDir = $PSScriptRoot
$longLeg = Split-Path -Parent $kitDir
$results = Join-Path $longLeg 'results'
New-Item -ItemType Directory -Force -Path $results | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $longLeg 'LONGLEG_DRIVE.txt'))) {
    throw "$longLeg is not a filled long-leg drive folder (LONGLEG_DRIVE.txt is missing); run fill_drive.ps1 on the home box"
}
$transcript = Join-Path $results ('kit-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
Start-Transcript -LiteralPath $transcript | Out-Null
try {
    Write-Line "SuperSLM long legs; drive folder $longLeg"
    Write-Line 'checking prerequisites'
    $missing = @(Show-Prerequisites (Get-Prerequisites $Toolchains))
    if ($missing.Count -gt 0) {
        if ($Resume -or $CheckOnly) { exit 2 }
        $go = $Install
        if (-not $go) { $go = (Read-Host 'Install them now? [Y/n]') -notmatch '^[nN]' }
        if (-not $go) { exit 2 }
        Install-Missing $missing
        $missing = @(Show-Prerequisites (Get-Prerequisites $Toolchains))
        if ($missing.Count -gt 0) { Write-Line 'still missing after the install; open a new window and run the kit again'; exit 2 }
    }
    Write-Line 'checking the drive''s inputs'
    $problems = @(Test-Inputs $longLeg $results)
    if ($problems.Count -gt 0) { $problems | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }; exit 3 }
    if ($CheckOnly) { Write-Line 'prerequisites and inputs are in place'; exit 0 }

    # Work root: fixed on the first run (state.txt); a drive-relative root follows the drive's letter.
    $state = Read-KeyValue (Join-Path $longLeg 'state.txt')
    if (-not $state['work']) {
        if ($WorkRoot) { $state['work'] = $WorkRoot }
        else {
            $c = Get-PSDrive -Name C -ErrorAction SilentlyContinue
            if ($c -and $c.Free -ge 30GB) { $state['work'] = 'C:\_scratch\longleg' } else { $state['work'] = '<drive>\work' }
        }
        Write-KeyValue (Join-Path $longLeg 'state.txt') $state
    }
    $work = $state['work'].Replace('<drive>', $longLeg)
    New-Item -ItemType Directory -Force -Path $work | Out-Null
    Write-Line "work root $work"

    Enter-Lock $results
    Set-StayAwake
    Install-ResumeHook

    $source = Initialize-Source $longLeg $work
    Write-Line "source $($source.Commit) ($($source.Branch))"
    $vs = Get-VsInfo
    Import-VcVars $vs.Vcvars

    $pieces = @(Get-Pieces $Toolchains $Only)
    $built = @{}
    foreach ($tc in ($script:ToolchainOrder | Where-Object { $Toolchains -contains $_ })) {
        if (-not ($pieces | Where-Object { $_.Toolchain -eq $tc })) { continue }
        $b = Invoke-Build $work $source.Src $source.Commit $tc $results
        $built[$tc] = ($b['status'] -eq 'ok')
    }
    $runnable = @($pieces | Where-Object { $built[$_.Toolchain] })
    $pendingHours = ($runnable | Where-Object { -not (Test-Path -LiteralPath (Get-PieceFile $results $_.Id '.done')) } | Measure-Object -Property Hours -Sum).Sum
    if (-not $pendingHours) { $pendingHours = 0 }
    Write-Line ("{0} piece(s); about {1:N0} CPU-hours left, {2} at a time" -f $runnable.Count, $pendingHours, $Parallel)
    [void](Write-Summary $longLeg $results $pieces $source $work)

    $inputs = Join-Path $longLeg 'inputs'
    $launchFor = {
        param($p)
        $env = @{
            SUPERSLM_PRECACHE_SYNTHETIC_DIR   = (Join-Path $inputs 'synthetic')
            SUPERSLM_PAGED_KV_FIXTURE_DIR     = (Join-Path $inputs 'pkv')
            SUPERSLM_PRECACHE_E0_ARTIFACT_DIR = (Join-Path $inputs 'e0')
            SUPERSLM_PRECACHE_CHECKPOINT_DIR  = $null
        }
        if ($p.Checkpointed) { $env['SUPERSLM_PRECACHE_CHECKPOINT_DIR'] = (Join-Path $work ('checkpoints\' + $p.Id)) }
        @{ Exe = (Get-BinaryPath $work $p.Toolchain $p.Binary); Dir = (Get-BuildDir $work $p.Toolchain)
           Arguments = @('"=' + $p.Cell + '"'); Env = $env }
    }.GetNewClosure()
    $onChange = { [void](Write-Summary $longLeg $results $pieces $source $work) }.GetNewClosure()
    Invoke-Queue $runnable $launchFor $results $Parallel $onChange

    $open = Write-Summary $longLeg $results $pieces $source $work
    if ($open -eq 0) {
        Remove-ResumeHook
        Write-Utf8File (Join-Path $results 'ALL-DONE.txt') ("All pieces done " + (Get-Date).ToString('s') + "`n")
        Write-Line "ALL DONE. Bring back $(Join-Path $results 'SUMMARY.md')"
    } else {
        Write-Line "$open piece(s) could not run (see SUMMARY.md: a failed build skips its toolchain's pieces)"
        Remove-ResumeHook
    }
} finally {
    Remove-Item -LiteralPath (Join-Path $results 'kit.lock') -ErrorAction SilentlyContinue
    Stop-Transcript | Out-Null
}
