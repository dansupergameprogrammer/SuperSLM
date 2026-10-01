# Paged-KV plan (rev 16.2) §8 and step C6: helpers the two box scripts share
# (tools/build_paged_kv_reference.ps1 and tools/run_paged_kv_c6_box.ps1). Dot-source it.
#
# The frozen reference library is built from a tag in a throwaway worktree under the scratch
# directory, never from the tree under work: the flat path through the new code is never the
# reference (§8). Every build runs in the Visual Studio developer shell
# (tests/t2130-g5-red-suite/resolve_toolchain.ps1) with CMake and Ninja, Release, CPU only.

. (Join-Path $PSScriptRoot '..\tests\t2130-g5-red-suite\resolve_toolchain.ps1')

# The real artifacts the C6 cells read, by the exact file names the cells name
# (tests/paged-kv/c6_dim10_cohort.cpp, c6_dim7_*.cpp).
$PkvArtifact05B = 'qwen2.5-0.5b-instruct-cap4096-aex.sslm'
$PkvArtifact15B = 'qwen2.5-1.5b-instruct.sslm'

function Get-PkvRepo {
    return (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}

$script:PkvDevShellEntered = $false
function Enter-PkvDevShell {
    if ($script:PkvDevShellEntered) { return }
    Enter-SuperSlmVsDevShell | Out-Host
    foreach ($tool in @('cmake', 'ninja', 'cl', 'git')) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool not on PATH in the developer shell" }
    }
    $script:PkvDevShellEntered = $true
}

# Runs a native program with its stdout and stderr streamed to files (so a crash keeps what was
# written) and returns its exit code. Arguments must not contain spaces.
function Invoke-PkvLogged {
    param([string]$Exe, [string[]]$Arguments, [string]$Log, [string]$WorkingDirectory = '')
    if (-not $WorkingDirectory) { $WorkingDirectory = Split-Path -Parent $Log }
    $start = @{
        FilePath               = $Exe
        RedirectStandardOutput = $Log
        RedirectStandardError  = "$Log.err"
        WorkingDirectory       = $WorkingDirectory
        NoNewWindow            = $true
        PassThru               = $true
    }
    if ($Arguments -and $Arguments.Count -gt 0) { $start.ArgumentList = $Arguments }
    $p = Start-Process @start
    $null = $p.Handle  # keeps ExitCode readable after WaitForExit on Windows PowerShell 5.1
    $p.WaitForExit()
    return $p.ExitCode
}

# Builds the library at $Tag in <WorkDir>\src-<Tag> (a detached worktree of this repository) into
# <WorkDir>\build-<Tag>, and returns @{ Src; Build; Lib }. Reuses both when present, after checking
# the worktree still sits exactly on the tag with no local change.
function Initialize-PkvFrozenLibrary {
    param([string]$Tag, [string]$WorkDir)
    Enter-PkvDevShell
    $repo = Get-PkvRepo
    $src = Join-Path $WorkDir "src-$Tag"
    $build = Join-Path $WorkDir "build-$Tag"
    New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
    $want = & git -C $repo rev-parse --verify --quiet "refs/tags/$Tag^{commit}"
    if ($LASTEXITCODE -ne 0 -or -not $want) { throw "tag $Tag not found in $repo (run: git -C $repo fetch --tags)" }
    if (-not (Test-Path -LiteralPath $src)) {
        & git -C $repo worktree prune | Out-Host
        & git -C $repo worktree add --detach $src $Tag | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "git worktree add $src $Tag failed" }
    }
    $have = & git -C $src rev-parse HEAD
    if ($have -ne $want) { throw "$src is at $have, not $Tag ($want): remove it and rerun" }
    $dirty = & git -C $src status --porcelain
    if ($dirty) { throw "$src has local changes: remove it and rerun" }
    if (-not (Test-Path -LiteralPath (Join-Path $build 'CMakeCache.txt'))) {
        & cmake -S $src -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DSUPERSLM_BUILD_GPU=OFF | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "configure of $Tag failed" }
    }
    & cmake --build $build --target superslm | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "build of $Tag's superslm failed" }
    $lib = Join-Path $build 'superslm.lib'
    if (-not (Test-Path -LiteralPath $lib)) { throw "missing $lib after the build" }
    return [pscustomobject]@{ Src = $src; Build = $build; Lib = $lib }
}

# Compiles $Sources against $Lib into $Out with CMake's Release flags for MSVC (/MD /O2 /DNDEBUG)
# and the project's own (/std:c++20 /fp:precise). Objects go to <Out>.obj\. $ForceInclude, when
# given, is passed as /FI.
function Build-PkvExe {
    param([string]$Out, [string[]]$Sources, [string[]]$Includes, [string]$Lib, [string]$ForceInclude = '')
    Enter-PkvDevShell
    $objDir = "$Out.obj"
    New-Item -ItemType Directory -Force -Path $objDir | Out-Null
    $clArgs = @('/nologo', '/std:c++20', '/O2', '/MD', '/DNDEBUG', '/EHsc', '/fp:precise', '/W3')
    foreach ($i in $Includes) { $clArgs += "/I$i" }
    if ($ForceInclude) { $clArgs += "/FI$ForceInclude" }
    $clArgs += $Sources
    $clArgs += $Lib
    $clArgs += "/Fe$Out"
    Push-Location $objDir
    try {
        & cl @clArgs | Out-Host
        $code = $LASTEXITCODE
    } finally { Pop-Location }
    if ($code -ne 0 -or -not (Test-Path -LiteralPath $Out)) { throw "cl failed for $Out (exit $code)" }
}

function Get-PkvSha256 {
    param([string]$Path)
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}
