#Requires -Version 7.0
<#
.SYNOPSIS
    Builds and runs SuperSLM's C++ test suite on the SHIPPED WINDOWS MSVC-ABI
    PATH, instrumented with AddressSanitizer + UndefinedBehaviorSanitizer, via
    clang-cl (an MSVC-ABI-compatible driver -- not a second, unrelated
    toolchain: it links against the same MSVC STL/CRT headers and libraries
    build.bat's cl.exe does, so this is a sanitizer build of the real shipped
    binary's ABI, not a port to a different one).

.DESCRIPTION
    T-1593 (owed since T-1406, named again by D-SLM566 and the cd2e75a
    confirmation review's Critical 1): CI's ASan/UBSan legs (tests.yml
    linux-x64-asan) run only on Linux clang++, which never compiles the
    shipped Windows MSVC path -- so a genuine stack-use-after-return
    (D-SLM487) lived in this tree with clang applying NRVO where MSVC did
    not, invisible to the Linux instrument for two independent reasons
    (NRVO divergence, and detect_stack_use_after_return being opt-in and
    unconfirmed). This script is the missing instrument: an MSVC-ABI
    sanitizer build, runnable locally on demand (the project's CI spending
    limit rules out adding a Windows sanitizer leg to Actions).

    Compiles tests/test_main.cpp and the production sources UNMODIFIED
    (nothing about the shipped suite is patched to make this script succeed)
    with clang-cl, -fsanitize=address,undefined -fno-sanitize-recover=all,
    /MD (dynamic CRT -- required for ASan's Windows allocator interception;
    a static-CRT build fails to intercept malloc/free/HeapAlloc and is not
    usable), copies the matching ASan runtime DLL next to the output
    executable (clang-cl's dynamic-CRT ASan build depends on it at load
    time and will not find it via the LLVM install directory alone), and
    runs the suite with detect_stack_use_after_return=1 set -- the exact
    flag D-SLM487's diagnosis names as "opt-in and unconfirmed".

    KNOWN LIMITATION, established by direct diagnosis rather than assumed
    (T-1593 build log, D:\Wizard\Claude\Brunel\): on this exact toolchain
    (LLVM/clang-cl 18.1.8) and this exact Windows build, ASan's Windows
    function-interception layer fails to hook one SSE-dispatched ucrtbase.dll
    routine ("interception_win: unhandled instruction" at a fixed address,
    reproduced by a 12-line C++ program containing nothing but a
    throw/catch). Any C++ exception thrown and caught while ASan is active
    crashes the process with STATUS_BREAKPOINT, with no ASan report --
    confirmed independent of SuperSLM's code. The shipped suite contains
    twenty-two cells (the S-HARDEN-7 "bad_alloc contract" family,
    tests/test_main.cpp, `TestBadAllocContract*` and
    `TestBadAllocContractOpenFromMemoryPassthroughClauseIsSpecific`) that
    deliberately throw and catch an exception to exercise that contract, and
    the suite's call list in main() is unconditional -- so a plain run of
    this script crashes partway through the full 454-function suite (measured
    at the first such cell, ~277 of 454 functions completed cleanly first)
    rather than completing to the summary line. This script does not paper
    over that: it runs the REAL, unmodified suite and reports whatever
    happens, including a crash, as the honest result. See the build log for
    the full diagnosis (a symbolized backtrace localizing the crash site, and
    the independent minimal repro) and for a second, diagnostic-only build
    that excludes exactly those twenty-two cells to obtain a real pass/fail
    count over the remaining ~432 (23,262 checks, 0 failures, golden hashes
    unchanged from the non-sanitized baseline) -- that diagnostic variant is
    NOT this script and is not meant to be run routinely; it exists only to
    show the instrument works cleanly everywhere outside its one named blind
    spot.

    UBSan does not link standalone on this toolchain/OS (its
    clang_rt.ubsan_standalone runtime is missing several Windows-only
    compiler-rt symbols -- __coe_win::ContinueOnError, RawWrite -- that
    are only satisfied when linked alongside the ASan runtime), so this
    script always builds the two combined, matching the flags CI's
    linux-x64-asan job already uses (tests.yml).

.PARAMETER RepoRoot
    The SuperSLM checkout to run against. Defaults to the directory this
    script lives in.

.PARAMETER ClangCl
    Path to clang-cl.exe. Defaults to $env:SUPERSLM_CLANGCL if set, then
    clang-cl on PATH, then LLVM's own Windows installer default location
    (C:\Program Files\LLVM\bin\clang-cl.exe).

.PARAMETER SkipRun
    Build only; do not execute the resulting binary.

.EXAMPLE
    pwsh -File build_asan.ps1
    Builds out\superslm_tests_asan.exe and runs it with
    ASAN_OPTIONS=detect_stack_use_after_return=1. On this toolchain/OS the
    run is expected to crash partway through the suite (see DESCRIPTION);
    that crash is itself the honest, reportable result of this instrument,
    not a script bug.
#>

[CmdletBinding()]
param(
    [string]$RepoRoot = $PSScriptRoot,
    [string]$ClangCl = '',
    [switch]$SkipRun
)

$ErrorActionPreference = 'Stop'

function Find-ClangCl {
    param([string]$Explicit)
    if ($Explicit -and (Test-Path -LiteralPath $Explicit -PathType Leaf)) { return $Explicit }
    if ($env:SUPERSLM_CLANGCL -and (Test-Path -LiteralPath $env:SUPERSLM_CLANGCL -PathType Leaf)) {
        return $env:SUPERSLM_CLANGCL
    }
    $onPath = Get-Command 'clang-cl' -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    $known = @(
        'C:\Program Files\LLVM\bin\clang-cl.exe'
        (Join-Path ${env:ProgramFiles} 'LLVM\bin\clang-cl.exe')
    )
    foreach ($p in $known) {
        if ($p -and (Test-Path -LiteralPath $p -PathType Leaf)) { return $p }
    }
    return $null
}

$chosen = Find-ClangCl -Explicit $ClangCl
if (-not $chosen) {
    Write-Error "No usable clang-cl.exe found on PATH, in `$env:SUPERSLM_CLANGCL, or in LLVM's known install location. This script builds the Windows MSVC-ABI ASan/UBSan path via clang-cl specifically (see the script header for why) -- install LLVM for Windows or pass -ClangCl explicitly."
    exit 1
}
Write-Host "=== SuperSLM MSVC-ABI ASan+UBSan instrument (T-1593) ==="
Write-Host "clang-cl              : $chosen"

# The matching ASan dynamic runtime DLL, alongside clang-cl in
# <llvm_root>\lib\clang\<major>\lib\windows\. Located relative to the chosen
# clang-cl rather than hardcoded, so a different LLVM install still resolves.
$llvmRoot = Split-Path -Parent (Split-Path -Parent $chosen)  # .../LLVM
$asanDll = Get-ChildItem -Path (Join-Path $llvmRoot 'lib\clang') -Recurse -Filter 'clang_rt.asan_dynamic-x86_64.dll' -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $asanDll) {
    Write-Error "Could not find clang_rt.asan_dynamic-x86_64.dll under $llvmRoot\lib\clang -- this build uses /MD, which requires it alongside the output executable at run time."
    exit 1
}
Write-Host "ASan runtime DLL      : $($asanDll.FullName)"

if (-not (Test-Path -LiteralPath $RepoRoot)) {
    Write-Error "RepoRoot not found: $RepoRoot"
    exit 1
}
Push-Location -LiteralPath $RepoRoot
try {
    $outDir = Join-Path $RepoRoot 'out'
    if (-not (Test-Path -LiteralPath $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
    Copy-Item -LiteralPath $asanDll.FullName -Destination $outDir -Force

    $sources = @(
        'src\artifact.cpp', 'src\sha256.cpp', 'src\tokenizer.cpp', 'src\model.cpp',
        'src\intmath.cpp', 'src\silu_lut.cpp', 'src\matmul.cpp', 'src\proof_manifest.cpp',
        'src\trace_hook.cpp', 'src\forward\checked_chain_funnel.cpp',
        'src\forward\forward_sites.cpp', 'src\decode_digest.cpp', 'tests\test_main.cpp'
    )
    $exe = Join-Path $outDir 'superslm_tests_asan.exe'
    $clArgs = @(
        '/nologo', '/std:c++20', '/O2', '/W4', '/fp:precise', '/EHsc', '/MD',
        '/Iinclude', '/Itests', '/DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION',
        # This machine's installed MSVC STL (14.44) requires Clang 19+;
        # LLVM 18.1.8 is what is actually installed. The STL's own
        # compatibility check is a conservative floor, not a proven
        # incompatibility at this pairing -- overridden deliberately, not
        # blindly (build.bat's own cl.exe path is unaffected; this override
        # is scoped to this script's clang-cl invocation only).
        '/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH',
        '-fsanitize=address,undefined', '-fno-sanitize-recover=all'
    ) + $sources + @("/Fo$outDir\", "/Fe$exe")

    Write-Host ''
    Write-Host "Building $exe ..."
    & $chosen @clArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "clang-cl build failed with exit code $LASTEXITCODE."
        exit $LASTEXITCODE
    }
    Write-Host "Build OK."

    if ($SkipRun) {
        Write-Host "SkipRun set -- not executing $exe."
        exit 0
    }

    Write-Host ''
    Write-Host "Running $exe (ASAN_OPTIONS=detect_stack_use_after_return=1) ..."
    Write-Host "Expected on this toolchain/OS: a crash partway through the suite at the"
    Write-Host "S-HARDEN-7 bad_alloc-contract cells (see this script's header) -- that is"
    Write-Host "the honest, documented result, not a failure of this script."
    Write-Host ''
    $env:ASAN_OPTIONS = 'detect_stack_use_after_return=1'
    & $exe
    $runExit = $LASTEXITCODE
    Write-Host ''
    Write-Host "Run exit code: $runExit"
    exit $runExit
} finally {
    Pop-Location
}
