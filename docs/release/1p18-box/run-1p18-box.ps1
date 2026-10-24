# Engine 1.18.0 box sitting (delta prefix saves, plan superslm-prefix-delta-1p18 §8 S3): the C verb inventory, MSVC
# and clang-cl builds, the X2 export check, the E6 cells, and the full precache E1 target (E2/E3/E6, D12 included)
# under the E4 box environment. The run sheet is README.md beside this file.
# Resumable: each step writes <Scratch>\out\<step>.done and is skipped when that file exists; delete it to re-run.
# Run under Windows PowerShell 5.1 from any directory:
#   powershell -ExecutionPolicy Bypass -File D:\_scratch\delta-1p18\src\docs\release\1p18-box\run-1p18-box.ps1
# Steps: verbs, build-msvc, e6-msvc, x2, e1-msvc, build-clangcl, e6-clangcl (default: all, in that order).
# Exit 0 only when every step passed. It never commits, pushes or deletes anything outside <Scratch>\out.
param(
    [string[]]$Steps = @('verbs', 'build-msvc', 'e6-msvc', 'x2', 'e1-msvc', 'build-clangcl', 'e6-clangcl'),
    [string]$Scratch = 'D:\_scratch\delta-1p18',
    # The E4 box environment (MSVC 14.34 dev shell and the SUPERSLM_PRECACHE_* paths), from the 1.14 post-M1 line.
    [string]$E4Env = 'D:\_scratch\e4\src\tests\precache\e4_box_env.ps1',
    [string]$ClangCl = 'clang-cl'
)
$ErrorActionPreference = 'Stop'
$src = Join-Path $Scratch 'src'
$out = Join-Path $Scratch 'out'
$status = Join-Path $out 'status.txt'
$fixtures = Join-Path $Scratch 'pkv-fixtures'
New-Item -ItemType Directory -Force -Path $out | Out-Null
function Say($t) { $line = "[$(Get-Date -Format o)] $t"; Write-Output $line; Add-Content -LiteralPath $status -Value $line }
function Run([string]$Log, [string]$Exe, [string[]]$A) {
    & $Exe @A *> $Log
    if ($LASTEXITCODE -ne 0) { throw "$Exe exited $LASTEXITCODE ($Log)" }
}
# Every E6 cell id the cloud runs (tools/ci/precache_coverage_cells.json), so the box runs the same population.
function E6Cells() {
    $j = Get-Content (Join-Path $src 'tools\ci\precache_coverage_cells.json') -Raw | ConvertFrom-Json
    return @($j.e1 | Where-Object { $_ -like '*/E6' })
}
function RunCells([string]$Exe, [string[]]$Cells, [string]$Log) {
    & $Exe @Cells *> $Log
    $code = $LASTEXITCODE
    $last = (Get-Content $Log | Select-Object -Last 1)
    Say "  $last"
    if ($code -ne 0) { throw "$Exe exited $code ($Log)" }
}

$failed = $false
try {
    if (-not (Test-Path (Join-Path $src '.git'))) { throw "$src is not a worktree: run piece P0 of the run sheet first" }
    if (@(git -C $src status --porcelain --untracked-files=no).Count -ne 0) { throw "$src has local changes" }
    $busy = @(Get-Process -Name 'UnrealEditor', 'UnrealEditor-Cmd', 'UnrealBuildTool', 'cl', 'link' -ErrorAction SilentlyContinue)
    if ($busy.Count -gt 0 -or (Test-Path 'D:\_ssu_build_lock')) { throw 'box not clear (an editor, a build or the build lock is present)' }
    $head = (git -C $src rev-parse HEAD).Trim()
    Say "start: tree $head, steps $($Steps -join ',')"
    if (-not (Test-Path (Join-Path $fixtures 'pkv_def.sslm'))) {
        Say 'generating the paged-KV fixtures (tools/gen_paged_kv_fixture.py)'
        Run (Join-Path $out 'fixtures.log') 'python' @((Join-Path $src 'tools\gen_paged_kv_fixture.py'), $fixtures)
    }

    foreach ($step in $Steps) {
        if (Test-Path (Join-Path $out "$step.done")) { Say "$step already done: skipped"; continue }
        Say "$step start"
        $bdMsvc = Join-Path $Scratch 'build-msvc'
        $bdClang = Join-Path $Scratch 'build-clangcl'
        if ($step -eq 'verbs') {
            # Regenerate the C verb list into out\ (never over the committed file); the box session commits it.
            $verbs = Join-Path $out 'c_abi_verbs.txt'
            $consumed = Join-Path $out 'consumed_symbols.txt'
            Run (Join-Path $out 'verbs.log') 'powershell' @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                (Join-Path $src 'tests\t2807-api-slot\gen_consumed_symbols.ps1'), '-Engine', $src,
                '-Recorded', (Join-Path $src 'tests\t2807-api-slot\recorded'), '-Out', $consumed, '-CVerbsOut', $verbs)
            $new = @(Get-Content $verbs | Where-Object { $_ -and $_ -notmatch '^#' })
            $old = @(Get-Content (Join-Path $src 'tests\t2807-api-slot\c_abi_verbs.txt') | Where-Object { $_ -and $_ -notmatch '^#' })
            $added = @($new | Where-Object { $old -notcontains $_ }); $gone = @($old | Where-Object { $new -notcontains $_ })
            Say "verbs: $($new.Count) (was $($old.Count)); added $($added -join ', '); removed $($gone -join ', ')"
            if (($added -join ',') -ne 'sslm_prefix_assemble,sslm_prefix_save_delta' -or $gone.Count -ne 0) {
                throw 'STOP: the verb change is not exactly the two 1.18.0 verbs'
            }
            $line = [array]::IndexOf($new, 'sslm_seq_jump_forward') + 1
            Say "verbs: sslm_seq_jump_forward is at line $line of the list's names (cell 9.X2 expects 48 after this change)"
        } elseif ($step -like 'build-*') {
            . $E4Env | Out-Null
            $bd = if ($step -eq 'build-msvc') { $bdMsvc } else { $bdClang }
            $cc = if ($step -eq 'build-msvc') { 'cl' } else { $ClangCl }
            if (-not (Test-Path (Join-Path $bd 'build.ninja'))) {
                Run (Join-Path $out "$step-configure.log") 'cmake' @('-S', $src, '-B', $bd, '-G', 'Ninja',
                    '-DBUILD_TESTING=ON', '-DCMAKE_BUILD_TYPE=Release', "-DCMAKE_C_COMPILER=$cc", "-DCMAKE_CXX_COMPILER=$cc")
            }
            Run (Join-Path $out "$step.log") 'cmake' @('--build', $bd, '--target', 'superslm', 'superslm_precache_e1',
                'superslm_precache_unpinned', 'superslm_tests')
        } elseif ($step -like 'e6-*') {
            $bd = if ($step -eq 'e6-msvc') { $bdMsvc } else { $bdClang }
            $env:SUPERSLM_PAGED_KV_FIXTURE_DIR = $fixtures
            $env:SUPERSLM_PRECACHE_DELTA_OUT = Join-Path $out "$step-d15"
            New-Item -ItemType Directory -Force -Path $env:SUPERSLM_PRECACHE_DELTA_OUT | Out-Null
            RunCells (Join-Path $bd 'superslm_precache_e1.exe') (E6Cells) (Join-Path $out "$step.log")
            Run (Join-Path $out "$step-oracle.log") 'python' @((Join-Path $src 'tests\precache\check_spd1.py'),
                $env:SUPERSLM_PRECACHE_DELTA_OUT, '--self-test')
            Say "  $(Get-Content (Join-Path $out "$step-oracle.log") | Select-Object -Last 1)"
        } elseif ($step -eq 'x2') {
            . $E4Env | Out-Null
            # X2 exports against the regenerated verb list from step 'verbs'.
            Run (Join-Path $out 'x2.log') 'powershell' @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                (Join-Path $src 'tests\t2807-api-slot\build_x2.ps1'), '-Engine', $src, '-Out', (Join-Path $out 'x2'),
                '-CVerbs', (Join-Path $out 'c_abi_verbs.txt'), '-List', (Join-Path $out 'consumed_symbols.txt'))
        } elseif ($step -eq 'e1-msvc') {
            # The whole E1 target (E2, E3, E6 and D12) under the E4 environment, which points at the box's synthetic
            # models, prompts and references; the E6 cells read the generated fixtures through the same variable.
            . $E4Env | Out-Null
            Run (Join-Path $out 'e1-list.log') (Join-Path $bdMsvc 'superslm_precache_e1.exe') @('--list')
            RunCells (Join-Path $bdMsvc 'superslm_precache_e1.exe') @() (Join-Path $out 'e1-msvc.log')
        } else {
            throw "unknown step $step"
        }
        New-Item -ItemType File -Force -Path (Join-Path $out "$step.done") | Out-Null
        Say "$step done"
    }
} catch {
    $failed = $true
    Say "FAILED: $($_.Exception.Message)"
}
if ($failed) { exit 1 }
Say 'all steps passed'
exit 0
