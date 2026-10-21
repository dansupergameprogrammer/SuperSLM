# Builds, from the sfd-results worktree (f03a5358 plus the bench's draft sources): (1) the timed bench, a fresh Release
# build (no seams) in build-exact; (2) the D2 table builder, the bench compiled with SFD_BENCH_DRAFT_TABLE_BUILDER
# against superslm_scalar_forced_seams (a test library; it only walks streams and reads forced flags). Then builds the
# table from B4a's on-file off streams and records its sha256. Run under Windows PowerShell 5.1.
$ErrorActionPreference = 'Stop'
$wt = 'D:\SuperSLM\.worktrees\sfd-results'
$bd = "$wt\build-exact"
$out = 'D:\_scratch\decode-j1x-ab\exact'
$model = 'D:\_artifacts\SuperSLM\aex\qwen2.5-0.5b-instruct-cap4096-aex-pr.sslm'
New-Item -ItemType Directory -Force -Path $out | Out-Null
. "$wt\tests\t2130-g5-red-suite\resolve_toolchain.ps1"
Enter-SuperSlmVsDevShell | Out-Null
"source $((git -C $wt rev-parse HEAD).Trim()) + working tree: $((git -C $wt status --short) -join '; ')"
if (-not (Test-Path "$bd\CMakeCache.txt")) {
    & cmake -S $wt -B $bd -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DSUPERSLM_BUILD_GPU=OFF *> "$out\configure.log"
    if ($LASTEXITCODE -ne 0) { throw 'configure failed' }
}
& cmake --build $bd --target superslm sslm_schema_fast_decode_bench superslm_scalar_forced_seams *> "$out\build.log"
if ($LASTEXITCODE -ne 0) { throw 'build failed (build.log)' }
"timed bench sha256 $((Get-FileHash "$bd\sslm_schema_fast_decode_bench.exe").Hash.ToLower())"
$builder = "$bd\sfd_draft_table_builder.exe"
New-Item -ItemType Directory -Force -Path "$bd\builder-obj" | Out-Null
Push-Location "$bd\builder-obj"
try {
    & cl /nologo /std:c++20 /O2 /MD /DNDEBUG /EHsc /fp:precise /W3 /DSFD_BENCH_DRAFT_TABLE_BUILDER /DSUPERSLM_FORCE_SCALAR_MATMUL `
        /DSUPERSLM_ENABLE_MATVEC_TEST_SEAMS /DSUPERSLM_ENABLE_SCHEMA_FAST_DECODE_TEST_SEAMS /DSUPERSLM_CPU_G5_FINISH_ROW_FAULT_INJECTION `
        /DSUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS "/I$wt\include" "/I$wt\tests" "$wt\tools\schema_fast_decode_bench.cpp" `
        "$bd\superslm_scalar_forced_seams.lib" "/Fe$builder" *> "$out\builder-build.log"
    $code = $LASTEXITCODE
} finally { Pop-Location }
if ($code -ne 0) { throw "builder failed (builder-build.log)" }
$streams = "$wt\tools\schema_fast_decode_box\results\2026-10-20\B4a.json"
& $builder $model drafttable potion_shop_order --streams $streams --out "$out\d2-table.txt" 2>&1 | Tee-Object "$out\d2-table.log"
if ($LASTEXITCODE -ne 0) { throw 'drafttable failed' }
"d2 table sha256 $((Get-FileHash "$out\d2-table.txt").Hash.ToLower())"
"B4a.json sha256 $((Get-FileHash $streams).Hash.ToLower())"
