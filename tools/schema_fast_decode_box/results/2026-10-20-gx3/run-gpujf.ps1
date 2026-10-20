# GPU jump-forward off/on, headless: build sslm_schema_fast_decode_gpu_jf_bench (build.bat's t2899 CPU-common sources plus
# the CPU ABI with the schema fast-decode test seams for the forced-table read, and the GPU sources), shaders beside the
# exe (compiled for GX.3 already), then one run. Run under Windows PowerShell 5.1. Status in status-gpujf.txt.
$ErrorActionPreference = 'Stop'
# The engine from 36fe4b3: src and include are unchanged from there to the branch tip, whose test seam header is ahead of
# the library (red cells) and does not compile with the seams on. The bench file comes from the results worktree.
$src = 'D:\SuperSLM\.worktrees\sfd-arm-36fe4b37'
$tool = 'D:\SuperSLM\.worktrees\sfd-results\tools\schema_fast_decode_gpu_jf_bench.cpp'
$root = 'D:\_scratch\schema-fast-decode-gx3'
$bin = "$root\bin"
$exe = "$bin\sslm_schema_fast_decode_gpu_jf_bench.exe"
$model = 'D:\_artifacts\SuperSLM\aex\qwen2.5-0.5b-instruct-cap4096-aex-pr.sslm'
$status = "$root\status-gpujf.txt"
function Say($t) { for ($i = 0; $i -lt 20; $i++) { try { Add-Content -LiteralPath $status -Value "[$(Get-Date -Format o)] $t" -ErrorAction Stop; return } catch { Start-Sleep -Milliseconds 250 } } }
New-Item -ItemType Directory -Force -Path "$bin\objjf" | Out-Null
Say "start pid $PID; source $src at $((git -C $src rev-parse HEAD).Trim()) (+ $tool)"
try {
    if (-not (Test-Path "$bin\shaders\logits_site.cso")) { throw 'shaders missing beside the exe (run gx3.ps1 first)' }
    . "$src\tests\t2130-g5-red-suite\resolve_toolchain.ps1"
    Enter-SuperSlmVsDevShell | Out-Null
    $srcs = @('src\artifact.cpp', 'src\sha256.cpp', 'src\tokenizer.cpp', 'src\model.cpp', 'src\intmath.cpp', 'src\silu_lut.cpp',
        'src\matmul.cpp', 'src\proof_manifest.cpp', 'src\trace_hook.cpp', 'src\forward\checked_chain_funnel.cpp',
        'src\forward\forward_sites.cpp', 'src\forward\parallel_split.cpp', 'src\gpu\gpu_inflight_registry.cpp',
        'src\decode_digest.cpp', 'src\kv_pages.cpp', 'src\damped_greedy_antilm.cpp', 'src\damped_greedy_topk.cpp', 'src\damped_greedy_phaseD.cpp', 'src\damped_greedy_phaseD_loop.cpp',
        'src\sslm_abi.cpp', 'src\gpu\superslm_gpu.cpp', 'src\gpu\gpu_1p0.cpp') | ForEach-Object { "$src\$_" }
    $srcs += $tool
    Push-Location $src
    try {
        & cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /MP4 /DSUPERSLM_ENABLE_SCHEMA_FAST_DECODE_TEST_SEAMS "/I$src\include" "/I$src\src" "/I$src\src\gpu" "/I$src\tests" @srcs "/Fo:$bin\objjf\\" "/Fe:$exe" /link d3d12.lib dxgi.lib dxguid.lib *> "$root\build-gpujf.log"
        $code = $LASTEXITCODE
    } finally { Pop-Location }
    if ($code -ne 0) { throw "cl exited $code (build-gpujf.log)" }
    Say "built: sha256 $((Get-FileHash $exe).Hash.ToLower())"
    Say "model sha256 $((Get-FileHash $model).Hash.ToLower())"
    $p = Start-Process -FilePath $exe -ArgumentList $model, '--pairs', '15', '--warmup', '1', '--max-new', '64', '--json', "$root\gpujf.json" `
        -RedirectStandardOutput "$root\gpujf.log" -RedirectStandardError "$root\gpujf.log.err" -NoNewWindow -PassThru
    $null = $p.Handle
    if (-not $p.WaitForExit(15 * 60000)) { $p.Kill(); throw 'bench passed 15 min; killed' }
    if ($p.ExitCode -ne 0) { throw "bench exited $($p.ExitCode) (gpujf.log.err)" }
    Say 'end PASS'
}
catch { Say "end FAIL $($_.Exception.Message)" }
