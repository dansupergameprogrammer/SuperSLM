# GX.3 headless GPU per-submit timing: build sslm_schema_fast_decode_gx3_bench from the sfd-results worktree (shaders
# compiled beside the exe, as build.bat's dxc loop does; the source list is build.bat's t2113_b7 line), then one run.
# Run under Windows PowerShell 5.1 (the toolchain helper's shell). Status in status-slices.txt; never throws on a held log.
$ErrorActionPreference = 'Stop'
$src = 'D:\SuperSLM\.worktrees\sfd-results'
$root = 'D:\_scratch\schema-fast-decode-gx3'
$bin = "$root\bin"
$model = 'D:\_artifacts\SuperSLM\aex\qwen2.5-0.5b-instruct-cap4096-aex-pr.sslm'
$status = "$root\status-slices.txt"
$exe = "$bin\sslm_schema_fast_decode_gx3_bench_slices.exe"
function Say($t) { for ($i = 0; $i -lt 20; $i++) { try { Add-Content -LiteralPath $status -Value "[$(Get-Date -Format o)] $t" -ErrorAction Stop; return } catch { Start-Sleep -Milliseconds 250 } } }
New-Item -ItemType Directory -Force -Path "$bin\shaders", "$bin\obj" | Out-Null
Say "start pid $PID; source $src at $((git -C $src rev-parse HEAD).Trim())"
try {
    $dxc = 'C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x64\dxc.exe'
    if (-not (Test-Path $dxc)) { throw "dxc not found at $dxc" }
    $n = 0
    foreach ($f in Get-ChildItem "$src\src\gpu\shaders\*.hlsl") {
        & $dxc -T cs_6_2 -E main -Fo "$bin\shaders\$($f.BaseName).cso" $f.FullName -O3 -HV 2018 -WX *> "$bin\obj\dxc-$($f.BaseName).log"
        if ($LASTEXITCODE -ne 0) { throw "dxc failed on $($f.Name) (exit $LASTEXITCODE)" }
        $n++
    }
    Say "shaders compiled: $n"
    . "$src\tests\t2130-g5-red-suite\resolve_toolchain.ps1"
    Enter-SuperSlmVsDevShell | Out-Null
    $srcs = @('src\artifact.cpp', 'src\sha256.cpp', 'src\tokenizer.cpp', 'src\model.cpp', 'src\intmath.cpp', 'src\silu_lut.cpp',
        'src\matmul.cpp', 'src\proof_manifest.cpp', 'src\trace_hook.cpp', 'src\forward\checked_chain_funnel.cpp',
        'src\forward\forward_sites.cpp', 'src\forward\parallel_split.cpp', 'src\gpu\gpu_inflight_registry.cpp',
        'src\decode_digest.cpp', 'src\kv_pages.cpp', 'src\gpu\superslm_gpu.cpp', 'src\gpu\gpu_1p0.cpp',
        'tools\schema_fast_decode_gx3_bench.cpp') | ForEach-Object { "$src\$_" }
    Push-Location $src
    try {
        & cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /MP4 "/I$src\include" "/I$src\tests" @srcs "/Fo:$bin\obj\\" "/Fe:$exe" /link d3d12.lib dxgi.lib dxguid.lib *> "$root\build-slices.log"
        $code = $LASTEXITCODE
    } finally { Pop-Location }
    if ($code -ne 0) { throw "cl exited $code (build-slices.log)" }
    Say "built: sha256 $((Get-FileHash $exe).Hash.ToLower())"
    Say "model sha256 $((Get-FileHash $model).Hash.ToLower())"
    $p = Start-Process -FilePath $exe -ArgumentList $model, '--pairs', '15', '--warmup', '1', '--slices-only', '--slice-layers', '4,8,12,24', '--json', "$root\gx3-slices.json" `
        -RedirectStandardOutput "$root\gx3-slices.log" -RedirectStandardError "$root\gx3-slices.log.err" -NoNewWindow -PassThru
    $null = $p.Handle
    if (-not $p.WaitForExit(15 * 60000)) { $p.Kill(); throw 'bench passed 15 min; killed' }
    if ($p.ExitCode -ne 0) { throw "bench exited $($p.ExitCode) (gx3-slices.log.err)" }
    Say 'end PASS'
}
catch { Say "end FAIL $($_.Exception.Message)" }
