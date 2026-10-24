# F2 GPU twins on the RTX 2080 SUPER. usage: pwsh -NoProfile -File run.ps1 -Piece <build|f2|schema|mutant|g4|g5> [-Force]
#   build   the engine library and shaders (CMake, MSVC 14.34), then build_paged_kv_gpu.bat: F2, pkvg_g4, pkvg_g5
#   f2      the F2 cells with the D3D12 debug layer: format, pool, busy, faults, budget_snapshot, restore_derived
#   schema  pcg_cell_schema with the E0 artifact directory (debug layer on)
#   mutant  11.5 on real hardware: no-entry-guard applied to a copy of src\gpu, pcg_cell_busy built against it, run:
#           it must go red (3.4); the worktree's own src\gpu is never touched
#   g4, g5  pkvg_g4 and pkvg_g5 with the debug layer, adapter 0: the step's own cells ([G4] or [G5]); g4 leaves 1.9 out
#   g19     cell 1.9 alone, debug layer OFF: under GPU-based validation it ran 90 min on this card (2026-10-13) and
#           both earlier box crashes came under that load
param([Parameter(Mandatory)][ValidateSet('build', 'f2', 'schema', 'mutant', 'g4', 'g5', 'g19')][string]$Piece, [switch]$Force)
. D:\_scratch\f2-2080\pieces\hdr.ps1
if ((F2-Done $Piece) -and -not $Force) { "already done: $(Get-Content (Join-Path $F2 "$Piece.done"))"; exit 0 }
F2-Busy
$log = F2-Log $Piece
if (Test-Path $log) { Remove-Item $log }
$env:SSLM_GPU_ADAPTER_INDEX = '0'
$verdict = 'PASS'
F2-Lock
try {
    switch ($Piece) {
        'build' {
            $head = (git -C $F2Src rev-parse HEAD).Trim()
            E4-Append $log "worktree HEAD $head"
            if ($head -notlike '5599f7e4*') { throw "the worktree is not at 5599f7e4 ($head)" }
            E4-Run $log cmake @('-S', $F2Src, '-B', "$F2Src\build", '-G', 'Visual Studio 17 2022', '-A', 'x64', '-T', 'v143,version=14.34', '-DSUPERSLM_BUILD_GPU=ON', '-DBUILD_TESTING=OFF') | Out-Null
            E4-Run $log cmake @('--build', "$F2Src\build", '--config', 'Release', '--target', 'superslm', 'superslm_gpu_shaders') | Out-Null
            E4-Run $log cmd @('/c', "$F2Src\tests\paged-kv-gpu\build_paged_kv_gpu.bat", ('"--out=' + $F2Out + '"'), 'F2', 'pkvg_g4', 'pkvg_g5') | Out-Null
            $summary = Select-String -LiteralPath $log -Pattern 'PKVG BUILD: (\d+) linked, (\d+) red by link, (\d+) red by compile' | Select-Object -Last 1
            if (-not $summary) { throw 'no PKVG BUILD summary' }
            E4-Append $log "SUMMARY $($summary.Line)"
            if ($summary.Matches[0].Groups[2].Value -ne '0' -or $summary.Matches[0].Groups[3].Value -ne '0') { $verdict = "FAIL $($summary.Line)" }
        }
        'f2' {
            $env:SSLM_GPU_ENABLE_DEBUG_LAYER = '1'
            $results = @()
            foreach ($cell in 'pcg_cell_format', 'pcg_cell_pool', 'pcg_cell_busy', 'pcg_cell_faults', 'pcg_cell_budget_snapshot', 'pcg_cell_restore_derived') {
                $c = F2-Cell $log "$F2Bin\$cell.exe" @("--fixture-dir=$F2FX") 20
                $results += "$cell exit $c"
                if ($c -ne 0) { $verdict = 'FAIL' }
            }
            E4-Append $log ('SUMMARY ' + ($results -join '; '))
        }
        'schema' {
            $env:SSLM_GPU_ENABLE_DEBUG_LAYER = '1'
            $env:SUPERSLM_PRECACHE_E0_ARTIFACT_DIR = $F2E0
            $c = F2-Cell $log "$F2Bin\pcg_cell_schema.exe" @() 10
            E4-Append $log "SUMMARY pcg_cell_schema exit $c"
            if ($c -ne 0) { $verdict = 'FAIL' }
        }
        'mutant' {
            $mdir = Join-Path $F2 'mutant'
            if (Test-Path $mdir) { Remove-Item -Recurse -Force $mdir }
            New-Item -ItemType Directory -Force $mdir | Out-Null
            Copy-Item -Recurse "$F2Src\src" "$mdir\src"  # whole: gpu_1p0.cpp includes ../forward/ by relative path
            E4-Run $log 'C:\Users\dansu\AppData\Local\Programs\Python\Python312\python.exe' @('-I', "$F2Src\tests\precache-gpu\apply_gpu_mutant.py", "$mdir\src\gpu\gpu_1p0.cpp", 'no-entry-guard') | Out-Null
            if ((F2-LastExit $log) -ne 0) { throw 'the mutant did not apply' }
            E4-Run $log cmd @('/c', "$F2Src\tests\paged-kv-gpu\build_paged_kv_gpu.bat", ('"--out=' + "$mdir\pkvg" + '"'), ('"--gpu-dir=' + "$mdir\src\gpu" + '"'), 'pcg_cell_busy') | Out-Null
            if (-not (Test-Path "$mdir\pkvg\bin\pcg_cell_busy.exe")) { throw 'the mutant cell did not link' }
            $env:SSLM_GPU_ENABLE_DEBUG_LAYER = '1'
            $c = F2-Cell $log "$mdir\pkvg\bin\pcg_cell_busy.exe" @("--fixture-dir=$F2FX") 20
            $redOn34 = [bool](Select-String -LiteralPath $log -Pattern '3\.4' -Quiet)
            E4-Append $log "SUMMARY mutant no-entry-guard: pcg_cell_busy exit $c (must be non-zero: the mutant is killed); 3.4 named in the output: $redOn34"
            if ($c -eq 0) { $verdict = 'FAIL the mutant survived' }
        }
        { $_ -in 'g4', 'g5', 'g19' } {
            $env:SSLM_GPU_ENABLE_DEBUG_LAYER = $(if ($Piece -eq 'g19') { $null } else { '1' })
            $exeStep = $(if ($Piece -eq 'g19') { 'g4' } else { $Piece })
            $exe = "$F2Bin\pkvg_$exeStep.exe"
            # Only this step's own cells (the bracket [G4] or [G5]): the earlier steps' cells ran on this card at
            # their own releases. Each by exact id.
            $step = $exeStep.ToUpper()
            Push-Location $F2Src
            $ids = @(& $exe --list 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -match "^\[$step " } | ForEach-Object { '=' + (($_ -replace '^\[[^\]]+\]\s+', '').Trim()) })
            Pop-Location
            if ($Piece -eq 'g4') { $ids = @($ids | Where-Object { $_ -ne '=1.9' }) }
            if ($Piece -eq 'g19') { $ids = @('=1.9') }
            E4-Append $log "selected $($ids.Count) [$step] cells: $($ids -join ' ')"
            $c = F2-Cell $log $exe (@("--pkv-fixtures=$F2FX", "--pkv-reference=$F2Src\tests\paged-kv\reference", '--gpu-ref=D:\_artifacts\pkv-gpu-refs', "--model05=$F2M05", "--model15=$F2M15", '--adapter=D:\_scratch\pkv-gpu-113\adapters\pkv_def_zero_adapter.sslm', '--adapter-other-base=D:\_scratch\pkv-gpu-113\adapters\pkv_qk_zero_adapter.sslm') + $ids) 25
            E4-Append $log "SUMMARY pkvg_$Piece exit $c"
            if ($c -ne 0) { $verdict = "FAIL exit $c" }
        }
    }
}
catch { $verdict = "FAIL $($_.Exception.Message)"; E4-Append $log "ERROR $($_.Exception.Message)" }
finally { F2-Unlock }
F2-Mark $Piece $verdict
"$Piece $verdict"
if ($verdict -like 'PASS*') { exit 0 } else { exit 1 }
