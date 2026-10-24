# F3 certification (4.4, 4.6, 6.5, 10.4) and 9.5 [F4] on the RTX 2080 SUPER. PowerShell 7.
#   usage: pwsh -NoProfile -File run.ps1 -Piece <piece> [-Force] [-KitResults DIR] [-Dest DIR] [-Replace]
#
# CPU pieces (no GPU work):
#   build      engine library and shaders (CMake, MSVC 14.34), build_paged_kv_gpu.bat F2 E5 F3, adapters.exe, and the
#              Ninja test build's superslm_precache_unpinned (9.5's MSVC half)
#   build-mut  the two engine mutants, each on a copy of src\ (the worktree's own src\ is never touched):
#              restore-prefills -> pcg_cell_10_4_timing (10.4's must-reject); upload-skip -> pcg_cell_6_5 and
#              pcg_cell_10_4_real (the run-time red of 6.5 and of 10.4's two-process leg)
#   cpu95      9.5 [F4] on MSVC: superslm_precache_unpinned =9.5/F4 =9.5/F4:must-reject
#   kit        assemble the work-machine kit at -Dest (default F:\kit-f3-xtx; -Replace rebuilds it); needs real15 and
#              real05 done (the 2080 SUPER's 4.6 blobs travel with it)
# GPU pieces (each bounded at 15 min, at least 5 min after the previous GPU piece, no debug layer):
#   f2set      4.4 on this card: the six F2 cells, pcg_cell_schema (E0 artifacts), pcg_cell_9_10 (the v1.13.1 captures
#              of this card), and pcg_cell_6_5 on pkv_def's shapes
#   real15     1.5B: pcg_cell_4_6 (a) and pcg_cell_6_5 on the real artifact
#   real05     0.5B: the same
#   r10x       10.4's "10.1-10.2 on both GPUs": pcg_cell_10_4_real 10.1 and 10.2, capture then restore, two processes
#   t-acc-15, t-acc-05     10.4 timing, the must-accept (this build), 1.5B / 0.5B
#   t-rej-15, t-rej-05     10.4 timing, the must-reject (the restore-prefills mutant build), 1.5B / 0.5B
#   t-grade-15, t-grade-05 10.4 timing, the readings graded once the debunker signs PC-GPU-TIMING-1
#   mutants    upload-skip: pcg_cell_6_5 on pkv_def and pcg_cell_10_4_real 10.1's restore must both go red, each on its
#              named assertion (exit 1, a failing suite line, 6.2's or 10.1's stream message); a timeout, a skip or a
#              start-up failure is "not killed"
#   cross      after the kit returns (-KitResults F:\kit-f3-xtx\results\<computer>): the drive checked against its
#              manifest, then 4.6 (b) and (c) here with the 7900 XTX's blobs, both models
#   all        build, build-mut, cpu95, then every GPU piece except cross, in the order above
param([Parameter(Mandatory)][ValidateSet('build', 'build-mut', 'cpu95', 'kit', 'f2set', 'real15', 'real05', 'r10x', 't-acc-15', 't-acc-05',
        't-rej-15', 't-rej-05', 't-grade-15', 't-grade-05', 'mutants', 'cross', 'all')][string]$Piece,
      [switch]$Force, [string]$KitResults, [string]$Dest = 'F:\kit-f3-xtx', [switch]$Replace)
. (Join-Path $PSScriptRoot 'hdr.ps1')

$Gpu = @('f2set', 'real15', 'real05', 'r10x', 't-acc-15', 't-rej-15', 't-acc-05', 't-rej-05', 't-grade-15', 't-grade-05', 'mutants', 'cross')
if ($Piece -eq 'all') {
    foreach ($p in @('build', 'build-mut', 'cpu95') + ($Gpu | Where-Object { $_ -ne 'cross' })) {
        $args2 = @('-NoProfile', '-File', $PSCommandPath, '-Piece', $p)
        if ($Force) { $args2 += '-Force' }
        & pwsh @args2
        $code = $LASTEXITCODE
        if ($code -ne 0 -and $p -in 'build', 'build-mut') { "stopping: $p failed"; exit 1 }
        if ($code -eq 4) { "stopping: $p could not kill a cell that overran its bound"; exit 1 }
        if ($code -eq 3) { "stopping: $p refused to start (box busy or lock not taken); rerun -Piece all to resume"; exit 1 }
    }
    'all pieces before the kit trip are done; next: -Piece kit'
    exit 0
}

if ((F3-Done $Piece) -and -not $Force) { "already done: $(Get-Content (Join-Path $F3Logs "$Piece.done"))"; exit 0 }
$log = F3-Log $Piece
$isGpu = $Piece -in $Gpu
# Code review B4: a GPU piece's log without its .done is a crash (a reboot skips every finally). Rest the full five
# minutes from now before rerunning it, whatever the stamp says.
if ($isGpu -and (Test-Path $log) -and -not (F3-Done $Piece)) { F3-RestStamp; E4-Append (F3-Log 'rest') "crash evidence: $Piece.log without $Piece.done" }
if ($isGpu) { F3-Rest (F3-Log 'rest') }
if (Test-Path $log) { Remove-Item $log }
$verdict = 'PASS'
$findings = ''
try { F3-Lock } catch { "$Piece refused: $($_.Exception.Message)"; exit 3 }
# Code review N6: the busy check runs after the rest and after the lock is taken, so a process started during the
# rest is seen.
try { F3-Busy } catch { F3-Unlock; "$Piece refused: $($_.Exception.Message)"; exit 3 }
$started = Get-Date
# Code review B4: stamp at the start too, so the stamp never predates a piece that has begun GPU work.
if ($isGpu) { F3-RestStamp }
try {
    if ($isGpu) {
        $global:F3Deadline = $started.AddMinutes($F3GpuMinutes)
        $card = F3-Adapter $log
    }
    $fail = { param($what) if ($script:verdict -eq 'PASS') { $script:verdict = "FAIL $what" } else { $script:verdict += "; $what" } }
    $note46 = { param($r) if ($r[0] -eq 'finding') { $script:findings += " 4.6-FINDING: $($r[1]);" } elseif ($r[0] -eq 'fail') { & $fail $r[1] } }
    switch ($Piece) {
        'build' {
            $head = (git -C $F3Src rev-parse HEAD).Trim()
            E4-Append $log "worktree HEAD $head"
            & git -C $F3Src merge-base --is-ancestor $F3Cells HEAD
            if ($LASTEXITCODE -ne 0) { throw "the worktree does not contain $F3Cells (the F3 seams)" }
            if ((git -C $F3Src status --porcelain)) { throw 'the worktree has local changes' }
            E4-Run $log cmake @('-S', $F3Src, '-B', "$F3Src\build", '-G', 'Visual Studio 17 2022', '-A', 'x64', '-T', 'v143,version=14.34', '-DSUPERSLM_BUILD_GPU=ON', '-DBUILD_TESTING=OFF') | Out-Null
            E4-Run $log cmake @('--build', "$F3Src\build", '--config', 'Release', '--target', 'superslm', 'superslm_gpu_shaders') | Out-Null
            if ((F3-LastExit $log) -ne 0) { throw 'the engine library or shaders did not build' }
            E4-Run $log cmd @('/c', "$F3Src\tests\paged-kv-gpu\build_paged_kv_gpu.bat", ('"--out=' + $F3Out + '"'), 'F2', 'E5', 'F3') | Out-Null
            $summary = Select-String -LiteralPath $log -Pattern 'PKVG BUILD: (\d+) linked, (\d+) red by link, (\d+) red by compile' | Select-Object -Last 1
            if (-not $summary) { throw 'no PKVG BUILD summary' }
            E4-Append $log "SUMMARY $($summary.Line)"
            if ($summary.Matches[0].Groups[2].Value -ne '0' -or $summary.Matches[0].Groups[3].Value -ne '0') { & $fail $summary.Line }
            E4-Run $log cl.exe @('/nologo', '/EHsc', '/O2', "/Fo$F3\adapters.obj", "/Fe$F3\adapters.exe", "$F3Src\docs\release\f3-f4-1p16\work-machine-kit\adapters.cpp", 'dxgi.lib') | Out-Null
            if ((F3-LastExit $log) -ne 0) { & $fail 'adapters.exe' }
            Remove-Item "$F3\adapters.obj" -ErrorAction SilentlyContinue
            E4-Run $log cmake @('-S', $F3Src, '-B', $F3Cmake, '-G', 'Ninja', '-DBUILD_TESTING=ON', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl') | Out-Null
            E4-Run $log ninja @('-C', $F3Cmake, 'superslm_precache_unpinned') | Out-Null
            if ((F3-LastExit $log) -ne 0) { & $fail 'superslm_precache_unpinned' }
        }
        'build-mut' {
            $plan = @{ 'restore-prefills' = @('pcg_cell_10_4_timing'); 'upload-skip' = @('pcg_cell_6_5', 'pcg_cell_10_4_real') }
            foreach ($m in $plan.Keys) {
                $mdir = Join-Path $F3Mut $m
                if (Test-Path $mdir) { Remove-Item -Recurse -Force $mdir }
                New-Item -ItemType Directory -Force $mdir | Out-Null
                Copy-Item -Recurse "$F3Src\src" "$mdir\src"  # whole: gpu_1p0.cpp includes ../forward/ by relative path
                E4-Run $log $F3Py @('-I', "$F3Src\tests\precache-gpu\apply_gpu_mutant.py", "$mdir\src\gpu\gpu_1p0.cpp", $m) | Out-Null
                if ((F3-LastExit $log) -ne 0) { & $fail "$m did not apply"; continue }
                E4-Run $log cmd (@('/c', "$F3Src\tests\paged-kv-gpu\build_paged_kv_gpu.bat", ('"--out=' + "$mdir\pkvg" + '"'), ('"--gpu-dir=' + "$mdir\src\gpu" + '"')) + $plan[$m]) | Out-Null
                foreach ($x in $plan[$m]) { if (-not (Test-Path "$mdir\pkvg\bin\$x.exe")) { & $fail "$m $x did not link" } }
            }
        }
        'cpu95' {
            Push-Location $F3Src
            E4-Run $log "$F3Cmake\superslm_precache_unpinned.exe" @('=9.5/F4', '=9.5/F4:must-reject') | Out-Null
            Pop-Location
            if ((F3-LastExit $log) -ne 0) { & $fail 'superslm_precache_unpinned 9.5' }
        }
        'f2set' {
            foreach ($cell in 'pcg_cell_format', 'pcg_cell_pool', 'pcg_cell_busy', 'pcg_cell_faults', 'pcg_cell_budget_snapshot', 'pcg_cell_restore_derived') {
                $c = F3-Cell $log "$F3Bin\$cell.exe" @("--fixture-dir=$F3FX") 5
                if ($c -ne 0) { & $fail "$cell exit $c" }
            }
            $env:SUPERSLM_PRECACHE_E0_ARTIFACT_DIR = $F3E0
            $c = F3-Cell $log "$F3Bin\pcg_cell_schema.exe" @() 3
            if ($c -ne 0) { & $fail "pcg_cell_schema exit $c" }
            $c = F3-Cell $log "$F3Bin\pcg_cell_9_10.exe" @("--fixture-dir=$F3FX", "--captures=$F3Src\tests\precache-gpu\captures\v1.13.1_$card") 5
            if ($c -ne 0) { & $fail "pcg_cell_9_10 exit $c" }
            $c = F3-Cell $log "$F3Bin\pcg_cell_6_5.exe" @("--fixture-dir=$F3FX") 5
            if ($c -ne 0) { & $fail "pcg_cell_6_5 pkv_def exit $c" }
        }
        { $_ -in 'real15', 'real05' } {
            $tag = $(if ($Piece -eq 'real15') { '1p5' } else { '0p5' })
            $model = $(if ($Piece -eq 'real15') { $F3M15 } else { $F3M05 })
            $ids = Join-Path $F3Prompts $(if ($Piece -eq 'real15') { 'world.1p5.ids' } else { 'world.ids' })
            $c = F3-Cell $log "$F3Bin\pcg_cell_4_6.exe" @("--model=$model", "--tag=$tag", "--ids=$ids", "--out=$F3Blobs") 10
            & $note46 (F3-Cell46 $log $c "pcg_cell_4_6 (a) $tag")
            $c = F3-Cell $log "$F3Bin\pcg_cell_6_5.exe" @("--model=$model", "--tag=$tag", "--world=$ids", ('--persona=' + (Join-Path $F3Prompts 'persona00.ids'))) 10
            if ($c -ne 0) { & $fail "pcg_cell_6_5 $tag exit $c" }
        }
        'r10x' {
            $b = Join-Path $F3Blobs 'r10x'
            New-Item -ItemType Directory -Force $b | Out-Null
            foreach ($cell in '10.1', '10.2') {
                foreach ($phase in 'capture', 'restore') {
                    $c = F3-Cell $log "$F3Bin\pcg_cell_10_4_real.exe" @("--cell=$cell", "--phase=$phase", "--model=$F3M05", "--prompts=$F3Prompts", "--blobs=$b") 8
                    if ($c -ne 0) { & $fail "pcg_cell_10_4_real $cell $phase exit $c"; break }
                }
            }
        }
        { $_ -like 't-*' } {
            $null = $Piece -match '^t-(acc|rej|grade)-(15|05)$'
            $role = @{ acc = 'accept'; rej = 'reject'; grade = 'grade' }[$Matches[1]]
            $tag = $(if ($Matches[2] -eq '15') { '1p5' } else { '0p5' })
            $model = $(if ($tag -eq '1p5') { $F3M15 } else { $F3M05 })
            $ids = Join-Path $F3Prompts $(if ($tag -eq '1p5') { 'world.1p5.ids' } else { 'world.ids' })
            $exe = $(if ($role -eq 'reject') { "$F3Mut\restore-prefills\pkvg\bin\pcg_cell_10_4_timing.exe" } else { "$F3Bin\pcg_cell_10_4_timing.exe" })
            $c = F3-Cell $log $exe @("--model=$model", "--tag=$tag", "--ids=$ids", "--role=$role") 15
            # The instrument decides nothing; tools\precache_gpu\grade_f3.py does, after commissioning. Exit 0 here
            # means only that the run completed and every reading resolved.
            if ($c -ne 0) { & $fail "pcg_cell_10_4_timing $role $tag exit $c" }
        }
        'mutants' {
            $bin = "$F3Mut\upload-skip\pkvg\bin"
            $c = F3-Cell $log "$bin\pcg_cell_6_5.exe" @("--fixture-dir=$F3FX") 5
            $k = F3-Killed $log $c 'pcg_cell_6_5' "6\.2 the restored prefix's adopter differs"
            E4-Append $log "SUMMARY upload-skip pcg_cell_6_5 exit ${c}: $k (killed only on 6.2's assertion)"
            if ($k -ne 'killed') { & $fail "upload-skip pcg_cell_6_5 $k" }
            if (-not (Test-Path (Join-Path $F3Blobs 'r10x\10.1.refs'))) { & $fail 'r10x has not captured 10.1' }
            else {
                $c = F3-Cell $log "$bin\pcg_cell_10_4_real.exe" @('--cell=10.1', '--phase=restore', "--model=$F3M05", "--prompts=$F3Prompts", ('--blobs=' + (Join-Path $F3Blobs 'r10x'))) 8
                $k = F3-Killed $log $c 'pcg_cell_10_4_real' "10\.1 .*the restored prefix's adopter differs from the capture's|10\.1: \d+ of 50 adopter streams"
                E4-Append $log "SUMMARY upload-skip pcg_cell_10_4_real 10.1 restore exit ${c}: $k (killed only on 10.1's stream assertion)"
                if ($k -ne 'killed') { & $fail "upload-skip pcg_cell_10_4_real $k" }
            }
        }
        'cross' {
            if (-not $KitResults -or -not (Test-Path $KitResults)) { throw 'pass -KitResults <kit>\results\<computer>' }
            $kitRoot = Split-Path -Parent (Split-Path -Parent $KitResults)
            $bad = 0
            foreach ($line in Get-Content (Join-Path $kitRoot 'MANIFEST.sha256')) {
                $h, $rel = $line -split '  ', 2
                $f = Join-Path $kitRoot $rel
                if (-not (Test-Path -LiteralPath $f) -or (Get-FileHash -Algorithm SHA256 -LiteralPath $f).Hash.ToLower() -ne $h) { $bad++; E4-Append $log "MANIFEST MISMATCH $rel" }
            }
            E4-Append $log "manifest: $bad mismatch(es)"
            if ($bad -ne 0) { throw "the kit came back with $bad file(s) off its manifest" }
            $out = Join-Path $F3Blobs 'cross'
            New-Item -ItemType Directory -Force $out | Out-Null
            foreach ($tag in '1p5', '0p5') {
                $model = $(if ($tag -eq '1p5') { $F3M15 } else { $F3M05 })
                $ids = Join-Path $F3Prompts $(if ($tag -eq '1p5') { 'world.1p5.ids' } else { 'world.ids' })
                $theirs = @(Get-ChildItem (Join-Path $KitResults 'blobs46') -Filter "gpu-$tag-1002_*.spx1")
                if ($theirs.Count -ne 1) { & $fail "no single 7900 XTX $tag blob in the kit results"; continue }
                $c = F3-Cell $log "$F3Bin\pcg_cell_4_6.exe" @("--model=$model", "--tag=$tag", "--ids=$ids", "--out=$out", "--other=$($theirs[0].FullName)", "--foreign=$($theirs[0].FullName)") 7
                & $note46 (F3-Cell46 $log $c "pcg_cell_4_6 (b, c) $tag")
            }
        }
        'kit' {
            foreach ($p in 'build', 'build-mut', 'real15', 'real05') { if (-not (F3-Done $p)) { throw "$p has not run" } }
            $k = Join-Path $F3Src 'docs\release\f3-f4-1p16\work-machine-kit\make-kit.ps1'
            $a = @('-NoProfile', '-File', $k, '-Dest', $Dest)
            if ($Replace) { $a += '-Replace' }
            E4-Run $log pwsh $a | Out-Null
            if ((F3-LastExit $log) -ne 0) { & $fail 'make-kit' }
        }
    }
}
catch { $verdict = "FAIL $($_.Exception.Message)"; E4-Append $log "ERROR $($_.Exception.Message)" }
finally {
    F3-Unlock
    if ($isGpu) { F3-RestStamp }
}
$mins = ((Get-Date) - $started).TotalMinutes
E4-Append $log ('piece took {0:N1} min' -f $mins)
if ($isGpu -and $mins -gt $F3GpuMinutes) { $verdict = "FAIL over the 15-minute bound ($('{0:N1}' -f $mins) min); $verdict" }
if ($findings) { $verdict += $findings }
F3-Mark $Piece $verdict
"$Piece $verdict"
if ($verdict -like 'PASS*') { exit 0 } else { exit 1 }
