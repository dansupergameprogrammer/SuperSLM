. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$xc = Join-Path $E4Work 'xc'; $jb = Join-Path $E4Work 'build-4.3-msvc'
$env:CC = 'cl'; $env:CXX = 'cl'
Remove-Item -Recurse -Force $jb -ErrorAction SilentlyContinue
E4-Run (E4-Log 'B10-4.3-msvc-job') python @((Join-Path $E4Src 'tests\precache\capture_compiler_job.py'),
  '--source', $E4Src, '--build-dir', $jb, '--bundle-dir', $xc, '--generator', 'Ninja')
$env:SUPERSLM_PRECACHE_CROSS_COMPILER_DIR = $xc
$env:SUPERSLM_PRECACHE_COMPILER_PHASE = 'compare'
$env:SUPERSLM_PRECACHE_RUNNER_EXE = Join-Path $jb 'superslm_precache_e1.exe'
E4-Run (E4-Log 'B10-4.3-compare') $env:SUPERSLM_PRECACHE_RUNNER_EXE @('=4.3/E4')
