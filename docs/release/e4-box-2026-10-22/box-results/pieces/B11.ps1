. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$m = Join-Path $E4Work 'msrc'; $mb = Join-Path $E4Work 'build-mutant'
if (-not (Test-Path $m)) { git -C D:\SuperSLM worktree add --detach $m e8c029ed68f6d3fb0a728673c9a41d16d433916d }
git -C $m checkout -- .
python (Join-Path $E4Src 'tests\precache\apply_e4_mutant.py') $m matvec-last-short
$log = E4-Log 'B11-matvec-mutant'
E4-Run $log git @('-C', $m, 'diff', '--stat')
E4-Run $log cmake @('-S', $m, '-B', $mb, '-G', 'Ninja', '-DBUILD_TESTING=ON', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl')
E4-Run $log ninja @('-C', $mb, 'superslm_precache_e1')
E4-Run $log (Join-Path $mb 'superslm_precache_e1.exe') @('=6.6/E4', '=6.6/E4:rule-b')
git -C $m checkout -- .
