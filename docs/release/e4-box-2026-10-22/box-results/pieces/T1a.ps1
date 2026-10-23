# E4 T1a (debunker): build the 10.3 must-reject mutant like B1, in its own worktree and build directory.
. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$mutSrc = Join-Path $E4Work 'mut-src'
$mutBuild = Join-Path $E4Work 'build-mut'
$log = E4-Log 'T1a-mutant-build'
if (-not (Test-Path -LiteralPath (Join-Path $mutSrc '.git'))) {
    E4-Run $log git @('-C', 'D:\SuperSLM', 'worktree', 'add', '--detach', $mutSrc, 'e8c029ed68f6d3fb0a728673c9a41d16d433916d')
    E4-Run $log 'C:\Users\dansu\AppData\Local\Programs\Python\Python312\python.exe' @('-I', 'D:\_scratch\e4\t1\make_mutant.py', $mutSrc)
}
E4-Run $log git @('-C', $mutSrc, 'diff', '--stat')
E4-Run $log cmake @('-S', $mutSrc, '-B', $mutBuild, '-G', 'Ninja', '-DBUILD_TESTING=ON', '-DCMAKE_BUILD_TYPE=Release',
                    '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl')
E4-Run $log ninja @('-C', $mutBuild, 'superslm_precache_e1')
E4-Run $log (Join-Path $mutBuild 'superslm_precache_e1.exe') @('--list') | Select-String '10.3'
