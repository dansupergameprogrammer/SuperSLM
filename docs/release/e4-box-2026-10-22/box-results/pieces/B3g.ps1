. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$GenSrc = Join-Path $E4Work 'gensrc'; $GenBuild = Join-Path $E4Work 'build-gen'
if (-not (Test-Path $GenSrc)) { git -C D:\SuperSLM worktree add --detach $GenSrc 47bdcc9e24917f91e5f9bf67e8a0141662ca5114 }
git -C $GenSrc rev-parse HEAD
git -C $GenSrc merge-base HEAD origin/main
git -C $GenSrc status --porcelain
$log = E4-Log 'B3g-gen-build'
Remove-Item $log -ErrorAction SilentlyContinue
E4-Run $log cmake @('-S', $GenSrc, '-B', $GenBuild, '-G', 'Ninja', '-DBUILD_TESTING=ON', '-DCMAKE_BUILD_TYPE=Release',
                    '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl')
E4-Run $log ninja @('-C', $GenBuild, 'superslm_precache_reference_gen')
Select-String -Path $log -Pattern 'unanchored'
E4-Run $log (Join-Path $GenBuild 'superslm_precache_reference_gen.exe') @()
