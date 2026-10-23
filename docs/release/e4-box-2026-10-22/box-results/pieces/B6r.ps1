. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$G1Src = Join-Path $E4Work 'g1src'; $G1Build = Join-Path $E4Work 'build-g1'
if (-not (Test-Path $G1Src)) { git -C D:\SuperSLM fetch origin; git -C D:\SuperSLM worktree add --detach $G1Src 1c145b53c7cf2e81e0b0ad8535356d9763c68eeb }
git -C $G1Src rev-parse HEAD              # must print: 1c145b53c7cf2e81e0b0ad8535356d9763c68eeb
git -C $G1Src merge-base HEAD origin/main # must print: 1948e1f2ea4528790dc6701e26c47bda5739b003
git -C $G1Src status --porcelain          # must print nothing
$log = E4-Log 'B6r-build'
Remove-Item $log -ErrorAction SilentlyContinue
E4-Run $log cmake @('-S', $G1Src, '-B', $G1Build, '-G', 'Ninja', '-DBUILD_TESTING=ON', '-DCMAKE_BUILD_TYPE=Release',
                    '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl')
E4-Run $log ninja @('-C', $G1Build, 'superslm_precache_e1', 'superslm_precache_reference_header')
Remove-Item -Recurse -Force $env:SUPERSLM_PRECACHE_BLOB_DIR -ErrorAction SilentlyContinue
Push-Location $G1Build
E4-Run (E4-Log 'B6r-two-process') ctest @('-R', '^superslm_precache_real_two_process$', '-V', '--timeout', '1500')
Pop-Location
