# E4 run sheet B1, verbatim: build, MSVC 14.34 Release.
. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$log = E4-Log 'B1-build'
Remove-Item $log -ErrorAction SilentlyContinue          # a re-run reads only its own log
E4-Run $log cmake @('-S', $E4Src, '-B', $E4Build, '-G', 'Ninja', '-DBUILD_TESTING=ON', '-DCMAKE_BUILD_TYPE=Release',
                    '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl')
E4-Run $log ninja @('-C', $E4Build, 'superslm_precache_e1', 'superslm_precache_reference_gen', 'superslm_precache_reference_header')
E4-Run $log $E4Runner @('--list') | Select-String 'E4'
Select-String -Path $log -Pattern 'unanchored'
