. D:\_scratch\e5-1p14\pieces\hdr.ps1
$log = E5-Log 'E1-build'; Remove-Item $log -ErrorAction SilentlyContinue
E4-Run $log cmake @('-S', $E5Src, '-B', "$E5\build-msvc", '-G', 'Ninja', '-DBUILD_TESTING=ON', '-DCMAKE_BUILD_TYPE=Release',
                    '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl')
E4-Run $log ninja @('-C', "$E5\build-msvc", 'superslm_precache_unpinned')
