. D:\_scratch\e5-1p14\pieces\hdr.ps1
$V = "$E5\v1131"; $log = E5-Log 'G1-v1131-build'; Remove-Item $log -ErrorAction SilentlyContinue
E4-Run $log cmake @('-S', $V, '-B', "$V\build", '-G', 'Visual Studio 17 2022', '-A', 'x64', '-T', 'v143,version=14.34',
                    '-DSUPERSLM_BUILD_GPU=ON', '-DBUILD_TESTING=OFF')
E4-Run $log cmake @('--build', "$V\build", '--config', 'Release', '--target', 'superslm', 'superslm_gpu_shaders')
E4-Run $log cmd @('/c', "$E5Src\tests\paged-kv-gpu\build_paged_kv_gpu.bat", "--out=$E5\pkvg", "--capture910-tree=$V", 'capture910')
