. D:\_scratch\e5-1p14\pieces\hdr.ps1
$log = E5-Log 'G3-release-gpu-build'; Remove-Item $log -ErrorAction SilentlyContinue
E4-Run $log cmake @('-S', $E5Src, '-B', "$E5Src\build", '-G', 'Visual Studio 17 2022', '-A', 'x64', '-T', 'v143,version=14.34',
                    '-DSUPERSLM_BUILD_GPU=ON', '-DBUILD_TESTING=OFF')
E4-Run $log cmake @('--build', "$E5Src\build", '--config', 'Release', '--target', 'superslm', 'superslm_gpu_shaders')
E4-Run $log cmd @('/c', "$E5Src\tests\paged-kv-gpu\build_paged_kv_gpu.bat", ('"--out=' + "$E5\pkvg" + '"'), 'e5')
