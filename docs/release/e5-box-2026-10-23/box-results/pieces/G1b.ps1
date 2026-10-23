. D:\_scratch\e5-1p14\pieces\hdr.ps1
$V = "$E5\v1131"; $log = E5-Log 'G1b-v1131-capture-link'; Remove-Item $log -ErrorAction SilentlyContinue
E4-Run $log cmd @('/c', "$E5Src\tests\paged-kv-gpu\build_paged_kv_gpu.bat", ('"--out=' + "$E5\pkvg" + '"'), ('"--capture910-tree=' + $V + '"'), 'capture910')
