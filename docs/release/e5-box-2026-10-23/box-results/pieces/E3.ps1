. D:\_scratch\e5-1p14\pieces\hdr.ps1
$env:SUPERSLM_PAGED_KV_FIXTURE_DIR = $FX
Push-Location $E5Src
E4-Run (E5-Log 'E3-9.9') "$E5\build-msvc\superslm_precache_unpinned.exe" @('=9.9/E5')
Pop-Location
