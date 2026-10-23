. D:\_scratch\e5-1p14\pieces\hdr.ps1
Push-Location "$E5\build-msvc"
E4-Run (E5-Log 'E2-ctest') ctest @('-R', '^superslm_precache_(api_ordinals|api_ordinals_locator|gpu_prefix_labels|version_header|version_header_commission)$', '-V')
Pop-Location
E4-Run (E5-Log 'E2-717-dryrun') python @("$E5Src\docs\release\e5-box-2026-10-23\7.17\run_717_constructions.py", $E5Src)
