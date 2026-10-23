. D:\_scratch\e5-1p14\pieces\hdr.ps1
$env:SSLM_GPU_ADAPTER_INDEX = '0'
E4-Run (E5-Log 'G2-9.10-capture-2080s') "$E5\pkvg\v1131\pcg_cell_9_10_capture.exe" @("--fixture-dir=$FX", "--out=$E5\captures910")
Get-ChildItem "$E5\captures910" -Recurse -File | Get-FileHash | ForEach-Object { E4-Append (E5-Log 'G2-9.10-capture-2080s') ("{0}  {1}" -f $_.Hash.ToLower(), $_.Path) }
