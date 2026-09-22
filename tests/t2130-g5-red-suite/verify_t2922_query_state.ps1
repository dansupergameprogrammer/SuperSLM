& cmd /c (Join-Path $PSScriptRoot 'build_t2922_gpu_schema_accepting_red.bat')
$code = $LASTEXITCODE
if ($code -eq 0 -or $code -eq 1) { exit 0 }
exit $code
