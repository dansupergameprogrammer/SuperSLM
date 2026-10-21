# Relinks each arm's bench with CMake's own link line plus /MAP (same objects, same order, same flags), into scratch,
# and lists where the matmul.obj functions land (address and address mod 64).
$ErrorActionPreference = 'Stop'
$out = 'D:\_scratch\decode-j1x-ab\maps'
New-Item -ItemType Directory -Force -Path $out | Out-Null
. 'D:\SuperSLM\.worktrees\schema-fast-decode\tests\t2130-g5-red-suite\resolve_toolchain.ps1'
Enter-SuperSlmVsDevShell | Out-Null
foreach ($arm in 'a4692482', '2af41659', 'f4f69efb', 'bfdec2a6') {
    $bd = "D:\SuperSLM\.worktrees\sfd-arm-$arm\build-fresh"
    $obj = "$bd\CMakeFiles\sslm_schema_fast_decode_bench.dir\tools\schema_fast_decode_bench.cpp.obj"
    & link /nologo /machine:x64 /INCREMENTAL:NO /subsystem:console "/OUT:$out\$arm.exe" "/MAP:$out\$arm.map" $obj "$bd\superslm.lib" kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "link $arm failed" }
    $same = (Get-FileHash "$out\$arm.exe").Hash -eq (Get-FileHash "$bd\sslm_schema_fast_decode_bench.exe").Hash
    "== $arm (relinked exe identical to the CMake one: $same)"
    Get-Content "$out\$arm.map" | Where-Object { $_ -match 'matmul\.cpp\.obj|matmul\.obj' -and $_ -match 'Gemm|DotRow' } | ForEach-Object {
        $f = ($_ -split '\s+') | Where-Object { $_ }
        $addr = [Convert]::ToInt64($f[2], 16)
        "{0,-70} {1} mod64={2}" -f ($f[1].Substring(0, [math]::Min(70, $f[1].Length))), $f[2], ($addr % 64)
    }
}
