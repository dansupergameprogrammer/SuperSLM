# Builds the pin commit fresh in its own worktree, relinks its bench with /MAP, and shows where the GEMM kernels land.
param([Parameter(Mandatory)][string]$Commit)
$ErrorActionPreference = 'Stop'
$out = 'D:\_scratch\decode-j1x-ab\maps'
New-Item -ItemType Directory -Force -Path $out | Out-Null
. 'D:\SuperSLM\.worktrees\schema-fast-decode\tests\t2130-g5-red-suite\resolve_toolchain.ps1'
Enter-SuperSlmVsDevShell | Out-Null
$full = (git -C D:\SuperSLM rev-parse $Commit).Trim()
$wt = "D:\SuperSLM\.worktrees\sfd-arm-$($full.Substring(0,8))"
if (-not (Test-Path $wt)) { git -C D:\SuperSLM worktree add -q --detach $wt $full }
$bd = "$wt\build-fresh"
if (-not (Test-Path "$bd\sslm_schema_fast_decode_bench.exe")) {
    & cmake -S $wt -B $bd -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DSUPERSLM_BUILD_GPU=OFF | Out-Null
    & cmake --build $bd --target superslm sslm_schema_fast_decode_bench | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'build failed' }
}
$obj = "$bd\CMakeFiles\sslm_schema_fast_decode_bench.dir\tools\schema_fast_decode_bench.cpp.obj"
$tag = $full.Substring(0, 8)
& link /nologo /machine:x64 /INCREMENTAL:NO /subsystem:console "/OUT:$out\$tag.exe" "/MAP:$out\$tag.map" $obj "$bd\superslm.lib" kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib | Out-Null
"bench $bd\sslm_schema_fast_decode_bench.exe"
Get-Content "$out\$tag.map" | Where-Object { $_ -match '^\s*[0-9A-F]{4}:' -and $_ -match '\.sslmgem|DotRowAvx2@|GemmInt8AccumulateCols@|TiledGemmAvx2@|ActiveGemmTier@' -and $_ -notmatch 'unwind|cppxdata|ip2state|stateUnwind|chain|dtor' } | ForEach-Object {
    $f = ($_ -split '\s+') | Where-Object { $_ }
    if ($f.Count -ge 3 -and $f[2] -match '^[0-9a-f]{16}$') { "{0,-60} {1} mod64={2}" -f $f[1].Substring(0, [math]::Min(60, $f[1].Length)), $f[2], ([Convert]::ToInt64($f[2], 16) % 64) } else { $_ }
}
Get-Content "$out\$tag.map" | Select-String '\.sslmgem' | Select -First 3 | % { $_.Line }
