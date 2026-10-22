# MSVC GEMM placement pin: builds sslm_decode_threading_bench from a source tree (a headless C ABI bench whose prefill
# and decode run the whole forward pass, so it links matmul.cpp's forward kernels through superslm.lib), relinks it
# with /MAP, and checks where the GEMM kernels land mod 64 against the pinned placement (README.md).
# Run from a Visual Studio x64 developer shell, or let tests/t2130-g5-red-suite/resolve_toolchain.ps1 enter one.
# Exit 0: every kernel at its expected offset. Exit 1: a kernel moved, or a symbol was not found in the map.
param(
    [Parameter(Mandatory)][string]$Worktree,   # the SuperSLM source tree to check (a checkout or a worktree)
    [Parameter(Mandatory)][string]$BuildDir,   # its MSVC Ninja build directory; configured here if it is empty
    [string]$MapDir = ''                       # where the relinked bench and its map go; default <BuildDir>\pinmap
)
$ErrorActionPreference = 'Stop'
$Worktree = (Resolve-Path $Worktree).Path
if (-not $MapDir) { $MapDir = Join-Path $BuildDir 'pinmap' }
New-Item -ItemType Directory -Force -Path $BuildDir, $MapDir | Out-Null
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    . (Join-Path $Worktree 'tests\t2130-g5-red-suite\resolve_toolchain.ps1')
    Enter-SuperSlmVsDevShell | Out-Null
}
if ((Test-Path (Join-Path $BuildDir 'CMakeCache.txt')) -and -not (Test-Path (Join-Path $BuildDir 'build.ninja'))) {
    throw "$BuildDir holds a non-Ninja CMake configuration; give an empty or Ninja build directory"
}
if (-not (Test-Path (Join-Path $BuildDir 'build.ninja'))) {
    & cmake -S $Worktree -B $BuildDir -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DSUPERSLM_BUILD_GPU=OFF | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'cmake configure failed' }
}
& cmake --build $BuildDir --target sslm_decode_threading_bench | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'build failed' }

$obj = Join-Path $BuildDir 'CMakeFiles\sslm_decode_threading_bench.dir\tools\decode_threading_bench.cpp.obj'
$lib = Join-Path $BuildDir 'superslm.lib'
$exe = Join-Path $MapDir 'sslm_decode_threading_bench_pinmap.exe'
$map = Join-Path $MapDir 'sslm_decode_threading_bench_pinmap.map'
& link /nologo /machine:x64 /INCREMENTAL:NO /subsystem:console "/OUT:$exe" "/MAP:$map" $obj $lib `
    kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'relink with /MAP failed' }

# The pinned placement (README.md): offsets of each kernel's address mod 64, measured on another line's forward-pass
# bench; a box run has to confirm them on this binary (README.md).
$expected = [ordered]@{ 'GemmInt8AccumulateCols' = 0; 'DotRowAvx2' = 16; 'TiledGemmAvx2' = 32 }
$found = @{}
Get-Content $map | Where-Object { $_ -match '^\s*[0-9A-F]{4}:' -and $_ -notmatch 'unwind|cppxdata|ip2state|stateUnwind|chain|dtor' } | ForEach-Object {
    $f = ($_ -split '\s+') | Where-Object { $_ }
    if ($f.Count -lt 3 -or $f[2] -notmatch '^[0-9a-f]{16}$') { return }
    foreach ($name in $expected.Keys) {
        if ($f[1] -match ('(\?|\$)' + $name + '@') -and -not $found.ContainsKey($name)) {
            $found[$name] = [Convert]::ToInt64($f[2], 16) % 64
            "{0,-24} {1} mod64={2}" -f $name, $f[2], $found[$name]
        }
    }
}
Get-Content $map | Select-String '\.sslmgem' | Select-Object -First 3 | ForEach-Object { $_.Line }
$ok = $true
foreach ($name in $expected.Keys) {
    if (-not $found.ContainsKey($name)) { "MISSING $name (not in the map)"; $ok = $false }
    elseif ($found[$name] -ne $expected[$name]) { "MOVED $name at $($found[$name]) mod 64, expected $($expected[$name])"; $ok = $false }
}
if ($ok) { 'PIN OK: the GEMM kernels are at the pinned placement'; exit 0 } else { 'PIN FAILED'; exit 1 }
