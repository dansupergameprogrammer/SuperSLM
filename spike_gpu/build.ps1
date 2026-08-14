# Build the T-1979 GPU walking-skeleton harness: compile shaders/qproj_site.hlsl
# -> .cso via dxc, and compile+link t1979_gpu_harness.cpp via MSVC. Mirrors
# Claude/Laplace/gpu-determinism/build.ps1's own dxc/cl recipe exactly (same
# dxc path, same -T cs_6_2 -O3 -HV 2018, same vcvars64 one-shot cl invocation)
# -- the sibling substrate this spike's gpu.hpp was copied from commits its
# recipe; this file is the same solution for this spike.
#
# T-1987 (closing T-1983 review S-3): this script did not exist anywhere in
# HEAD, so nothing reproduced the shader/harness half of the T-1979 build from
# a fresh checkout.
#
# Usage: build.ps1                          (build only)
#        build.ps1 -Run <dump.bin>          (build, then run the harness against dump.bin)
param([string]$Run = "")
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root
$dxc = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x64\dxc.exe"

# Compile shaders/*.hlsl -> shaders/*.cso (cs_6_2 for int64 + templated ByteAddressBuffer loads).
Get-ChildItem "$root\shaders\*.hlsl" | ForEach-Object {
    $cso = [IO.Path]::ChangeExtension($_.FullName, ".cso")
    & $dxc -T cs_6_2 -E main -Fo $cso $_.FullName -O3 -HV 2018
    if ($LASTEXITCODE -ne 0) { throw "dxc failed on $($_.Name)" }
}

# Compile+link the C++ harness through vcvars64 (one shot).
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
$exe = "$root\t1979_gpu_harness.exe"
$cmd = "call `"$vcvars`" >nul && cl /nologo /std:c++17 /EHsc /O2 /Fe:`"$exe`" `"$root\t1979_gpu_harness.cpp`" /link d3d12.lib dxgi.lib dxguid.lib"
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { throw "cl failed" }

if ($Run -ne "") {
    Write-Output "=== running t1979_gpu_harness ==="
    & $exe $Run "$root\shaders\qproj_site.cso"
    Write-Output "=== exit $LASTEXITCODE ==="
}
