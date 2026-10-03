<#
.SYNOPSIS
T-3103 S0-SF1: build the reference harness against SuperFAISS master cd42dc2 and execute the
resolved-batch round trip. Exit 0 = receipt written and sf1_check.py accepted it; 1 = checker
refused; 2 = build/harness error; 3 = box timing lock present (nothing run).

Never builds from the SuperFAISS working tree: the pinned commit is extracted with git archive
into D:\_scratch\T-3103\superfaiss-cd42dc2. The receipt goes to D:\_artifacts\SuperSLM\t3103\.
#>
$ErrorActionPreference = 'Stop'
$Pinned = 'cd42dc228244d1db54864ef69b5ee5f07d16fc2d'
$Here = $PSScriptRoot
$Scratch = 'D:\_scratch\T-3103'
$Src = Join-Path $Scratch 'superfaiss-cd42dc2'
$Out = Join-Path $Scratch 'sf1-build'
$Artifacts = 'D:\_artifacts\SuperSLM\t3103'
$Receipt = Join-Path $Artifacts 'sf1-receipt.json'

if (Test-Path 'D:\_scratch\chains\box-timing.lock') { Write-Output 'LOCK_PRESENT: box-timing.lock exists; SF1 run owed'; exit 3 }

$master = (git -C D:\SuperFAISS rev-parse master).Trim()
if ($master -ne $Pinned) { Write-Output "PIN_MISMATCH: master is $master, fixture pins $Pinned"; exit 2 }
New-Item -ItemType Directory -Force $Src, $Out, $Artifacts | Out-Null
if (-not (Test-Path (Join-Path $Src 'include\superfaiss\scratch.h'))) {
    git -C D:\SuperFAISS archive --format=tar --output (Join-Path $Scratch 'superfaiss-cd42dc2.tar') $Pinned src include
    tar -x -f (Join-Path $Scratch 'superfaiss-cd42dc2.tar') -C $Src
}

$py = (Get-Command py -ErrorAction Stop).Source
& $py -3 (Join-Path $Here 'sf1_cells.py') (Join-Path $Here 'sf1-fixture.json') (Join-Path $Out 'cells.txt')
if ($LASTEXITCODE -ne 0) { exit 2 }

$vsdev = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat'
if (-not (Test-Path $vsdev)) { Write-Output "VsDevCmd.bat not found at $vsdev"; exit 2 }
# Same TU list and flags as master CMakeLists.txt (MSVC branch), plus the harness.
$tus = 'kernels','alloc','validate','bake','query','compose','analytics','pca','scratch','graph','novelty','matching','diversity'
$common = '/nologo /std:c++17 /O2 /W4 /fp:precise /EHsc /c'
$lines = @("call `"$vsdev`" -arch=x64 -no_logo", "cd /d `"$Out`"")
foreach ($tu in $tus) { $lines += "cl $common /I`"$Src\include`" `"$Src\src\$tu.cpp`" || exit /b 1" }
$lines += "cl $common /arch:AVX2 /I`"$Src\include`" `"$Src\src\kernels_avx2.cpp`" || exit /b 1"
$lines += "cl $common /I`"$Src\include`" `"$Here\sf1_harness.cpp`" || exit /b 1"
$lines += "link /nologo /OUT:sf1_harness.exe *.obj || exit /b 1"
$bat = Join-Path $Out 'build.cmd'
Set-Content -Path $bat -Value $lines -Encoding ascii
cmd /c "`"$bat`"" | Out-File (Join-Path $Out 'build.log') -Encoding utf8
if ($LASTEXITCODE -ne 0) { Write-Output "BUILD_FAILED: see $Out\build.log"; exit 2 }

& (Join-Path $Out 'sf1_harness.exe') (Join-Path $Out 'cells.txt') | Out-File $Receipt -Encoding utf8NoBOM
if ($LASTEXITCODE -ne 0) { Write-Output 'HARNESS_FAILED'; exit 2 }

& $py -3 (Join-Path $Here 'sf1_check.py') (Join-Path $Here 'sf1-fixture.json') $Receipt
exit $LASTEXITCODE
