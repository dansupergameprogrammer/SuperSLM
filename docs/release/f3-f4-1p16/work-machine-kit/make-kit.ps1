# Assembles the F3 work-machine kit at <Dest> (default F:\kit-f3-xtx) from the box run's build and inputs: run.ps1's
# `-Piece kit` calls it after build, build-mut, real15 and real05. Copies only; never deletes anything on the drive
# outside <Dest>, and refuses a <Dest> that exists unless -Replace.
param([string]$Dest = 'F:\kit-f3-xtx', [switch]$Replace)
$ErrorActionPreference = 'Stop'
$F3 = 'D:\_scratch\f3-f4-1p16'
$Src = 'D:\SuperSLM\.worktrees\f3-f4-1p16'
$KitSrc = Join-Path $Src 'docs\release\f3-f4-1p16\work-machine-kit'
$Prompts = 'D:\_scratch\e4\t3035\tests\t3035-precache-refs\prompts'
# Code review N7: -Replace deletes <Dest> whole, so <Dest> must be a kit folder: its leaf is kit-f3-xtx, and an existing
# one carries the marker this script writes. A mistyped -Dest (F:\, or another folder) is refused.
if ((Split-Path -Leaf $Dest) -ne 'kit-f3-xtx') { throw "-Dest must end in \kit-f3-xtx (got $Dest)" }
if (Test-Path $Dest) {
    if (-not $Replace) { throw "$Dest exists; pass -Replace to rebuild it" }
    if (-not (Test-Path (Join-Path $Dest 'KIT-F3-XTX.marker'))) { throw "$Dest has no KIT-F3-XTX.marker: not a kit this script made; not deleting it" }
    Remove-Item -Recurse -Force $Dest   # only this kit's own folder
}
New-Item -ItemType Directory -Force $Dest | Out-Null
Set-Content (Join-Path $Dest 'KIT-F3-XTX.marker') "made by make-kit.ps1 $(Get-Date -Format s)"
foreach ($d in 'bin', 'bin\shaders', 'fixtures', 'e0', 'models', 'prompts', 'captures', 'blobs', 'results') {
    New-Item -ItemType Directory -Force (Join-Path $Dest $d) | Out-Null
}
Copy-Item (Join-Path $KitSrc 'run.ps1'), (Join-Path $KitSrc 'run.cmd'), (Join-Path $KitSrc 'README.txt') $Dest
foreach ($x in 'pcg_cell_format', 'pcg_cell_pool', 'pcg_cell_busy', 'pcg_cell_faults', 'pcg_cell_schema', 'pcg_cell_budget_snapshot',
               'pcg_cell_restore_derived', 'pcg_cell_9_10', 'pcg_cell_4_6', 'pcg_cell_6_5', 'pcg_cell_10_4_real', 'pcg_cell_10_4_timing') {
    Copy-Item "$F3\pkvg\bin\$x.exe" (Join-Path $Dest 'bin')
}
# 10.4's must-reject: the timing cell built on the restore-prefills mutant engine, under its own name.
Copy-Item "$F3\mutants\restore-prefills\pkvg\bin\pcg_cell_10_4_timing.exe" (Join-Path $Dest 'bin\pcg_cell_10_4_timing_restore_prefills.exe')
Copy-Item "$F3\adapters.exe" (Join-Path $Dest 'bin')
Copy-Item "$F3\pkvg\bin\shaders\*" (Join-Path $Dest 'bin\shaders') -Recurse
foreach ($dll in 'vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll') { Copy-Item (Join-Path $env:windir "System32\$dll") (Join-Path $Dest 'bin') }
Copy-Item 'D:\_artifacts\SuperSLM\t3035-precache-refs\fixtures\*' (Join-Path $Dest 'fixtures')
Copy-Item 'D:\_artifacts\SuperSLM\t3030-commission\must-*.sslm' (Join-Path $Dest 'e0')
Copy-Item 'D:\_artifacts\SuperSLM\te442-qwen2.5-0.5b-instruct-dg\qwen2.5-0.5b-instruct-dg.sslm' (Join-Path $Dest 'models')
Copy-Item 'D:\_artifacts\SuperSLM\t2901-qwen2.5-1.5b-dg\qwen2.5-1.5b-instruct-dg-g5.sslm' (Join-Path $Dest 'models')
Copy-Item "$Prompts\*.ids" (Join-Path $Dest 'prompts')
Copy-Item -Recurse "$Src\tests\precache-gpu\captures\v1.13.1_1002_744c" (Join-Path $Dest 'captures')
foreach ($tag in '0p5', '1p5') {
    $b = "$F3\blobs\gpu-$tag-10de_1e81.spx1"
    if (-not (Test-Path $b)) { throw "$b is missing: run real15 and real05 on the box first" }
    Copy-Item $b (Join-Path $Dest 'blobs')
}
# A manifest of every file's SHA-256, so the dev box can check the drive came back intact (run.ps1 -Piece cross).
$files = Get-ChildItem -Recurse -File $Dest | Where-Object { $_.FullName -notlike "*\results\*" }
$files | ForEach-Object { '{0}  {1}' -f (Get-FileHash -Algorithm SHA256 $_.FullName).Hash.ToLower(), $_.FullName.Substring($Dest.Length + 1) } |
    Set-Content (Join-Path $Dest 'MANIFEST.sha256')
"kit at $Dest : $($files.Count) files, $([math]::Round(($files | Measure-Object Length -Sum).Sum / 1GB, 2)) GB"
