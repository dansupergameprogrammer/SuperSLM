# Assembles the F2 work-machine kit at <Dest> (default F:\kit-f2-xtx) from this run's build and inputs. Copies only;
# never deletes anything on the drive outside <Dest>, and refuses a <Dest> that exists unless -Replace.
param([string]$Dest = 'F:\kit-f2-xtx', [switch]$Replace)
$ErrorActionPreference = 'Stop'
$F2 = 'D:\_scratch\f2-2080'
if (Test-Path $Dest) {
    if (-not $Replace) { throw "$Dest exists; pass -Replace to rebuild it" }
    Remove-Item -Recurse -Force $Dest   # only this kit's own folder
}
$dirs = 'bin', 'bin\shaders', 'fixtures', 'e0', 'models', 'adapters', 'blobs', 'paged-kv-reference', 'results'
foreach ($d in $dirs) { New-Item -ItemType Directory -Force (Join-Path $Dest $d) | Out-Null }
Copy-Item "$F2\kit\run.ps1", "$F2\kit\run.cmd", "$F2\kit\README.txt" $Dest
foreach ($x in 'pcg_cell_format', 'pcg_cell_pool', 'pcg_cell_busy', 'pcg_cell_faults', 'pcg_cell_schema', 'pcg_cell_budget_snapshot',
               'pcg_cell_restore_derived', 'pcg_cross_card', 'pkvg_g4', 'pkvg_g5') {
    Copy-Item "$F2\pkvg\bin\$x.exe" (Join-Path $Dest 'bin')
}
Copy-Item "$F2\adapters.exe" (Join-Path $Dest 'bin')
Copy-Item "$F2\pkvg\bin\shaders\*" (Join-Path $Dest 'bin\shaders') -Recurse
foreach ($dll in 'vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll') { Copy-Item (Join-Path $env:windir "System32\$dll") (Join-Path $Dest 'bin') }
Copy-Item 'D:\_artifacts\SuperSLM\t3035-precache-refs\fixtures\*' (Join-Path $Dest 'fixtures')
Copy-Item 'D:\_artifacts\SuperSLM\t3030-commission\must-*.sslm' (Join-Path $Dest 'e0')
Copy-Item 'D:\_artifacts\SuperSLM\te442-qwen2.5-0.5b-instruct-dg\qwen2.5-0.5b-instruct-dg.sslm' (Join-Path $Dest 'models')
Copy-Item 'D:\_artifacts\SuperSLM\t2901-qwen2.5-1.5b-dg\qwen2.5-1.5b-instruct-dg-g5.sslm' (Join-Path $Dest 'models')
Copy-Item 'D:\_scratch\pkv-gpu-113\adapters\*.sslm' (Join-Path $Dest 'adapters')
Copy-Item "$F2\blob-10de_1e81.spx1" (Join-Path $Dest 'blobs')
Copy-Item 'D:\SuperSLM\.worktrees\f2-2080\tests\paged-kv\reference\*' (Join-Path $Dest 'paged-kv-reference') -Recurse
# A manifest of every file's SHA-256, so the dev box can check the drive came back intact.
$files = Get-ChildItem -Recurse -File $Dest | Where-Object { $_.FullName -notlike "*\results\*" }
$files | ForEach-Object { '{0}  {1}' -f (Get-FileHash -Algorithm SHA256 $_.FullName).Hash.ToLower(), $_.FullName.Substring($Dest.Length + 1) } |
    Set-Content (Join-Path $Dest 'MANIFEST.sha256')
"kit at $Dest : $($files.Count) files, $([math]::Round(($files | Measure-Object Length -Sum).Sum / 1GB, 2)) GB"
