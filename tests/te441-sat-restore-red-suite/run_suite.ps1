# TE-441 -- runs the saturation-census cells, one process per job, and scores each by exit status.
#
# Usage:
#   pwsh -File run_suite.ps1 -Bin <build_suite.bat output> -Out <new or empty dir>
#        [-Qwen25 PATH] [-Qwen3 PATH] [-Blobs DIR] [-CpuV181PerSite saved|zero]
#        [-GpuV181PerSite saved|zero] [-Only pattern[,pattern]]
#
# Verdicts: GREEN (exit 0: every check held), RED (exit 1: at least one check failed), INVALID
# (exit 2: the cell could not decide -- a missing artifact, an unpinned blob, a prompt that did
# not saturate), CRASH (anything else). The at-fix column is the verdict each job must give at the
# fix: GREEN for every cell, RED for the mutant (vitality) leg. Each job's full output is
# <Out>\<job>.txt; the summary is <Out>\summary.txt. Jobs run one at a time, CPU before GPU.
#
# The v181 jobs read the pinned v1.8.1 blobs (te441_cpu_cells.cpp / te441_gpu_cells.cpp,
# kPinnedV181*Blobs) from -Blobs. -CpuV181PerSite (the 'SSB4' blob) and -GpuV181PerSite (the 'SLM5'
# blobs) are passed through once the builder has named what a restored v1.8.1 blob's per-site
# counts read; until then the v181 jobs assert only that the restore succeeds and the next tokens
# are unchanged.
param(
	[Parameter(Mandatory = $true)][string]$Bin,
	[Parameter(Mandatory = $true)][string]$Out,
	[string]$Qwen25 = 'D:/hf_cache/superslm_artifacts/qwen2.5-1.5b-instruct.sslm',
	[string]$Qwen3 = 'D:/_artifacts/superslm/_t2743conv/flow-final/qwen3-embedding-0.6b-1p5.sslm',
	[string]$Blobs = 'D:/_artifacts/superslm/te441-v181-blobs',
	[string]$CpuV181PerSite = '',
	[string]$GpuV181PerSite = '',
	[string]$Only = '*'
)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$cpu = Join-Path $Bin 'te441_cpu_cells.exe'
$gpu = Join-Path $Bin 'te441_gpu_cells.exe'
foreach ($p in @($cpu, $gpu, $Qwen25, $Qwen3)) {
	if (-not (Test-Path $p)) { throw "missing: $p" }
}
$cpuPersite = @()
if ($CpuV181PerSite) { $cpuPersite = @("--v181-persite=$CpuV181PerSite") }
$gpuPersite = @()
if ($GpuV181PerSite) { $gpuPersite = @("--v181-persite=$GpuV181PerSite") }

# Fix = the verdict every job must give at the fix. The mutant leg is the vitality check: GPU save
# and restore with the blob's per-site counters zeroed must be RED, or the GPU cells cannot fire.
$jobs = @(
	@{ Name = 'cpu-cells-qwen25'; Fix = 'GREEN'; Exe = $cpu; Args = @('cells', "--model=$Qwen25") },
	@{ Name = 'cpu-cells-qwen3'; Fix = 'GREEN'; Exe = $cpu; Args = @('cells', "--model=$Qwen3") },
	@{ Name = 'cpu-v181-qwen25'; Fix = 'GREEN'; Exe = $cpu; Args = @('v181', "--model=$Qwen25", "--blob=$Blobs/qwen2.5-1.5b-instruct.cpu-ssb4.blob") + $cpuPersite },
	@{ Name = 'gpu-cells-qwen25'; Fix = 'GREEN'; Exe = $gpu; Args = @('cells', "--model=$Qwen25") },
	@{ Name = 'gpu-cells-qwen3'; Fix = 'GREEN'; Exe = $gpu; Args = @('cells', "--model=$Qwen3") },
	@{ Name = 'gpu-mutant-qwen3'; Fix = 'RED'; Exe = $gpu; Args = @('cells', "--model=$Qwen3", '--mutant=zero-persite') },
	@{ Name = 'gpu-v181-qwen25'; Fix = 'GREEN'; Exe = $gpu; Args = @('v181', "--model=$Qwen25", "--blob=$Blobs/qwen2.5-1.5b-instruct.gpu-slm5.blob") + $gpuPersite },
	@{ Name = 'gpu-v181-qwen3'; Fix = 'GREEN'; Exe = $gpu; Args = @('v181', "--model=$Qwen3", "--blob=$Blobs/qwen3-embedding-0.6b-1p5.gpu-slm5.blob") + $gpuPersite }
)
$patterns = $Only -split ','
$summary = @()
foreach ($j in $jobs) {
	$match = $false
	foreach ($pat in $patterns) { if ($j.Name -like $pat) { $match = $true } }
	if (-not $match) { continue }
	$txt = Join-Path $Out "$($j.Name).txt"
	$t0 = Get-Date
	$output = & $j.Exe @($j.Args) 2>&1 | ForEach-Object { "$_" }
	$code = $LASTEXITCODE
	$secs = [int]((Get-Date) - $t0).TotalSeconds
	$verdict = switch ($code) { 0 { 'GREEN' } 1 { 'RED' } 2 { 'INVALID' } default { 'CRASH' } }
	$header = "JOB $($j.Name): $($j.Exe) $($j.Args -join ' ')"
	(@($header) + $output + @("EXIT $code -> $verdict ($secs s)")) | Set-Content -Encoding utf8 $txt
	$fails = @($output | Where-Object { $_ -like 'FAIL *' }).Count
	$line = '{0,-18} {1,-8} exit={2} failed_checks={3} at-fix={4} ({5} s)' -f $j.Name, $verdict, $code, $fails, $j.Fix, $secs
	Write-Output $line
	$summary += $line
}
$summary | Set-Content -Encoding utf8 (Join-Path $Out 'summary.txt')
