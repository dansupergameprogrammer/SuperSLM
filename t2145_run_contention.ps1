# T-2145 (Laplace) -- disposable contention driver. Scratch, never product tooling.
#
# Measures BOTH directions of the contention question, because the deployment cost lands on both:
#   (a) what a memory-bandwidth competitor does to decode throughput, and
#   (b) what the model's weight sweep does to the competitor -- the cost the host pays.
#
# Protocol: competitor alone -> decode alone -> both concurrently. The competitor reports its own
# achieved GB/s once per second throughout, so its degradation is read from the same run in which
# the decode's degradation is read. Every sample is reported.

param(
	[string]$Model,
	[string]$Tag,
	[int]$CompThreads = 4,
	[int]$Reps = 2,
	[string]$Variant = "avx2"
)

$tok  = "tests\fixtures\qwen2.5-1.5b.tok.sslm"
$bench = ".\out\t2145\t2145_bench_$Variant.exe"
$gen   = ".\out\t2145\sslm_generate_$Variant.exe"
$prompt = "The shopkeeper looks up from the counter and says"

function Get-DecodeSeconds([int]$n) {
	$out = & $gen $Model $tok $prompt "--max-new" $n 2>&1
	return [double]($out | Select-String -Pattern "^wall_time_seconds: (.*)$").Matches.Groups[1].Value
}

Write-Host "=== PHASE 1: competitor ALONE ($CompThreads threads) ==="
$alone = & $bench compete 4096 $CompThreads 14 2>&1 | Select-String -Pattern "^compete t="
$alone | ForEach-Object { Write-Host "  $_" }
$aloneVals = $alone | ForEach-Object { [double](($_ -split "GB_per_s=")[1]) }
$aloneMean = ($aloneVals | Measure-Object -Average).Average
Write-Host ("  competitor_alone_GB_per_s mean={0:F3} min={1:F3} max={2:F3}" -f $aloneMean, `
	($aloneVals | Measure-Object -Minimum).Minimum, ($aloneVals | Measure-Object -Maximum).Maximum)

Write-Host ""
Write-Host "=== PHASE 2: decode ALONE ($Tag, $Variant) ==="
$soloPer = @()
for ($r = 0; $r -lt $Reps; $r++) {
	$lo = Get-DecodeSeconds 8
	$hi = Get-DecodeSeconds 32
	$p = ($hi - $lo) / 24.0
	$soloPer += $p
	Write-Host ("  rep=$r wall8=$lo wall32=$hi s_per_token={0:F4} tok_per_s={1:F2}" -f $p, (1/$p))
}

Write-Host ""
Write-Host "=== PHASE 3: decode WITH competitor running concurrently ==="
$compOut = "out\t2145\compete_during_$Tag.txt"
$proc = Start-Process -FilePath $bench -ArgumentList @("compete","4096","$CompThreads","600") `
	-RedirectStandardOutput $compOut -PassThru -NoNewWindow
Start-Sleep -Seconds 4
$contPer = @()
for ($r = 0; $r -lt $Reps; $r++) {
	$lo = Get-DecodeSeconds 8
	$hi = Get-DecodeSeconds 32
	$p = ($hi - $lo) / 24.0
	$contPer += $p
	Write-Host ("  rep=$r wall8=$lo wall32=$hi s_per_token={0:F4} tok_per_s={1:F2}" -f $p, (1/$p))
}
Stop-Process -Id $proc.Id -Force
Start-Sleep -Seconds 1
$during = (Get-Content $compOut | Select-String -Pattern "^compete t=") | ForEach-Object { [double](($_ -split "GB_per_s=")[1]) }
# Drop the first 3 samples: they cover the 4 s before the decode started.
$during = $during | Select-Object -Skip 3
$duringMean = ($during | Measure-Object -Average).Average

Write-Host ""
Write-Host "=== RESULT ($Tag, $Variant, competitor=$CompThreads threads) ==="
$soloMean = ($soloPer | Measure-Object -Average).Average
$contMean = ($contPer | Measure-Object -Average).Average
Write-Host ("  decode alone      : {0:F4} s/token  ({1:F2} tok/s)" -f $soloMean, (1/$soloMean))
Write-Host ("  decode contended  : {0:F4} s/token  ({1:F2} tok/s)" -f $contMean, (1/$contMean))
Write-Host ("  decode slowdown   : {0:F2}x" -f ($contMean / $soloMean))
Write-Host ("  competitor alone  : {0:F3} GB/s" -f $aloneMean)
Write-Host ("  competitor during : {0:F3} GB/s  (n={1} samples)" -f $duringMean, $during.Count)
Write-Host ("  competitor loss   : {0:F1}%" -f ((1 - $duringMean/$aloneMean) * 100))
