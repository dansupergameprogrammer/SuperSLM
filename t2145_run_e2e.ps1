# T-2145 (Laplace) -- disposable end-to-end measurement driver. Scratch, never product tooling.
#
# Runs the REAL production decode driver (sslm_generate) on a REAL .sslm artifact and reports
# what a consumer would actually receive. Per-token decode cost is isolated by differencing two
# token counts against the SAME prompt, so the fixed cost (artifact read, marshal, prefill) --
# which is identical across the two runs -- cancels:
#
#     s_per_token = (wall(N_hi) - wall(N_lo)) / (N_hi - N_lo)
#
# Every repetition is an INDEPENDENT process invocation, matching the protocol the GPU cell used.
# Every repetition is reported; nothing is filtered to a best run.

param(
	[string]$Model,
	[string]$Tag,
	[int]$Nlo = 8,
	[int]$Nhi = 32,
	[int]$Reps = 3,
	[string[]]$Variants = @("scalar","sse2","avx2"),
	[string]$Prompt = "The shopkeeper looks up from the counter and says",
	[switch]$DumpLogits
)

$tok = "tests\fixtures\qwen2.5-1.5b.tok.sslm"
$results = @()

foreach ($v in $Variants) {
	$exe = ".\out\t2145\sslm_generate_$v.exe"
	foreach ($n in @($Nlo, $Nhi)) {
		for ($r = 0; $r -lt $Reps; $r++) {
			$args = @($Model, $tok, $Prompt, "--max-new", $n)
			if ($DumpLogits -and $n -eq $Nhi -and $r -eq 0) {
				$args += @("--dump-logits", "out\t2145\logits_${Tag}_$v.bin")
			}
			$out = & $exe @args 2>&1
			$wall = ($out | Select-String -Pattern "^wall_time_seconds: (.*)$").Matches.Groups[1].Value
			$toks = ($out | Select-String -Pattern "^output_tokens \(\d+\):(.*)$").Matches.Groups[1].Value.Trim()
			$results += [pscustomobject]@{
				variant = $v; n = $n; rep = $r; wall = [double]$wall; tokens = $toks
			}
			Write-Host ("run tag=$Tag variant=$v max_new=$n rep=$r wall_s=$wall")
		}
	}
}

Write-Host ""
Write-Host "=== PER-TOKEN DECODE COST (tag=$Tag, N_hi=$Nhi minus N_lo=$Nlo) ==="
foreach ($v in $Variants) {
	$lo = $results | Where-Object { $_.variant -eq $v -and $_.n -eq $Nlo }
	$hi = $results | Where-Object { $_.variant -eq $v -and $_.n -eq $Nhi }
	# Every lo/hi pairing, so the spread reported is the spread of the DERIVED quantity,
	# not the spread of one side of it.
	$per = @()
	foreach ($a in $lo) { foreach ($b in $hi) { $per += ($b.wall - $a.wall) / ($Nhi - $Nlo) } }
	$min = ($per | Measure-Object -Minimum).Minimum
	$max = ($per | Measure-Object -Maximum).Maximum
	$avg = ($per | Measure-Object -Average).Average
	Write-Host ("{0,-7} s_per_token min={1:F4} max={2:F4} mean={3:F4}  ->  tok_per_s min={4:F2} max={5:F2} mean={6:F2}" -f `
		$v, $min, $max, $avg, (1/$max), (1/$min), (1/$avg))
}

Write-Host ""
Write-Host "=== OUTPUT TOKEN IDENTITY (all variants must agree exactly) ==="
$byVariant = @{}
foreach ($v in $Variants) {
	$byVariant[$v] = ($results | Where-Object { $_.variant -eq $v -and $_.n -eq $Nhi } | Select-Object -First 1).tokens
	Write-Host ("$v : " + $byVariant[$v])
}
$distinct = ($byVariant.Values | Sort-Object -Unique)
Write-Host ("distinct_token_sequences=" + $distinct.Count + "  (1 == every variant produced identical output)")
