# T-2105 (Laplace, DISPOSABLE): lane sweep. One variable (the lane split), everything else fixed.
param(
  [int[]]$Lanes = @(8,16,32,64,128,256),
  [int]$Steps = 32,
  [string]$Model = "D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct.sslm"
)
foreach ($L in $Lanes) {
  $env:SSLM_T2105_LANES_QPROJ=$L; $env:SSLM_T2105_LANES_OPROJ=$L
  $env:SSLM_T2105_LANES_DOWN=$L;  $env:SSLM_T2105_LANES_GATE=$L; $env:SSLM_T2105_LANES_UP=$L
  $out = & .\out\t2100_gpu_throughput.exe $Model $Steps 0 2>&1 | Out-String
  $tok = ([regex]::Match($out,'GPU \(RunLayerLoopGpu\)\s+([0-9.]+) s/token\s+([0-9.]+) tok/s')).Groups
  $eq  = ([regex]::Match($out,'per-step CPU/GPU equality over (\d+) timed steps: (\w+) \((\d+) mismatched\)')).Groups
  $busy= ([regex]::Match($out,'GPU-busy\s+([0-9.]+) ms')).Groups[1].Value
  $gemm = @()
  foreach ($site in @('down_proj_gemm','gate_proj_gemm','up_proj_gemm','q_proj_gemm','o_proj_gemm')) {
    $m = [regex]::Match($out, "  $site\s+([0-9.]+)\s+[0-9.]+%\s+[0-9.]+\s+([0-9.]+)")
    if ($m.Success) { $gemm += ("{0}={1}ms/{2}GBs" -f $site.Replace('_proj_gemm',''), $m.Groups[1].Value, $m.Groups[2].Value) }
  }
  "lanes={0,3}  {1} tok/s  busy={2}ms  equality={3}({4} mismatched over {5})  {6}" -f `
    $L, $tok[2].Value, $busy, $eq[2].Value, $eq[3].Value, $eq[1].Value, ($gemm -join ' ')
}
