# T-2105 (Laplace, DISPOSABLE): the time-slice invariance sweep. One variable -- where the token's
# dispatch chain is cut into separately-submitted, separately-fenced command lists -- against the
# same CPU int8 oracle, per timed step.
param([int[]]$Every = @(0,1,3,7,13,24,101), [int]$Steps = 16,
      [string]$Model = "D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct.sslm")
$env:SSLM_T2105_LANES_QPROJ=32; $env:SSLM_T2105_LANES_OPROJ=32; $env:SSLM_T2105_LANES_DOWN=64
$env:SSLM_T2105_LANES_GATE=32;  $env:SSLM_T2105_LANES_UP=32;    $env:SSLM_T2105_LANES_KV=32
foreach ($k in $Every) {
  $env:SSLM_T2105_SLICE_EVERY = "$k"
  $o = & .\out\t2100_gpu_throughput.exe $Model $Steps 0 2>&1 | Out-String
  $tok  = [regex]::Match($o,'GPU \(RunLayerLoopGpu\)\s+[0-9.]+ s/token\s+([0-9.]+) tok/s').Groups[1].Value
  $eq   = [regex]::Match($o,'equality over (\d+) timed steps: (\w+) \((\d+) mismatched\)').Groups
  $cuts = [regex]::Match($o,'([0-9]+) suspend/resume cuts').Groups[1].Value
  "slice_every={0,4}  cuts/step={1,5}  {2,6} tok/s   equality={3} ({4} mismatched over {5} steps)" -f `
    $k, $cuts, $tok, $eq[2].Value, $eq[3].Value, $eq[1].Value
}
$env:SSLM_T2105_SLICE_EVERY = "0"
