param(
    [string]$R05 = 'D:\hf_cache\superslm_artifacts\example\qwen2.5-0.5b-instruct-cap4096-aex-tok.sslm',
    [string]$R15 = 'D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct.sslm',
    [string]$Adapter = 'D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-shopkeeper-lora-v2-t2102-runtime.sslm',
    [string]$Shaders = 'D:\_slm170\build\t2957-seams\gpu-shaders-staged'
)
$ErrorActionPreference = 'Stop'
$suite = Split-Path -Parent $MyInvocation.MyCommand.Path
$out = Join-Path $suite 'out\green'
New-Item -ItemType Directory -Path $out -Force | Out-Null
$bin = Join-Path $suite 'out'
$ru = Join-Path $suite 'out\ru\ru.sslm'
$missing = Join-Path $suite 'out\missing-shader'
New-Item -ItemType Directory -Path $missing -Force | Out-Null
Copy-Item -Path (Join-Path $Shaders '*.cso') -Destination $missing -Force
Remove-Item -LiteralPath (Join-Path $missing 'logits_site.cso') -ErrorAction SilentlyContinue
$cases = @(
    @('cell_status_ordinals'),
    @('cell_flags',$R05),
    @('cell_setters',$R05),
    @('cell_setter_atomicity',$R05),
    @('cell_rows',$R05),
    @('cell_rows',$R15),
    @('cell_rows',$ru),
    @('cell_device_logits',$R05,$R15,$ru),
    @('cell_alloc_faults',$R15,$Adapter),
    @('cell_residency',$R05,$R15,$ru),
    @('cell_hook_lifecycle',$R05),
    @('cell_gpu_hook_lifecycle',$R05),
    @('cell_batch_four',$R05),
    @('cell_close_faults'),
    @('cell_missing_shader',$R05,$missing)
)
for ($index = 0; $index -lt $cases.Count; ++$index) {
    $case = $cases[$index]
    $name = $case[0]
    $arguments = @()
    if ($case.Count -gt 1) { $arguments = $case[1..($case.Count-1)] }
    $exe = Join-Path $bin "$name.exe"
    $log = Join-Path $out ("{0:D2}_{1}.txt" -f $index,$name)
    $ErrorActionPreference = 'Continue' # fault cells intentionally write diagnostics on stderr
    & $exe @arguments *> $log
    $status = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    Write-Output "$index $name exit=$status"
    if ($status -ne 0) { Get-Content $log -Tail 20; exit $status }
}
Write-Output "PASS green cells=$($cases.Count)"
