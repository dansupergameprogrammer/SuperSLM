param([Parameter(Mandatory=$true)][string]$Id)
$ErrorActionPreference = 'Continue' # MSVC setup and deliberate fault cells write to stderr
$suite = Split-Path -Parent $MyInvocation.MyCommand.Path
$tree = 'D:\_t2956-mutants\src'
$log = Join-Path 'D:\_t2956-mutants\logs' $Id
New-Item -ItemType Directory -Path $log -Force | Out-Null
$python = 'C:\Users\dansu\AppData\Local\Programs\Python\Python313\python.exe'
$aex = 'D:\hf_cache\superslm_artifacts\example\qwen2.5-0.5b-instruct-cap4096-aex-tok.sslm'
$r15 = 'D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct.sslm'
$adapter = 'D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-shopkeeper-lora-v2-t2102-runtime.sslm'
$ru = Join-Path $suite 'out\ru\ru.sslm'
$base = Join-Path $suite 'out\cell_real_decode_base.exe'
$cpu = Join-Path $tree 'build\t2956-mutant\superslm.lib'
$gpu = Join-Path $tree 'build\t2956-mutant\superslm_gpu.lib'
$shaders = Join-Path $tree 'build\t2956-mutant\gpu-shaders-staged'
$bin = Join-Path 'D:\_t2956-mutants\bin' $Id

& $python (Join-Path $suite 'apply_mutant.py') $Id *> (Join-Path $log 'mutation.txt')
if ($LASTEXITCODE -ne 0) { Get-Content (Join-Path $log 'mutation.txt'); exit 2 }
if ($Id -in @('q_cpu','q_gpu')) {
    $out = Join-Path 'D:\_t2956-mutants\build' "allmasked_$Id"
    & cmd /c (Join-Path $suite 'build_all_masked.bat') $tree $out candidate *> (Join-Path $log 'build.txt')
    if ($LASTEXITCODE -ne 0) { Get-Content (Join-Path $log 'build.txt') -Tail 30; exit 3 }
    $env:T2956_ALL_MASKED = '1'
    $env:T2956_SHADER_DIR = Join-Path $out 'shaders'
    $backend = if ($Id -eq 'q_cpu') { 'cpu' } else { 'gpu' }
    & (Join-Path $out 'cell_real_decode.exe') $backend $aex 'potion_shop_order' '1' '0' '0' *> (Join-Path $log 'cell.txt')
    $cellExit = $LASTEXITCODE
    Remove-Item Env:T2956_ALL_MASKED
    Remove-Item Env:T2956_SHADER_DIR
    Get-Content (Join-Path $log 'mutation.txt')
    Write-Output "MUTANT $Id build=0 link=0 cell_exit=$cellExit"
    Get-Content (Join-Path $log 'cell.txt')
    if ($cellExit -eq 0 -or (Get-Content -Raw (Join-Path $log 'cell.txt')) -notmatch "FAIL all-masked $backend") { exit 1 }
    exit 0
}
& cmd /c (Join-Path $suite 'build_mutant.bat') $tree *> (Join-Path $log 'build.txt')
if ($LASTEXITCODE -ne 0) { Get-Content (Join-Path $log 'build.txt') -Tail 30; exit 3 }

if ($Id -in @('a','c')) { $cell = 'cell_rows'; $cellArgs = @($aex) }
elseif ($Id -eq 'c_exact') { $cell = 'cell_real_decode'; $cellArgs = @() }
elseif ($Id -in @('b','b2')) {
    $cell = 'cell_real_decode'
    $kind = if ($Id -eq 'b') { 'omit0' } else { 'dupinplace' }
    $cellArgs = @('cpu',$aex,'-','1','3','0','-',$kind)
}
elseif ($Id -in @('d','o2')) { $cell = 'cell_device_logits'; $cellArgs = @($aex,$r15,$ru) }
elseif ($Id -in @('e','m')) { $cell = 'cell_residency'; $cellArgs = @($aex,$r15,$ru) }
elseif ($Id -eq 'f') { $cell = 'cell_flags'; $cellArgs = @($aex) }
elseif ($Id -eq 'g') { $cell = 'cell_hook_lifecycle'; $cellArgs = @($aex) }
elseif ($Id -in @('h','h2')) { $cell = 'cell_real_decode'; $cellArgs = @() }
elseif ($Id -in @('i','o')) { $cell = 'cell_real_decode'; $cellArgs = @('gpu',$aex,'-','1','0','1','-') }
elseif ($Id -in @('j','k','n','n2_upload','n3_upload','n2_restore','n3_restore') -or $Id -match '^l[1-6]$') {
    $cell = 'cell_alloc_faults'; $cellArgs = @($r15,$adapter)
    if ($Id -match '^l([1-6])$') { $cellArgs += @('model_head',$Matches[1]) }
    if ($Id -eq 'n') { $cellArgs += @('model_head','1') }
    if ($Id -eq 'n2_upload') { $cellArgs += @('model_clear','1') }
    if ($Id -eq 'n3_upload') { $cellArgs += @('model_clear','2') }
    if ($Id -eq 'n2_restore') { $cellArgs += @('sequence_restore','5') }
    if ($Id -eq 'n3_restore') { $cellArgs += @('sequence_restore','4') }
}
elseif ($Id -eq 'p') { $cell = 'cell_close_faults'; $cellArgs = @() }
else { throw "No cell mapping for mutant $Id" }

& cmd /c (Join-Path $suite 'build_one.bat') $cell $cpu $gpu $bin $shaders *> (Join-Path $log 'link.txt')
if ($LASTEXITCODE -ne 0) { Get-Content (Join-Path $log 'link.txt') -Tail 30; exit 4 }
$exe = Join-Path $bin "$cell.exe"
if ($Id -eq 'c_exact') {
    & $python (Join-Path $suite 'run_integrated_tie.py') --base $base --candidate $exe *> (Join-Path $log 'cell.txt')
}
elseif ($Id -in @('h','h2')) {
    $compare = Join-Path $suite 'cell_ru_tokens.py'
    if ($Id -eq 'h') { & $python $compare --base $base --candidate $exe --device *> (Join-Path $log 'cell.txt') }
    else { & $python $compare --base $base --candidate $exe *> (Join-Path $log 'cell.txt') }
}
else {
    if ($Id -eq 'i') { $env:T2956_OVERFLOW = '1' }
    & $exe @cellArgs *> (Join-Path $log 'cell.txt')
    if ($Id -eq 'i') { Remove-Item Env:T2956_OVERFLOW }
}
$cellExit = $LASTEXITCODE
Get-Content (Join-Path $log 'mutation.txt')
Write-Output "MUTANT $Id build=0 link=0 cell_exit=$cellExit"
$cellText = Get-Content -Raw (Join-Path $log 'cell.txt')
($cellText -split "`r?`n") | Where-Object { $_ -match 'FAIL ' } | Select-Object -First 3
$expected = switch -Regex ($Id) {
    '^a$' { 'FAIL rows vector=0 max_tasks=3'; break }
    '^b$' { 'FAIL malformed CPU omit0'; break }
    '^b2$' { 'FAIL malformed CPU dupinplace'; break }
    '^c$' { 'FAIL tie '; break }
    '^c_exact$' { 'FAIL integrated tie cpu/T3 got=\[64'; break }
    '^d$' { 'FAIL wide row '; break }
    '^e$' { 'FAIL VRAM '; break }
    '^f$' { 'FAIL unknown bit '; break }
    '^g$' { 'FAIL schema-content prefill hook calls'; break }
    '^h2?$' { 'FAIL RU '; break }
    '^i$' { 'FAIL overflow status=0'; break }
    '^j$' { 'FAIL model_head k=7 got=8 want=16'; break }
    '^k$' { 'FAIL removed model_head '; break }
    '^l[1-6]$' { 'FAIL model_head k='; break }
    '^m$' { 'FAIL head copy '; break }
    '^n$' { 'FAIL retry model_head '; break }
    '^n[23]_' { 'FAIL retry '; break }
    '^o$' { 'FAIL finish device allocations='; break }
    '^o2$' { 'FAIL device run .*allocations=6'; break }
    '^p$' { 'FAIL Close creation:'; break }
    default { throw "No expected failure for $Id" }
}
if ($cellExit -eq 0 -or $cellText -notmatch $expected) {
    Write-Output "UNEXPECTED MUTANT READING expected=$expected"
    Get-Content (Join-Path $log 'cell.txt') -Tail 20
    exit 1
}
exit 0
