# E4 L1 (debunker): the loader's must-accept and must-rejects on scratch copies of the population.
#   -Step accept : capture + restore of 10.1 and 10.2 through run_real_two_process.py (fresh blobs), exact copy
#   -Step edit   : restore-only 10.1 against the edited, re-licensed copy (blobs from the accept capture)
#   -Step v1     : restore-only 10.1 against the copy licensed over the v1 record hashes
param([string]$Step)
. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$py = 'C:\Users\dansu\AppData\Local\Programs\Python\Python312\python.exe'
$w = Join-Path $E4Work 'l1c'; New-Item -ItemType Directory -Force $w | Out-Null
$cand = Join-Path $E4Work 'l1\candidate'
$dest = Join-Path $w ('pop-' + $Step)
$log = E4-Log ('L1c-' + $Step)
Remove-Item $log -ErrorAction SilentlyContinue
$mb = Join-Path $w 'merge-base.txt'
E4-Run $log $py @('-I', (Join-Path $E4Src 'tests\precache\write_merge_base.py'), $E4Src, 'origin/main', $mb) | Out-Null
$vargs = @('-I', (Join-Path $E4Work 'l1\variant.py'), $E4Refs, $cand, $dest, $Step)
if ($Step -eq 'v1') { $vargs += (Join-Path $E4T3035 'tests\t3035-precache-refs\discrimination') }
E4-Run $log $py $vargs | Out-Null
$env:SUPERSLM_PRECACHE_MERGE_BASE_FILE = $mb
$env:SUPERSLM_PRECACHE_REFERENCE_DIR = $dest
$env:SUPERSLM_PRECACHE_REFERENCE_MANIFEST = Join-Path $dest 'manifest.json'
$env:SUPERSLM_PRECACHE_DISCRIMINATION_RECEIPT = Join-Path $dest 'license.txt'
$env:SUPERSLM_PRECACHE_BLOB_DIR = Join-Path $w 'blobs'
if ($Step -eq 'accept') {
    E4-Run $log $py @('-I', (Join-Path $E4Src 'tests\precache\run_real_two_process.py'), $E4Runner) | Out-Null
} else {
    $env:SUPERSLM_PRECACHE_PHASE = 'restore'
    $env:SUPERSLM_PRECACHE_CAPTURE_MANIFEST = Join-Path $env:SUPERSLM_PRECACHE_BLOB_DIR 'capture-manifest.json'
    E4-Run $log $E4Runner @('=10.1/E4') | Out-Null
}
