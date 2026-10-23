. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$E4Record = '{id}.stream.json'
$Gen = Join-Path $E4Work 'build-gen\superslm_precache_reference_gen.exe'
$V2 = Join-Path $E4T3035 'tests\t3035-precache-refs\discrimination-v2'
$refused = "$E4Refs-v1bound-refused"
if ((Test-Path $E4Refs) -and -not (Test-Path $refused)) { Move-Item $E4Refs $refused }
$out = Join-Path $E4Refs 'P1-real0p5-flat'; New-Item -ItemType Directory -Force $out | Out-Null
$tok = 'TokenizerView::Encode(rel10,4da4c4c)'
foreach ($run in 'run1', 'run2') {
  E4-Run (E4-Log "B3-P1-$run") $Gen @($E4Model05, $E4Prompts, $out, 'flat', $run, $tok, 'ids', (Join-Path $V2 "P1-real0p5-flat\$E4Record"))
}
