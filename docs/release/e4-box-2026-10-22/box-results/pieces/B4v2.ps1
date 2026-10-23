. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$E4Record = '{id}.stream.json'
$Gen = Join-Path $E4Work 'build-gen\superslm_precache_reference_gen.exe'
$V2 = Join-Path $E4T3035 'tests\t3035-precache-refs\discrimination-v2'
$out = Join-Path $E4Refs 'P2-real0p5-nested'; New-Item -ItemType Directory -Force $out | Out-Null
$tok = 'TokenizerView::Encode(rel10,4da4c4c)'
foreach ($run in 'run1', 'run2') {
  E4-Run (E4-Log "B4-P2-$run") $Gen @($E4Model05, $E4Prompts, $out, 'nested', $run, $tok, 'ids', (Join-Path $V2 "P2-real0p5-nested\$E4Record"))
}
