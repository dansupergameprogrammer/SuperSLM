. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$out = Join-Path $E4Refs 'P1-real0p5-flat'; New-Item -ItemType Directory -Force $out | Out-Null
$tok = 'TokenizerView::Encode(rel10,4da4c4c)'
foreach ($run in 'run1', 'run2') {
  E4-Run (E4-Log "B3-P1-$run") $E4Gen @($E4Model05, $E4Prompts, $out, 'flat', $run, $tok, 'ids', (Join-Path $E4Disc 'real0p5-flat'))
}
