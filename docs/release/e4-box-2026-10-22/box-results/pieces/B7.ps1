. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
0..9 | ForEach-Object { $f = Join-Path $E4Prompts ('flat{0:D2}.ids' -f $_)
  '{0}  flat{1:D2}.ids' -f (Get-FileHash $f).Hash.ToLower(), $_ } |
  Set-Content -Encoding ascii $env:SUPERSLM_PRECACHE_TIMING_PROMPT_MANIFEST
Get-Content $env:SUPERSLM_PRECACHE_TIMING_PROMPT_MANIFEST
(Get-FileHash $env:SUPERSLM_PRECACHE_TIMING_INSTRUMENT).Hash.ToLower()
