. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$log = E4-Log 'B5-pairs'; $bad = 0
Get-ChildItem $E4Refs -Recurse -Filter '*-run1.ref' | ForEach-Object {
  $twin = $_.FullName -replace '-run1\.ref$', '-run2.ref'
  $a = (Get-FileHash $_.FullName).Hash; $b = (Get-FileHash $twin).Hash
  if ($a -ne $b) { $bad++ }
  "{0} {1} {2}" -f $a, ($(if ($a -eq $b) { 'pair-equal' } else { 'PAIR-DIFFERS' })), $_.FullName.Substring($E4Refs.Length + 1)
} | ForEach-Object { E4-Append $log $_; $_ }
"pairs differing: $bad" | ForEach-Object { E4-Append $log $_; $_ }
