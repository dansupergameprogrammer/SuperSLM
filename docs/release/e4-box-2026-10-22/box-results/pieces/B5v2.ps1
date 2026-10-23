. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$log = E4-Log 'B5-pairs'; $bad = 0
Get-ChildItem $E4Refs -Recurse -Filter '*-run1.ref' | ForEach-Object {
  $twin = $_.FullName -replace '-run1\.ref$', '-run2.ref'
  $a = (Get-FileHash $_.FullName).Hash; $b = (Get-FileHash $twin).Hash
  if ($a -ne $b) { $bad++ }
  "{0} {1} {2}" -f $a, ($(if ($a -eq $b) { 'pair-equal' } else { 'PAIR-DIFFERS' })), $_.FullName.Substring($E4Refs.Length + 1)
} | ForEach-Object { E4-Append $log $_; $_ }
"pairs differing: $bad" | ForEach-Object { E4-Append $log $_; $_ }
$E4Record = '{id}.stream.json'
$V2 = Join-Path $E4T3035 'tests\t3035-precache-refs\discrimination-v2'; $unbound = 0
Get-ChildItem $E4Refs -Recurse -Filter '*-run1.ref' | ForEach-Object {
  $stem = $_.Name -replace '-run1\.ref$', ''; $grp = $_.Directory.Name
  $want = (Get-FileHash (Join-Path $V2 ("$grp\" + $E4Record.Replace('{id}', $stem)))).Hash.ToLower()
  $got = ([regex]'discrimination_record_sha256=([0-9a-f]{64})').Match((Get-Content $_.FullName -TotalCount 1)).Groups[1].Value
  if ($got -ne $want) { $unbound++ }
  "{0} {1} {2}" -f $got, ($(if ($got -eq $want) { 'binds-v2' } else { 'NOT-V2' })), $stem
} | ForEach-Object { E4-Append $log $_; $_ }
"headers not bound to their v2 sidecar: $unbound" | ForEach-Object { E4-Append $log $_; $_ }
