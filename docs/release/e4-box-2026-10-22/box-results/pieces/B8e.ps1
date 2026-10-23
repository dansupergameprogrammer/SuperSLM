. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$log = E4-Log 'B8e-coverage'; Remove-Item $log -ErrorAction SilentlyContinue
$logs = '0-4', '5-7', '8-8', '9-9' | ForEach-Object { E4-Log ('B8-timing-' + $_) }
$lines = Select-String -Path $logs -Pattern '^10\.3 receipt=(\w+) runner=(\w+) .* (flat\d\d) .*ratio95=\[([\d.]+),'
$slices = Select-String -Path $logs -Pattern '^10\.3 slice flat\d\d-flat\d\d: (\d+) of (\d+) .* runner=(\w+) receipt=(\w+)'
$names = @($lines | ForEach-Object { $_.Matches[0].Groups[3].Value })
$checks = [ordered]@{
  'ten distinct prefixes 00-09' = (($names | Sort-Object -Unique) -join ',') -eq ((0..9 | ForEach-Object { 'flat{0:D2}' -f $_ }) -join ',') -and $names.Count -eq 10
  'every lower >= 1000'         = @($lines | Where-Object { [double]$_.Matches[0].Groups[4].Value -lt 1000 }).Count -eq 0
  'one receipt'                 = @(@($lines | ForEach-Object { $_.Matches[0].Groups[1].Value }) + @($slices | ForEach-Object { $_.Matches[0].Groups[4].Value }) | Sort-Object -Unique).Count -eq 1
  'one runner'                  = @(@($lines | ForEach-Object { $_.Matches[0].Groups[2].Value }) + @($slices | ForEach-Object { $_.Matches[0].Groups[3].Value }) | Sort-Object -Unique).Count -eq 1
  'four slices, each N of N'    = $slices.Count -eq 4 -and @($slices | Where-Object { $_.Matches[0].Groups[1].Value -ne $_.Matches[0].Groups[2].Value }).Count -eq 0
}
$checks.GetEnumerator() | ForEach-Object { $l = '{0}: {1}' -f $_.Key, $(if ($_.Value) { 'PASS' } else { 'FAIL' }); E4-Append $log $l; $l }
'runner = ' + (Get-FileHash $E4Runner).Hash.ToLower() | ForEach-Object { E4-Append $log $_; $_ }
