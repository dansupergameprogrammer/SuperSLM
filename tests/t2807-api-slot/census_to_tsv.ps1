# T-2825, T-2843 (Curie) -- the mechanical conversion of a recorded object census into the recorded-read TSV that
# gen_consumed_symbols.ps1 -Recorded takes (plan Sec3.7 item 3: a recorded read is admitted only as a mechanical,
# hash-pinned conversion of a recorded census). The generator dot-sources this file and re-runs the conversion on
# every -Recorded input, so a TSV row that the census does not produce refuses the input.
#
# The census format is the planner's census.ps1 output (Claude/Vitruvius/t2823-probe/census-output.txt): one
# "  [<module>] <object>.obj (<time>): <n>" line per importing object, then its imported engine names, each on a
# line indented six spaces. A TSV row is <module> TAB <object> TAB <name>, one per (object, name), in the census's
# own order. Lines starting with '#' in a TSV are provenance comments; the header names the census's SHA-256.
#
# Usage (write a TSV):  census_to_tsv.ps1 -FromCensus <census.txt> -ToTsv <tsv> [-Note '<provenance sentence>']
# Dot-sourced:          ConvertFrom-ObjectCensus <census.txt>  -> the rows, without the header
param(
    [string]$FromCensus = '',
    [string]$ToTsv = '',
    [string]$Note = ''
)

function ConvertFrom-ObjectCensus([string]$Path) {
    $rows = New-Object System.Collections.Generic.List[string]
    $module = $null; $obj = $null
    foreach ($l in (Get-Content -LiteralPath $Path)) {
        if ($l -match '^\s{2}\[(\S+)\]\s+(\S+\.obj)\s+\(') { $module = $Matches[1]; $obj = $Matches[2]; continue }
        if ($obj -and $l -match '^\s{6}(\S+)\s*$') { $rows.Add("$module`t$obj`t$($Matches[1])"); continue }
        if ($l -notmatch '^\s') { $obj = $null }
    }
    return $rows.ToArray()
}

if ($FromCensus -and $ToTsv) {
    $ErrorActionPreference = 'Stop'
    $sha = (Get-FileHash -Algorithm SHA256 -LiteralPath $FromCensus).Hash.ToLower()
    $rows = ConvertFrom-ObjectCensus $FromCensus
    $head = @("# Converted mechanically by tests/t2807-api-slot/census_to_tsv.ps1 from the census $FromCensus",
              "# (sha256 $sha).")
    if ($Note) { $head += "# $Note" }
    [System.IO.File]::WriteAllLines($ToTsv, [string[]]($head + $rows))
    Write-Output "wrote $ToTsv : $($rows.Count) rows from census sha256 $sha"
}
