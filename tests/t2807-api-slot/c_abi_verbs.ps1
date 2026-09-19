# T-2825 (Curie) -- the C ABI's declared verbs, read from the header (plan Sec3.7 item 3, T-2823 fold).
# Dot-source it: `. $PSScriptRoot\c_abi_verbs.ps1`, then `Get-CAbiVerbs <include\superslm dir>`.
#
# Every C verb the ABI declares lives in exactly two files, include/superslm/sslm_abi_functions.inc and
# sslm_abi_functions_g5_comparable.inc (sslm_abi.h includes the first inside its extern "C" block, and the
# first includes the second). A verb is a DECLARATION: after /* ... */ and // comments are blanked (line
# numbers kept), a line whose first token is an optional SUPERSLM_API, then a return type (an identifier,
# optionally `const`, optionally followed by `*`), then `sslm_<name>(`. A mention of a verb in a comment,
# in another verb's argument list or in a macro line is never a declaration. The slot prefix is optional,
# so the same set is read from a tree with or without the slot, and from an X3 mutant with one slot dropped.
#
# Returns objects { Name; File; Line } sorted by Name (ordinal). Throws when a name is declared twice or
# when either file is missing -- a list read from half the ABI is refused, not returned.
function Get-CAbiVerbs([string]$IncDir) {
    $files = @('sslm_abi_functions.inc', 'sslm_abi_functions_g5_comparable.inc')
    $decl = [regex]'^\s*(?:SUPERSLM_API\s+)?(?:const\s+)?[A-Za-z_][A-Za-z0-9_]*\s*\**\s+\**\s*(sslm_[a-z0-9_]+)\s*\('
    $out = @()
    $seen = @{}
    foreach ($f in $files) {
        $p = Join-Path $IncDir $f
        if (-not (Test-Path $p)) { throw "Get-CAbiVerbs: $p not found" }
        $text = [System.IO.File]::ReadAllText($p)
        # Blank comments, keeping every newline so line numbers stay true.
        $blank = [regex]::Replace($text, '(?s)/\*.*?\*/', { param($m) [regex]::Replace($m.Value, '[^\n]', ' ') })
        $blank = [regex]::Replace($blank, '//[^\n]*', '')
        $n = 0
        foreach ($l in ($blank -split "`n")) {
            $n++
            if ($l -match '^\s*#') { continue }
            $m = $decl.Match($l)
            if ($m.Success) {
                $name = $m.Groups[1].Value
                if ($seen.ContainsKey($name)) { throw "Get-CAbiVerbs: $name declared twice ($($seen[$name]) and ${f}:$n)" }
                $seen[$name] = "${f}:$n"
                $out += [pscustomobject]@{ Name = $name; File = $f; Line = $n }
            }
        }
    }
    $sorted = New-Object 'System.Collections.Generic.SortedDictionary[string,object]' ([System.StringComparer]::Ordinal)
    foreach ($o in $out) { $sorted[$o.Name] = $o }
    return @($sorted.Values)
}
