# T-2825, T-2835 (Curie) -- the C ABI's declared verbs, read from the header (plan Sec3.7 item 3, T-2823 fold;
# the second declaration form and the split-line form by T-2830 / T-2824 G6 / T-2825 F-3).
# Dot-source it: `. $PSScriptRoot\c_abi_verbs.ps1`, then `Get-CAbiVerbs <include\superslm dir>`.
#
# Every C verb the ABI declares lives in exactly two files, include/superslm/sslm_abi_functions.inc and
# sslm_abi_functions_g5_comparable.inc (sslm_abi.h includes the first inside its extern "C" block, and the
# first includes the second).
#
# A verb is a DECLARATION, read at statement level, not line by line:
#   1. /* ... */ and // comments are blanked, keeping every newline so line numbers stay true;
#   2. every preprocessor line is blanked and ends a statement;
#   3. a declaration is a statement start (the file's start, or just after `;`, `{`, `}` or a preprocessor
#      line), then ONE OR MORE declarator tokens -- identifiers, `*` -- which may span lines, then
#      `sslm_<name>(`. That admits the three forms a declaration takes:
#        SUPERSLM_API sslm_status sslm_x(          (the slot)
#        sslm_status                                 (the return type on the line before)
#        sslm_x(
#        SSLM_C_EXPORT sslm_status sslm_x(          (an unknown prefix macro)
#      and never a mention: a comment is blanked, and a name inside another verb's argument list follows
#      `(` or `,`, never a statement start.
# The slot prefix is not required, so the same set is read from a tree with or without the slot, and from
# an X3 mutant with one slot dropped.
#
# The reader is an instrument (it decides X2's expected C set), so X2 cross-checks the list it wrote against
# an independent mention set (build_x2.ps1, X2 CVERBS-MENTION), and fixtures\ghost_reader\ is its
# must-reject: test_c_abi_reader.ps1 requires exactly the fixture's two declarations and not its comment.
#
# Returns objects { Name; File; Line } sorted by Name (ordinal); Line is the line of the name itself.
# Throws when a name is declared twice or when either file is missing -- a list read from half the ABI is
# refused, not returned.
function Get-CAbiVerbs([string]$IncDir) {
    $files = @('sslm_abi_functions.inc', 'sslm_abi_functions_g5_comparable.inc')
    # A statement start, then declarator tokens (at least one), then the name and its open parenthesis.
    $decl = [regex]'(?:\A|[;{}])\s*((?:[A-Za-z_][A-Za-z0-9_]*\s*\**\s*)+?)\b(sslm_[a-z0-9_]+)\s*\('
    $out = @()
    $seen = @{}
    foreach ($f in $files) {
        $p = Join-Path $IncDir $f
        if (-not (Test-Path $p)) { throw "Get-CAbiVerbs: $p not found" }
        $text = [System.IO.File]::ReadAllText($p).Replace("`r`n", "`n")
        # Comments blanked, newlines kept.
        $blank = [regex]::Replace($text, '(?s)/\*.*?\*/', { param($m) [regex]::Replace($m.Value, '[^\n]', ' ') })
        $blank = [regex]::Replace($blank, '//[^\n]*', { param($m) ' ' * $m.Value.Length })
        # Preprocessor lines blanked; each becomes a statement boundary (a ';' at its first column).
        $blank = [regex]::Replace($blank, '(?m)^[ \t]*#[^\n]*', { param($m) ';' + (' ' * [Math]::Max(0, $m.Value.Length - 1)) })
        foreach ($m in $decl.Matches($blank)) {
            $name = $m.Groups[2].Value
            $idx = $m.Groups[2].Index
            $line = 1 + ([regex]::Matches($blank.Substring(0, $idx), "`n")).Count
            if ($seen.ContainsKey($name)) { throw "Get-CAbiVerbs: $name declared twice ($($seen[$name]) and ${f}:$line)" }
            $seen[$name] = "${f}:$line"
            $out += [pscustomobject]@{ Name = $name; File = $f; Line = $line }
        }
    }
    $sorted = New-Object 'System.Collections.Generic.SortedDictionary[string,object]' ([System.StringComparer]::Ordinal)
    foreach ($o in $out) { $sorted[$o.Name] = $o }
    return @($sorted.Values)
}
