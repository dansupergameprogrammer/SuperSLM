# T-2835 (Curie) -- commissioning of the C verb reader (plan Sec3.7 item 3, T-2824 G6, T-2825 F-3).
#
# The reader (c_abi_verbs.ps1) decides X2's expected C set, so it is an instrument, and this is its
# must-reject: over fixtures\ghost_reader\ it must return EXACTLY { sslm_ghost_macro, sslm_ghost_split }:
#   sslm_ghost_split   the return type on the line before the name   (a line reader misses it)
#   sslm_ghost_macro   behind an unknown prefix macro                 (a SUPERSLM_API-only reader misses it)
#   sslm_ghost_comment a comment mention                              (must NOT be counted)
# Its must-accept: over -Engine's real include\superslm it returns exactly the committed c_abi_verbs.txt.
#
# Usage: test_c_abi_reader.ps1 [-Reader <c_abi_verbs.ps1>] [-Engine <root>]
#   -Reader lets a different reader be graded by the same cases (how the reader filed at 5ffed4c was shown
#   to fail the must-reject). Exit 0 only when both cases pass.
param(
    [string]$Reader = (Join-Path $PSScriptRoot 'c_abi_verbs.ps1'),
    [string]$Engine = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
)
$ErrorActionPreference = 'Stop'
. $Reader
$ok = $true
$want = @('sslm_ghost_macro', 'sslm_ghost_split')
$got = @(Get-CAbiVerbs (Join-Path $PSScriptRoot 'fixtures\ghost_reader') | ForEach-Object Name)
$d = @(Compare-Object -CaseSensitive $want $got | ForEach-Object { "$($_.SideIndicator) $($_.InputObject)" })
Write-Output ("READER MUST-REJECT (fixtures\ghost_reader): {0} -- read {1} [{2}], want {3} [{4}]" -f ($(if ($d.Count -eq 0) { 'PASS' } else { 'FAIL' })), $got.Count, ($got -join ', '), $want.Count, ($want -join ', '))
$d | ForEach-Object { Write-Output "  READER $_  (=> read but not a declaration, <= a declaration the reader missed)" }
if ($d.Count -ne 0) { $ok = $false }
$committed = @(Get-Content (Join-Path $PSScriptRoot 'c_abi_verbs.txt') | Where-Object { $_ -and $_ -notmatch '^#' })
$real = @(Get-CAbiVerbs (Join-Path $Engine 'include\superslm') | ForEach-Object Name)
$d2 = @(Compare-Object -CaseSensitive $committed $real | ForEach-Object { "$($_.SideIndicator) $($_.InputObject)" })
Write-Output ("READER MUST-ACCEPT ({0}\include\superslm): {1} -- read {2}, c_abi_verbs.txt holds {3}" -f $Engine, ($(if ($d2.Count -eq 0) { 'PASS' } else { 'FAIL' })), $real.Count, $committed.Count)
$d2 | ForEach-Object { Write-Output "  READER $_" }
if ($d2.Count -ne 0) { $ok = $false }
exit ($(if ($ok) { 0 } else { 1 }))
