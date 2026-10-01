# Paged-KV plan (rev 16.2) step C6: the box run's per-leg provenance records and the standing derived
# from them (tools/run_paged_kv_c6_box.ps1). Pure functions only (no file system beyond the marker
# read and write, no processes), so tools/test_paged_kv_c6_provenance.ps1 can run them anywhere pwsh
# runs. Dot-source it.
#
# A leg's done-marker is its provenance record: the git HEAD, the sha256 of the test binary, the
# sha256 of every file the leg read (artifacts, references, baselines), the commissioned instrument
# ids in force, the assertion environment variables actually set, and -Max74Ratio, plus the outcome.
# A leg is skipped on resume only when its stored record equals the record this invocation would
# produce; its standing is derived from its stored record alone, never from the current arguments.

# The instruments -Commissioned may name, by the registry entry each stands for, and the environment
# variable through which the cell asserts that instrument's verdict (empty: no variable; the oracle
# and the admission instruments assert unconditionally). The lifecycle timings
# (c6_lifecycle_timing.cpp) are not here: they have no registry entry and no commissioning, record
# figures only, and are always quarantined.
$PkvC6Instruments = [ordered]@{
    oracle    = @{ Registry = 'PKV-ORACLE-R0'; Env = '' }
    admission = @{ Registry = 'PKV-LEGACY-COUNT-C4, PKV-FILL-PROBE-C5, PKV-FILL-PROBE-ONESTATE-C5'; Env = '' }
    timing74  = @{ Registry = 'PKV-TIMING-74-C6'; Env = 'SUPERSLM_PAGED_KV_74_MAX_RATIO' }
    timing79  = @{ Registry = 'PKV-TIMING-79-C6'; Env = 'SUPERSLM_PAGED_KV_TIMING_COMMISSIONED' }
}
$PkvC6RecordSchema = 'pkv-c6-leg/1'
# The fields a stored record must match for a leg to be skipped. Outcome and the binary's path are
# kept for the reader, not compared.
$PkvC6RecordKeys = @('Schema', 'Leg', 'Head', 'BinarySha256', 'Inputs', 'Commissioned', 'AssertEnv', 'Max74Ratio')

# Splits comma-joined lists and rejects any id that is not a commissionable instrument.
function Get-PkvCommissionedIds {
    param([string[]]$Ids)
    $out = @($Ids | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    foreach ($c in $out) {
        if ($c -eq 'lifecycle') {
            throw "-Commissioned lifecycle: the lifecycle timings have no registry entry and no commissioning; they record figures only and are always quarantined"
        }
        if (-not $PkvC6Instruments.Contains($c)) {
            throw "-Commissioned takes $(@($PkvC6Instruments.Keys) -join ', '); not '$c'"
        }
    }
    return , @($out | Sort-Object -Unique)
}

# The assertion variables a leg resting on $Rests gets: only for an instrument that is both rested on
# and commissioned. 7.4 asserts only with a ratio, so timing74 without -Max74Ratio sets nothing.
function Get-PkvLegAssertionEnv {
    param([string[]]$Rests, [string[]]$Commissioned, [string]$Max74Ratio = '')
    $envs = [ordered]@{}
    foreach ($r in @($Rests | Sort-Object -Unique)) {
        if ($Commissioned -notcontains $r -or -not $PkvC6Instruments.Contains($r)) { continue }
        switch ($r) {
            'timing79' { $envs['SUPERSLM_PAGED_KV_TIMING_COMMISSIONED'] = '1' }
            'timing74' { if ($Max74Ratio) { $envs['SUPERSLM_PAGED_KV_74_MAX_RATIO'] = $Max74Ratio } }
        }
    }
    return $envs
}

function ConvertTo-PkvPairs {
    param($Map)
    if (-not $Map) { return '' }
    return (@($Map.Keys | Sort-Object | ForEach-Object { "$_=$($Map[$_])" }) -join ';')
}

# The record this invocation would write for a leg. $Inputs maps a label (role:file name) to the
# file's sha256. Every value is a plain string, so the record survives a JSON round trip unchanged.
function New-PkvLegRecord {
    param(
        [Parameter(Mandatory = $true)][string]$Leg,
        [string[]]$Rests = @(),
        [string]$Head = '',
        [string]$Binary = '',
        [string]$BinarySha256 = '',
        [System.Collections.IDictionary]$Inputs = @{},
        [string[]]$Commissioned = @(),
        [string]$Max74Ratio = ''
    )
    $envs = Get-PkvLegAssertionEnv -Rests $Rests -Commissioned $Commissioned -Max74Ratio $Max74Ratio
    return [ordered]@{
        Schema       = $PkvC6RecordSchema
        Leg          = $Leg
        Head         = $Head
        Binary       = $Binary
        BinarySha256 = $BinarySha256
        Inputs       = (ConvertTo-PkvPairs $Inputs)
        Commissioned = (@($Commissioned | Sort-Object -Unique) -join ',')
        AssertEnv    = (ConvertTo-PkvPairs $envs)
        Max74Ratio   = $Max74Ratio
        Outcome      = ''
    }
}

# The assertion variables a record says were set, as a hashtable (what the runner puts in the env).
function Get-PkvRecordEnv {
    param($Record)
    $envs = [ordered]@{}
    if (-not $Record) { return $envs }
    foreach ($p in @("$($Record.AssertEnv)" -split ';' | Where-Object { $_ })) {
        $k, $v = $p -split '=', 2
        $envs[$k] = $v
    }
    return $envs
}

function Get-PkvRecordField {
    param($Record, [string]$Key)
    if ($Record -is [System.Collections.IDictionary]) { return "$($Record[$Key])" }
    $p = $Record.PSObject.Properties[$Key]
    if ($p) { return "$($p.Value)" }
    return ''
}

# True only when both records exist, are of this schema, and agree on every compared field.
function Test-PkvLegRecordCurrent {
    param($Stored, $Current)
    if (-not $Stored -or -not $Current) { return $false }
    foreach ($k in $PkvC6RecordKeys) {
        if ((Get-PkvRecordField $Stored $k) -cne (Get-PkvRecordField $Current $k)) { return $false }
    }
    return ((Get-PkvRecordField $Stored 'Schema') -eq $PkvC6RecordSchema)
}

# The compared fields on which two records differ (for the runner's STALE line).
function Get-PkvLegRecordDiff {
    param($Stored, $Current)
    return , @($PkvC6RecordKeys | Where-Object { (Get-PkvRecordField $Stored $_) -cne (Get-PkvRecordField $Current $_) })
}

# A leg's standing, from its stored record alone. Standing needs every instrument the leg rests on to
# be commissioned in that record and, for an instrument asserted through a variable, that variable to
# have been set for that run. No record (a marker from before this schema, or none): quarantined.
function Get-PkvLegStanding {
    param([string[]]$Rests, $Record)
    if (-not $Record -or (Get-PkvRecordField $Record 'Schema') -ne $PkvC6RecordSchema) { return 'quarantined (no provenance record)' }
    $commissioned = @((Get-PkvRecordField $Record 'Commissioned') -split ',' | Where-Object { $_ })
    $envs = Get-PkvRecordEnv $Record
    $missing = @()
    foreach ($r in $Rests) {
        if (-not $PkvC6Instruments.Contains($r)) { $missing += "$r (never commissioned)"; continue }
        if ($commissioned -notcontains $r) { $missing += $r; continue }
        $var = $PkvC6Instruments[$r].Env
        if ($var -and -not $envs.Contains($var)) {
            if ($r -eq 'timing74') { $missing += 'ratio' } else { $missing += "$r ($var unset)" }
        }
    }
    if ($missing.Count -eq 0) { return 'standing' }
    return 'quarantined (' + ($missing -join ', ') + ')'
}

# Reads a marker; $null when it is absent or not a record of this schema (a pre-provenance marker
# holds only "GREEN exit=0 ...", which can never match, so its leg reruns).
function Read-PkvLegRecord {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try { $r = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json } catch { return $null }
    if (-not $r -or (Get-PkvRecordField $r 'Schema') -ne $PkvC6RecordSchema) { return $null }
    return $r
}

function Write-PkvLegRecord {
    param([string]$Path, $Record)
    Set-Content -LiteralPath $Path -Encoding utf8 -Value ([pscustomobject]$Record | ConvertTo-Json)
}
