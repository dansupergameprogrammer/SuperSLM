# Paged-KV plan (rev 16.7) step C6: the box run's per-leg provenance records and the standing derived
# from them (tools/run_paged_kv_c6_box.ps1). The functions can be tested without a timing run.
# Dot-source this file.
#
# A leg's done-marker is its provenance record: the git HEAD, the sha256 of the test binary, the
# sha256 of every file the leg read (artifacts, references, baselines), the commissioned instrument
# ids in force that the leg rests on, the assertion environment variables actually set, and
# the outcome. Recording only what
# bears on the leg means commissioning one instrument reruns only the legs that rest on it.
# A leg is skipped on resume only when its stored record equals the record this invocation would
# produce; its standing is derived from its stored record alone, never from the current arguments.

# The instruments the verified receipt may name, by the registry entry each stands for, and the environment
# variable through which the cell asserts that instrument's verdict (empty: no variable; the oracle
# and the admission instruments assert unconditionally). Lifecycle has no performance bar;
# commissioning makes its measured change report eligible to stand, while its NO RESULT/RESOLVED
# verdicts continue to assert nothing.
$PkvC6Instruments = [ordered]@{
    oracle    = @{ Registry = 'PKV-ORACLE-R0'; Env = '' }
    admission = @{ Registry = 'PKV-LEGACY-COUNT-C4, PKV-FILL-PROBE-C5, PKV-FILL-PROBE-ONESTATE-C5'; Env = '' }
    timing74  = @{ Registry = 'PKV-TIMING-74-C6-R168'; Env = '' }
    lifecycle = @{ Registry = 'PKV-LIFECYCLE-C6-R168'; Env = 'SUPERSLM_PAGED_KV_LIFE_COMMISSIONED' }
}
$PkvC6RecordSchema = 'pkv-c6-leg/4'
# The fields a stored record must match for a leg to be skipped. Outcome and the binary's path are
# kept for the reader, not compared.
$PkvC6RecordKeys = @('Schema', 'Leg', 'Head', 'BinarySha256', 'SourceManifestSha256', 'Inputs', 'Commissioned', 'AssertEnv', 'HookState')
$PkvC6TimingLock = 'D:\_scratch\SuperSLM\pkv-c6-timing.lock'

# The box timing lock (D-SLM8464): one format and one protocol in every program that takes it
# (run_paged_kv_rows.py's timing_lock, run_paged_kv_pairs.py's PidLock and Lock74 in
# tests/paged-kv/c6_dim7_reset_adopt_timing.cpp are the others). The file holds the owner's
# bare decimal pid. A fresh acquire moves a temporary file holding the pid into place without replacing, so the lock is
# never visible empty. An existing file is recovered only under an exclusive non-blocking OS lock on it (one byte at
# offset 1 MiB, the same byte the Python legs lock). Then a file modified before the system's last boot is stale
# whatever pid it holds, since that pid may now name an unrelated process (D-SLM8466 (3)); otherwise a pid liveness
# check (a live pid refuses). Then an overwrite with this pid. An empty file is stale, so a lock left empty by a crash
# recovers; content other than ASCII digits naming a pid from 1 to [int]::MaxValue, with only space, tab, CR and LF
# trimmed around it ($PkvC6LockTrim), is torn (T-3206 N-4, T-3210 N-1), and so is a file longer than
# $PkvC6LockMaxBytes, whatever its first $PkvC6LockMaxBytes hold (T-3211 O-1). A release removes the file when it still
# holds this pid and never truncates it; a file longer than $PkvC6LockMaxBytes holds no pid. A sharing conflict on the
# recovery open (another process holding the file without write sharing) is retried, and refused by name when every
# attempt meets one (T-3204 O-3).
$PkvC6LockByte = 1048576
$PkvC6LockAttempts = 3
# T-3210 N-1 (D-SLM8494): the only characters trimmed around the pid, as the Python legs' LOCK_TRIM. .NET's bare Trim()
# trims every Unicode white-space character, U+3000 among them.
$PkvC6LockTrim = [char[]]@([char]0x20, [char]0x09, [char]0x0D, [char]0x0A)
# T-3211 O-1 (D-SLM8496): a lock file longer than this is torn, as the Python legs' LOCK_MAX_BYTES, on the recovery read
# and on the release check.
$PkvC6LockMaxBytes = 4096

# At most $PkvC6LockMaxBytes + 1 bytes of the stream from its current position, as the Python legs' _read_capped; more
# than $PkvC6LockMaxBytes means the file is too long.
function Read-PkvLockBytes {
    param([IO.Stream]$Stream)
    $buffer = New-Object byte[] ($PkvC6LockMaxBytes + 1)
    $read = 0
    while ($read -lt $buffer.Length) {
        $n = $Stream.Read($buffer, $read, $buffer.Length - $read)
        if ($n -le 0) { break }
        $read += $n
    }
    $bytes = New-Object byte[] $read
    [Array]::Copy($buffer, $bytes, $read)
    return , $bytes
}

# The system's last boot (UTC): now less the uptime, as the Python legs read it (GetTickCount64, which counts sleep and
# hibernation). [Environment]::TickCount64 needs PowerShell 7; Windows PowerShell falls back to the CIM boot time.
function Get-PkvBootTimeUtc {
    $ticks = $null
    try { $ticks = [Environment]::TickCount64 } catch { $ticks = $null }
    if ($null -ne $ticks) { return [DateTime]::UtcNow.AddMilliseconds(-[double]$ticks) }
    return (Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime()
}

function Enter-PkvTimingLock {
    param([string]$Path = $PkvC6TimingLock)
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $sharing = 0
    for ($attempt = 0; $attempt -lt $PkvC6LockAttempts; ++$attempt) {
        $temp = "$Path.$PID.new"
        [IO.File]::WriteAllBytes($temp, [Text.Encoding]::ASCII.GetBytes("$PID"))
        try {
            [IO.File]::Move($temp, $Path)   # no overwrite: fails when the lock exists
            return $Path
        } catch [IO.IOException], [UnauthorizedAccessException] {
            # an existing (or deleting) lock: recover it below
        } finally {
            if ([IO.File]::Exists($temp)) { [IO.File]::Delete($temp) }
        }
        try {
            $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::ReadWrite)
        } catch [IO.FileNotFoundException], [UnauthorizedAccessException] {
            continue
        } catch [IO.IOException] {
            # a sharing conflict: another process has the file open without write sharing; retry shortly
            ++$sharing
            Start-Sleep -Milliseconds 50
            continue
        }
        try {
            try { $stream.Lock($PkvC6LockByte, 1) }
            catch [IO.IOException] { throw "second C6 timing runner refused: $Path is being recovered by another runner" }
            try {
                # Windows refuses to delete a file this stream holds open (no delete sharing), so the path still
                # names this file.
                $stale = [IO.File]::GetLastWriteTimeUtc($Path) -lt (Get-PkvBootTimeUtc)
                $bytes = Read-PkvLockBytes $stream
                if ($bytes.Length -gt $PkvC6LockMaxBytes -and -not $stale) {
                    throw "torn C6 timing lock: $Path is longer than $PkvC6LockMaxBytes bytes"
                }
                $text = [Text.Encoding]::ASCII.GetString($bytes).Trim($PkvC6LockTrim)
                if ($text -and -not $stale) {
                    # T-3206 N-4: ASCII digits naming a pid from 1 to [int]::MaxValue, as the Python legs read it;
                    # anything else (an out-of-range number, zero) is torn, never a raw conversion error
                    [long]$owner = 0
                    if ($text -cnotmatch '^[0-9]+$' -or
                        -not [long]::TryParse($text, [Globalization.NumberStyles]::None, [Globalization.CultureInfo]::InvariantCulture, [ref]$owner) -or
                        $owner -lt 1 -or $owner -gt [int]::MaxValue) { throw "torn C6 timing lock: $Path" }
                    if (Get-Process -Id $owner -ErrorAction SilentlyContinue) { throw "second C6 timing runner refused: pid $owner" }
                }
                $bytes = [Text.Encoding]::ASCII.GetBytes("$PID")
                $stream.SetLength(0)
                $stream.Position = 0
                $stream.Write($bytes, 0, $bytes.Length)
                $stream.Flush()
                return $Path
            } finally { $stream.Unlock($PkvC6LockByte, 1) }
        } finally { $stream.Dispose() }
    }
    if ($sharing -eq $PkvC6LockAttempts) {
        throw "second C6 timing runner refused: $Path was held open by another process (a sharing conflict) in each of $PkvC6LockAttempts attempts"
    }
    throw "cannot acquire C6 timing lock: $Path"
}

function Exit-PkvTimingLock {
    param([string]$Path)
    try {
        if (-not [IO.File]::Exists($Path)) { return }
        # FileShare.Read, as File.ReadAllText opens it
        $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
        try { $bytes = Read-PkvLockBytes $stream } finally { $stream.Dispose() }
        if ($bytes.Length -le $PkvC6LockMaxBytes -and
            [Text.Encoding]::ASCII.GetString($bytes).Trim($PkvC6LockTrim) -ceq "$PID") {
            [IO.File]::Delete($Path)
        }
    } catch [IO.IOException], [UnauthorizedAccessException] {
        # a reader holds it open: it keeps this pid, stale once this process exits
    }
}

# Splits comma-joined lists and rejects any id that is not a commissionable instrument.
function Get-PkvCommissionedIds {
    param([string[]]$Ids)
    $out = @($Ids | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    foreach ($c in $out) {
        if (-not $PkvC6Instruments.Contains($c)) {
            throw "-Commissioned takes $(@($PkvC6Instruments.Keys) -join ', '); not '$c'"
        }
    }
    return , @($out | Sort-Object -Unique)
}

# Only the receipt-verifying tool's computed status can confer standing.
# -Status prints stored fields; -Verify rechecks the receipt and live hash.
function Get-PkvVerifiedIds {
    param([string[]]$Lines)
    $found = @()
    foreach ($key in $PkvC6Instruments.Keys) {
        $registryIds = @($PkvC6Instruments[$key].Registry -split ',' | ForEach-Object { $_.Trim() })
        $verified = @($registryIds | Where-Object { $id = $_; @($Lines | Where-Object { $_ -ceq "COMMISSIONED  $id" }).Count })
        if ($verified.Count -eq $registryIds.Count) { $found += $key }
    }
    return ,$found
}

function Get-PkvRunnerCommissionedIds {
    param([string[]]$Requested = @(), [string[]]$VerifyLines = @())
    # Requested is accepted for command-line compatibility only. A verified
    # receipt line is the sole path to standing.
    return (Get-PkvVerifiedIds $VerifyLines)
}

function Get-PkvNormalizedSha256 {
    param([string]$Path)
    $bytes = [IO.File]::ReadAllBytes($Path)
    $normalized = New-Object System.Collections.Generic.List[byte] ($bytes.Length)
    for ($i = 0; $i -lt $bytes.Length; ++$i) {
        if ($bytes[$i] -eq 13 -and $i + 1 -lt $bytes.Length -and $bytes[$i + 1] -eq 10) { continue }
        $normalized.Add($bytes[$i])
    }
    $hash = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($hash.ComputeHash($normalized.ToArray())).Replace('-', '').ToLowerInvariant() }
    finally { $hash.Dispose() }
}

# The receipt verifier establishes the registry entry; this check establishes
# that its source is in this link's source set and the invoked bytes still match.
function Test-PkvSourceBinding {
    param([string]$Binary, [string]$RegistryId, [string]$Repo,
          [string]$RegistryPath = 'D:\Wizard\Claude\Tooling\instrument-registry.json')
    $checker = Join-Path $Repo 'tools/verify_paged_kv_binding.py'
    if (-not (Test-Path -LiteralPath $checker) -or -not (Test-Path -LiteralPath $RegistryPath)) { return $false }
    try {
        $result = & python $checker --binary $Binary --repo $Repo --registry $RegistryPath --id $RegistryId 2>$null
        if ($LASTEXITCODE -ne 0) { return $false }
        return [bool](($result | ConvertFrom-Json).bound)
    } catch { return $false }
}

# The assertion variables a leg resting on $Rests gets: only for an instrument that is both rested on
# and commissioned.
function Get-PkvLegAssertionEnv {
    param([string[]]$Rests, [string[]]$Commissioned)
    $envs = [ordered]@{}
    foreach ($r in @($Rests | Sort-Object -Unique)) {
        if ($Commissioned -notcontains $r -or -not $PkvC6Instruments.Contains($r)) { continue }
        switch ($r) {
            'lifecycle' { $envs['SUPERSLM_PAGED_KV_LIFE_COMMISSIONED'] = '1' }
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
# file's sha256. Commissioned keeps only the ids the leg rests on. Every value is a plain string, so the record survives a JSON round trip
# unchanged.
function New-PkvLegRecord {
    param(
        [Parameter(Mandatory = $true)][string]$Leg,
        [string[]]$Rests = @(),
        [string]$Head = '',
        [string]$Binary = '',
        [string]$BinarySha256 = '',
        [string]$SourceManifestSha256 = '',
        [System.Collections.IDictionary]$Inputs = @{},
        [string[]]$Commissioned = @(),
        [string]$HookState = ''
    )
    $envs = Get-PkvLegAssertionEnv -Rests $Rests -Commissioned $Commissioned
    $relevant = @($Commissioned | Where-Object { $Rests -contains $_ } | Sort-Object -Unique)
    return [ordered]@{
        Schema       = $PkvC6RecordSchema
        Leg          = $Leg
        Head         = $Head
        Binary       = $Binary
        BinarySha256 = $BinarySha256
        SourceManifestSha256 = $SourceManifestSha256
        Inputs       = (ConvertTo-PkvPairs $Inputs)
        Commissioned = ($relevant -join ',')
        AssertEnv    = (ConvertTo-PkvPairs $envs)
        HookState    = $HookState
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
    if (Get-PkvRecordField $Record 'HookState') { return 'quarantined (mutant construction hook)' }
    $commissioned = @((Get-PkvRecordField $Record 'Commissioned') -split ',' | Where-Object { $_ })
    $envs = Get-PkvRecordEnv $Record
    $missing = @()
    foreach ($r in $Rests) {
        if (-not $PkvC6Instruments.Contains($r)) { $missing += "$r (never commissioned)"; continue }
        if ($commissioned -notcontains $r) { $missing += $r; continue }
        $var = $PkvC6Instruments[$r].Env
        if ($var -and -not $envs.Contains($var)) {
            $missing += "$r ($var unset)"
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
    # -InputObject: a dictionary serializes as one object on Windows PowerShell 5.1 and on pwsh; every
    # value is a string, so the default depth is never reached.
    Set-Content -LiteralPath $Path -Encoding utf8 -Value (ConvertTo-Json -InputObject $Record -Depth 3)
}
