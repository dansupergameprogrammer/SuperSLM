# Regression tests for C6 provenance and resume.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'paged_kv_c6_provenance.ps1')
$script:failures = 0
function Check([bool]$ok, [string]$label) { if ($ok) { Write-Output "ok $label" } else { Write-Output "FAIL $label"; $script:failures++ } }
# The box runner supplies this helper from paged_kv_box_common.ps1 before it
# loads provenance. Keep this cell standalone without entering its build shell.
function Get-PkvSha256([string]$Path) { return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant() }
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("pkv-c6-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    $base = @{ Head = ('a' * 40); Binary = 'C:\test.exe'; BinarySha256 = ('b' * 64); Inputs = [ordered]@{ artifact = ('c' * 64) } }
    $marker = Join-Path $tmp 'leg.done'
    $old = New-PkvLegRecord -Leg '7.4/C6' -Rests @('timing74') @base
    $old.Outcome = 'GREEN exit=0'
    Write-PkvLegRecord $marker $old
    $read = Read-PkvLegRecord $marker
    Check (Test-PkvLegRecordCurrent $read $old) 'record round trip'
    Check ((Get-PkvLegStanding -Rests @('timing74') -Record $read) -like 'quarantined*') 'uncommissioned 7.4 quarantined'
    $verified = Get-PkvRunnerCommissionedIds -Requested @('timing74') -VerifyLines @('QUARANTINED   PKV-TIMING-74-C6')
    Check ($verified.Count -eq 0) 'caller flag without receipt-verifier result confers no standing'
    # A local clone supplies a clean committed tree even while this test file is
    # being edited. The manifest is link evidence; registry.source_binding is
    # the independent claim over that evidence and the invoked environment.
    $repo = Join-Path $tmp 'current-source'
    $origin = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
    & git clone --quiet --shared $origin $repo
    if ($LASTEXITCODE -ne 0) { throw 'fixture clone failed' }
    $buildDir = Join-Path $tmp 'build'
    New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
    $instrument = Join-Path $repo 'tests/paged-kv/c6_dim7_reset_adopt_timing.cpp'
    $runner = Join-Path $repo 'tools/run_paged_kv_c6_box.ps1'
    $binary = Join-Path $buildDir 'superslm_pkv_c6.exe'
    [IO.File]::WriteAllBytes($binary, [byte[]](74, 168, 4, 1))
    $sourceSha = Get-PkvNormalizedSha256 $instrument
    $runnerSha = Get-PkvSha256 $runner
    $binarySha = Get-PkvSha256 $binary
    $successorId = 'PKV-TIMING-74-C6-R168'
    $registryPath = Join-Path $tmp 'registry.json'
    $head = (& git -C $repo rev-parse HEAD).Trim()
    $tree = (& git -C $repo rev-parse 'HEAD^{tree}').Trim()
    $recipe = [ordered]@{ commands = [ordered]@{ 'superslm_pkv_c6.exe' = 'cl /O2 /DNDEBUG c6_dim7_reset_adopt_timing.cpp' }; compiler_sha256 = ('d' * 64) }
    $manifest = [ordered]@{
        schema = 'pkv-c6-source-set/2'
        git_head = $head
        git_tree = $tree
        build_recipe = $recipe
        sources = [ordered]@{ 'tests/paged-kv/c6_dim7_reset_adopt_timing.cpp' = $sourceSha; 'tools/run_paged_kv_c6_box.ps1' = (Get-PkvNormalizedSha256 $runner) }
        invoked = [ordered]@{ 'superslm_pkv_c6.exe' = $binarySha; 'run_paged_kv_c6_box.ps1' = $runnerSha }
    }
    $utf8 = New-Object Text.UTF8Encoding($false)
    [IO.File]::WriteAllText("$binary.source-set.json", ($manifest | ConvertTo-Json -Depth 5), $utf8)
    $recipeJson = ConvertTo-Json -InputObject $recipe -Compress -Depth 5
    $recipeBytes = [Text.Encoding]::UTF8.GetBytes($recipeJson)
    $sha = [Security.Cryptography.SHA256]::Create()
    try { $recipeSha = [BitConverter]::ToString($sha.ComputeHash($recipeBytes)).Replace('-', '').ToLowerInvariant() }
    finally { $sha.Dispose() }
    $environment = [ordered]@{}
    $named = @('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES','CL','CXXFLAGS','LDFLAGS')
    foreach ($variable in (Get-ChildItem Env: | Sort-Object Name)) {
        if (($variable.Name.StartsWith('SUPERSLM_PAGED_KV_') -or $variable.Name.StartsWith('PKV_DEBUNK_') -or $named -contains $variable.Name) -and
            $variable.Name -notin @('PKV_C6_REGISTRY','PKV_C6_RECEIPT_VERIFIER')) { $environment[$variable.Name] = $variable.Value }
    }
    # The registrar bound() grades is the one beside the registry it is handed (SuperSLM cb7dd42, no fallback). The suite
    # copies the Wizard tree's registrar, named by PKV_COMMISSION_TOOL (required; there is no default), beside its
    # temporary registry and claims its hash (T-3211 N-1; the test author's record, T-3189 section 20).
    if (-not $env:PKV_COMMISSION_TOOL -or -not (Test-Path -LiteralPath $env:PKV_COMMISSION_TOOL -PathType Leaf)) {
        Check $false 'PKV_COMMISSION_TOOL names the instrument-commission.ps1 of the Wizard tree under test (no default)'
        throw 'the provenance suite needs PKV_COMMISSION_TOOL: the registrar of the Wizard tree under test'
    }
    $verifier = Join-Path $tmp 'instrument-commission.ps1'
    Copy-Item -LiteralPath $env:PKV_COMMISSION_TOOL -Destination $verifier
    $registry = [ordered]@{
        'PKV-TIMING-74-C6' = @{ instrument = $instrument; instrument_sha256 = $sourceSha }
        $successorId = @{ instrument = $instrument; instrument_sha256 = $sourceSha;
            source_binding = @{ manifest_sha256 = (Get-PkvSha256 "$binary.source-set.json"); git_head = $head;
                git_tree = $tree; build_recipe_sha256 = $recipeSha; verifier_sha256 = (Get-PkvSha256 $verifier);
                measurement_env = $environment } }
    }
    [IO.File]::WriteAllText($registryPath, ($registry | ConvertTo-Json -Depth 8), $utf8)
    $currentBase = @{} + $base
    $currentBase.Binary = $binary
    $currentBase.BinarySha256 = $binarySha
    $currentBase.Inputs = [ordered]@{ instrument = $sourceSha; runner = $runnerSha }
    # A T-3093 receipt is withdrawn even if its registry entry names the
    # current instrument and the leg records the current invoked bytes.
    $legacyVerified = Get-PkvRunnerCommissionedIds -VerifyLines @('COMMISSIONED  PKV-TIMING-74-C6')
    $legacyRecord = New-PkvLegRecord -Leg '7.4/C6' -Rests @('timing74') -Commissioned $legacyVerified @currentBase
    Check ((Get-PkvLegStanding -Rests @('timing74') -Record $legacyRecord) -like 'quarantined*') 'superseded T-3093 receipt quarantines current 7.4 leg'
    $successorVerified = Get-PkvRunnerCommissionedIds -VerifyLines @("COMMISSIONED  $successorId")
    $hasSourceBinding = [bool](Get-Command Test-PkvSourceBinding -ErrorAction SilentlyContinue)
    $bound = $false
    if ($hasSourceBinding) {
        $bound = Test-PkvSourceBinding -Binary $binary -RegistryId $successorId -Repo $repo -RegistryPath $registryPath
    }
    Check ($successorVerified -contains 'timing74' -and $bound) 'source-bound successor receipt accepts current source and invoked bytes'
    $successorRecord = New-PkvLegRecord -Leg '7.4/C6' -Rests @('timing74') -Commissioned @($successorVerified | Where-Object { $bound }) @currentBase
    Check ((Get-PkvLegStanding -Rests @('timing74') -Record $successorRecord) -eq 'standing') 'source-bound successor confers 7.4 standing'
    if ($hasSourceBinding) {
        [IO.File]::AppendAllText($instrument, "// changed source`n", [Text.Encoding]::ASCII)
        Check (-not (Test-PkvSourceBinding -Binary $binary -RegistryId $successorId -Repo $repo -RegistryPath $registryPath)) 'changed source quarantines successor'
        & git -C $repo checkout -- 'tests/paged-kv/c6_dim7_reset_adopt_timing.cpp'
        [IO.File]::AppendAllText($runner, "`n# changed invocation`n", [Text.Encoding]::ASCII)
        Check (-not (Test-PkvSourceBinding -Binary $binary -RegistryId $successorId -Repo $repo -RegistryPath $registryPath)) 'changed invoked runner quarantines successor'
        & git -C $repo checkout -- 'tools/run_paged_kv_c6_box.ps1'
        [IO.File]::WriteAllBytes($binary, [byte[]](74, 168, 4, 2))
        Check (-not (Test-PkvSourceBinding -Binary $binary -RegistryId $successorId -Repo $repo -RegistryPath $registryPath)) 'changed executable quarantines successor'
    } else {
        Check $false 'changed source quarantines successor'
        Check $false 'changed invoked runner quarantines successor'
        Check $false 'changed executable quarantines successor'
    }
    $verified = Get-PkvVerifiedIds @('COMMISSIONED  PKV-LEGACY-COUNT-C4')
    Check ($verified -notcontains 'admission') 'partial grouped admission receipts confer no standing'
    $next = New-PkvLegRecord -Leg '7.4/C6' -Rests @('timing74') -Commissioned @('timing74') @base
    Check (-not (Test-PkvLegRecordCurrent $read $next)) 'commissioning invalidates marker'
    Check ((Get-PkvLegRecordDiff $read $next) -join ',' -eq 'Commissioned') 'commissioning difference named'
    Check ((Get-PkvLegStanding -Rests @('timing74') -Record $next) -eq 'standing') 'commissioned 7.4 standing without arbitrary ratio'
    Check ((Get-PkvRecordEnv $next).Count -eq 0) '7.4 has no assertion environment'
    $hooked = New-PkvLegRecord -Leg '7.4/C6' -Rests @('timing74') -Commissioned @('timing74') -HookState 'adopt=1;reset=' @base
    Check ((Get-PkvLegStanding -Rests @('timing74') -Record $hooked) -like 'quarantined*') 'mutant hook never publishes a graded 7.4 reading'
    Check (-not (Test-PkvLegRecordCurrent $hooked $next)) 'hook state changes resume identity'
    Check (Test-PkvLegRecordCurrent $next (New-PkvLegRecord -Leg '7.4/C6' -Rests @('timing74') -Commissioned @('timing74','oracle') @base)) 'unrelated commission does not invalidate'
    foreach ($key in 'Head', 'BinarySha256') {
        $changed = @{} + $base; $changed[$key] = ('d' * $base[$key].Length)
        Check (-not (Test-PkvLegRecordCurrent $next (New-PkvLegRecord -Leg '7.4/C6' -Rests @('timing74') -Commissioned @('timing74') @changed))) "$key invalidates marker"
    }
    $changed = @{} + $base; $changed.Inputs = [ordered]@{ artifact = ('d' * 64) }
    Check (-not (Test-PkvLegRecordCurrent $next (New-PkvLegRecord -Leg '7.4/C6' -Rests @('timing74') -Commissioned @('timing74') @changed))) 'input digest invalidates marker'
    $life = New-PkvLegRecord -Leg 'lifecycle/C6' -Rests @('lifecycle') -Commissioned @('lifecycle') @base
    Check ((Get-PkvRecordEnv $life)['SUPERSLM_PAGED_KV_LIFE_COMMISSIONED'] -eq '1') 'lifecycle asserts only its own variable'
    Check ((Get-PkvLegStanding -Rests @('lifecycle') -Record $life) -eq 'standing') 'lifecycle commissioned standing'
    $life.AssertEnv = ''
    Check ((Get-PkvLegStanding -Rests @('lifecycle') -Record $life) -like 'quarantined*') 'missing lifecycle assertion quarantined'
    $lifeMutant = New-PkvLegRecord -Leg 'lifecycle/C6' -Rests @('lifecycle') -Commissioned @('lifecycle') -HookState 'reset=1' @base
    Check ((Get-PkvLegStanding -Rests @('lifecycle') -Record $lifeMutant) -like 'quarantined*') 'lifecycle reset mutant cannot publish standing'
    $life['LoadBefore'] = 'load=88; offenders=UnrealEditor'
    $life['LoadAfter'] = 'load=91; offenders=cl'
    Check ($life.LoadBefore -match '88' -and $life.LoadAfter -match 'cl') 'load and offenders record even when busy'
    $lock = Join-Path $tmp 'box-global.lock'
    $held = Enter-PkvTimingLock -Path $lock
    try {
        try { Enter-PkvTimingLock -Path $lock | Out-Null; Check $false 'distinct timing leg refused by global lock' }
        catch { Check ($_.Exception.Message -match 'second C6 timing runner') 'distinct timing leg refused by global lock' }
    } finally { Exit-PkvTimingLock $held }
    Check (-not (Test-Path -LiteralPath $lock)) 'global lock released after publication'
    Set-Content -LiteralPath $lock -Value '99999999' -Encoding ascii
    $held = Enter-PkvTimingLock -Path $lock
    Check ($held -eq $lock) 'dead pid lock reclaimed'
    Exit-PkvTimingLock $held
    Set-Content -LiteralPath $marker -Value 'GREEN exit=0' -Encoding utf8
    Check ($null -eq (Read-PkvLegRecord $marker)) 'legacy marker invalid'
    try { Get-PkvCommissionedIds @('timing79') | Out-Null; Check $false 'obsolete timing79 refused' }
    catch { Check ($_.Exception.Message -match 'not .timing79.') 'obsolete timing79 refused' }
} finally { Remove-Item -LiteralPath $tmp -Recurse -Force }
if ($script:failures) { throw "$script:failures provenance checks failed" }
