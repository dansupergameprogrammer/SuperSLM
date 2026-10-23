# SuperSLM 1.14.0, E5 W1-W4: cell 9.10's GPU half on the work machine's RX 7900 XTX, from the F: drive.
#
# This is G1-G4 of docs/release/e5-box-2026-10-23/e5-box-runsheet-2026-10-23.md, run as W1-W4: the same commands,
# including the quoting fix for --name=value arguments in G1 and G3. The adapter is the AMD one, found by vendor id
# 0x1002 rather than by a fixed index. The drive carries the source (one git bundle: the release-step tree ce82c898
# and v1.13.1 d697927f) and the pkv_def fixture. Builds go to the work machine's own disk. Logs, the W2 captures, a
# SUMMARY and SHA256SUMS go to <drive>\e5-w\results\.
#
#   <drive>\e5-w\START.cmd                  precheck, adapter, W1-W4 (finished pieces are skipped on a rerun)
#   <drive>\e5-w\START.cmd -CheckOnly       precheck and adapter detection only
#   -WorkRoot <dir>                         build root (default C:\_scratch\e5w)
#
# Each piece writes results\<piece>.done when it passes and is skipped on any rerun. The first red stops the run;
# rerunning retries that piece. Windows PowerShell 5.1.
param([switch]$CheckOnly, [string]$WorkRoot = 'C:\_scratch\e5w')

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$Release = 'ce82c89831ac880a5dd58b0a76599332343fb6ea'
$V1131 = 'd697927f6d5ce5fa73ecbb8660020a425518b923'
$FixtureSha = '994b58b49d3660bccfdd59dc6315028ea0203f37f4705d70eb7ce4dbecdb1db3'
$MinVramMB = 4096    # the RX 7900 XTX has 24 GB; a smaller 0x1002 adapter is the 7950X3D's integrated GPU

$kit = $PSScriptRoot
$root = Split-Path -Parent $kit                     # <drive>\e5-w
$inputs = Join-Path $root 'inputs'
$results = Join-Path $root 'results'
$logs = Join-Path $results 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$transcript = Join-Path $results ('kit-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')

function Write-Line([string]$Text) { Write-Host ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $Text) }
function Append([string]$Log, [string]$Line) {
    [System.IO.File]::AppendAllText($Log, $Line + "`r`n", (New-Object System.Text.UTF8Encoding($false)))
}
# Runs a program, appending its output to $Log line by line (UTF-8, no BOM). Returns the exit code.
function Run([string]$Log, [string]$Exe, [string[]]$Arguments) {
    Append $Log ('[{0}] {1} {2}' -f (Get-Date -Format 's'), $Exe, ($Arguments -join ' '))
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    & $Exe @Arguments 2>&1 | ForEach-Object { Append $Log ("$_") }
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    Append $Log ('[{0}] exit {1}' -f (Get-Date -Format 's'), $code)
    return $code
}
function LogOf([string]$Name) { return (Join-Path $logs ($Name + '.log')) }
function Done([string]$Piece) { return (Test-Path -LiteralPath (Join-Path $results ($Piece + '.done'))) }
function Stamp([string]$Piece, [string]$Text) {
    [System.IO.File]::WriteAllText((Join-Path $results ($Piece + '.done')), "$Text`r`nfinished=$((Get-Date).ToString('s'))`r`n",
                                   (New-Object System.Text.UTF8Encoding($false)))
}
function Red([string]$Piece, [string]$Why) {
    Write-Line "RED: $Piece -- $Why"
    Write-Line "Nothing after $Piece ran. Bring the drive back as it is (results\ holds the logs)."
    Write-Summary
    exit 1
}
function Write-Summary {
    $lines = @("# SuperSLM 1.14.0 E5 W1-W4 on the work machine", "", "written $((Get-Date).ToString('s'))", "")
    foreach ($p in @('W0-source', 'W1-v1131-build', 'W2-9.10-capture', 'W3-release-gpu-build', 'W4-9.10')) {
        $f = Join-Path $results ($p + '.done')
        $state = 'not done'
        if (Test-Path -LiteralPath $f) { $state = 'PASS: ' + (([System.IO.File]::ReadAllLines($f)) -join '; ') }
        $lines += "- ${p}: $state"
    }
    $a = Join-Path $results 'adapter.txt'
    if (Test-Path -LiteralPath $a) { $lines += ''; $lines += '## Adapter'; $lines += [System.IO.File]::ReadAllLines($a) }
    [System.IO.File]::WriteAllText((Join-Path $results 'SUMMARY.md'), (($lines -join "`r`n") + "`r`n"), (New-Object System.Text.UTF8Encoding($false)))
    $sums = Get-ChildItem -LiteralPath $results -Recurse -File |
        Where-Object { $_.Name -ne 'SHA256SUMS.txt' -and $_.Name -notlike 'kit-*.log' } |
        ForEach-Object { '{0}  {1}' -f (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant(),
                                        $_.FullName.Substring($results.Length + 1).Replace('\', '/') }
    [System.IO.File]::WriteAllText((Join-Path $results 'SHA256SUMS.txt'), ((@($sums) -join "`n") + "`n"), (New-Object System.Text.UTF8Encoding($false)))
}

# ---------------------------------------------------------------------------------------------------------------
# Precheck: everything the four pieces need, all reported at once.
# ---------------------------------------------------------------------------------------------------------------
function Test-Prerequisites {
    $missing = New-Object System.Collections.ArrayList
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $script:Vcvars = $null
    $script:VsPath = $null
    if (-not (Test-Path -LiteralPath $vswhere)) {
        [void]$missing.Add('Visual Studio 2022 (no vswhere.exe; install VS 2022 with "Desktop development with C++")')
    } else {
        $installs = @(& $vswhere -version '[17.0,18.0)' -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
        if ($installs.Count -eq 0) { [void]$missing.Add('Visual Studio 2022 with the C++ x64 tools (workload "Desktop development with C++")') }
        foreach ($vs in $installs) {
            $vc = Join-Path $vs 'VC\Auxiliary\Build\vcvarsall.bat'
            if (-not (Test-Path -LiteralPath $vc)) { continue }
            $null = & cmd.exe /d /c "call `"$vc`" x64 10.0.22621.0 -vcvars_ver=14.34 >nul 2>&1"
            if ($LASTEXITCODE -eq 0) { $script:Vcvars = $vc; $script:VsPath = $vs; break }
        }
        if ($installs.Count -gt 0 -and -not $script:Vcvars) {
            [void]$missing.Add('MSVC 14.34 toolset with Windows SDK 10.0.22621.0 (VS Installer > Individual components: "MSVC v143 - VS 2022 C++ x64/x86 build tools (v14.34-17.4)" and "Windows 11 SDK (10.0.22621.0)")')
        }
    }
    # build_paged_kv_gpu.bat calls VsDevCmd.bat at this fixed path (the release tree as pinned).
    $devcmd = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat'
    if (-not (Test-Path -LiteralPath $devcmd)) {
        [void]$missing.Add("$devcmd (the GPU cells' batch file calls Visual Studio 2022 Community's VsDevCmd at this path; install VS 2022 Community)")
    }
    $dxc = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin\10.0.22621.0\x64\dxc.exe'
    if (-not (Test-Path -LiteralPath $dxc)) { [void]$missing.Add("$dxc (Windows 11 SDK 10.0.22621.0, for the shaders)") }
    $cmake = Get-Command cmake -ErrorAction SilentlyContinue
    if (-not $cmake -and $script:VsPath) {
        $bundled = Join-Path $script:VsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
        if (Test-Path -LiteralPath (Join-Path $bundled 'cmake.exe')) { $env:PATH = "$bundled;$env:PATH"; $cmake = Get-Command cmake -ErrorAction SilentlyContinue }
    }
    if (-not $cmake) { [void]$missing.Add('cmake 3.24 or newer on PATH (or VS 2022''s "C++ CMake tools for Windows" component)') }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { [void]$missing.Add('git on PATH (Git for Windows)') }
    $py = @(Get-Command python -All -ErrorAction SilentlyContinue | Where-Object { $_.Source -notlike '*\WindowsApps\*' })
    if ($py.Count -eq 0) { [void]$missing.Add('python 3 on PATH (python.org installer; the Microsoft Store alias does not count)') }
    return ,$missing
}

function Import-VcVars([string]$Vcvars) {
    $lines = & cmd.exe /d /c "call `"$Vcvars`" x64 10.0.22621.0 -vcvars_ver=14.34 >nul 2>&1 && set"
    if ($LASTEXITCODE -ne 0) { throw 'vcvarsall x64 10.0.22621.0 -vcvars_ver=14.34 failed' }
    foreach ($line in $lines) {
        $at = $line.IndexOf('=')
        if ($at -gt 0) { Set-Item -Path ('env:' + $line.Substring(0, $at)) -Value $line.Substring($at + 1) }
    }
}

# ---------------------------------------------------------------------------------------------------------------
# Source: one repository on the work disk, cloned from the drive's bundle; src at the release step, v1131 at v1.13.1.
# ---------------------------------------------------------------------------------------------------------------
function Initialize-Source([string]$Work) {
    $repo = Join-Path $Work 'src'
    $v = Join-Path $Work 'v1131'
    $log = LogOf 'W0-source'
    $bundle = Join-Path $inputs 'e5w.bundle'
    if (-not (Test-Path -LiteralPath (Join-Path $repo '.git'))) {
        New-Item -ItemType Directory -Force -Path $repo | Out-Null
        if ((Run $log 'git' @('-C', $repo, 'init', '-q')) -ne 0) { Red 'W0-source' "git init failed; see $log" }
    }
    $have = $true
    foreach ($c in @($Release, $V1131)) {
        $null = & git -C $repo rev-parse -q --verify "$c^{commit}" 2>$null
        if ($LASTEXITCODE -ne 0) { $have = $false }
    }
    if (-not $have) {
        Write-Line 'unpacking the source from the drive''s bundle (full history)'
        if ((Run $log 'git' @('-C', $repo, 'fetch', '-q', $bundle, '+refs/*:refs/bundle/*')) -ne 0) { Red 'W0-source' "git fetch from the bundle failed; see $log" }
    }
    if ((Run $log 'git' @('-C', $repo, 'checkout', '-q', '--detach', '--force', $Release)) -ne 0) { Red 'W0-source' "checkout $Release failed; see $log" }
    if (-not (Test-Path -LiteralPath (Join-Path $v '.git'))) {
        if ((Run $log 'git' @('-C', $repo, 'worktree', 'add', '--detach', $v, $V1131)) -ne 0) { Red 'W0-source' "worktree add v1131 failed; see $log" }
    }
    $shallow = (& git -C $repo rev-parse --is-shallow-repository).Trim()
    $srcHead = (& git -C $repo rev-parse HEAD).Trim()
    $vHead = (& git -C $v rev-parse HEAD).Trim()
    $mergeBase = (& git -C $repo merge-base HEAD $V1131).Trim()
    Append $log "shallow=$shallow src=$srcHead v1131=$vHead"
    if ($shallow -ne 'false') { Red 'W0-source' "the clone at $repo is shallow ($shallow)" }
    if ($srcHead -ne $Release) { Red 'W0-source' "src HEAD is $srcHead, not $Release" }
    if ($vHead -ne $V1131) { Red 'W0-source' "v1131 HEAD is $vHead, not $V1131" }
    $null = $mergeBase
    return @{ Src = $repo; V = $v }
}

# ---------------------------------------------------------------------------------------------------------------
# Adapter: the engine's own enumeration (tools/t2116_list_adapters.cpp, raw EnumAdapters1 order, the order
# SSLM_GPU_ADAPTER_INDEX selects by). Pick the usable vendor-0x1002 hardware adapter with the most dedicated memory.
# ---------------------------------------------------------------------------------------------------------------
function Find-AmdAdapter([string]$Src, [string]$Work) {
    $out = Join-Path $Work 'adapters'
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $exe = Join-Path $out 't2116_list_adapters.exe'
    $log = LogOf 'adapter'
    if (Test-Path -LiteralPath $log) { Remove-Item -LiteralPath $log }
    $code = Run $log 'cl.exe' @('/nologo', '/EHsc', '/O2', "/Fe$exe", ("/Fo" + (Join-Path $out 't2116_list_adapters.obj')),(Join-Path $Src 'tools\t2116_list_adapters.cpp'), 'dxgi.lib', 'd3d12.lib')
    if ($code -ne 0) { Red 'adapter' "building the adapter lister failed; see $log" }
    $lines = @(& $exe)
    foreach ($l in $lines) { Append $log $l }
    $rows = @()
    foreach ($l in $lines) {
        $m = [regex]::Match($l, '^INDEX=(\d+) SOFTWARE=0 USABLE=(\d) VENDOR=0x([0-9a-fA-F]{4}) DEVICEID=0x([0-9a-fA-F]{4}) DEDICATED_VIDEO_MB=(\d+) NAME=(.*)$')
        if ($m.Success) {
            $rows += [pscustomobject]@{ Index = [int]$m.Groups[1].Value; Usable = $m.Groups[2].Value -eq '1'; Vendor = $m.Groups[3].Value.ToLower()
                                        Device = $m.Groups[4].Value.ToLower(); VramMB = [long]$m.Groups[5].Value; Name = $m.Groups[6].Value }
        }
    }
    Write-Line 'D3D12 hardware adapters (raw EnumAdapters1 order):'
    foreach ($r in $rows) { Write-Host ("    index {0}: vendor 0x{1} device 0x{2} {3} MB usable={4} {5}" -f $r.Index, $r.Vendor, $r.Device, $r.VramMB, $r.Usable, $r.Name) }
    $amd = @($rows | Where-Object { $_.Vendor -eq '1002' -and $_.Usable } | Sort-Object -Property VramMB -Descending)
    if ($amd.Count -eq 0) { return $null }
    if ($amd[0].VramMB -lt $MinVramMB) {
        Write-Line ("the only usable AMD adapter has {0} MB dedicated memory: that is an integrated GPU, not the RX 7900 XTX" -f $amd[0].VramMB)
        return $null
    }
    return $amd[0]
}

# ---------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------
Start-Transcript -LiteralPath $transcript | Out-Null
try {
    Write-Line "SuperSLM 1.14.0 E5 W1-W4 kit; drive folder $root; work root $WorkRoot"
    Write-Line 'checking prerequisites'
    $missing = Test-Prerequisites
    if ($missing.Count -gt 0) {
        Write-Line 'MISSING (install these, then run START.cmd again):'
        foreach ($m in $missing) { Write-Host "    - $m" -ForegroundColor Yellow }
        exit 2
    }
    Write-Line "prerequisites: ok (VS at $script:VsPath, MSVC 14.34, cmake, git, python)"
    foreach ($f in @('e5w.bundle', 'fixtures\pkv_def.sslm')) {
        if (-not (Test-Path -LiteralPath (Join-Path $inputs $f))) { Write-Line "the drive is missing inputs\${f}: refill it on the home box (fill_drive.ps1)"; exit 3 }
    }
    $fx = Join-Path $inputs 'fixtures'
    if ((Get-FileHash -LiteralPath (Join-Path $fx 'pkv_def.sslm') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $FixtureSha) {
        Write-Line 'inputs\fixtures\pkv_def.sslm does not hash to 994b58b4...: refill the drive'; exit 3
    }
    Import-VcVars $script:Vcvars
    New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null

    $source = Initialize-Source $WorkRoot
    if (-not (Done 'W0-source')) { Stamp 'W0-source' "src=$Release`r`nv1131=$V1131`r`nwork=$WorkRoot" }

    $amd = Find-AmdAdapter $source.Src $WorkRoot
    if (-not $amd) {
        Write-Line 'NO AMD ADAPTER: no usable D3D12 hardware adapter with vendor id 0x1002 and at least 4 GB. W1-W4 need the RX 7900 XTX; stopping before any build.'
        exit 4
    }
    $adapterText = "SSLM_GPU_ADAPTER_INDEX=$($amd.Index)`r`nvendor=0x$($amd.Vendor)`r`ndevice=0x$($amd.Device)`r`ndedicated_mb=$($amd.VramMB)`r`nname=$($amd.Name)`r`nmachine=$env:COMPUTERNAME"
    [System.IO.File]::WriteAllText((Join-Path $results 'adapter.txt'), $adapterText + "`r`n", (New-Object System.Text.UTF8Encoding($false)))
    Write-Line ("AMD adapter: index {0}, {1} ({2} MB)" -f $amd.Index, $amd.Name, $amd.VramMB)
    if ($CheckOnly) { Write-Line 'precheck and adapter detection: ok (-CheckOnly: nothing built)'; exit 0 }

    $E5 = $WorkRoot; $E5Src = $source.Src; $V = $source.V; $FX = $fx
    $pkvg = Join-Path $E5 'pkvg'
    $captures = Join-Path $E5 'captures910'

    # W1 = G1: v1.13.1 GPU build for the capture leg.
    if (-not (Done 'W1-v1131-build')) {
        Write-Line 'W1: v1.13.1 GPU build and the capture program (about 10 minutes)'
        $log = LogOf 'W1-v1131-build'; if (Test-Path -LiteralPath $log) { Remove-Item -LiteralPath $log }
        if ((Run $log 'cmake' @('-S', $V, '-B', "$V\build", '-G', 'Visual Studio 17 2022', '-A', 'x64', '-T', 'v143,version=14.34',
                                '-DSUPERSLM_BUILD_GPU=ON', '-DBUILD_TESTING=OFF')) -ne 0) { Red 'W1' "cmake configure failed; see $log" }
        if ((Run $log 'cmake' @('--build', "$V\build", '--config', 'Release', '--target', 'superslm', 'superslm_gpu_shaders')) -ne 0) { Red 'W1' "cmake build failed; see $log" }
        if (-not (Test-Path -LiteralPath "$V\build\Release\superslm.lib") -or @(Get-ChildItem "$V\build\Release\shaders\*.cso" -ErrorAction SilentlyContinue).Count -eq 0) { Red 'W1' "no superslm.lib or shaders under $V\build\Release" }
        $null = Run $log 'cmd' @('/c', "$E5Src\tests\paged-kv-gpu\build_paged_kv_gpu.bat", ('"--out=' + $pkvg + '"'), ('"--capture910-tree=' + $V + '"'), 'capture910')
        $linked = Select-String -LiteralPath $log -Pattern 'LINKED \(v1\.13\.1\): .*pcg_cell_9_10_capture\.exe' -SimpleMatch:$false
        if (-not $linked -or -not (Test-Path -LiteralPath "$pkvg\v1131\pcg_cell_9_10_capture.exe")) { Red 'W1' "the capture program did not link; see $log" }
        Stamp 'W1-v1131-build' 'LINKED (v1.13.1) pcg_cell_9_10_capture.exe'
    }

    $env:SSLM_GPU_ADAPTER_INDEX = [string]$amd.Index

    # W2 = G2: the v1.13.1 capture on the RX 7900 XTX.
    if (-not (Done 'W2-9.10-capture')) {
        Write-Line "W2: 9.10 capture on adapter $($amd.Index)"
        $log = LogOf 'W2-9.10-capture-7900xtx'; if (Test-Path -LiteralPath $log) { Remove-Item -LiteralPath $log }
        $code = Run $log "$pkvg\v1131\pcg_cell_9_10_capture.exe" @("--fixture-dir=$FX", "--out=$captures")
        Get-ChildItem $captures -Recurse -File -ErrorAction SilentlyContinue | Get-FileHash | ForEach-Object { Append $log ("{0}  {1}" -f $_.Hash.ToLower(), $_.Path) }
        if ($code -ne 0) { Red 'W2' "the capture exited $code; see $log" }
        $dirs = @(Get-ChildItem $captures -Directory -Filter 'v1.13.1_1002_*' -ErrorAction SilentlyContinue)
        if ($dirs.Count -ne 1) { Red 'W2' "expected one captures910\v1.13.1_1002_<device>\ folder, found $($dirs.Count)" }
        $bins = @(Get-ChildItem $dirs[0].FullName -Filter '*.bin')
        if (-not (Test-Path -LiteralPath (Join-Path $dirs[0].FullName 'captures.txt')) -or $bins.Count -ne 5) { Red 'W2' "$($dirs[0].Name) lacks captures.txt or five .bin files" }
        $dest = Join-Path $results ('captures910\' + $dirs[0].Name)
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        Copy-Item -Path (Join-Path $dirs[0].FullName '*') -Destination $dest -Force
        Stamp 'W2-9.10-capture' ("captures=" + $dirs[0].Name)
    }

    # W3 = G3: the release-tree GPU build and the 9.10 cell.
    if (-not (Done 'W3-release-gpu-build')) {
        Write-Line 'W3: release-tree GPU build and the 9.10 cell (about 10 minutes)'
        $log = LogOf 'W3-release-gpu-build'; if (Test-Path -LiteralPath $log) { Remove-Item -LiteralPath $log }
        if ((Run $log 'cmake' @('-S', $E5Src, '-B', "$E5Src\build", '-G', 'Visual Studio 17 2022', '-A', 'x64', '-T', 'v143,version=14.34',
                                '-DSUPERSLM_BUILD_GPU=ON', '-DBUILD_TESTING=OFF')) -ne 0) { Red 'W3' "cmake configure failed; see $log" }
        if ((Run $log 'cmake' @('--build', "$E5Src\build", '--config', 'Release', '--target', 'superslm', 'superslm_gpu_shaders')) -ne 0) { Red 'W3' "cmake build failed; see $log" }
        $null = Run $log 'cmd' @('/c', "$E5Src\tests\paged-kv-gpu\build_paged_kv_gpu.bat", ('"--out=' + $pkvg + '"'), 'e5')
        $linked = Select-String -LiteralPath $log -Pattern 'pcg_cell_9_10: LINKED'
        if (-not $linked -or -not (Test-Path -LiteralPath "$pkvg\bin\pcg_cell_9_10.exe")) { Red 'W3' "pcg_cell_9_10 did not link; see $log" }
        Stamp 'W3-release-gpu-build' 'pcg_cell_9_10: LINKED'
    }

    # W4 = G4: 9.10 on the RX 7900 XTX.
    if (-not (Done 'W4-9.10')) {
        Write-Line "W4: 9.10 on adapter $($amd.Index)"
        $log = LogOf 'W4-9.10-7900xtx'; if (Test-Path -LiteralPath $log) { Remove-Item -LiteralPath $log }
        $code = Run $log "$pkvg\bin\pcg_cell_9_10.exe" @("--fixture-dir=$FX", "--captures=$captures")
        $pass = Select-String -LiteralPath $log -Pattern 'pcg_cell_9_10: checks=(\d+) failures=0 skips=0 -> PASS'
        if ($code -ne 0 -or -not $pass) { Red 'W4' "9.10 is not green (exit $code); see $log" }
        Stamp 'W4-9.10' $pass[0].Line
    }

    Write-Summary
    Write-Line 'ALL DONE: W1-W4 green. Bring the drive back (e5-w\results\).'
    exit 0
} finally {
    Stop-Transcript | Out-Null
}
