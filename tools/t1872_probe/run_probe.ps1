<#
.SYNOPSIS
  T-1872 portable probe -- drive-root entry point's real logic.

.DESCRIPTION
  Extracts T1872_Probe.tar (sitting next to this script, on the drive) to a
  clearly-named temporary folder under the current user's %TEMP%, runs the
  probe from there, then deletes that temporary folder -- on the success path
  AND on any failure, via try/finally, so a probe that dies partway through
  does not leave ~9 GB behind on a machine that is not Dan's own.

  WHY EXTRACT AT ALL, rather than run the interpreter straight off the drive:
  a native-Windows ROCm PyTorch install is a large tree of small files (the
  same shape that made a directory-tree COPY to this drive catastrophically
  slow -- 0.04 MB/s measured, a budget flash controller's random-write floor,
  not its sequential one). Running Python's import machinery and the HIP
  runtime against that many small files sitting on the same slow flash medium
  would be exactly as painful. Local disk (SSD or HDD, either one) handles
  many small files far better than USB flash does, so the fix for the copy
  problem (ship one big sequential file) and the fix for the run-time problem
  (execute off local disk) are the same fix, applied once.

  RESULTS.txt and the results\ folder are written back to THIS SCRIPT'S OWN
  directory (the drive), never to the temporary extraction folder -- see
  orchestrator.py's T1872_RESULTS_DIR handling. That is what makes the results
  survive the cleanup below.
#>

$ErrorActionPreference = "Stop"
$DriveRoot = $PSScriptRoot
$Archive = Join-Path $DriveRoot "T1872_Probe.tar"
$RunId = [Guid]::NewGuid().ToString().Substring(0, 8)
$TempDir = Join-Path $env:TEMP "T1872_PortableProbe_run_$RunId"
$RequiredFreeGB = 10.0
$exitCode = 1

Write-Host "T-1872 portable probe -- launcher"
Write-Host "Drive root: $DriveRoot"
Write-Host ""

if (-not (Test-Path $Archive)) {
    Write-Host "ERROR: could not find $Archive"
    Write-Host "This script expects T1872_Probe.tar in the same folder as itself."
    exit 1
}

# --- Free-space courtesy check, BEFORE writing anything ---
$tempDriveLetter = (Get-Item $env:TEMP).PSDrive.Name
$freeGB = (Get-PSDrive $tempDriveLetter).Free / 1GB
Write-Host ("Temp drive {0}: has {1:N1} GB free; this probe needs about 9 GB " -f $tempDriveLetter, $freeGB) `
    "temporarily (deleted automatically when this finishes, on success OR failure)."
if ($freeGB -lt $RequiredFreeGB) {
    Write-Host ("NOT ENOUGH FREE SPACE on {0}: -- need at least {1:N0} GB free, found {2:N1} GB. " -f `
        $tempDriveLetter, $RequiredFreeGB, $freeGB) "Stopping before extracting anything."
    exit 1
}

try {
    Write-Host "Extracting probe to a temporary local folder: $TempDir"
    New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
    & tar.exe -xf $Archive -C $TempDir
    if ($LASTEXITCODE -ne 0) {
        throw "tar extraction failed with exit code $LASTEXITCODE -- is tar.exe on PATH? " +
              "(Windows ships one at C:\Windows\System32\tar.exe by default.)"
    }
    Write-Host "Extraction complete. Running the probe from local disk..."
    Write-Host ""

    $env:T1872_RESULTS_DIR = $DriveRoot
    & "$TempDir\python\python.exe" "$TempDir\probe\orchestrator.py"
    $exitCode = $LASTEXITCODE
}
catch {
    Write-Host ""
    Write-Host "ERROR: $($_.Exception.Message)"
    $exitCode = 1
}
finally {
    if (Test-Path $TempDir) {
        Write-Host ""
        Write-Host "Cleaning up temporary extraction folder (this always runs, pass or fail)..."
        Remove-Item -Recurse -Force $TempDir -ErrorAction SilentlyContinue
        if (Test-Path $TempDir) {
            Write-Host "WARNING: could not fully remove $TempDir -- you may want to delete it by hand."
        } else {
            Write-Host "Done -- no trace left on this machine's local disk."
        }
    }
}

exit $exitCode
