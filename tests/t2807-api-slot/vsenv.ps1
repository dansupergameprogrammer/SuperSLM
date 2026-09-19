# Imports the MSVC x64 developer environment (cl, link, lib, dumpbin) into the calling PowerShell
# session. Dot-source it: `. $PSScriptRoot\vsenv.ps1`. Does nothing when cl.exe is already on PATH.
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $candidates = @()
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $inst = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($inst) { $candidates += Join-Path $inst 'Common7\Tools\VsDevCmd.bat' }
    }
    $candidates += 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat'
    $candidates += 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat'
    $devcmd = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $devcmd) { throw 'vsenv.ps1: no VsDevCmd.bat found (MSVC x64 is required)' }
    cmd /c "`"$devcmd`" -arch=x64 -no_logo >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
    }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { throw "vsenv.ps1: cl.exe not on PATH after $devcmd" }
}
