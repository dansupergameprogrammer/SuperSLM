function Resolve-SuperSlmPython {
    $requested = if ($env:SUPERSLM_PYTHON) { $env:SUPERSLM_PYTHON } else { 'python' }
    $command = Get-Command $requested -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $command) { throw "Python interpreter not found: $requested" }
    Write-Host "PYTHON RESOLVED $($command.Source)"
    return $command.Source
}

function Enter-SuperSlmVsDevShell {
    $vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
        throw "Visual Studio locator not found: $vswhere"
    }
    $installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0 -or -not $installation) {
        throw 'Visual Studio installation with Microsoft.VisualStudio.Component.VC.Tools.x86.x64 not found by vswhere'
    }
    $devShell = Join-Path ($installation | Select-Object -First 1) 'Common7\Tools\Launch-VsDevShell.ps1'
    if (-not (Test-Path -LiteralPath $devShell -PathType Leaf)) {
        throw "Visual Studio developer shell not found: $devShell"
    }
    Write-Output "VS DEV SHELL RESOLVED $devShell"
    $env:PATH = "$(Split-Path -Parent $vswhere);$env:PATH"
    & $devShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null
    if (-not (Get-Command cl -CommandType Application -ErrorAction SilentlyContinue)) {
        throw "Visual Studio developer shell did not put cl on PATH: $devShell"
    }
}
