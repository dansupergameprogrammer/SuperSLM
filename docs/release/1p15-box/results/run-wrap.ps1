# Loads the BuildTools x64 environment (MSVC 19.44, as B7's build), then runs the 1.15 box driver unchanged.
$vc = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat'
$lines = & cmd.exe /d /c "call `"$vc`" x64 >nul 2>&1 && set"
foreach ($l in $lines) { $at = $l.IndexOf('='); if ($at -gt 0) { Set-Item -Path ('env:' + $l.Substring(0, $at)) -Value $l.Substring($at + 1) } }
"VCToolsVersion=$env:VCToolsVersion"
(& cmd.exe /d /c 'cl 2>&1' | Select-Object -First 1)
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\_scratch\sfd-1p15\src\docs\release\1p15-box\run-1p15-box.ps1'
"driver exit: $LASTEXITCODE"
