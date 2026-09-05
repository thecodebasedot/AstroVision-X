# Install AstroVision-X for the current user on Windows.
#
#   powershell -ExecutionPolicy Bypass -File packaging\install.ps1
#
# Needs Python 3.9 or newer (https://www.python.org/downloads/windows/,
# tick "Add python.exe to PATH").  Everything goes into one folder,
# %LOCALAPPDATA%\AstroVision-X, with shortcuts on the Desktop and in the
# Start Menu.  Delete that folder and the shortcuts to uninstall.
$ErrorActionPreference = "Stop"

$Prefix = if ($env:ASTROVISION_HOME) { $env:ASTROVISION_HOME } else { Join-Path $env:LOCALAPPDATA "AstroVision-X" }
$Repo   = if ($env:ASTROVISION_REPO) { $env:ASTROVISION_REPO } else { "https://github.com/thecodebasedot/AstroVision-X.git" }
$Extras = if ($env:ASTROVISION_EXTRAS) { $env:ASTROVISION_EXTRAS } else { "science,ml" }
$Source = $env:ASTROVISION_SOURCE
if (-not $Source) {
    $here = Split-Path -Parent $MyInvocation.MyCommand.Path
    if (Test-Path (Join-Path $here "..\pyproject.toml")) { $Source = (Resolve-Path (Join-Path $here "..")).Path }
}

function Find-Python {
    foreach ($cmd in @("py -3.12", "py -3.11", "py -3.10", "py -3.9", "py -3", "python", "python3")) {
        try {
            $ok = & cmd /c "$cmd -c ""import sys; print(sys.version_info >= (3, 9))"" 2>nul"
            if ($ok -match "True") { return $cmd }
        } catch {}
    }
    return $null
}
$Python = Find-Python
if (-not $Python) { Write-Error "AstroVision-X needs Python 3.9 or newer. Install it from https://www.python.org/downloads/windows/ (tick 'Add python.exe to PATH') and run this again." }
Write-Host "Using $Python"

New-Item -ItemType Directory -Force -Path $Prefix | Out-Null
$Venv = Join-Path $Prefix "venv"
$VPy  = Join-Path $Venv "Scripts\python.exe"
$VPyw = Join-Path $Venv "Scripts\pythonw.exe"
if (-not (Test-Path $VPy)) {
    Write-Host "Creating a private Python environment in $Venv"
    & cmd /c "$Python -m venv ""$Venv"""
}
& $VPy -m pip install --quiet --upgrade pip
if ($Source) {
    Write-Host "Installing from $Source with extras [$Extras]"
    & $VPy -m pip install --quiet --upgrade "$Source[$Extras]"
} else {
    Write-Host "Installing from $Repo with extras [$Extras]"
    & $VPy -m pip install --quiet --upgrade "astrovision-x[$Extras] @ git+$Repo"
}

# Launchers: a console one (shows the log, Ctrl-C stops) and shortcuts.
$Launcher = Join-Path $Prefix "AstroVision-X.cmd"
"@echo off`r`n""$VPy"" -m astrovision.gui %*`r`n" | Set-Content -Path $Launcher -Encoding ASCII
$Cli = Join-Path $Prefix "astrovision.cmd"
"@echo off`r`n""$VPy"" -m astrovision.cli.main %*`r`n" | Set-Content -Path $Cli -Encoding ASCII

$Shell = New-Object -ComObject WScript.Shell
foreach ($dir in @([Environment]::GetFolderPath("Desktop"), (Join-Path ([Environment]::GetFolderPath("StartMenu")) "Programs"))) {
    $lnk = $Shell.CreateShortcut((Join-Path $dir "AstroVision-X.lnk"))
    $lnk.TargetPath = $Launcher
    $lnk.WorkingDirectory = [Environment]::GetFolderPath("MyDocuments")
    $lnk.Description = "AstroVision-X desktop application"
    $lnk.Save()
}

& $VPy -m astrovision.cli.main info | Select-Object -Skip 8
Write-Host ""
Write-Host "Installed. Start AstroVision-X from the Desktop or Start Menu shortcut, or run:"
Write-Host "  $Launcher"
