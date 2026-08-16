$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $launcher) {
        & $launcher.Source -3.13 -m venv (Join-Path $root ".venv")
    } else {
        $installedPython = Join-Path $env:LOCALAPPDATA "Python\pythoncore-3.13-64\python.exe"
        if (-not (Test-Path -LiteralPath $installedPython)) {
            throw "Python 3.13 was not found. Install it or create .venv manually."
        }
        & $installedPython -m venv (Join-Path $root ".venv")
    }
}
& $python -c "import sys; assert sys.version_info[:2] == (3, 13), f'Python 3.13 is required; found {sys.version}'"
if ($LASTEXITCODE -ne 0) {
    throw "The existing .venv is not Python 3.13. Remove .venv and run bootstrap.ps1 again."
}
& $python -m pip install --upgrade pip
& $python -m pip install -r (Join-Path $root "requirements-lock.txt")
& $python -m pip install -e $root
Write-Host "AudioAlignTool .venv is ready. Run start.bat."
