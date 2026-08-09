[CmdletBinding()]
param(
    [string]$Python = "",
    [string]$Version = "",
    [switch]$SkipInstall,
    [switch]$SkipTests,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$distRoot = Join-Path $projectRoot "dist"
$workRoot = Join-Path $projectRoot "build"
$artifactRoot = Join-Path $projectRoot "artifacts"
$bundleDir = Join-Path $distRoot "AudioAlignTool"

function Assert-ChildPath {
    param([Parameter(Mandatory)][string]$Path)
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $rootWithSeparator = $projectRoot.TrimEnd('\') + '\'
    if (-not $fullPath.StartsWith($rootWithSeparator, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to modify a path outside the repository: $fullPath"
    }
    return $fullPath
}

function Remove-BuildPath {
    param([Parameter(Mandatory)][string]$Path)
    $safePath = Assert-ChildPath $Path
    if (Test-Path -LiteralPath $safePath) {
        Remove-Item -LiteralPath $safePath -Recurse -Force
    }
}

Push-Location $projectRoot
try {
    if ([string]::IsNullOrWhiteSpace($Python)) {
        if (-not (Test-Path -LiteralPath $venvPython)) {
            $launcher = Get-Command py -ErrorAction SilentlyContinue
            if ($null -eq $launcher) {
                $installedPython = Join-Path $env:LOCALAPPDATA "Python\pythoncore-3.14-64\python.exe"
                if (-not (Test-Path -LiteralPath $installedPython)) {
                    throw "Neither .venv nor Python 3.14 was found. Install Python 3.14 or pass -Python."
                }
            }
            Write-Host "Creating the Python 3.14 virtual environment: .venv"
            if ($null -ne $launcher) {
                & $launcher.Source -3.14 -m venv (Join-Path $projectRoot ".venv")
            } else {
                & $installedPython -m venv (Join-Path $projectRoot ".venv")
            }
            if ($LASTEXITCODE -ne 0) { throw "Failed to create the virtual environment." }
        }
        $pythonPath = $venvPython
    } else {
        $pythonCommand = Get-Command $Python -ErrorAction Stop
        $pythonPath = $pythonCommand.Source
    }

    & $pythonPath -c "import sys; assert sys.version_info[:2] == (3, 14), f'Python 3.14 is required; found {sys.version}'"
    if ($LASTEXITCODE -ne 0) { throw "Python version validation failed." }

    if (-not $SkipInstall) {
        Write-Host "Installing locked runtime and packaging dependencies..."
        & $pythonPath -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }
        & $pythonPath -m pip install -r requirements-build.txt
        if ($LASTEXITCODE -ne 0) { throw "Failed to install packaging dependencies." }
        & $pythonPath -m pip install --no-deps -e .
        if ($LASTEXITCODE -ne 0) { throw "Failed to install AudioAlignTool." }
    }

    if ([string]::IsNullOrWhiteSpace($Version)) {
        $versionLine = Select-String -Path "pyproject.toml" -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1
        if ($null -eq $versionLine) { throw "Could not read the version from pyproject.toml." }
        $Version = $versionLine.Matches[0].Groups[1].Value
    }
    $safeVersion = $Version -replace '[^0-9A-Za-z._-]', '-'
    if ([string]::IsNullOrWhiteSpace($safeVersion)) { throw "The version is not valid for an artifact filename." }

    if ($Clean) {
        Write-Host "Removing old packaging output..."
        Remove-BuildPath $distRoot
        Remove-BuildPath $workRoot
        Remove-BuildPath $artifactRoot
    }

    if (-not $SkipTests) {
        Write-Host "Running tests..."
        $env:QT_QPA_PLATFORM = "offscreen"
        $env:PYTHONDONTWRITEBYTECODE = "1"
        & $pythonPath -m unittest discover -s tests -v
        if ($LASTEXITCODE -ne 0) { throw "Tests failed; packaging stopped." }
    }

    Write-Host "Building the PyInstaller onedir bundle..."
    & $pythonPath -m PyInstaller --noconfirm --clean --distpath $distRoot --workpath $workRoot AudioAlignTool.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

    $executable = Join-Path $bundleDir "AudioAlignTool.exe"
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "The build finished without producing: $executable"
    }

    New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null
    $zipPath = Join-Path $artifactRoot "AudioAlignTool-$safeVersion-Windows-x64-portable.zip"
    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath (Assert-ChildPath $zipPath) -Force
    }

    Write-Host "Creating the portable ZIP..."
    Compress-Archive -LiteralPath $bundleDir -DestinationPath $zipPath -CompressionLevel Optimal

    $hash = Get-FileHash -LiteralPath $zipPath -Algorithm SHA256
    $checksumPath = Join-Path $artifactRoot "SHA256SUMS.txt"
    "{0}  {1}" -f $hash.Hash.ToLowerInvariant(), [System.IO.Path]::GetFileName($zipPath) |
        Set-Content -LiteralPath $checksumPath -Encoding utf8

    Write-Host ""
    Write-Host "Portable package complete:"
    Write-Host "  $zipPath"
    Write-Host "  $checksumPath"
} finally {
    Pop-Location
}
