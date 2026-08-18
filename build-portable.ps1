[CmdletBinding()]
param(
    [string]$Python = "",
    [string]$Version = "",
    [switch]$SkipInstall,
    [switch]$SkipTests,
    [switch]$Clean,
    [switch]$Standard
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$distRoot = Join-Path $projectRoot "dist"
$workRoot = Join-Path $projectRoot "build"
$artifactRoot = Join-Path $projectRoot "artifacts"
$bundleDir = Join-Path $distRoot "AudioAlignTool"
$runtimeWheelRoot = Join-Path $projectRoot "runtime-packages\wheels"

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
                    $installedPython = Join-Path $env:LOCALAPPDATA "Python\pythoncore-3.13-64\python.exe"
                }
                if (-not (Test-Path -LiteralPath $installedPython)) {
                    throw "Neither .venv nor Python 3.13/3.14 was found. Install it or pass -Python."
                }
            }
            Write-Host "Creating the Python 3.14 virtual environment: .venv"
            if ($null -ne $launcher) {
                & $launcher.Source -3.14 -m venv (Join-Path $projectRoot ".venv")
                if ($LASTEXITCODE -ne 0) {
                    & $launcher.Source -3.13 -m venv (Join-Path $projectRoot ".venv")
                }
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

    & $pythonPath -c "import sys; assert (3, 13) <= sys.version_info[:2] < (3, 15), f'Python 3.13 or 3.14 is required; found {sys.version}'"
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

    # OmegaConf requires antlr4-python3-runtime 4.9.x, which PyPI publishes
    # only as a source archive. A frozen executable cannot host pip's PEP 517
    # build subprocess, so build this tiny pure-Python wheel here.
    Remove-BuildPath $runtimeWheelRoot
    New-Item -ItemType Directory -Path $runtimeWheelRoot -Force | Out-Null
    Write-Host "Preparing the source-only runtime compatibility wheel..."
    & $pythonPath -m pip wheel --no-deps --wheel-dir $runtimeWheelRoot antlr4-python3-runtime==4.9.3
    if ($LASTEXITCODE -ne 0) { throw "Failed to prepare the antlr4 runtime wheel." }

    # COLLECT does not reliably remove unrelated files from an existing onedir
    # target.  Always recreate the bundle so a prior portable ZIP, model, log,
    # or project cannot become nested in the next release archive.
    Remove-BuildPath $bundleDir
    Write-Host "Building the PyInstaller onedir bundle..."
    & $pythonPath -m PyInstaller --noconfirm --clean --distpath $distRoot --workpath $workRoot AudioAlignTool.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

    $executable = Join-Path $bundleDir "AudioAlignTool.exe"
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "The build finished without producing: $executable"
    }
    # PyInstaller 6 places collected data under _internal by default.  The
    # runtime component catalog is intentionally writable and belongs beside
    # the executable, so copy the source catalog into the portable layout
    # explicitly instead of depending on PyInstaller's internal data layout.
    $sourceRuntimeIndex = Join-Path $projectRoot "runtime-packages\runtime-index.json"
    if (-not (Test-Path -LiteralPath $sourceRuntimeIndex -PathType Leaf)) {
        throw "The source runtime index is missing: $sourceRuntimeIndex"
    }
    $runtimeDirectory = Join-Path $bundleDir "runtime-packages"
    New-Item -ItemType Directory -Path $runtimeDirectory -Force | Out-Null
    $runtimeIndex = Join-Path $bundleDir "runtime-packages\runtime-index.json"
    Copy-Item -LiteralPath $sourceRuntimeIndex -Destination $runtimeIndex -Force
    $runtimeWheelDestination = Join-Path $runtimeDirectory "wheels"
    Copy-Item -LiteralPath $runtimeWheelRoot -Destination $runtimeWheelDestination -Recurse -Force
    if (-not (Test-Path -LiteralPath $runtimeIndex -PathType Leaf)) {
        throw "The build finished without the local runtime index: $runtimeIndex"
    }
    if (-not (Get-ChildItem -LiteralPath $runtimeWheelDestination -Filter "antlr4_python3_runtime-4.9.3-*.whl" -File)) {
        throw "The build finished without the antlr4 runtime compatibility wheel."
    }
    if ($Standard) {
        Write-Host "Installing the shared Qwen CPU runtime beside the standard package..."
        $previousPythonPath = $env:PYTHONPATH
        $env:PYTHONPATH = Join-Path $projectRoot "src"
        try {
            & $pythonPath -c "import sys; from pathlib import Path; from audioalign.core.paths import ApplicationPaths; from audioalign.core.runtime_addons import install_runtime_component, load_runtime_index; p=ApplicationPaths(Path(sys.argv[1])); c=next(x for x in load_runtime_index(p) if x.id=='ai-qwen-cpu-win-x64'); install_runtime_component(c,p,lambda v,m: print(f'{v:.1%} {m}'))" $bundleDir
            if ($LASTEXITCODE -ne 0) { throw "Failed to prepare the shared Qwen CPU runtime." }
        } finally {
            $env:PYTHONPATH = $previousPythonPath
        }
    }

    New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null
    $flavor = if ($Standard) { "standard" } else { "portable" }
    $zipPath = Join-Path $artifactRoot "AudioAlignTool-$safeVersion-Windows-x64-$flavor.zip"
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
