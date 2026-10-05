$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw 'Install uv from https://docs.astral.sh/uv/getting-started/installation/ or use Python 3.12 with pip as described in README.md.'
    }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot '.runtime/python'
    if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
        uv venv .venv --python 3.12 --cache-dir tmp/uv-cache
        if ($LASTEXITCODE -ne 0) { throw 'Virtual environment setup failed' }
    }
    uv pip install --python .venv/Scripts/python.exe --cache-dir tmp/uv-cache -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
    uv pip install --python .venv/Scripts/python.exe --cache-dir tmp/uv-cache --no-deps -e .
    if ($LASTEXITCODE -ne 0) { throw 'Package installation failed' }
    if (-not (Test-Path -LiteralPath 'configs/cameras.json')) {
        Copy-Item -LiteralPath 'configs/cameras.example.json' -Destination 'configs/cameras.json'
    }
    & .venv/Scripts/python.exe -m sentrycare.doctor
    if ($LASTEXITCODE -ne 0) { throw 'Runtime checks failed' }
} finally {
    Pop-Location
}
