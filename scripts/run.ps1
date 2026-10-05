param(
    [ValidateSet('webcam', 'cameras', 'doctor')]
    [string]$Mode = 'webcam',
    [string]$Source = '0',
    [switch]$FaceId,
    [switch]$Headless
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    $pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run scripts/setup.ps1 first' }
    if ($Mode -eq 'doctor') {
        & $pythonPath -m sentrycare.doctor --camera 0
    } elseif ($Mode -eq 'cameras') {
        & $pythonPath -m sentrycare.cameras
    } else {
        $launchArgs = @('-m', 'sentrycare.app', '--source', $Source, '--camera-id', 'laptop')
        if (-not $FaceId) { $launchArgs += '--no-face-id' }
        if ($Headless) { $launchArgs += '--headless' }
        & $pythonPath @launchArgs
    }
    if ($LASTEXITCODE -ne 0) { throw "SentryCare exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
