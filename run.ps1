# estimationFace launcher (offline). Runs the app with the bundled runtime.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$py = Join-Path $root "runtime\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "Bundled runtime not found at $py. This build is incomplete."
    exit 1
}
& $py (Join-Path $root "app.py") @args
