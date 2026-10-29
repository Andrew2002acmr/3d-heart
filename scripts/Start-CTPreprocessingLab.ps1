$ErrorActionPreference = 'Stop'
$ctLabRoot = Split-Path -Parent $PSScriptRoot
$ctLabPython = Join-Path $ctLabRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ctLabPython)) {
    throw 'Create .venv and install requirements-ct-lab.txt first.'
}
Push-Location -LiteralPath $ctLabRoot
try {
    & $ctLabPython (Join-Path $ctLabRoot 'ct_lab\launch.py') @args
}
finally {
    Pop-Location
}
