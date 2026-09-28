$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& "$PSScriptRoot\.venv\Scripts\python.exe" -m heart3d view "$PSScriptRoot\outputs\ct_1001_v1"
exit $LASTEXITCODE
