param([switch]$IsolatedBrowser, [switch]$ViewerOnly, [string]$Study = '', [switch]$PasswordStdin, [switch]$AnatomyReview, [switch]$EditBoundaries)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Install the project Python environment first.' }
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'web\node_modules'))) { throw 'Run npm.cmd ci in web first.' }
$viewerArguments = @((Join-Path $projectRoot 'web\scripts\local-viewer.mjs'), 'http://127.0.0.1:5173')
if ($Study) { $viewerArguments += $Study }
if ($AnatomyReview) { $viewerArguments += '--anatomy' }
if ($EditBoundaries) { $viewerArguments += '--edit-boundaries' }
if ($PasswordStdin) {
    if (-not $Study) { throw 'A pseudonymous study is required for stdin login.' }
    $viewerArguments += '--password-stdin'
}
if ($ViewerOnly) {
    & node @viewerArguments
    exit $LASTEXITCODE
}
foreach ($port in @(8000,5173)) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $port is already listening. Reuse or stop that server before starting this script."
    }
}
New-Item -ItemType Directory -Path (Join-Path $projectRoot 'outputs') -Force | Out-Null
$apiProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m','uvicorn','heart3d.web.app:create_app','--factory','--host','127.0.0.1','--port','8000','--no-access-log') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $projectRoot 'outputs\annotation-api.log') -RedirectStandardError (Join-Path $projectRoot 'outputs\annotation-api-errors.log')
try {
    Write-Host 'Heart3D Annotation: http://127.0.0.1:5173'
    Write-Host 'Keep this terminal open. Ctrl+C stops the owned servers.'
    if ($IsolatedBrowser) {
        $nodeExecutable = (Get-Command node -ErrorAction Stop).Source
        $frontendProcess = Start-Process -FilePath $nodeExecutable -ArgumentList @((Join-Path $projectRoot 'web\node_modules\vite\bin\vite.js')) -WorkingDirectory (Join-Path $projectRoot 'web') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $projectRoot 'outputs\annotation-vite.log') -RedirectStandardError (Join-Path $projectRoot 'outputs\annotation-vite-errors.log')
        $ready = $false
        for ($attempt = 0; $attempt -lt 80; $attempt++) {
            if ($apiProcess.HasExited -or $frontendProcess.HasExited) { throw 'A local server exited; inspect outputs/annotation-*.log.' }
            try {
                $apiStatus = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 1
                $pageStatus = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:5173/' -TimeoutSec 1
                if ($apiStatus.StatusCode -eq 200 -and $pageStatus.StatusCode -eq 200) { $ready = $true; break }
            } catch {}
            Start-Sleep -Milliseconds 250
        }
        if (-not $ready) { throw 'Local server readiness timed out.' }
        & $nodeExecutable @viewerArguments
    } else {
        Set-Location -LiteralPath (Join-Path $projectRoot 'web')
        npm.cmd run dev
    }
} finally {
    if ($frontendProcess -and -not $frontendProcess.HasExited) { Stop-Process -Id $frontendProcess.Id }
    if (-not $apiProcess.HasExited) { Stop-Process -Id $apiProcess.Id }
    Set-Location -LiteralPath $projectRoot
}
