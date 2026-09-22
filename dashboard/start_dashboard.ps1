$ErrorActionPreference = "Stop"

$dashboardDirectory = $PSScriptRoot
$bundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$localPython = Join-Path $dashboardDirectory ".venv\Scripts\python.exe"

if (Test-Path -LiteralPath $localPython) {
    $pythonExecutable = $localPython
} elseif (Test-Path -LiteralPath $bundledPython) {
    $pythonExecutable = $bundledPython
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $pythonCommand) {
        throw "Python was not found. Install Python 3.11+ or create dashboard\.venv first."
    }
    $pythonExecutable = $pythonCommand.Source
}

Push-Location $dashboardDirectory
try {
    & $pythonExecutable -u run_server.py
} finally {
    Pop-Location
}
