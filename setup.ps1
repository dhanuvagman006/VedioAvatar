# Windows bootstrap. Run from a PowerShell prompt inside the repo:
#   powershell -ExecutionPolicy Bypass -File setup.ps1
# Extra arguments are passed to scripts/setup_envs.py, e.g.  -File setup.ps1 --lipsync both
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { throw "Python 3.10+ is required. Install from https://www.python.org/downloads/ and tick 'Add python.exe to PATH'." }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "git is required. Install from https://git-scm.com/downloads" }

if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe scripts\setup_envs.py @args
& .\.venv\Scripts\python.exe -m avatar_pipeline.cli doctor
Write-Host ""
Write-Host "Ready. Activate with:  .\.venv\Scripts\Activate.ps1"
Write-Host "Then:  python -m avatar_pipeline.cli serve   (web UI on http://localhost:8000)"
