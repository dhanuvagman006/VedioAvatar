# Windows bootstrap. Run from cmd or PowerShell inside the repo folder:
#   powershell -ExecutionPolicy Bypass -File setup.ps1
# Extra arguments are passed to scripts/setup_envs.py, e.g.  -File setup.ps1 --lipsync both
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$scriptArgs = $args

function Test-RealPython {
    # Returns @(exe, args...) for a working Python >= 3.10, or $null.
    # Rejects the Microsoft Store shortcut that Windows installs under WindowsApps.
    param([string]$Exe, [string[]]$ExtraArgs)
    $ErrorActionPreference = "SilentlyContinue"
    $cmd = Get-Command $Exe -ErrorAction SilentlyContinue
    if (-not $cmd) { return $null }
    if ($cmd.Source -like "*WindowsApps*") { return $null }
    $out = & $Exe @ExtraArgs -c "import sys; print(sys.version_info[0]*100 + sys.version_info[1])" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }
    $ver = 0
    if (-not [int]::TryParse("$out".Trim(), [ref]$ver)) { return $null }
    if ($ver -lt 310) { return $null }
    return @($Exe) + $ExtraArgs
}

function Invoke-Checked {
    param([string]$Exe, [string[]]$ArgList)
    Write-Host "> $Exe $($ArgList -join ' ')" -ForegroundColor DarkGray
    & $Exe @ArgList
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit code $LASTEXITCODE): $Exe $($ArgList -join ' ')" }
}

$python = $null
foreach ($candidate in @(
        @("py", @("-3.11")), @("py", @("-3.12")), @("py", @("-3.10")), @("py", @("-3")),
        @("python", @()), @("python3", @()))) {
    $python = Test-RealPython -Exe $candidate[0] -ExtraArgs $candidate[1]
    if ($python) { break }
}
if (-not $python) {
    Write-Host ""
    Write-Host "Python 3.10 or newer was not found." -ForegroundColor Red
    Write-Host "(If 'python' printed a Microsoft Store message, that is only a shortcut, not a real Python.)"
    Write-Host ""
    Write-Host "Install Python 3.11 from https://www.python.org/downloads/windows/ and in the installer tick:"
    Write-Host "    [x] Add python.exe to PATH        [x] py launcher (default)"
    Write-Host "Alternatively:  winget install -e --id Python.Python.3.11"
    Write-Host "Then open a NEW terminal window and run this script again."
    exit 1
}
$pyExe = $python[0]
$pyArgs = @()
if ($python.Length -gt 1) { $pyArgs = $python[1..($python.Length - 1)] }
Write-Host "Using Python: $($python -join ' ')" -ForegroundColor Green

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "git is required: https://git-scm.com/downloads" -ForegroundColor Red
    exit 1
}

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Invoke-Checked -Exe $pyExe -ArgList ($pyArgs + @("-m", "venv", ".venv"))
}
if (-not (Test-Path $venvPy)) { throw "Virtual environment creation failed: $venvPy does not exist" }

Invoke-Checked -Exe $venvPy -ArgList @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked -Exe $venvPy -ArgList @("-m", "pip", "install", "-r", "requirements.txt")
Invoke-Checked -Exe $venvPy -ArgList (@("scripts\setup_envs.py") + $scriptArgs)

& $venvPy -m avatar_pipeline.cli doctor
Write-Host ""
Write-Host "Setup complete. From cmd or PowerShell in this folder (no activation needed):" -ForegroundColor Green
Write-Host "  avatar doctor"
Write-Host "  avatar run --video examples\sample_face.mp4 --script examples\sample_script.txt --out out.mp4"
Write-Host "  avatar serve        (web UI + API on http://localhost:8000)"
Write-Host "In PowerShell type .\avatar instead of avatar."
