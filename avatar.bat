@echo off
rem Runs the VedioAvatar CLI with the project's virtual environment (no activation needed).
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo The virtual environment is missing. Run:  powershell -ExecutionPolicy Bypass -File setup.ps1
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -m avatar_pipeline.cli %*
