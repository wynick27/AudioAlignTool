@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo The project-local .venv is missing. Run bootstrap.ps1 first.
  exit /b 1
)
"%PYTHON%" "%ROOT%run.py"
