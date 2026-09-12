@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please follow README.md to create the Python 3.12 environment first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m assembly_workbench %*
if errorlevel 1 pause
