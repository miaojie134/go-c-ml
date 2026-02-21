@echo off
setlocal
set SCRIPT_DIR=%~dp0

where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -3 "%SCRIPT_DIR%run_ml.py" %*
  exit /b %ERRORLEVEL%
)

where python >nul 2>nul
if %ERRORLEVEL%==0 (
  python "%SCRIPT_DIR%run_ml.py" %*
  exit /b %ERRORLEVEL%
)

echo python launcher not found. Install Python 3 and ensure "py" or "python" is in PATH.
exit /b 1
