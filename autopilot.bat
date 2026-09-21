@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"
rem pythonw = no console window; output is silenced in main.py
rem Prefer the project venv interpreter (has the package installed)
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" main.py ui
  exit /b 0
)
where pythonw >nul 2>nul
if %errorlevel%==0 (
  start "" pythonw.exe main.py ui
  exit /b 0
)
where pyw >nul 2>nul
if %errorlevel%==0 (
  start "" pyw -3 main.py ui
  exit /b 0
)
echo [ERROR] pythonw not found.
pause
endlocal