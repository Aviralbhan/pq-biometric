@echo off
rem Demo UI (review use only): start the demo server and open http://127.0.0.1:8000/ui
rem Stop it with stop_demo.bat.
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo The .venv folder is missing. See README.md, "Quick start".
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat

powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }"
if errorlevel 1 (
  echo The demo server is already running. Opening the page.
  start "" http://127.0.0.1:8000/ui
  exit /b 0
)

echo Starting the demo server (a minimised window titled "PQ demo server")...
start "PQ demo server" /min python scripts\demo.py server

echo Loading the models, this takes about 30-60 seconds...
set /a tries=0
:wait
ping -n 3 127.0.0.1 >nul
set /a tries+=1
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:8000/health | Out-Null; exit 0 } catch { exit 1 }"
if not errorlevel 1 goto ready
if %tries% lss 90 goto wait
echo The server did not start within 3 minutes. Open the "PQ demo server" window to see why.
pause
exit /b 1

:ready
echo Server ready. Opening http://127.0.0.1:8000/ui
start "" http://127.0.0.1:8000/ui
