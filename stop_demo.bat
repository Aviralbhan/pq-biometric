@echo off
rem Stop only the demo server started by start_demo.bat (other Python jobs keep running).
powershell -NoProfile -Command "$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and ($_.CommandLine -match 'demo\.py.{0,3}server' -or ($_.CommandLine -match '-m server' -and $_.CommandLine -match 'var.demo.config\.yaml')) }; if (-not $p) { 'The demo server is not running.' } else { $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; 'Stopped the demo server (' + @($p).Count + ' processes).' }"
