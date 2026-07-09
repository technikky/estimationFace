@echo off
REM estimationFace launcher (100% offline)
setlocal
set ROOT=%~dp0
"%ROOT%runtime\python.exe" "%ROOT%app.py" %*
if errorlevel 1 pause
endlocal
