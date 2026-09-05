@echo off
rem Install AstroVision-X on Windows: double-click this file.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
if errorlevel 1 (
  echo.
  echo The installation did not finish. The message above says why.
)
pause
