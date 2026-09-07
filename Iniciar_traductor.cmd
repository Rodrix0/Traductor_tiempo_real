@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -X utf8 traductor.py
if errorlevel 1 (
  echo.
  echo No se pudo iniciar. Consulta la instalacion en README.md.
  pause
)
