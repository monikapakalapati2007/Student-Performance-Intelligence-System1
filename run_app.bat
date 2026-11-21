@echo off
REM Start the Student Performance Intelligence System on http://127.0.0.1:8000
REM Uses the project virtual environment (no global installs, no extra dependencies).
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  set PY=.venv\Scripts\python.exe
) else (
  set PY=python
)
"%PY%" app\server.py --open %*
endlocal
