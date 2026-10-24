@echo off
rem Usage: scheduler\run.bat "humanoid whole-body control"
rem Runs from wherever the repo is cloned; prefers the project venv.
set PYTHONUTF8=1
cd /d "%~dp0.."
if not exist logs mkdir logs
set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
"%PY%" pipeline.py %* >> logs\pipeline.log 2>&1
exit /b %ERRORLEVEL%
