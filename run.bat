@echo off
cd /d "%~dp0"
if not exist .venv python -m venv .venv
rem install requirements on first run and whenever requirements.txt changes
fc /b requirements.txt .venv\installed-requirements.txt >nul 2>&1
if errorlevel 1 (
    .venv\Scripts\python -m pip install -r requirements.txt && copy /y requirements.txt .venv\installed-requirements.txt >nul
)
start "" .venv\Scripts\pythonw.exe main.py
