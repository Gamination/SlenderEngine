@echo off
rem One-step setup of this fork for Qwen3.8-Flash-Next GSQ-RCO IQ3_S on a 32 GB PC. Nothing of the model is downloaded:
rem the first run asks where its files are (or drag their folder onto this file), builds the engine, then tunes for a
rem 32 GB PC and offers dual SSD streaming. Later runs only tune. tools\fast_setup.py does the work; --help lists its options.
setlocal
set "PYTHONUTF8=1"
title SlenderEngine fast setup
cd /d "%~dp0"
rem Python and the .venv first (installed for this user if missing, no admin needed); nothing else is set up here
call START-HERE.bat --python-only
if errorlevel 1 exit /b 1
set "PY=.venv\Scripts\python.exe"
if exist "python\python.exe" set "PY=python\python.exe"
"%PY%" tools\fast_setup.py %*
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
