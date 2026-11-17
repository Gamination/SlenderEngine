@echo off
rem One-step setup of Qwen3.6-35B-A3B (Unsloth UD-Q4_K_M with its MTP block). Nothing of the model is downloaded: the
rem first run asks where the file is (or drag it onto this file; a copy in the Hugging Face cache is used without asking),
rem builds the engine for this card, then tools\fast_setup_q36.py writes run-qwen36-fast.bat with the switches measured in
rem docs\fork\QWEN36-12GB.md for this PC's card, RAM and CPU. Later runs only tune; --help lists the options.
setlocal
set "PYTHONUTF8=1"
title SlenderEngine fast setup: Qwen3.6-35B-A3B
cd /d "%~dp0"
rem Python and the .venv first (installed for this user if missing, no admin needed); nothing else is set up here
call START-HERE.bat --python-only
if errorlevel 1 exit /b 1
set "PY=.venv\Scripts\python.exe"
if exist "python\python.exe" set "PY=python\python.exe"
"%PY%" tools\fast_setup_q36.py %*
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
