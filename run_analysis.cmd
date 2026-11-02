@echo off
"%~dp0.venv\Scripts\python.exe" -u "%~dp0run_analysis.py" %*
exit /b %errorlevel%
