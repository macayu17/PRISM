@echo off
setlocal

python check_a4000_ready.py %*
exit /b %errorlevel%
