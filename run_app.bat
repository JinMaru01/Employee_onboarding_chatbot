@echo off
setlocal

REM Check if argument is provided
if "%~1"=="" (
    echo Usage: %~nx0 filename.py
    goto :eof
)

REM Run python from the virtual environment directly (adjust path)
env\Scripts\python.exe -m %1

endlocal