@echo off
rem Local voice stack: mic -> STT -> Qwen 27B -> Pocket TTS -> speakers
rem Ctrl-C to quit. First run of the day: make sure llama-server (port 8080)
rem is running first.
cd /d "%~dp0"
python -m voice_stack %*
echo.
pause
