@echo off
cd /d %~dp0
powershell -NoProfile -File voice_stack\ensure_coding_voice.ps1 -Workspace "%CD%" --tts-engine qwen3ggufbase %*
