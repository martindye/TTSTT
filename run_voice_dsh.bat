@echo off
rem Voice bridge: mic -> Kyutai STT -> DSH voice session -> Pocket TTS -> speakers
cd /d %~dp0
python -X utf8 -m voice_stack.voice_dsh %*
