---
name: start-voice
description: Start, stop, or check the status of the TTSTT voice assistant (mic -> local DSH agent -> speakers on the user's machine).
whenToUse: When the user asks to start, launch, boot, or wake the voice assistant; stop or shut it down; or asks whether the voice assistant is running.
---

# Voice assistant (TTSTT project)

The voice assistant is a bridge process: microphone -> Kyutai STT (GPU) -> a
dedicated DSH session (persona: `voice_persona\`, home: `TTSTT\.dsh-voice`)
-> Pocket TTS -> speakers. Project root:
`C:\Users\press\OneDrive\Projects\TTSTT`.

## Check status first

    Get-Content C:\Users\press\OneDrive\Projects\TTSTT\tests\live_dsh_log.txt -Tail 5

It is running if the last lines are recent 10-second heartbeat lines
(`mic raw rms=... | stt ms/frame=...`) with timestamps within the last minute.
`Voice bridge ready` marks a successful boot. An old/absent log = not running.
Never start a second instance while one is running (two processes would fight
over the microphone) — stop the old one first.

## Start

1. Prerequisite — the local LLM server must be up (shared GPU):

       Invoke-WebRequest http://127.0.0.1:8080/v1/models -UseBasicParsing

   If it fails, tell the user the llama.cpp server must be started first and
   stop here.

2. Start the bridge as a background pwsh job (workdir
   `C:\Users\press\OneDrive\Projects\TTSTT`, `run_in_background: true`):

       python -X utf8 -m voice_stack.voice_dsh 2>&1 | Out-File -Encoding utf8 tests\live_dsh_log.txt

3. Give it 35-45 seconds (STT ~10s, TTS ~5s, DSH runtime ~5s, mic open), then
   confirm `Voice bridge ready` appears in `tests\live_dsh_log.txt` and tell
   the user to speak.

Useful options (append to the command):
- `--tts-voice <name>` — voice; default `eve`. Female voices: anna, vera,
  fantine, eponine, azelma, mary, jane, eve, cosette, caro_davy.
- `--mic-gain <factor>` — default 1.0 (leave at 1.0 unless the user's mic is
  too quiet; do not chase gain on your own).

## Stop

    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
      Where-Object { $_.CommandLine -like '*voice_stack.voice_dsh*' } |
      ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

This also stops the DSH SDK runtime coprocess. Confirm the heartbeat lines
stopped in the log, then report.

## Notes

- Each start is a fresh conversation for the voice agent (no session resume).
- The voice agent is invisible in this GUI; its session journal lives under
  `TTSTT\.dsh-voice\sessions`. Its logs: `tests\live_dsh_log.txt`.
- Voice -> coding-agent handoff: the voice agent can run
  `python -X utf8 -m voice_stack.handoff "<task>"`, which starts a full DSH
  coding session in TTSTT that appears in the GUI session list.
