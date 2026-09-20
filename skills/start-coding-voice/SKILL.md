---
name: start-coding-voice
description: Start, stop, or check the status of the TTSTT coding voice (speak into the open coding chat; the coding agent's visible replies are spoken out loud).
whenToUse: When the user asks to start coding voice / talk to the coding agent by voice / have this chat read its replies aloud; or to stop it or check whether it is running.
---

# Coding voice (TTSTT project)

The coding voice is a bridge process: microphone -> Kyutai STT (GPU) -> the
**open coding chat** (injected through the GUI gateway, so the user's words
appear in the chat window) -> the coding agent's visible replies are followed
live and spoken with Pocket TTS (thinking and tool activity are never spoken).
Project root: `C:\Users\press\OneDrive\Projects\TTSTT`.

This is different from the voice assistant (the `start-voice` skill): that one
talks to a separate invisible voice agent; the coding voice talks to the
coding agent in the currently open chat window.

## Check status first

    Get-Content C:\Users\press\OneDrive\Projects\TTSTT\tests\coding_voice_log.txt -Tail 5

It is running if the last line is recent and recent lines show no repeated
`follow stream error`. `coding-voice: ready — speaking into session ...`
marks a successful boot. An old/absent log = not running. Never start a second
instance (two processes would fight over the microphone) — stop the old one
first.

## Start

1. Prerequisite — the DSH GUI gateway must be up (it serves the chat window
   the user is looking at):

        Invoke-WebRequest http://127.0.0.1:3080/ -UseBasicParsing

   If it fails, tell the user the DSH web app must be open first and stop here.

2. Start the bridge as a background pwsh job (workdir
   `C:\Users\press\OneDrive\Projects\TTSTT`, `run_in_background: true`):

        python -X utf8 -m voice_stack.coding_voice 2>&1 | Out-File -Encoding utf8 tests\coding_voice_log.txt

3. Give it 30-40 seconds (STT ~10s, TTS ~5s, follow stream open), then confirm
   `ready — speaking into session session-...` appears in
   `tests\coding_voice_log.txt` and tell the user to speak.

Useful options (append to the command):
- `--session <id>` — target a specific session id. Default: the most recently
  written TTSTT session (the chat the user is looking at).
- `--tts-voice <name>` — voice; default `eve`.
- `--gui <url>` — default `http://127.0.0.1:3080`.

## Stop

    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
      Where-Object { $_.CommandLine -like '*voice_stack.coding_voice*' } |
      ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

Confirm the process is gone, then report.

## Notes

- Half-duplex: while the bridge is holding the floor (from the user's
  utterance until the reply's audio has finished playing) the microphone is
  closed; the user speaks again after the reply is done.
- Spoken = the coding agent's visible text in that chat window, sentence by
  sentence. Thinking blocks and tool activity are never spoken.
- The bridge also speaks replies triggered any other way (typed in the GUI,
  handoffs) — everything the window shows is spoken. That is by design.
- Logs: `tests\coding_voice_log.txt`. Session events are only consumed, the
  gateway is never restarted by this bridge.
