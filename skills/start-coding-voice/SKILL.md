---
name: start-coding-voice
description: Start, stop, or check the status of the TTSTT coding voice (speak into the open coding chat; the coding agent's visible replies are spoken out loud). Also lists available TTS voices.
whenToUse: When the user asks to start coding voice / talk to the coding agent by voice / have this chat read its replies aloud; to stop it or check whether it is running; or to ask which voices are available / change the voice.
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

## List available voices (no bridge needed)

    cd C:\Users\press\OneDrive\Projects\TTSTT
    python -X utf8 -m voice_stack.coding_voice --list-voices

If the user wants a different voice, add `--tts-voice <name>` to the start
command (below) for that run. `eve` is the default.

## Check status first

    Get-Content C:\Users\press\.dsh\logs\coding_voice.log -Tail 5 -ErrorAction SilentlyContinue

It is running if the last line is recent and recent lines show no repeated
`follow stream error`. `coding-voice: ready — speaking into session ...`
marks a successful boot. An old/absent log = not running. Never start a second
instance (two processes would fight over the microphone) — stop the old one
first.

## Start

1. Prerequisite — the DSH GUI gateway must be up (it serves the chat window
   the user is looking at):

       Invoke-WebRequest http://127.0.0.1:3080/ -UseBasicParsing

   (A 401 is fine — it means the gateway is up and wants a cookie.)
   If it fails, tell the user the DSH web app must be open first and stop here.

2. Make sure the log dir exists, then start the bridge as a background pwsh
   job (workdir `C:\Users\press\OneDrive\Projects\TTSTT`,
   `run_in_background: true`). The log lives OUTSIDE the OneDrive-synced
   project dir on purpose:

       New-Item -ItemType Directory -Force C:\Users\press\.dsh\logs | Out-Null
       python -X utf8 -m voice_stack.coding_voice 2>&1 | Out-File -Encoding utf8 C:\Users\press\.dsh\logs\coding_voice.log

3. Give it 30-40 seconds (STT ~10s, TTS ~5s, follow stream open), then confirm
   `ready — speaking into session session-...` appears in
   `C:\Users\press\.dsh\logs\coding_voice.log` and tell the user to speak.

Useful options (append to the command):
- `--session <id>` — target a specific session id. Default: the most recently
  written TTSTT session (the chat the user is looking at).
- `--tts-voice <name>` — voice; default `eve`. Run `--list-voices` for the
  full list (anna, vera, fantine, eponine, azelma, mary, jane, eve,
  cosette, caro_davy, alba, jean, charles, paul, george, michael, marius,
  javert, bill_boerst, peter_yearsley, stuart_bell, ...).
- `--utterance-max <seconds>` — how long the user may keep talking before
  their utterance is sent anyway (default 120 = two minutes).
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
- While the coding agent is mid-turn (thinking/working) the bridge keeps
  the GPU 100% free for the agent: the STT model is OFF (zero GPU), TTS is
  off, the agent's reply text is buffered, and if the user speaks a canned
  "still working, one moment" is said. Anything the user says while the
  agent works is captured to a mailbox (last 5 minutes of mic audio) and
  transcribed the moment the turn ends, so nothing is lost. When the turn
  ends the buffered reply is spoken.
- Utterance ending is lenient: a short complete utterance (ends in . ! ? and
  is ~14 words or fewer) sends after ~0.6-1.2 s; anything longer or
  unpunctuated keeps the door open for pauses up to `--utterance-max`
  (default 2 minutes) before being sent anyway.
- The bridge also speaks replies triggered any other way (typed in the GUI,
  handoffs) — everything the window shows is spoken. That is by design.
- Logs: `C:\Users\press\.dsh\logs\coding_voice.log`. Session events are only
  consumed; the gateway is never restarted by this bridge.
