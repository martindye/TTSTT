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
command (below) for that run. `anna` is the default.

## Start (fast path: at most 3 commands, well under a minute)

Target = the DSH workspace of the chat that asked for the voice = the
agent's own working directory (TTSTT only when this chat IS the TTSTT
project). Pass it as `--workspace <that dir>` — the bridge's built-in
default (TTSTT) is wrong for every other chat, and a voice aimed at the
wrong workspace is the classic "it took 20 minutes" failure.

1. One-shot check — running, and aimed at the right workspace? The
   sessions dir for a workspace is its path with `:` removed and every
   other non-alphanumeric replaced by `-`, wrapped in `--` (DSH_TESTS →
   `--C-Users-press-OneDrive-Projects-DSH_TESTS--`); each session is a
   subdirectory named `session-...`. Run (substitute `$ws`):

       $ws = '--C-Users-press-OneDrive-Projects-DSH_TESTS--'   # per workspace
       $lf = 'C:\Users\press\.dsh\logs\coding_voice.log'
       if (Test-Path $lf) {
         # whole file (it is small; the supervisor re-truncates it each start)
         $m = [regex]::Match((Get-Content $lf) -join "`n", 'target session: (session-[0-9a-f-]+)')
         $t = if ($m.Success) { $m.Groups[1].Value } else { '' }
         'fresh=' + ((Get-Item $lf).LastWriteTime -gt (Get-Date).AddMinutes(-15)) +
         ' target=' + $t + ' right=' + ($t -ne '' -and (Test-Path "C:\Users\press\.dsh\sessions\$ws\$t"))
       } else { 'not running' }

   - `fresh=True right=True` → already live and aimed here: tell the user,
     do nothing else. STOP HERE.
   - `right=False` or not fresh → steps 2-3. This one-liner is the WHOLE
     check — no session forensics (no mtime archaeology, no id listings).
2. If it was running: Stop (below) first. Then start it as a background pwsh
   job, workdir `C:\Users\press\OneDrive\Projects\TTSTT` (that is where the
   CODE lives; `--workspace` is where the SESSION lives):

       New-Item -ItemType Directory -Force C:\Users\press\.dsh\logs | Out-Null
       powershell -NoProfile -File voice_stack\supervise_coding_voice.ps1 --workspace 'C:\Users\press\OneDrive\Projects\DSH_TESTS'

   The supervisor restarts the bridge if it dies and keeps the dead run's
   log as `coding_voice.prev.log`. The gateway must be up first:
   `Invoke-WebRequest http://127.0.0.1:3080/ -UseBasicParsing` — a 401 is
   fine (up, wants a cookie); a failure means the DSH web app is closed.
3. Wait 30-40 s (STT ~10s, TTS ~5s, follow stream open), confirm
   `ready — speaking into session ...` in `C:\Users\press\.dsh\logs\coding_voice.log`
   (its session id must sit under the workspace's sessions dir from step 1)
   and tell the user to speak.

Useful options (append to the command):
- `--workspace <path>` — the DSH workspace whose newest session the voice
  follows (e.g. `C:\Users\press\OneDrive\Projects\DSH_TESTS`). Pass it
  whenever the requesting chat is not in the TTSTT workspace — which is
  almost always, since the agent's working directory IS the workspace.
  The built-in default (TTSTT) only fits this skill's own chat.
- `--session <id>` — pin a specific session id. Default: the most recently
  written session of the target workspace, and the bridge then AUTO-FOLLOWS
  (see below) — passing `--session` disables auto-follow.
- `--stt-model <fast|accurate>` — speech recognition model. `fast` (default)
  is kyutai/stt-1b-en_fr: quick, ~0.5 s delay. `accurate` is
  kyutai/stt-2.6b-en: noticeably more accurate but ~2.5 s delay, needs ~7 GB
  of FREE GPU memory (the big LLM server usually takes most of it — if the
  bridge refuses to start, that's why; free the GPU or stay on fast), and
  downloads the model on first use. Phrasing: "start coding voice with the
  accurate model".
- `--tts-voice <name>` — voice; default `anna`. Run `--list-voices` for the
  full list (anna, vera, fantine, eponine, azelma, mary, jane, eve,
  cosette, caro_davy, alba, jean, charles, paul, george, michael, marius,
  javert, bill_boerst, peter_yearsley, stuart_bell, ...).
- `--utterance-max <seconds>` — how long the user may keep talking before
  their utterance is sent anyway (default 120 = two minutes).
- `--gui <url>` — default `http://127.0.0.1:3080`.

## Auto-follow (any window in the target workspace)

Unless pinned with `--session`, the bridge keeps re-resolving the
**newest-written** session of the target workspace (a session's file mtime
moves when it receives activity), debounced over two polls and never
mid-turn. In practice: the user opens a new window -> the bridge follows it
within a few seconds; an agent's active reply keeps its window newest while
it works; a quiet window is picked up the moment anything lands in it.
Switches log `auto-follow: speaking into session-...` in the log. Limitation:
a window that is merely looked at (nothing has landed in it yet) is only
picked up once something lands in it — a true focus signal would need a
browser-side heartbeat, which the DSH gateway does not expose.

## "Start coding voice in <workspace>"

Same fast path with a different `--workspace`: resolve the name under
`C:\Users\press\OneDrive\Projects\<name>` (case-insensitive; if nothing
matches, look under `C:\Users\press\OneDrive\Projects` and say so), use its
sessions-dir name in the step-1 check, and pass the path to `--workspace`.
Stop any running bridge first. The bridge follows ONE workspace at a time
(auto-following between its sessions), so there is no "go back" — the next
`/start-coding-voice` from any chat just re-aims it.

## Stop

Create the stop sentinel FIRST (so the supervisor does not restart the
bridge), then kill the python process:

    New-Item C:\Users\press\.dsh\logs\coding_voice.STOP -ItemType File -Force
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
      Where-Object { $_.CommandLine -like '*voice_stack.coding_voice*' } |
      ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

Then confirm `supervisor stopped` appears in
`C:\Users\press\.dsh\logs\coding_voice_supervisor.log` (the supervisor
removes the sentinel itself when it exits).

## Notes

- The supervisor auto-restarts the bridge on unexpected death; the dead
  run's log is kept as `coding_voice.prev.log` and supervisor actions are
  in `coding_voice_supervisor.log`. If the voice is dead, check those two
  files first.
- Half-duplex: while the bridge is holding the floor (from the user's
  utterance until the reply's audio has finished playing) the microphone is
  closed; the user speaks again after the reply is done.
- Spoken = the coding agent's visible text in that chat window, sentence by
  sentence. Thinking blocks and tool activity are never spoken.
- While the bridge is speaking it does not listen at all (mic frames are
  dropped before anything else happens — even while the agent is mid-turn —
  so its own voice can never be buffered or transcribed).
- While the coding agent is mid-turn (thinking/working) the bridge keeps
  the GPU 100% free for the agent: the STT model is OFF (zero GPU), TTS is
  off, and the agent's reply text is buffered until the turn ends. Anything
  the user says while the agent works goes to a mailbox (ring buffer); when
  the turn ends the most recent 60 s of it is transcribed and sent, so
  nothing recent is lost without replaying minutes of old audio.
- Utterance ending is lenient: a short utterance that ends in . ! ? sends
  after ~0.6 s; an unpunctuated short utterance (3-12 words, a command)
  sends only after 6 s of real silence; fragments shorter than 3 words are
  never sent on their own — they merge with whatever the user says next.
  Everything else holds the door open until `--utterance-max`
  (default 2 minutes).
- The bridge also speaks replies triggered any other way (typed in the GUI,
  handoffs) — everything the window shows is spoken. That is by design.
- Logs: `C:\Users\press\.dsh\logs\coding_voice.log`. Session events are only
  consumed; the gateway is never restarted by this bridge.
