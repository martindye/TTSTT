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
Project root: {{TTSTT_ROOT}}.

This is different from the voice assistant (the `start-voice` skill): that one
talks to a separate invisible voice agent; the coding voice talks to the
coding agent in the currently open chat window.

## List available voices (no bridge needed)

    cd {{TTSTT_ROOT}}
    python -X utf8 -m voice_stack.coding_voice --list-voices

If the user wants a different voice, add `--tts-voice <name>` to the start
command (below) for that run. `anna` is the default.

## Start — one command, idempotent (no thinking)

Target = the DSH workspace of the chat that asked for the voice = the
agent's own working directory (the project root only when this chat IS the
project's own chat). Run exactly this (substitute the workspace dir), from
anywhere:

    powershell -NoProfile -File {{TTSTT_ROOT}}\voice_stack\ensure_coding_voice.ps1 -Workspace '<the requesting chat's workspace>'

It handles EVERY starting state by itself — already running (reports it,
changes nothing), stale/dead (stops it, starts fresh), aimed at the wrong
workspace (re-aims), not running (starts). It also installs this skill into
the DSH home on first use, so the package is complete after one run. It
launches the supervisor detached (NOT as a chat background job, so the voice
outlives this chat session), waits until the bridge is actually ready, and
exits:

- exit 0 → the command printed which session it is speaking into. Tell the
  user to speak. STOP HERE.
- exit 1 → the reason and the relevant log tails are printed. Fix that one
  thing, run the same command again. That is the only "thinking" allowed.

Useful options (append to the command):
- `--workspace <path>` — the DSH workspace whose newest session the voice
  follows. Pass it whenever the requesting chat is not in the project's own
  workspace — which is almost always, since the agent's working directory
  IS the workspace.
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
- `--final-silence <seconds>` — real silence required after a full stop
  before the sentence is sent (default 3). The STT drops a full stop on any
  pause, including a mid-sentence breath; a smaller value re-introduces
  mid-sentence cut-offs.
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

The same one command with a different `-Workspace`. Resolve the name against
the user's projects folder (case-insensitive; if nothing matches, ask). No
stopping first — the script re-aims any running bridge for you. The bridge
follows ONE workspace at a time (auto-following between its sessions), so
there is no "go back": the next `/start-coding-voice` from any chat just
re-aims it again.

## Stop

Create the stop sentinel FIRST (so the supervisor does not restart the
bridge), then kill the bridge by the PID the supervisor recorded in
`coding_voice.state` (no WMI needed; only if that file is absent — a legacy
stack — kill any `python`):

    New-Item {{DASHOME}}\logs\coding_voice.STOP -ItemType File -Force
    $s = $null
    if (Test-Path {{DASHOME}}\logs\coding_voice.state) {
      $s = Get-Content {{DASHOME}}\logs\coding_voice.state -Raw | ConvertFrom-Json
    }
    if ($s) { Stop-Process -Id $s.bridge_pid -Force }
    else { Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force }

Then confirm `supervisor stopped` appears in
{{DASHOME}}\logs\coding_voice_supervisor.log (the supervisor
removes the sentinel and its state file itself when it exits).

## Notes

- The supervisor auto-restarts the bridge on unexpected death; the dead
  run's log is kept as `coding_voice.prev.log` and supervisor actions are
  in `coding_voice_supervisor.log` (both under {{DASHOME}}\logs). If the
  voice is dead, check those two files first.
- Liveness without forensics: the bridge writes a `heartbeat` log line every
  30 s, and while running the supervisor writes `coding_voice.state`
  (supervisor PID, bridge PID, workspace). "Alive" = state file present,
  PIDs live, log mtime < 90 s. `ensure_coding_voice.ps1` does exactly that
  check and nothing else before deciding to (re)start.
- Models load OFFLINE: the supervisor sets `HF_HUB_OFFLINE=1`, so startup
  never touches huggingface.co (both STT models and the TTS voices are in
  the local HF cache). On a FRESH machine the cache is empty, so the first
  start must be allowed to download: set `HF_HUB_OFFLINE` to 0 in
  `supervise_coding_voice.ps1` once, let the models download, then set it
  back to 1. A missing model is a clear local error, not a network timeout.
- Half-duplex: while the bridge is holding the floor (from the user's
  utterance until the reply's audio has finished playing) the microphone is
  closed; the user speaks again after the reply is done.
- Spoken = the coding agent's visible text in that chat window, sentence by
  sentence. Thinking blocks and tool activity are never spoken.
- While the bridge is speaking it does not listen at all (mic frames are
  dropped before anything else happens — even while the agent is mid-turn —
  so its own voice can never be buffered or transcribed).
- While the coding agent is mid-turn (thinking/working) the reply is
  SPOKEN LIVE as it streams in, while the STT model stays OFF (GPU for the
  agent). The mic is closed while reply audio is in the air (echo guard)
  and open in the gaps (between sentences, tool calls); anything the user
  says in a gap goes to a mailbox (ring buffer), and when the turn ends the
  most recent 60 s of it is transcribed and sent, so nothing recent is lost
  without replaying minutes of old audio.
- Utterance ending is lenient: a sentence ending in . ! ? sends only after
  `--final-silence` s of real silence (default 3 — the STT drops a full stop
  on any pause, so a bare full stop never sends on its own); an unpunctuated
  short utterance (3-12 words, a command) sends only after 6 s of real
  silence; fragments shorter than 3 words are never sent on their own — they
  merge with whatever the user says next. Everything else holds the door open
  until `--utterance-max` (default 2 minutes).
- The bridge also speaks replies triggered any other way (typed in the GUI,
  handoffs) — everything the window shows is spoken. That is by design.
- Logs: {{DASHOME}}\logs\coding_voice.log. Session events are only
  consumed; the gateway is never restarted by this bridge.
