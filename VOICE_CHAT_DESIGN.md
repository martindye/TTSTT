# Voice Chat — design (ChatGPT-style voice mode)

Status: **design for the next step; a simple predecessor is live.** Written 30/09/2026.
02/10: the predecessor is the unified composer button (not this design): one
34px blue button beside the mic (the native send arrow is hidden by the
plugin) whose glyph follows state — a voice-waveform entry on an empty draft
(tap starts a voice turn), an up-arrow while typing (send), a stop-square
while the agent is running OR the reply is being spoken (tap halts the turn
and/or the speech — "speaking" is detected via dsh-voice-output's
"Stop speaking" speaker button), and a cross while a voice turn is active
(mic tap starts; tap = stop + send; the reply is auto-read via
dsh-voice-output and the `dsh.voice-chat.v1` localStorage flag). Typing
always reverts to the send arrow so a reply can be sent while thinking or
speaking. 02/10 (later): basic client-side VAD — the recording stops and
sends itself about 2 s after the last word (250 ms RMS polling of the live
mic level, threshold 0.02), and a microphone that never delivers speech gives
up after 15 s; the cross still stops it early. No barge-in, no streaming TTS.
02/10 (latest): continuous listening, in the same plugin — once a voice turn
has sent, the next reply ending (agent done AND any spoken reply finished)
re-arms the microphone on its own, so the conversation continues without a
tap; one auto-open per reply end, a turn that drew no words settles back to
the waveform entry quietly, and a reply that starts speaking late voids the
already-open turn and reopens it once the speech is done. A no-speech
auto-open waits a 3 s settle after the reply ends (device-muted case); no
draft, no lock, and no in-flight turn are allowed to coexist with the
auto-open. No barge-in yet (that stays M3). Verified 02/10: 95/95 unit,
e2e rewritten to the auto-send flow with refreshed goldens (the
fresh-round-trip replay fixture additionally fails on this box for everyone —
`unknown tool "bash"` on Windows, a pre-existing environment issue, not the
voice plugin). M1–M4 below remain the roadmap for the real thing.

Goal: a hands-free, back-and-forth voice conversation with the agent, from the
phone (primary) and the PC (parity), feeling like the current voice mode in
ChatGPT: speak, pause, it answers, and you can interrupt it by just talking.

## 1. What we have today (baseline)

| Piece | Where | Behaviour |
|---|---|---|
| Coding voice bridge | `voice_stack\coding_voice.py` | PC mic always hot; Kyutai 1B STT; injects text into this chat; `--tts-engine none` (PC silent); floor/mic gating is turn-based |
| Web speaker system | `dsh-voice-output` plugin → `voice_stack\tts_server.py` (8188) → Qwen3 1.7B server (8095) | Per **finalized message**: whole message (≤4000 chars) rendered to one WAV, then played. No streaming, one-shot |
| Browser mic | `client-ui-voice-input` (voice-input-bundle) | Push-to-record in the composer; MediaRecorder → 16 kHz WAV → SenseVoice STT on the PC |
| Gateway | DSH web host (3080, LAN, trusted host) | Cookie-authenticated chat API; the bridge already injects user turns through it |

Known weaknesses to fix ("not super stable" today):

- The PC mic is the only ear, always hot, and PortAudio can die silently
  (callback alive, digital silence — seen 27/09). No watchdog yet.
- The phone mic is blocked: getUserMedia needs a secure context (HTTPS
  decision pending: self-signed vs Tailscale vs tunnel).
- Audio is turn-granular: the phone waits for the *whole* message before
  hearing a word (10–30+ s on long replies).
- No barge-in: you cannot talk over the machine.
- STT (Kyutai 1B) and the LLM share the GPU; VRAM headroom is ~0.7 GB.

## 2. Target experience

Milestones, each independently shippable:

- **M1 — tap-to-talk.** Hold the mic button, speak, release. Transcript
  appears in the draft, the answer is **spoken sentence-by-sentence as it is
  generated** (first sentence in ~3–5 s, not after the whole message).
- **M2 — hands-free.** No button: server-side VAD opens the ear, 0.7 s of
  silence ends the utterance, the loop repeats. Push-to-talk stays as fallback.
- **M3 — barge-in.** Talk while it is talking: it stops within ~0.5 s and
  listens; the interrupted answer is aborted in the chat (same path as the
  stop button).
- **M4 — polish.** Live partial transcript, backchannels ("uh-huh" don't
  trigger a turn), PC parity (PC mic + PC speakers, same loop), watchdogs,
  one-word acknowledgment fast path.

Latency budget to first audio (target, per voice turn):

| Stage | Budget |
|---|---|
| Endpoint tail (silence rule) | 0.6–0.8 s |
| STT (SenseVoice, CPU, 5 s utterance) | 0.3–0.8 s |
| LLM time-to-first-sentence (27B Q6, 5090) | 1–3 s — **dominant term** |
| TTS first sentence (Qwen3 1.7B, ~12 words) | 1–1.5 s |
| Network + jitter buffer | 0.1–0.3 s |
| **Total** | **~3–5 s** ("good" < 3 s; today is 10–30 s) |

The 27B model is the latency floor. If the feel is too slow, the lever is a
smaller voice-dedicated model (VRAM permitting) — a later decision, not M1.

## 3. Architecture

```
 phone browser                     PC (loopback)                         DSH host
┌──────────────────┐   WebSocket   ┌──────────────────────┐              ┌─────────────┐
│ mic (AEC on)      │ ── audio ───▶ │ VOICE LOOP (new)      │  inject     │ gateway /    │
│ speaker + jitter  │ ◀─ audio ──── │  • VAD + endpointing  │ ──────────▶ │ session API  │
│ push-to-talk btn  │   PCM chunks  │  • STT (SenseVoice,   │  sentence   │ (qwen 27B)   │
└──────────────────┘               │    CPU/ONNX)          │  stream     └──────┬──────┘
                                   │  • TTS: Qwen3 8095,    │ ◀──────────────────┘
                                   │    per-sentence stream │      token stream
                                   └──────────────────────┘
```

Components:

1. **Voice loop service (new, small).** Owns the state machine, VAD, STT
   dispatch, sentence streaming into TTS, and gateway injection. Python, a
   sibling of `tts_server.py` in `voice_stack` (e.g. `voice_loop.py`),
   supervised with the same supervisor/state-file pattern as the coding
   voice. Runs the STT on **CPU** (SenseVoice/ONNX via the sherpa-onnx
   runtime we already ship) so it never shares the GPU with the LLM — this
   also kills the known STT-vs-TTS CUDA-graph corruption class by
   construction.
2. **Transport through the DSH web host (new plugin bundle, e.g.
   `dsh-voice-chat`).** A WebSocket endpoint on 3080 (no new port, no new
   exposure): the host relays browser ↔ loop. Reuses the existing
   `dsh-voice-output` supervision pattern for the Python child. Auth is the
   session's; trusted-host already limits who can connect.
3. **Client (extend `client-ui-voice-input`).** AudioWorklet capture,
   16 kHz mono int16, 20–40 ms frames over the WS; playback is a ~150 ms
   jitter buffer on Web Audio. `getUserMedia` constraints:
   `echoCancellation: true, noiseSuppression: true, autoGainControl: true`.
   The existing mic button becomes: hold-to-talk (M1/M2) + a live "speaking"
   pill showing VAD state.
4. **VAD / endpointing.** Silero-VAD (ONNX, CPU) frame scores; speech
   start = 2 voiced frames; endpoint = 700 ms unvoiced after speech;
   min utterance 300 ms; max turn 30 s (hard cut + "I cut you off" note).
5. **TTS streaming.** The LLM token stream is split into sentences (same
   regex the Qwen3 engine already uses); each sentence is sent to the 8095
   server as it completes and its PCM is forwarded to the phone as it
   arrives (~250 ms of audio ahead of real time). No waiting for the final
   message. The per-message 4000-char cap of the speaker system does not
   apply to voice turns.
6. **Barge-in (M3).** The mic stays open during playback with AEC on; the
   VAD runs with a higher threshold while the speaker is on plus a 250 ms
   debounce (AEC glitches). On confirmed speech: client flushes the playback
   buffer; the loop tells the session to **abort the current turn** — the
   web UI already has this (the stop button ends turns with `reason:
   aborted, user`; see `apps/web/tests/message-actions.e2e.ts`) — then
   processes the new utterance. Fallback if iOS AEC proves unreliable:
   **tap-to-interrupt** (one tap on the speaking pill stops playback and
   opens the ear).

Turn state machine (shared by phone and PC endpoints):

```
IDLE → LISTENING ──(endpoint)→ THINKING ──(first sentence)→ SPEAKING
   ▲            ▲                                      │
   │            │  barge-in (M3): abort turn           │ silence
   └────────────┴──────────────────────────────────────┘
```

PC endpoint (M4): PortAudio in/out with the same state machine; while
SPEAKING the VAD threshold is raised (loudspeaker → mic self-hear), and if
that misbehaves the PC keeps today's behaviour (ear closed while speaking).

## 4. Stability plan (the "not super stable" part)

- **PortAudio watchdog** (PC): periodic level self-probe on a fresh stream;
  on digital-silence detection, restart the stream; supervisor restarts the
  loop process as last resort (sentinel + state file, exactly the coding
  voice pattern).
- **iOS realities**: Web Audio is suspended when the screen locks — hands-
  free means "screen on"; document it, show a "connection lost" pill on WS
  drop, auto-reconnect with a fresh VAD baseline. Battery: continuous mic +
  playback is real draw; M1 (tap-to-talk) is the low-battery mode.
- **GPU**: STT on CPU (no contention); TTS on 8095 (separate process,
  resident ~2 GB); if VRAM ever can't take the Qwen3 TTS, the existing
  pocket-CPU fallback in `tts_server.py` already degrades gracefully.
- **Network**: LAN WS (~5 ms). Different Wi-Fi → loop degrades to
  "speaker only" (today's behaviour) with a notice, never a dead end.

## 5. Open questions (resolve before building M3/M4)

1. Exact abort endpoint the UI stop button calls (route + auth) — confirm
   from the message-actions e2e path so barge-in reuses it verbatim.
2. iOS AEC quality with phone speaker + mic simultaneously — real-device
   test in M3; tap-to-interrupt is the sanctioned fallback.
3. 24 kHz playback on iOS Safari — verify (expect fine).
4. Backchannel policy: should "uh-huh"/"yeah" reach the LLM at all?
   Default: filter under 2 words unless the user config says otherwise.
5. Voice turns on the 27B always, or a smaller voice model later (VRAM).
6. PC mic: keep always-hot (with watchdog) or mirror the phone UX?

## 6. Prerequisites and ordering

0. **Phone HTTPS decision** (pending with Martin: self-signed local cert vs
   Tailscale vs tunnel vs no-mic). Nothing in M1–M4 moves without it.
1. M1 (1–2 focused sessions): loop service + WS relay + client hold-to-talk
   + sentence-streaming TTS.
2. M2 (~1): VAD + auto endpoint; push-to-talk remains as fallback.
3. M3 (1–2, AEC is the risk): barge-in + abort wiring + tap-to-interrupt
   fallback.
4. M4 (ongoing): partial transcript, backchannels, PC parity, watchdogs.

## 7. What this is not

- Not a native iOS app. Browser-based; a native app is the escape hatch if
  Web Audio/AEC limits bite.
- Not speech-to-speech (no end-to-end audio model). ASR → LLM → TTS is the
  pipeline; at our model sizes that is the pragmatic choice.
- Reuses, not replaces: the coding voice bridge stays the PC ear for text
  coding; the voice chat loop is the *conversation* surface and can run in
  parallel (separate mic discipline, separate session).
