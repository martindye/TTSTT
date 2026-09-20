# TTSTT — Local Voice Stack

**Mic → Kyutai STT 1B (GPU) → DSH voice session (Qwen 27B via llama.cpp) → Pocket TTS (CPU) → speakers**

Fully local. The "brain" is a real **DeepSeek Harness (DSH) session** — not a
bare LLM client — so the voice assistant has a persona (`voice_persona/`),
tools (files, shell), can hand coding work off to a full coding agent, and
persists its conversation per bridge run. Sentence-streamed: audio starts
while the model is still thinking through the answer.

```
        ┌──────────┐ 80ms frames  ┌──────────────┐ word-gap ┌──────────────────────────┐
 mic ─▶│ PortAudio │─────────────▶│ Kyutai STT 1B│ endpoint│ DSH voice session         │
        └──────────┘   12.5 fps   │   (GPU)      │  = 1.2s  │ qwen3.8-27b @ llama.cpp  │
                                 └──────────────┘  gap     │ (llama-server :8080)      │
        ┌──────────┐  24 kHz PCM ┌──────────────┐          └─────────────┬──────────────┘
        │ speakers ◀──────────────│ Pocket TTS  │◀── spoken text only ───┘
        └──────────┘             │ (CPU, INT4)  │    (thinking never
                                 └──────────────┘     spoken)
```

## Quick start

Prerequisite: the llama.cpp server running at `127.0.0.1:8080`
(model `qwen3.8-27b`). It is user-managed and has `--slot-save-path` state —
**never restart it without the user**.

```powershell
# from C:\Users\press\OneDrive\Projects\TTSTT
.\run_voice_dsh.bat            # or: python -X utf8 -m voice_stack.voice_dsh

# Coding voice: your words go into the open coding chat, and the coding
# agent's visible replies are spoken back (its thinking is never spoken).
python -X utf8 -m voice_stack.coding_voice
```

~35–45 s to boot (STT ~8–10 s, TTS ~5 s, DSH runtime ~5 s). "ready" =
listen and speak. **Ctrl-C quits.** Each start is a *fresh conversation*
for the voice agent (no session resume yet — SDK gap).

**New machine?** [SETUP.md](SETUP.md) takes you from zero: LLM server, DSH,
dependencies (`requirements.txt`), skills, and the troubleshooting table of
everything that actually bit us. [VIDEO.md](VIDEO.md) is the shot list for
the demo video.

### Starting it without typing

The `start-voice` DSH skill (user-level, `C:\Users\press\.dsh\skills\start-voice`)
knows the start/stop/status procedure — in any DSH chat session, just ask
"start the voice".

## Voice → coding-agent handoff

The voice assistant can delegate coding work to a full DSH coding agent:

```powershell
python -X utf8 -m voice_stack.handoff "fix the flaky endpointing test"
```

Two delivery modes:

- **Gateway mode (default):** if the web GUI gateway is running
  (`http://127.0.0.1:3080`), the task is queued as a normal (clearly labelled
  `[VOICE-HANDOFF]`) user message into the **most recently active TTSTT
  session** — i.e. the open coding chat when one is open, else the last
  written session; the gateway resumes a cold session automatically. The
  agent works it where you can watch, exactly like ChatGPT. The client
  authenticates with the same signed cookie the browser uses (secret from
  `C:\Users\press\.dsh\.credentials.yaml`; loopback-only gateway).
- **Spawn mode (fallback / `--spawn`):** the gateway is down, so a fresh
  `handoff-<ts>` DSH SDK session runs the task standalone in the same DSH
  home as the web GUI (`C:\Users\press\.dsh`) and shows up in the GUI's
  session list.

It prints `HANDOFF RESULT: injected | spawned ok | spawned failed | timeout`
+ `HANDOFF SUMMARY: …` for the voice to read back. The voice persona already
knows how to drive this (see `voice_persona/AGENTS.md`, section "Handing
work to the coding agent").

## Options (`python -m voice_stack.voice_dsh --help`)

| Flag | Default | Meaning |
|------|---------|---------|
| `--tts-voice` | `eve` | Pocket TTS voice. Female catalog voices: anna, vera, fantine, eponine, azelma, mary, jane, eve, cosette, caro_davy |
| `--mic-gain` | `1.0` | software gain on the mic (leave at 1.0) |
| `--stt-device` | `auto` | `auto`/`cuda`/`cpu` (CPU is ~3.4× slower than realtime — emergency only) |
| `--stt-repo` | `kyutai/stt-1b-en_fr` | STT checkpoint |
| `--llm-url` / `--model` | `:8080/v1` / `qwen3.8-27b` | LLM endpoint |
| `--dsh-root` | `../deepseek-harness` | DSH checkout (SDK runtime) |
| `--dsh-home` | `TTSTT\.dsh-voice` | voice session home |
| `--persona` | `voice_persona/` | persona workspace (AGENTS.md) |
| `--perm` | `workspace-write` | permission mode for the voice agent |
| `--once TEXT` | — | speak one canned prompt, print timings, exit |
| `--mic-device` / `--spk-device` | default | PortAudio device ids |

## How it works

- **STT** (`stt_engine.py`) — 80 ms frames (1920 samples @ 24 kHz) fed to
  Mimi + the 1B LM at 12.5 fps, ~20–25 ms/frame on GPU, zero backlog.
- **Endpointing — model-specific, important** — this STT build is *unreliable
  at signaling end-of-utterance*: the END token (0) fires mid-sentence after
  every 1–2 words and the SILENCE token (3) fires on *every* idle frame.
  Neither is usable. The bridge endpoints on **gaps between word tokens**
  (`WORD_GAP_S = 1.2` in `voice_dsh.py`). The gap is **adaptive**: after a
  sentence-final punctuation token (`. ? ! …`) the wait drops to
  `WORD_GAP_SENTENCE_S = 0.6` s, because the STT model reliably emits those
  after complete sentences. Don't "fix" this back to END/SILENCE-based
  endpointing.
- **Brain** (`voice_dsh.py` + `dsh_runtime.py`) — spawns the official DSH
  SDK runtime (`node --import tsx/esm <dsh-root>/apps/cli/src/bin.ts
  --profile sdk`, JSON-RPC over stdio) with `cwd=voice_persona`, so the
  persona's `AGENTS.md` is injected into the voice agent's context. Only
  `text` blocks are spoken; `reasoning` (thinking) and tool-call blocks are
  counted but never spoken.
- **Half-duplex** — while assistant audio is (or was within 1.2 s) audible,
  the mic is deaf, so the speakers don't feed back into the STT.
- **TTS** (`tts_engine.py`) — Pocket TTS on CPU (INT4), one voice state
  (default `eve`), sentences streamed as they are generated.
- **Legacy path** — `python -m voice_stack` (old `assistant.py` +
  `llm_client.py`, direct OpenAI calls) still exists for offline selftests:
  `python -m voice_stack --selftest tests/test_speech.wav`. It is not the
  live path.

### Two voice modes

- **Voice assistant** (`voice_dsh.py`, skill `start-voice`) — mic -> STT ->
  a *separate* invisible DSH session with the voice persona -> TTS. It never
  appears in the coding GUI.
- **Coding voice** (`coding_voice.py`, skill `start-coding-voice`) — mic ->
  STT -> the *open coding chat* (utterances injected through the GUI gateway,
  so they appear in the window) -> the coding agent's **visible** replies are
  followed live over the gateway's `session/follow` stream and spoken,
  sentence by sentence. Thinking and tool activity are never spoken. Half
  duplex: the mic is closed from the user's utterance until the reply's audio
  has drained (released on `turn/end`).
  - **Auto-follow (any window).** Unless pinned with `--session`, the bridge
    re-resolves the *newest-written* session of the target workspace every
    few seconds (debounced over two polls, never mid-turn), so it follows
    whichever chat window the user is talking to: a newly opened window is
    picked up as soon as it exists, an active reply keeps its window newest
    while the agent works, and a quiet window is picked up once anything
    lands in it. A switch reopens the follow stream within ~1 s; per-session
    watermarks are re-armed from the new session's first snapshot and
    leftover frames of the old session are dropped by session-id checks, so
    a switch never double-speaks or crosses streams.

## Latency & thinking

Time-to-first-audio is the metric. The first-audio path is:
**endpoint gap → prompt → LLM prefill+thinking → first sentence → TTS**.

Levers, in order of impact:

1. **Thinking suppression (biggest).** The model does **not** honor
   `/no_think` as a hard switch (measured: it can even *increase* thinking).
   The persona section "Answer directly. Do not reason."
   (`voice_persona/AGENTS.md`) cuts it ~20 %; the voice home
   `.dsh-voice\settings.yaml` sets `llm-deepseek: {reasoningEffort: low}`,
   which makes the DSH adapter send `reasoning_effort: "low"` on the wire —
   the llama.cpp server honors that (measured: ~50 % less thinking than
   default; `"none"` on the raw wire turns thinking off entirely, but the
   DSH deepseek adapter only offers `off|low|high|max`, and its `off` maps
   to a `thinking:{type:disabled}` field llama.cpp *ignores*).
2. **Endpoint gap** — 1.2 s base, 0.6 s after sentence-final punctuation
   (see endpointing above).
3. **TTS** — Pocket TTS INT4 on CPU runs ~3× faster than realtime, so it
   never gates first audio (first audio lands ~0.2 s after the first
   sentence).

Clean-slot reference numbers (`--once --no-speak`, room empty, measured
02:3x): warm slot, "How are you doing today?" → **total 4.7 s, first text
2.7 s, first audio 3.0 s, thinking 322 chars** (before the latency work:
11.6 s total / 6.6 s first text / 561 chars thinking on the same question).
Caveat: the *first* request to llama-server after a long idle can stall
~20 s before the first token (warm slot: ~1 s) — the first turn after the
room has been silent for a long time will feel slow; turns after it are fast.

## GPU / VRAM budget

| Component | Where | VRAM |
|-----------|-------|------|
| Qwen 27B LLM | GPU | ~22 GB + KV cache (run llama-server `-c 65536`) |
| Kyutai STT 1B | GPU | ~2.8 GB |
| Pocket TTS | CPU | 0 (438 MB RAM, INT4) |

STT must be on the GPU (CPU is ~3.4× slower than realtime). If STT OOMs on
load, the LLM context is still too big — lower `-c` further.

## Known limitations

- **One LLM slot.** The voice, the web GUI, and handoff sessions all share
  the single llama-server slot. Voice turns *slow down* (not fail) while
  another session is generating — don't run big coding work while talking.
- **Fresh conversation per start.** The SDK runtime creates sessions, never
  resumes them (a persisted id collides). Voice memory resets each start.
- **Echo** — with speakers (not headphones) the mic can pick the assistant
  up; the 1.2 s voice-tail guard covers most of it.
- **Gated model** — voice *cloning* needs the gated `kyutai/pocket-tts`
  repo; we use the `without-voice-cloning` variant + its voice catalog.
- **torch.compile is dead on this machine** (no Triton-for-CUDA, no MSVC):
  `stt_engine.py` always sets `NO_TORCH_COMPILE=1`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| STT load OOM on CUDA | restart llama-server with smaller `-c` (user action) |
| "LLM request failed / Connection refused" | llama-server isn't running on :8080 |
| Words transcribe but no answer | check `tests\live_dsh_log.txt` turn lines; check LLM slot contention |
| No transcript while speaking | mic device wrong? raw rms in the heartbeat log should jump while talking |
| Assistant hears itself | use headphones; check `VOICE_TAIL` in voice_dsh.py |
| Voice sounds wrong | `--tts-voice <name>` (see options); voices are downloaded from HF on first use |

## Layout

```
voice_stack/
  voice_dsh.py     live bridge (DSH brain, word-gap endpointing, speaking)
  dsh_runtime.py   DSH SDK JSON-RPC client (spawns node + tsx SDK runtime)
  handoff.py       voice -> coding-agent handoff (GUI-visible DSH session)
  stt_engine.py    Kyutai STT 1B streaming wrapper
  tts_engine.py    Pocket TTS wrapper (thread-serialized, streaming)
  audio_io.py      mic capture / speaker playback (PortAudio, 24 kHz)
  assistant.py     legacy: old direct-LLM turn loop (sentence splitter reused)
  llm_client.py    legacy: old direct OpenAI client (selftest path)
  __main__.py      legacy CLI + offline selftest
voice_persona/
  AGENTS.md        the voice agent's persona (incl. "answer directly" + handoff rules)
.dsh-voice/        voice session home (session journals; cleared once in the past)
tests/
  live_dsh_log.txt  live bridge log (rewritten each start)
  dsh_runtime.err.log / dsh_handoff.err.log   runtime stderr
run_voice_dsh.bat  launcher (python -X utf8 -m voice_stack.voice_dsh %*)
```
