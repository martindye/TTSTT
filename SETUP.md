# TTSTT — Setup Guide

A fully **local** voice interface for a coding agent, running on one Windows
machine with an NVIDIA GPU:

```
mic ──▶ Kyutai STT 1B (GPU) ──▶ DSH session (Qwen 27B via llama.cpp, GPU)
                                        │  visible text only (thinking never spoken)
      speakers ◀── Pocket TTS (CPU) ◀──┘
```

Three things you can do with it:

| Mode | What it is | How to start |
|---|---|---|
| **Voice assistant** | Talks to a separate, invisible DSH session with its own persona and tools. Invisible in the GUI. | `python -X utf8 -m voice_stack.voice_dsh` |
| **Coding voice** | Your words go into **the open coding chat**; the coding agent's visible replies are spoken back. | `python -X utf8 -m voice_stack.coding_voice` |
| **Handoff** | The voice assistant delegates a task to the coding agent (into your open chat). | `python -X utf8 -m voice_stack.handoff "the task"` |

Everything is on `127.0.0.1`: nothing leaves the machine.

---

## 1. What you need

- Windows (developed on Win 11) with an NVIDIA GPU. A 32 GB card (RTX 5090)
  is comfortable; the LLM is the big VRAM user.
- **Python 3.12** and **Node.js 20+ with pnpm** (for DSH, step 3).
- ~25 GB disk: STT model (~2 GB, auto-downloaded), Pocket TTS voices
  (auto-downloaded), and the LLM GGUF you already have.
- A local OpenAI-compatible LLM server that serves a **thinking-capable**
  chat model (step 2).

## 2. The LLM server (bring your own)

The voice stack talks to any OpenAI-compatible chat server on
`127.0.0.1:8080`. In this setup it's llama.cpp:

```
llama-server -m <your-model>.gguf --port 8080 --slot-save-path <slot file>
```

Notes learned the hard way:

- This build of llama.cpp honors **`reasoning_effort`** on the wire:
  `"low"` ≈ −40–50 % thinking, `"none"` = thinking fully off.
  `thinking: {type: "disabled"}` is **ignored**.
- Keep `--slot-save-path` so the slot survives restarts, and **don't let the
  voice stack restart your server** — it never will; it only sends prompts.
- One shared slot: voice turns, GUI chat and benches all serialize on it.
- First request after a long idle can stall ~20 s (server warm-up).

## 3. DSH (the brain) — `deepseek-harness`

1. Clone/checkout `deepseek-harness` and build it per its own docs
   (`pnpm i`, build the web app).
2. Start the GUI:

   ```
   dsh web   # listens on 127.0.0.1:3080
   ```

   For LAN access add `--trusted-host <your hostnames>`.
3. First start creates your DSH home (`C:\Users\<you>\.dsh`): sessions,
   `settings.yaml`, and `.credentials.yaml` (the browser-session signing
   secret the voice bridges use to mint their gateway cookies — auto-created,
   never share it).
4. In the GUI, make sure your local model is wired up (provider pointing at
   `http://127.0.0.1:8080/v1`, model name = whatever `llama-server` reports).

The voice features depend on two gateway endpoints on that server:

- `POST /api/session/prompt` — inject a user message into a session;
- `WS  /api/remote.mux` → `session/follow` — live stream of a session's
  events (this is what the coding voice listens to).

## 4. The voice code (this repo)

```
git clone <this repo>
cd TTSTT
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Models (STT + TTS voices) download themselves from Hugging Face on first use.

### DSH skills (optional but nice)

The two start/stop procedures are packaged as DSH skills in `skills/`.
Copy them into your DSH home's skills folder and any DSH chat can start
them for you:

```
copy /E skills\* C:\Users\<you>\.dsh\skills\
```

Then just say/ask: "start the voice" / "start coding voice".

## 5. Running it

From the repo root, with your LLM server (step 2) and the DSH GUI (step 3)
already running:

```powershell
# Voice assistant (separate persona)
python -X utf8 -m voice_stack.voice_dsh

# Coding voice (this chat's window gets the words; my replies are spoken)
python -X utf8 -m voice_stack.coding_voice

# One-shot handoff from the command line
python -X utf8 -m voice_stack.handoff "fix the failing endpointing test"
```

- Each bridge takes ~30–45 s to boot (STT ~10 s, TTS ~5 s); it prints
  `ready` when it's listening. **Ctrl-C stops it.**
- Coding voice picks the *newest* session in your DSH home as its target —
  i.e. the chat you're looking at. Override with `--session <id>`.
- Coding voice options: `--gui`, `--tts-voice`, `--stt-device`,
  `--mic-device`, `--spk-device` (see `--help`).

### Voices and tuning

- `--tts-voice` (default `eve`): anna, vera, fantine, eponine, azelma,
  mary, jane, eve, cosette, caro_davy.
- Endpointing (when a pause means "done talking"): word-gap based, 1.2 s
  normal / 0.6 s after sentence-final punctuation — constants at the top of
  `voice_stack/voice_dsh.py` and `coding_voice.py`.
- Thinking level of the voice assistant: `TTSTT\.dsh-voice\settings.yaml`
  → `llm-deepseek.reasoningEffort: low`.
- Persona: `voice_persona/AGENTS.md` (the voice assistant's instructions).

## 6. Troubleshooting (the things that actually bit us)

| Symptom | Cause / fix |
|---|---|
| It never hears you | Check the **Windows mic volume/mute first** — a system mute is invisible to the bridge. Then check `tests\live_dsh_log.txt` for `stt word` lines. |
| It hears *itself* (echo loop) | Use headphones, or lower the speakers. The bridge is half-duplex (mic closed while it speaks + 1 s tail), which covers most of it. |
| Reply cut off mid-sentence | The gateway drops idle follow streams after ~75 s; the bridge now cycles its stream every 50 s and replays the snapshot, so this self-heals. If you're on an older copy: upgrade. |
| Mic deaf after a dropped stream | Same fix — the half-duplex floor now self-releases (reply done, or 2.5 s idle, or 45 s max). |
| First turn after a long silence is slow | llama.cpp slot warm-up (~20 s one-off). |
| STT OOM on GPU | `--stt-device cpu` (≈3.4× realtime — emergency only). |
| `argument --once: expected one argument` | The flag takes a value: `--once "question"`. |
| Bench numbers look polluted | Never bench while using the machine — the shared LLM slot and the open mic both contaminate measurements. Use `--once ... --no-speak` only in an empty room. |

## 7. Repo layout

```
voice_stack/     the bridge(s)
  voice_dsh.py       voice assistant bridge (DSH SDK runtime brain)
  coding_voice.py    coding-voice bridge (gateway inject + follow stream)
  handoff.py         voice → coding-agent handoff (gateway or spawn)
  stt_engine.py      Kyutai STT 1B streaming engine
  tts_engine.py      Pocket TTS wrapper (streaming sentences)
  audio_io.py        PortAudio mic/speaker
  assistant.py       legacy sentence splitter + old offline assistant
  dsh_runtime.py     DSH SDK runtime (node child process, JSON-RPC)
voice_persona/     the voice assistant's AGENTS.md persona
tests/             benches + one-off probes (quiet_bench, probe_*, ...)
skills/            start-voice / start-coding-voice DSH skills
tests\quiet_bench.py   latency bench (waits for a quiet room, --no-speak)
```

## 8. Design notes worth knowing

- **Endpointing is word-gap, not VAD.** This STT build's END/SILENCE tokens
  are unreliable (END fires mid-sentence, SILENCE on every idle frame).
  Don't "fix" this back to token-based endpointing.
- **Sentence-streamed TTS**: sentences are spoken as they complete
  (`split_sentence`), so audio starts while the LLM is still generating.
- **Thinking is never spoken.** The coding voice only speaks `text` blocks;
  `reasoning` blocks and tool calls are counted and dropped.
- **Coding voice is half-duplex**: mic closed from your utterance until the
  reply's audio has drained (released on `turn/end` + drain, safety caps at
  45 s).
- **Latency reference** (measured, clean room, `qwen3.8-27B-Q6` @ llama.cpp):
  ~4.7 s end-to-end on a short question; ~2.7 s to first text, ~3.0 s to
  first audio; thinking ≈ 300 chars at `reasoningEffort: low`.
