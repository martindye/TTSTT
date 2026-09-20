# HANDOFF — re-orientation for TTSTT

Purpose: a fresh agent session (or the user, later) reads this + README and is
fully oriented without re-deriving anything. The README covers *how to run it*;
this file covers the **hard-won facts and landmines**.

## What this is

A fully local voice assistant: mic → Kyutai STT 1B (GPU) → a DSH voice session
(qwen3.8-27b via llama.cpp) → Pocket TTS (CPU) → speakers. The "brain" is a
real DSH session with a persona (`voice_persona/AGENTS.md`), tools, and the
ability to hand coding work off to a full DSH coding agent that appears in the
user's web GUI.

## Key facts (do not re-derive)

**LLM server**
- llama-server at `127.0.0.1:8080`, model `qwen3.8-27b`, **single slot**,
  `--slot-save-path` state. **User-managed: never restart or kill it** without
  the user. Voice + GUI + handoff sessions share this one slot → while one is
  generating, the others stall (they don't fail).

**STT (kyutai/stt-1b-en_fr)**
- Must run on GPU: CPU RTF ≈ 0.29 (3.4× slower than realtime). GPU cost ~2.8 GB.
- **Token behavior on this build (critical):** token 0 (END) fires mid-sentence
  after every 1–2 words; token 3 (SILENCE) fires on *every* idle frame. Neither
  is an utterance-boundary signal. Endpointing works only via **gaps between
  word tokens** (`WORD_GAP_S = 1.2` in `voice_dsh.py`).
- ~20–25 ms/frame at 12.5 fps, zero backlog. First 6 frames of a fresh stream
  are discarded (model's built-in 0.5 s delay).
- Raw mic rms: ambient ~0.009–0.014, user voice ~0.02–0.04 — no gain needed.

**Thinking / latency**
- `/nothink`, `/no_think` are **not** hard switches on this model (measured:
  user-prefixed placements *increase* thinking; system-prefix best ≈ 22 % less).
- Working lever: short "Answer directly. Do not reason. Do not plan." in the
  persona + `/no_think` as the first line of `voice_persona/AGENTS.md`.
- Resulting simple-turn latency: a few seconds first-audio; without the lever:
  13–17 s (1–2.5k thinking chars). Thinking is captured but never spoken.

**DSH wiring**
- Voice brain = official SDK runtime: `node --import tsx/esm
  <dsh-root>/apps/cli/src/bin.ts --profile sdk`, JSON-RPC over stdio
  (`dsh_runtime.py`). dsh-root = `C:\Users\press\OneDrive\Projects\deepseek-harness`
  (inspect-only unless the user says to extend DSH).
- Voice session: home `TTSTT\.dsh-voice`, cwd `voice_persona` (persona injected
  from AGENTS.md), session id `voice-YYYYmmdd-HHMMSS` — **fresh per bridge
  start; the SDK creates but never resumes sessions** (a persisted id
  collides). Voice memory resets each start. Accepted limitation.
- Web GUI (http://127.0.0.1:3080) uses DSH home `C:\Users\press\.dsh`.
  Handoff sessions (`voice_stack.handoff`) use that same home + cwd TTSTT, so
  they appear in the GUI session list.
- The voice conversation is invisible in the GUI (different DSH home); its
  journals live under `TTSTT\.dsh-voice\sessions` (zstd-concatenated frames,
  magic `28 b5 2f fd`).
- **Handoff gateway injection** (voice -> open coding chat): `voice_stack/
  handoff.py` posts to the GUI gateway `POST http://127.0.0.1:3080/api/session/
  prompt` with body `{type:'client-request', rpcId, method:'session/prompt',
  payload:{args:{request:{requestId, sessionId, mode:'queue',
  content:[{type:'text',text}]}}}}`. Auth = cookie `dsh-auth-<b64url(sha256(
  authority))>=v1.<b64url(json{version,authority,issuedAt,expiresAt})>.<b64url(
  hmac-sha256(payload, secret))>`, secret from
  `~/.dsh/.credentials.yaml` → `records.client-connection/browser-session/
  payload.secret` (32-byte b64url). The session it targets = most recently
  written session dir under `~/.dsh/sessions/--<cwd-encoded>--/` (dir name =
  session id; single file `session.jsonl.zstd` per session). The gateway
  resumes a cold session on prompt; target discovery is "most recently
  written" = the open chat when one is active. If the gateway is down, it
  falls back to spawning an SDK `handoff-*` session.

**Machine / torch**
- torch.compile does NOT work here (no Triton for CUDA, no MSVC cl.exe):
  `stt_engine.py` always sets `NO_TORCH_COMPILE=1`.
- GPU: RTX 5090 32 GB = ~22 GB LLM weights + KV (`-c 65536`) + ~2.8 GB STT.
  If STT OOMs on load, LLM context is still too big → lower `-c`.
- DSH file policy for project sessions: danger-full-access; approval prompts
  disabled (requests auto-reject — never attempt sandbox escalation).

**Voices**
- Pocket TTS (CPU, INT4, thread-serialized). Default voice `eve` (female);
  catalog of predefined voices in `pocket_tts/utils/utils.py`
  (`_ORIGINS_OF_PREDEFINED_VOICES`); female: anna, vera, fantine, eponine,
  azelma, mary, jane, eve, cosette, caro_davy. Voices download from HF on
  first use (needs internet).

## How to operate

- Start voice: `start-voice` DSH skill (user-level,
  `C:\Users\press\.dsh\skills\start-voice`) or `run_voice_dsh.bat` /
  `python -X utf8 -m voice_stack.voice_dsh` from TTSTT. ~35–45 s to "Voice
  bridge ready". Stop = Ctrl-C or kill the python process.
- Logs: `tests\live_dsh_log.txt` (rewritten each start: heartbeat lines with
  rms/stt-ms/backlog, `stt word '…'` lines, turn timings incl. thinking chars),
  `tests\dsh_runtime.err.log` (SDK stderr), `tests\dsh_handoff.err.log`.
- Debug a silent voice: if `stt word` lines appear but no `USER:` turn, it's
  endpointing; if turns are slow, check LLM slot contention.

## Coding voice (separate bridge)

- `python -X utf8 -m voice_stack.coding_voice` (skill: `start-coding-voice`).
  Mic -> STT -> utterance injected into the **open coding chat** via the GUI
  gateway (`POST /api/session/prompt`, cookie minted from
  `~/.dsh/.credentials.yaml`); the chat's **visible** assistant output is
  followed live over WS `session/follow` (`/api/remote.mux`, same cookie,
  `max_size` 200 MB — snapshots of long sessions exceed 1 MB) and spoken with
  the shared `SpokenStream` (sentence-level, thinking never spoken). Half
  duplex: floor held from inject until `turn/end` + audio drained (120 s
  safety release). Log: `tests\coding_voice_log.txt`.
- Session events wire shape (verified by `tests\probe_follow.py`): WS mux
  frame `{type:"item", value:{type:"event", event:{type,seq,data}}}`;
  `assistant/chunk` carries `data.chunk` = StreamChunk (block-start /
  text-delta / reasoning-delta / block-end); `turn/end` =
  `{turn, reason:{kind}}`.

## Open threads

- README/HANDOFF refreshed for the DSH-bridge architecture (2026-09-19).
- Possible future work: session resume for the voice (SDK gap); AEC/ducking for
  speaker (non-headphone) use; barge-in is currently full half-duplex mute
  (no mid-answer interruption).
