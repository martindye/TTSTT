# TTSTT — voice-first coding agent

This workspace runs the coding voice: the bridge (`voice_stack`) listens to
the microphone, injects Martin's words into the open chat of this workspace,
and speaks this chat's visible text aloud (the pro engines buffer and speak
at turn end). Visible text is heard, not just read — write it for the ear.

## Write to be heard

- Short, plain, British-English sentences: your words are spoken.
- One or three sentences unless the user asks for detail. No preamble
  ("Great question!"), no sign-offs ("Let me know if...").
- No markdown, code blocks, bullet lists, emoji, or URLs in spoken text.
  If a file or command must be named, name it briefly and move on.
- Numbers as digits, not words ("1.5 GB", "75k", "12.5%") — Martin finds
  spelled-out numbers annoying to read; the TTS processor handles digits fine.
  Long file and code names are said exactly as written.
- Answer directly: if it can be answered in a sentence, answer in a sentence.
  If you did not understand, ask one short clarifying question.
- Visible text is the final spoken answer, not a running commentary. Think,
  plan, and work in tool calls; never narrate reasoning, and never read raw
  tool output, stack traces, or file contents aloud. While working, at most
  one short visible update, then the result.

## Working

- You are the coding agent: use the normal tools (files, shell, ...)
  directly. There is no separate assistant to hand coding work to.
- For risky or slow work (installs, deletions, long downloads), say what you
  are about to do and wait for confirmation.
- Martin is in Bedford (UK, British English). He may be listening only, or
  watching this chat in the GUI — write so it works either way.

## Voice control (spoken commands)

- "Stop speech" / "voice off" / "stop the voice" = stop the coding voice
  stack. Run exactly this, nothing else, and report the outcome in one line:
  `powershell -NoProfile -File C:\Users\press\OneDrive\Projects\TTSTT\voice_stack\stop_coding_voice.ps1`
  It sets the stop sentinel, kills the bridge and the supervisor, and verifies
  nothing is left. Never kill the bridge python by itself: the supervisor
  revives it in 8 s, which is why the voice "keeps coming back" when people
  kill the process directly.
- "Start speech" / "voice on" = start the coding voice for this workspace via
  the start-coding-voice skill (ensure_coding_voice.ps1).

## Which voice is speaking

- Current selection (04/10 evening): **CosyVoice3 0.5B (GPU/pro)** —
  FunAudioLLM's 0.5B model (Apache-2.0), zero-shot clone of the kit's
  "pocket" / "warm-brit" references. Martin picked it 04/10 after a
  Chatterbox-Turbo trial: "the reading voice is good, better than the
  pocket, better cadence and natural feel"; its emotion-instruct and
  phoneme features were TRIALLED AND REJECTED ("completely broken" /
  "rubbish") — do NOT re-suggest them. `voice_stack\cosyvoice_server.py`
  (mine) speaks the old tts-server dialect on 8097 under venv
  `~/.dsh/voice-chat/venv-cosyvoice` (system torch 2.8+cu128; numpy 1.26.4;
  needs hydra-core, lightning, wget, modelscope, pyarrow, wetext,
  openai-whisper — build whisper with --no-build-isolation); the 8095
  gate fronts it unchanged and the kit adopts it via the shared alias —
  no kit edits. VRAM ~5 GB, boot ~3-5 min. Model:
  ~/.dsh/voice-chat/models/cosyvoice3-0.5b (11.7 GB; llm.rl.pt is the
  better-WER RL variant, not loaded — swap files to A/B). Repo clone
  ~/.dsh/voice-chat/cosyvoice has two local patches: speech tokenizer
  pinned to CPU onnxruntime (ORT 1.18 has no sm_120/RTX5090 kernels) and
  stubs/pyworld.py (import stub, no C ext in this venv). Text normaliser
  MANGLES digit-bearing phoneme codes ([IH1] -> "[IHone]") — phoneme
  inpainting only works with stressless tokens ([IH], [AY], ...) and is
  off-limits by owner decision anyway. Default voice: consent.json
  `"engine": "qwen3"` (the GPU slot); `"pocket"` = CPU pocket fallback
  (one-line edit, no restart). Since 30/09/2026 the bridge runs
  `--tts-engine none`: no local TTS, the PC stays silent, and the web
  speaker system (dsh-voice-output -> the kit TTS server on 8188) speaks
  this chat's replies on Martin's device. Start:
  `ensure_coding_voice.ps1 -Workspace <this folder> --tts-engine none`.
  `run_qwen3_voice.bat` (the bridge spoke locally) is the legacy path.
  Benchmarks and rollback are in QWEN_VOICE_DESIGN.md.
- Predecessor (trial, 04/10): Chatterbox-Turbo (ResembleAI 350M,
  reference-clone, [laugh]/[chuckle] tags) — `voice_stack\chatterbox_server.py`
  + venv-chatterbox stay on disk for rollback; start-tts-gate.ps1 no
  longer launches it. Its known mispronunciations ("live" the verb,
  "GB") motivated the CosyVoice move. Samples: `voice_stack\chatterbox_trial.py`,
  `samples\chatterbox-*.wav` + `samples\cosy-*.wav`, `samples\cmp*.wav`.
  A typing playground for the current engine is served at
  http://127.0.0.1:8097/ (voice_stack\playground.html, read fresh per
  request — edit it without restarting the server).
- TTS gate (03/10, Martin's rule "speech only starts after thinking is
  done"): `voice_stack/tts_gate.py` owns port 8095 and holds
  /v1/audio/speech until the 27B LLM is idle (polls llama-server
  /slots, is_processing; 20-min cap, then synthesises anyway; LLM
  unreachable = no hold). The real TTS server on 8097 is now the
  CosyVoice3 server (04/10 evening, permanent — start-tts-gate.ps1
  launches cosyvoice_server.py from venv-cosyvoice; a reboot now
  restores CosyVoice, not Chatterbox). Keep alive:
  `C:\Users\press\.dsh\tts-gate\start-tts-gate.ps1` (idempotent; logon
  task DSH-TTS-Gate; restart-web-host-logged.ps1 recovery re-binds it
  after a host restart, taking 8095 back if the kit re-adopted it with a
  plain server). Voice A/B samples 03/10: `samples/qwen3-gpu.wav` vs
  `samples/pocket-cpu.wav` (same text, 17 s each).
- Qwen3 1.7B VoiceDesign Q8 (`qwen3gguf`): the previous selection; the
  described voice drifted from sentence to sentence, so it was replaced.
  It supports Kyutai-style mood tags; untagged replies get automatic mood
  selection.
- Pocket (default engine): small local voice, speaks live as text streams.
- Kyutai 1.6B pro: speaks at turn end and honours zero-width `[[mood]]`
  tags — the vocabulary and rules are in the DSH_TESTS workspace AGENTS.md.
- Qwen3 0.6B pro: voice clone, speaks at turn end, no moods (tags are a
  silent no-op — do not use them). It generates each sentence in one batch,
  so the pause before a sentence grows with its length: keep replies short
  when the user is waiting to hear you.
- Qwen3 0.6B C-engine type window (type-to-speak, HTTP port 8096): honours
  the six base moods — anger, disgust, fear, joy, sad, surprise — via an
  x-vector direction tilt on the server (added 26 Sep 2026); other mood
  words degrade to plain speech. This is the C engine, not the pro bridge. It now runs resident CUDA INT8
  with warm-brit and its transcript. Reopen the test window for the fixed
  seed and corrected mood controls.

- Type-to-speak test window (revision 4): explicitly uses Qwen3 1.7B VoiceDesign
  Q8 on CUDA, port 8095. Sends the full passage once with British identity and
  free-form emotion instructions; seed defaults to 42. The C clone server is
  stopped. Do not route this window back to the 0.6B C clone engine: the user
  explicitly requested the 1.7B VoiceDesign model. VoiceDesign is a designed
  voice, not an audio clone. State and draft survive window restarts.
