# Video shot list — "a fully local voice interface for my coding agent"

~5–8 min. One take per section is fine; the bridge is always on screen.
Prereqs: mic **unmuted** (check Windows sound settings — the bridge can't see
a system mute), headphones off (or speakers quiet), room quiet.

## 1. Cold open (30 s)
Say something plain to the machine and let it answer. That's the whole pitch.

## 2. What it is (1 min)
Screen: the pipeline diagram from README.md.
"Everything local: mic → Kyutai STT 1B on the GPU → my coding agent (Qwen 27B
through llama.cpp) → Pocket TTS on the CPU. Nothing leaves this machine."

## 3. The voice assistant (1.5 min)
- Show `python -X utf8 -m voice_stack.voice_dsh` booting (~35 s — let the
  countdown run, don't cut it).
- Ask something simple ("what's the capital of France") → answer in ~3 s.
- Ask something that needs work ("check what's in the repo") → show it doing
  real tool work, then reporting back in two sentences.
- Point out: **thinking never gets spoken** (thinking happens, you never hear
  it).

## 4. Coding voice (2 min) — the hero shot
- Switch to the coding chat. Start `coding_voice`.
- Ask it to do something visible ("add a README note", "list the files").
  Watch: the **words appear in the chat window** as you say them, then its
  reply is **spoken** while you watch it appear.
- Best single shot: ask a question, watch the answer type itself into the
  window while it's being said aloud.

## 5. Handoff (1 min)
- To the voice assistant: "hand off a task to the coding agent — tell it to
  fix X".
- It says "it's in your coding chat" → cut to the coding chat where the agent
  is now working on it live. ChatGPT-style: you watch your agent work.

## 6. Under the hood (1 min)
- `git log` / repo layout: `voice_stack/`, `SETUP.md`, `requirements.txt`.
- Show one measured number: "~3 s from end of my sentence to first audio,
  fully local".
- One-line roadmap: barge-in, session resume.

## 7. End card
Repo + "fully local, see SETUP.md to build it on your machine".

## Production notes
- Keep a hand near Ctrl-C: the bridge owns the mic; stop it between takes.
- If it goes deaf: Windows mic mute is the #1 silent killer.
- Keep answers short in the video: "keep your answers short" works — it
  listens.
