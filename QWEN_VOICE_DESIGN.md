# Qwen voice integration

The selected deployment is `qwen3ggufbase`: Qwen3 1.7B **Base** Q8 on the GPU,
cloning from a *registered reference voice*, via qwentts.cpp. The previous
selection `qwen3gguf` (VoiceDesign Q8) re-derives the speaker from the text
description on every request, so the identity drifted between sentences; the
Base checkpoint instead keeps the speaker's latents in memory and clones the
same anchor for every sentence. Both deployments share the same binary,
port, and launch protocol (below). The coding bridge is live with
`--tts-engine qwen3ggufbase`.

## Reference-voice deployment (qwen3ggufbase) — 26 September 2026

Weights: `qwen-talker-1.7b-base-Q8_0.gguf` from the same pinned
`Serveurperso/Qwen3-TTS-GGUF` revision, next to the VoiceDesign talker in
`~/models/Qwen3-TTS-VoiceDesign-GGUF`. Same 12 Hz codec.

The Base checkpoint carries the ECAPA speaker encoder, so the server
accepts `POST /v1/audio/voices` registrations (16-bit mono 24 kHz WAV,
optionally with `ref_text` for ICL mode). Voices are in-memory and
re-registered idempotently on every engine start; they are lost with the
server process. Three are registered:

- `pocket` — currently active. The standard small coding voice (pocket
  TTS, voice "anna"), captured straight from the local engine
  (`voice_stack/ref_voices/pocket.wav` + `.txt`), registered with its
  source text (ICL).
- `warm-brit` — an 18 s take of the warm British female voice, recorded
  from the old VoiceDesign server and frozen as the reference
  (`voice_stack/ref_voices/warm_brit.wav` + `.txt`), registered with its
  source text (ICL).
- `p277` — the old plain pro voice (VCTK p277_023 from the HF cache),
  registered x-vector-only, as the male alternative.

Switching the bridge's voice is one line: `Qwen3GGUFBase(voice="pocket")`
in `voice_stack/coding_voice.py`. Registering a new voice is one POST, no
restart. Mood tags are a silent no-op on this engine (the Base prompt
builder takes no instructions).

Engine: `voice_stack/qwen3_gguf_base_engine.py` — starts or reuses its
hidden server on `127.0.0.1:8095` (alias `ttstt-qwen3-base-q8`, forced
`CUDA0`), logs to `tests/qwen3_gguf_base_server.log`, state in
`tests/qwen3_gguf_base_server.json`. `scripts/stop_qwen3_gguf.ps1` now reads
the alias from the state file (an optional `-StatePath` selects which
server to stop; the default path still stops the VoiceDesign server).

Validation on 26 September 2026, RTX 5090 with the LLM resident: all four
smoke sentences came back fully intelligible on the STT roundtrip
(`scripts/tmp_*` deleted after the run); a 4.6 s sentence synthesised in
0.66 s warm (RTF ≈ 0.15; server log shows RTF 0.095). `run_qwen3_voice.bat`
now launches `qwen3ggufbase`.

Rollback ladder: `--tts-engine qwen3gguf` (VoiceDesign, the previous
selection — drifts sentence to sentence), then `qwen3tts`, then
pocket/kyutai.

Validation on this RTX 5090, with the existing LLM and Kyutai STT loaded:

| Mood | Audio duration | Warm synthesis | RTF |
| --- | ---: | ---: | ---: |
| neutral | 2.85 s | 0.422 s | 0.148 |
| amusement | 3.04 s | 0.312 s | 0.103 |
| interest | 3.43 s | 0.375 s | 0.109 |

All three STT roundtrips recovered the test words correctly (punctuation can
differ). This checks intelligibility, not subjective voice preference. About
1.1 GiB of GPU memory remained free. The first synthesis after initially
starting the server took 9.91 s; startup now silently warms the server before
the bridge reports ready. These are synthesis times, excluding the LLM reply
and turn-end buffering. The new backend does not speak while the agent works.

A longer 32-word sentence generated 10.04 s of audio in 1.25 s (RTF 0.125)
with the live microphone bridge running, leaving about 1.0 GiB free. No audio
was played during the diagnostic tests, to avoid feeding it into the mic.

`scripts/bench_qwen3_gguf.py --transcribe` records the results and WAV samples
under `tests/`; run it with the live bridge stopped because it loads its own
STT model. Twenty-one offline and local-HTTP contract tests pass.

## GPU Q8 deployment

User preference: buffer the reply until turn end, then generate complete
sentences. `qwen3gguf` sends the complete sentence and voice instructions with
`response_format: wav`, and only hands audio to the speaker after the full WAV
arrives. It does not use the runtime's PCM streaming mode.

Source checkout: `tools/qwentts.cpp`, revision
`6a3e91283220197bad2ae1eb40ae1c1392bfd820`, ggml submodule
`765bc96f9bb8d4c397c91c23b4e5c52a93fcf9b0`. Built with existing Visual Studio
2022 Build Tools and CUDA 12.8 for the RTX 5090 (`sm_120a`). Source/tool output
is ignored by Git. No existing Python packages or LLM configuration are changed.

Weights: `Serveurperso/Qwen3-TTS-GGUF`, pinned revision
`b7ee2e8c7459c3bea99da23e3d178125a7d1713c`:

- `qwen-talker-1.7b-voicedesign-Q8_0.gguf`: 2,042,833,824 bytes.
- `qwen-tokenizer-12hz-Q8_0.gguf`: 291,150,624 bytes.

Download is about 2.33 GB total; runtime GPU use is additional and must be
measured separately. `scripts/download_qwen3_gguf.py --download` fetches only
these files into `~/models/Qwen3-TTS-VoiceDesign-GGUF`.

Start/restart the coding bridge with `run_qwen3_voice.bat`. It uses the existing
supervisor to stop the old bridge before loading the replacement. The voice
adapter starts or reuses its own hidden server on `127.0.0.1:8095`, forced to
`CUDA0`, with one generation at a time. This server stays resident across bridge
restarts. It has no connection to the LLM server on port 8080.

Server log: `tests/qwen3_gguf_server.log`; saved PID:
`tests/qwen3_gguf_server.json`. `scripts/stop_qwen3_gguf.ps1` verifies executable
and alias before stopping that server. To roll back, stop this server and use
the usual ensure script with `--tts-engine qwen3tts` (or Pocket).

Generate a complete WAV without microphone capture or playback:

```powershell
python -m voice_stack.qwen3_gguf_engine --mood amusement --out tests/qwen3_gguf_preview.wav
```

`--qwen3tts-instruct` customises the voice in either bridge; `--tts-moods off`
disables mood additions in the coding bridge. `qwen3gguf` is fixed to GPU Q8;
the generic `--tts-quantize`, `--qwen3tts-model` and `--qwen3tts-device` flags
do not configure its server.

Upstream runtime: https://github.com/ServeurpersoCom/qwentts.cpp

Quantised weights: https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF

## Findings on 25 September 2026

The running coding bridge selects `qwen3tts`: Qwen3-TTS 0.6B Base with an
x-vector-only clone of VCTK p277. It has no instruction/mood support. The
bridge buffers replies until turn end; Qwen then generates each entire sentence
before returning audio. Splitting the returned waveform into small pieces is
playback chunking, not streaming synthesis. Existing adapter notes record an
RTF around 4.6 with SDPA (generation takes several times the audio duration).
The live log also shows one short sentence taking about 42 seconds to produce
57 chunks (about 4.6 seconds of audio). This is one observation, not a controlled
benchmark; GPU contention and other load can affect it.

`qwen-tts` 0.1.1 is already installed with CUDA PyTorch 2.8.0. Its
`generate_voice_design` returns complete waveforms. Its `non_streaming_mode=False`
option explicitly does not enable true streaming generation. VoiceDesign can
improve expression; simply replacing 0.6B with 1.7B is not a speed fix.

The original full-precision checkpoint and bundled speech tokenizer total about 4.52 GB.
The RTX 5090 had only about 0.9 GiB free with the LLM, STT and existing TTS
running. The Python adapter conservatively requires 6 GiB free before loading bf16
on CUDA. Actual peak usage still needs measuring. Even replacing the old voice
may not free enough for that full-precision path. Replacing the current voice
does recover its allocation; the microphone's STT allocation must be retained.
Windows GPU counters showed the whole old bridge at about 6 GiB, including STT,
TTS and runtime overhead. That is not 6 GiB available for the new TTS alone.
The user-managed llama-server has not been touched.

CPU is an i9-13900K (24 cores / 32 threads). The provisional estimate given
for full-precision Python CPU synthesis was 30–90 seconds per 10 seconds of
audio, not a benchmark. A native quantised runtime may differ substantially.
The user chose GPU, so no CPU voice benchmark was run.

## Setup and preview

Inspect the pinned download without fetching weights:

```powershell
python scripts/download_qwen3_design.py
```

After approval for the model download:

```powershell
python scripts/download_qwen3_design.py --download
```

Default destination is `~/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign`. Setup uses
a normal local directory to avoid Windows snapshot-symlink permissions.
Bridge loading is local-only; it never downloads weights at startup.

When sufficient GPU memory is available, generate a WAV without playing it:

```powershell
python -m voice_stack.qwen3_design_engine --mood amusement --out tests/qwen3_design_preview.wav
```

For a slower preview that leaves the occupied GPU alone, add
`--qwen3tts-device cpu`. That uses float32 and also needs host RAM.
The preview reports synthesis time, audio duration and RTF. Test a few short
sentences with the same voice description and different moods before activation.

## Harness integration

For the coding bridge, after stopping its existing instance through the normal
supervisor flow and resolving memory headroom:

```powershell
python -X utf8 -m voice_stack.coding_voice --workspace C:\Users\press\OneDrive\Projects\TTSTT --tts-engine qwen3design
```

For the separate DSH voice assistant:

```powershell
python -X utf8 -m voice_stack.voice_dsh --tts-engine qwen3design
```

Both accept `--qwen3tts-model`, `--qwen3tts-instruct` and `--qwen3tts-device`.
The default instruction describes a warm, conversational British female voice.
`--tts-voice` and `--tts-quantize` do not configure VoiceDesign.

The coding bridge reuses its existing mood selection and inline `[[mood]]`
handling. Identity stays in the base description and the adapter appends a
delivery instruction. `--tts-moods off` disables these appended instructions.
The separate DSH voice assistant uses the base voice description only; it does
not share the coding bridge's mood-tag parser. Voice identity consistency across
sentences remains an audio-validation item, because each sentence is designed
afresh. Existing workspace instructions still describe the currently running
0.6B voice; update them when switching the actual engine.

VoiceDesign startup errors are explicit, rather than silently substituting a
different voice. The old `qwen3tts`, Pocket and Kyutai paths remain available.

## Validation

```powershell
python -m unittest discover -s tests -p "test_qwen3*.py" -v
```

Tests use a mocked model: routing, instruction/mood handling, audio format,
cancellation, worker errors, sample-rate rejection, checkpoint validation,
memory guard and the existing clone generation contract. They do not establish
real model performance or sound quality.

Upstream: https://github.com/QwenLM/Qwen3-TTS#voice-design

Model: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
