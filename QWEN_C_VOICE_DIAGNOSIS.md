# C voice diagnosis — 26 September 2026

The open Tk window was `scripts/qwen3_type_window.py` (PID 34592).
The WSL server was a CPU-only build, with `--int4`, loading `pocket.qvoice`.
It was not using the existing warm British reference. No GPU speech process was running.
Do not stop the user's llama-server to make room.

## Changes applied

- Server now loads `warm-brit.qvoice`, extracted from the existing synthetic
  `voice_stack/ref_voices/warm_brit.wav` plus its transcript. The profile contains
  227 ICL frames. This selects the intended reference; perceived accent is not
  independently verified and can still drift when emotion modifies the embedding.
- CPU fallback uses INT8, matching the local small-model emotion recipe. The source
  warns that INT4 Talker quantisation can damage language fidelity. No controlled
  comparison was run, so do not attribute all variability to quantisation.
- Emotion vectors are present and the live log confirms the joy direction was applied.
  The mechanism is a generic speaker-embedding offset, not native emotion instruction
  following. Dedicated same-speaker emotional references would be more faithful.
- App fixes: fixed seed 42; unsupported moods explicitly rejected instead of mapped
  to unrelated moods or silently neutral; British identity retained in VoiceDesign
  mood prompts; Stop cannot re-enable Speak while the old request is still pending.
- The emoting clone submits the complete passage in one non-streaming request.
  This preserves context but waits for the full passage before playback.
- WSL launcher runs in the foreground of its hidden Windows process: detached shell
  startup did not reliably survive the launching shell exiting here.

## Verification and remaining work

Four unit tests pass in `tests/test_type_window_controls.py`.
A live request for “Hello, Martin. Shall we try that again?” with joy and seed 42
produced a valid 3.76-second WAV in 7.73 seconds measured on the Windows client.
This is one diagnostic request, not a performance qualification. Audio saved in
`tests/warm_brit_c_joy.wav`; timing in `tests/warm_brit_c_diagnostic.json`.
No listening-based claim about the result has been made.

The server was restarted with the corrected British profile. The existing Tk process
was left open to preserve the user's text; reopen it to load the corrected controls
and the new voice label. Its old “pocket (emoting clone)” selection now reaches the
British server, because the C API uses the server-loaded reference.

GPU work is pending permission to install a WSL CUDA toolkit and build the engine.
Windows has CUDA 12.8, but WSL has no nvcc. Ubuntu's stock toolkit candidate is 12.0,
which is unsuitable for the build's sm_120 target; use an appropriate NVIDIA toolkit.
Review `tools/qwen3-tts/docs/serving/gpu-cuda.md` and `cuda-performance.md` before
choosing GPU flags. Do not simply add --backend cuda to the CPU launcher or carry
CPU INT8 flags across: the documentation warns that INT8 disables GPU resident paths.
The GPU path is upstream work in progress; validate output and VRAM alongside the
existing LLM before switching. No streaming is requested.

## GPU installation completed

User authorised the installation. Installed CUDA 12.8 compiler, runtime and cuBLAS
development files from NVIDIA's WSL repository, without installing a Linux driver.
Source: https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/
Reproduction: scripts/install_c_voice_cuda.sh and scripts/build_c_voice_cuda.sh.
Separate GPU build: /home/press/ttstt/qwen3-tts-cuda. CPU build remains intact.
CUDA primitive self-test passed. Source ef339be-dirty:713675d96ebe; binary SHA256
4b0f5e4c2d7e717672802d7357821ba4935e6524b95515eba8ecac2d7d7b4cae.

Correction to the earlier caution: main.c and the CUDA kernels confirm INT8 works
with the resident paths enabled. The bf16-only seam warning does not apply to those
kernels. Live startup confirms resident INT8 Talker and CP, CUDA graphs and GPU
decoder enabled. The launch script explicitly enables all three resident paths.

Port 8096 now uses GPU INT8 with warm-brit and 227 ICL frames. No streaming or batching
was introduced. The original llama-server (PID 32092) remains running. The candidate
server on 8097 and old CPU speech processes were stopped. For CPU rollback copy
scripts/serve_c_voice_cpu.sh to /home/press/ttstt/serve_c_voice.sh and restart only
the idle speech server.

Four candidate requests passed valid non-silent audio, differing emotion output
and byte-identical joy output repeated after sadness with seed 42. Candidate times
were 3.42-3.75 seconds for 3.36-4.00 seconds of audio. Production-port requests then
took 5.84 and 5.56 seconds for 3.52 seconds of audio: latency varies alongside the
existing GPU workload. About 1.4 GB VRAM remained after requests.
Evidence: tests/c_voice_gpu/run_manifest.json, speech_checks.json, live_check.json
and WAV files. These are smoke checks, not performance qualification or listening
validation of accent/emotion quality. Four UI regression tests also pass.

The existing Tk window remains open to preserve its text. Its C voice option now
reaches the GPU server. Reopen it for the corrected mood controls and British/GPU
labels. The clone emotion mechanism still alters speaker identity and can drift.

## Superseded: user clarified the test app must use 1.7B VoiceDesign

The cloning-engine work above did not meet the intended model choice. Revision 4
of scripts/qwen3_type_window.py now offers only Qwen3 1.7B VoiceDesign Q8 and sends
to the verified ttstt-qwen3-voicedesign-q8 server on port 8095. The experimental
C clone server on 8096 was stopped to free GPU memory. The LLM was untouched.

The previous window also split VoiceDesign text into separate synthesis calls.
Revision 4 submits the entire passage in one non-streaming request. Its identity
instruction describes a British female voice without a competing playful/brisk
mood. The emotion box accepts full descriptions unchanged, expands sad/happy/angry
aliases, and retains the identity description. VoiceDesign designs a voice; it is
not cloning the previously created British reference recording.

Seed defaults to 42. Negative values (including the random sentinel -1), invalid
text and out-of-range values are rejected. tts-server.h parses the seed and
instructions; tools/tts-server.cpp forwards them into p.seed and p.instruct.
A fixed seed guarantees neither the same speaker across changed prompts nor
perfect accent/emotion fidelity. Generating a complete passage avoids resetting
the voice between its sentences.

The open revision-3 app had meanwhile been changed by another process, adding a
seed field and allowing arbitrary moods through the C path. Those additions were
inspected and the seed field retained. Its current 106-character draft was saved
and restored into the launched revision-4 window. The UI screenshot and captured
request confirm the running window uses the correct model, full two-sentence text,
sad instruction and seed 42. GUI synthesis took 0.55 seconds for 6.72 seconds of
audio. Three direct checks took 0.50-0.61 seconds; repeated sad requests produced
byte-identical WAVs, and a mixed disappointment/reassurance instruction changed
the output. That validates routing and repeatability, not subjective emotion quality.
Six regression tests pass. Evidence: tests/voice_window_last_request.json,
tests/voicedesign_revision4/checks.json and tests/voicedesign_revision4.png.
Official model reference: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign

## Revision 6: verbatim instructions at user request
Compared the official test_model_12hz_voice_design.py and inference wrapper with
qwentts.cpp: HTTP instructions maps to p.instruct and the same user-role token
wrapper. Talker and subtalker defaults match (temperature .9, top-k 50, top-p 1).
User requested no added wording. Revision 6 forwards the instruction exactly,
without British identity, mood expansion or sample controls. Blank instructions
are rejected because VoiceDesign requires an instruction. Seed and whole-passage
behaviour remain. Six tests pass, including exact whitespace preservation and
full request routing. Do not claim perceptual emotional quality from routing tests.
