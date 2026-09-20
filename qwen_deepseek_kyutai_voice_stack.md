# Local Voice Stack Plan for Qwen / DeepSeek

## Goal

Build a fast, fully local speech interface around an existing Qwen or DeepSeek LLM running on an NVIDIA RTX 5090 (32 GB VRAM).

## Recommended architecture

```text
Microphone
   ↓
Kyutai STT 1B (GPU)
   ↓
Qwen / DeepSeek LLM (GPU)
   ↓
Pocket TTS (CPU)
   ↓
Speakers
```

## Speech-to-text: Kyutai STT 1B

Use:

```text
kyutai/stt-1b-en_fr
```

Why this model:

- About 1 billion parameters.
- Designed specifically for streaming / real-time transcription.
- Supports English and French.
- Fixed transcription delay of about 0.5 seconds.
- Includes semantic voice activity detection (VAD), useful for detecting when the user has finished speaking.
- Produces word-level timestamps.
- Better suited to a conversational voice assistant than Kyutai's larger 2.6B STT model because latency is much lower.

### Larger alternative

```text
kyutai/stt-2.6b-en
```

- About 2.6 billion parameters.
- English only.
- Optimized more heavily for transcription accuracy.
- About 2.5 seconds of built-in transcription delay.

For a live assistant, prefer the **1B model** unless maximum transcription accuracy matters more than conversational responsiveness.

## Text-to-speech: Kyutai Pocket TTS

Use:

```text
kyutai/pocket-tts
```

Pocket TTS is a very small streaming TTS model:

- About 100 million parameters.
- Designed to run efficiently on CPU.
- Roughly 200 ms to first audio chunk in Kyutai's published figures.
- Roughly 6× real-time generation on a MacBook Air M4 CPU.
- Uses only around two CPU cores in Kyutai's benchmark.
- Supports streaming output.
- Supports voice cloning.
- Current project documentation lists English, French, German, Portuguese, Italian and Spanish support.

### CPU or GPU?

Run Pocket TTS on the **CPU first**.

Kyutai explicitly notes that on CPUs with strong single-thread performance they did not observe a useful GPU speedup. Because the model is only around 100M parameters and inference uses batch size 1, moving it to the RTX 5090 may provide little practical benefit.

This leaves GPU resources available for:

1. Qwen / DeepSeek
2. Kyutai STT
3. LLM context/KV cache

The generated voice quality should not inherently improve merely because Pocket TTS is moved from CPU to GPU; the main potential difference is inference speed.

## Suggested implementation priorities

1. Get `kyutai/stt-1b-en_fr` streaming from the microphone on CUDA.
2. Feed finalized utterances into the existing Qwen / DeepSeek harness.
3. Stream LLM output by sentence or suitable text chunks.
4. Feed those chunks immediately into Pocket TTS on CPU.
5. Stream generated PCM audio directly to playback.
6. Add interruption / barge-in handling:
   - detect new user speech with STT semantic VAD;
   - stop current TTS playback;
   - cancel or pause current LLM generation;
   - begin the next conversational turn.

## Latency target

The important metric is not total generation time but **time to first useful response audio**.

A good pipeline should overlap work:

```text
User speaks
   ↓
STT streams partial transcription
   ↓
End-of-speech detected
   ↓
LLM starts immediately
   ↓
First usable phrase produced
   ↓
Pocket TTS starts streaming it
   ↓
Audio begins while the LLM continues generating
```

Do not wait for the LLM to finish its entire answer before starting TTS.

## Hardware strategy for RTX 5090 32 GB

Preferred allocation:

```text
RTX 5090:
  Kyutai STT 1B
  Qwen / DeepSeek
  LLM KV cache

CPU:
  Pocket TTS
  audio capture/playback
  orchestration
```

This should preserve substantially more VRAM for the main LLM than placing every component on CUDA.

## Important correction

Kyutai has **two** main STT checkpoints:

| Model | Approx. parameters | Languages | Built-in delay | Best use |
|---|---:|---|---:|---|
| `stt-1b-en_fr` | ~1B | English + French | ~0.5 s | Live conversational assistant |
| `stt-2.6b-en` | ~2.6B | English | ~2.5 s | Higher-accuracy transcription |

The larger model is **2.6B**, not 3B. Hugging Face may round/category-label it as 3B in some UI views, but Kyutai describes it as approximately 2.6B parameters.

## Sources

- Kyutai STT: https://kyutai.org/stt/
- Kyutai STT GitHub / delayed-streams-modeling: https://github.com/kyutai-labs/delayed-streams-modeling
- Kyutai Pocket TTS: https://github.com/kyutai-labs/pocket-tts
- Pocket TTS model: https://huggingface.co/kyutai/pocket-tts
