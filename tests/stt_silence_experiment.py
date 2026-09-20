"""Controlled STT experiment.

Phase A: 90s of digital silence (reproduces the long ambient-quiet period).
Phase B: tests/test_speech.wav (4.8s of real recorded speech).
Phase C: 15s of trailing silence (should trigger end-of-utterance).

Verdict:
  - words only in B  -> engine fine; live failure is state/env related
  - no words at all  -> engine itself is deaf in this environment right now
"""
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, r"C:\Users\press\OneDrive\Projects\TTSTT")
from voice_stack.stt_engine import KyutaiSTT, SttEventKind  # noqa: E402

FRAME = 1920


def feed_phase(stt, pcm: np.ndarray, label: str, max_print=20):
    print(f"--- phase {label}: {len(pcm)/24000:.1f}s of audio", flush=True)
    words, ends, silences = [], 0, 0
    shown = 0
    t0 = time.time()
    for i in range(0, len(pcm), FRAME):
        frame = pcm[i:i + FRAME]
        if len(frame) < FRAME:
            frame = np.pad(frame, (0, FRAME - len(frame)))
        for ev in stt.feed_frame(torch.from_numpy(frame.astype(np.float32))):
            if ev.kind is SttEventKind.WORD:
                words.append(ev.piece)
                if shown < max_print:
                    print("   word:", repr(ev.piece), flush=True)
                    shown += 1
            elif ev.kind is SttEventKind.END:
                ends += 1
            else:
                silences += 1
    text = "".join(words).replace("\u2581", " ")
    print(f"=== phase {label} done in {time.time()-t0:.1f}s: words={len(words)} ends={ends} silences={silences}")
    print("    text:", repr(text), flush=True)
    return text


def main():
    t0 = time.time()
    stt = KyutaiSTT(hf_repo="kyutai/stt-1b-en_fr", device="cuda")
    print(f"STT loaded in {time.time() - t0:.1f}s", flush=True)

    silence = np.zeros(90 * 24000, dtype=np.float32)
    feed_phase(stt, silence, "A: 90s silence")

    speech = sf.read(r"C:\Users\press\OneDrive\Projects\TTSTT\tests\test_speech.wav",
                    dtype="float32")[0]
    text_b = feed_phase(stt, speech, "B: real speech wav")

    trail = np.zeros(15 * 24000, dtype=np.float32)
    feed_phase(stt, trail, "C: 15s trailing silence")

    # control: feed the SAME speech again right now (state after B)
    text_b2 = feed_phase(stt, speech, "B2: same speech again")

    print("\nVERDICT:", "STT HEARS SPEECH" if (text_b.strip() or text_b2.strip())
          else "STT DEAF (no words from real speech)")


if __name__ == "__main__":
    main()
