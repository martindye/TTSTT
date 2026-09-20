"""Generate short samples of female candidate voices so the user can pick.

Loads the TTS model once, then for each candidate voice renders the same
sentence to tests/voice_samples/<voice>.wav. Listen and tell me which to keep.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\press\OneDrive\Projects\TTSTT")

import soundfile as sf  # noqa: E402

from voice_stack.tts_engine import TTSEngine  # noqa: E402

# Female candidates from the pocket-tts catalog (English).
VOICES = [
    "alba",      # current
    "anna",
    "vera",
    "fantine",
    "eponine",
    "azelma",
    "mary",
    "jane",
    "eve",
    "cosette",
    "caro_davy",
]

SENTENCE = "Hi, I'm your voice. How do I sound?"

OUT = Path(r"C:\Users\press\OneDrive\Projects\TTSTT\tests\voice_samples")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    engine = TTSEngine(language="english", voice="alba")  # load model once
    model = engine.model
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    for v in VOICES:
        out = OUT / f"{v}.wav"
        t1 = time.time()
        try:
            state = model.get_state_for_audio_prompt(v)
            chunks = list(model.generate_audio_stream(state, SENTENCE))
            import numpy as np
            audio = np.concatenate([np.asarray(c, dtype="float32").reshape(-1) for c in chunks])
            sf.write(out, audio, 24000)
            print(f"OK  {v:12s} {time.time() - t1:5.1f}s  -> {out.name} "
                  f"({len(audio) / 24000:.1f}s audio)", flush=True)
        except Exception as e:
            print(f"ERR {v:12s} {str(e)[:100]}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
