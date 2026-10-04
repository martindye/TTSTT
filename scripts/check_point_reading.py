"""Regression check: does the live pocket voice say "point" for a decimal,
and does a unit suffix change anything?

Loads the same engine the bridge runs (pocket, voice "anna", int4 — CPU
only, no GPU load) and synthesises three short samples into tests/ for
Martin to judge by ear:

  tests/point_test_bare.wav  -- "The value is 1.5."
  tests/point_test_unit.wav  -- "The value is 1.5 gigabytes."
  tests/point_test_mid.wav   -- "Free memory is 1.2 now."

Run:  python scripts/check_point_reading.py
Then play the WAVs in order (bare, unit, mid), e.g. via the start-coding-voice
helpers or any player.
"""
import sys
import threading
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = [
    ("bare", "The value is 1.5."),
    ("unit", "The value is 1.5 gigabytes."),
    ("mid", "Free memory is 1.2 now."),
]


def main() -> None:
    from voice_stack.tts_engine import TTSEngine

    tts = TTSEngine(language="english", voice="anna", quantize=True)
    out_dir = ROOT / "tests"
    out_dir.mkdir(exist_ok=True)
    stop = threading.Event()
    for name, text in CASES:
        audio = np.concatenate(list(tts.stream_sentence(text, stop)))
        out = out_dir / f"point_test_{name}.wav"
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(tts.sample_rate)
            w.writeframes(pcm.tobytes())
        seconds = len(pcm) / tts.sample_rate
        print(f"{name}: {seconds:.2f}s  {out}   (text: {text!r})", flush=True)


if __name__ == "__main__":
    main()
