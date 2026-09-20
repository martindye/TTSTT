"""Generate a test utterance with Pocket TTS (also serves as TTS smoke test)."""
import sys
import time
import numpy as np
import soundfile as sf

t0 = time.time()
from pocket_tts import TTSModel

model = TTSModel.load_model(language="english", quantize=True)
print(f"model loaded in {time.time()-t0:.1f}s (sample_rate={model.sample_rate})")

t0 = time.time()
voice = model.get_state_for_audio_prompt("alba")
print(f"voice state ready in {time.time()-t0:.1f}s")

text = ("Hello! This is a test of the local voice stack. "
        "What is the capital of France?")
t0 = time.time()
audio = model.generate_audio(voice, text)
gen_time = time.time() - t0
print(f"generated {audio.shape} in {gen_time:.1f}s "
      f"({len(audio)/model.sample_rate:.1f}s audio, "
      f"RTF {len(audio)/model.sample_rate/gen_time:.1f}x)")
out = sys.argv[1] if len(sys.argv) > 1 else "tests/test_speech.wav"
sf.write(out, audio.numpy(), model.sample_rate)
print(f"wrote {out}")
