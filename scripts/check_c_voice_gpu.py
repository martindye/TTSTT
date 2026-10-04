"""Small non-streaming speech check; saves audio for listening, not a soak test."""
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import requests
import soundfile as sf

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8097
out = Path(__file__).resolve().parents[1] / "tests" / "c_voice_gpu"
out.mkdir(exist_ok=True)
records = []
for mood in ("neutral", "joy", "sad", "joy"):
    payload = {"input": "Hello, Martin. Shall we try that again?", "seed": 42,
               "language": "English", "response_format": "wav"}
    if mood != "neutral":
        payload["emotion"] = mood
    start = time.perf_counter()
    response = requests.post(f"http://127.0.0.1:{port}/v1/audio/speech",
                             json=payload, timeout=(5, 180))
    response.raise_for_status()
    elapsed = time.perf_counter() - start
    audio, rate = sf.read(io.BytesIO(response.content))
    assert rate == 24000 and len(audio) > rate and np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.001, "Silent output"
    name = f"{len(records)}_{mood}.wav"
    (out / name).write_bytes(response.content)
    record = {"file": name, "emotion": mood, "seed": 42,
              "generation_seconds": round(elapsed, 3),
              "audio_seconds": round(len(audio) / rate, 3),
              "sha256": hashlib.sha256(response.content).hexdigest()}
    records.append(record)
    print(json.dumps(record), flush=True)
(out / "speech_checks.json").write_text(json.dumps(records, indent=2))
assert records[0]["sha256"] != records[1]["sha256"], "Mood has no effect"
assert records[1]["sha256"] == records[3]["sha256"], "Repeated mood changed with fixed seed"
print("Audio validity, mood effect and repeatability checks passed.")
