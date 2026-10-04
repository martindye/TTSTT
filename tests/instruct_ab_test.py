"""A/B check: does the qwentts.cpp VoiceDesign server actually use
`instructions` in the /v1/audio/speech request?

Same text and seed, two contrasting instructions. If the server honours
the field the two WAVs differ; if it ignores it they are byte-identical.
The repeat of instruction A is a seed-stability sanity check.
"""
import io

import requests
import soundfile as sf

URL = "http://127.0.0.1:8095/v1/audio/speech"
TEXT = "The package arrived much earlier than we had expected."


def synth(instruct: str):
    r = requests.post(URL, json={
        "model": "ttstt-qwen3-voicedesign-q8",
        "input": TEXT,
        "language": "English",
        "instructions": instruct,
        "response_format": "wav",
        "seed": 42,
        "max_new_tokens": 768,
    }, timeout=(5, 180))
    r.raise_for_status()
    audio, _ = sf.read(io.BytesIO(r.content), dtype="float32")
    return audio, r.content


A = "Speak in a very slow, deep, dramatic tone, like a movie trailer narrator."
B = "Speak very quickly and cheerfully, like an excited child."

a, raw_a = synth(A)
b, raw_b = synth(B)
a2, raw_a2 = synth(A)

print("bytes A:", len(raw_a), " B:", len(raw_b), " A-repeat:", len(raw_a2))
print("A == A-repeat (seed stable):", raw_a == raw_a2)
print("A == B (instructions ignored?):", raw_a == raw_b)
print("A duration:", round(len(a) / 24000, 2), "s   B duration:",
      round(len(b) / 24000, 2), "s")
