"""GPU Q8 VoiceDesign via local qwentts.cpp, complete sentences only.

The server stays resident across bridge restarts. Requests explicitly select
WAV, so synthesis finishes before any audio is handed to the speakers.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import wave

import numpy as np
import requests

from .qwen3_design_engine import DEFAULT_INSTRUCT, MOOD_INSTRUCTIONS, Qwen3VoiceDesign

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "tools/qwentts.cpp/build-win/Release/tts-server.exe"
MODEL_DIR = Path.home() / "models/Qwen3-TTS-VoiceDesign-GGUF"
MODEL = MODEL_DIR / "qwen-talker-1.7b-voicedesign-Q8_0.gguf"
CODEC = MODEL_DIR / "qwen-tokenizer-12hz-Q8_0.gguf"
URL = "http://127.0.0.1:8095"
ALIAS = "ttstt-qwen3-voicedesign-q8"


def _session():
    session = requests.Session()
    session.trust_env = False  # local speech must never go through a proxy
    return session


def _ready(session):
    try:
        response = session.get(URL + "/v1/models", timeout=2)
    except requests.ConnectionError:
        return False
    response.raise_for_status()
    ids = [item.get("id") for item in response.json().get("data", [])]
    if ids != [ALIAS]:
        raise RuntimeError(f"Port 8095 has a different speech model: {ids}")
    return True


def ensure_server():
    """Reuse this voice's server, or start it hidden on loopback with CUDA."""
    with _session() as session:
        if _ready(session):
            return
        for path in (SERVER, MODEL, CODEC):
            if not path.is_file():
                raise FileNotFoundError(f"Qwen GGUF setup is incomplete: {path}")
        logs = ROOT / "tests"
        env = dict(os.environ, GGML_BACKEND="CUDA0")
        args = [str(SERVER), "--model", str(MODEL), "--codec", str(CODEC),
                "--host", "127.0.0.1", "--port", "8095", "--alias", ALIAS,
                "--lang", "English", "--max-batch", "1"]
        with (logs / "qwen3_gguf_server.log").open("ab") as out:
            process = subprocess.Popen(
                args, cwd=str(SERVER.parent), env=env, stdin=subprocess.DEVNULL,
                stdout=out, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        (logs / "qwen3_gguf_server.json").write_text(
            json.dumps({"pid": process.pid, "model": str(MODEL), "url": URL}),
            encoding="utf-8")
        try:
            deadline = time.monotonic() + 75
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Qwen GGUF server exited; see tests/qwen3_gguf_server.log")
                if _ready(session):
                    # Pay the first CUDA kernel/graph setup cost before the
                    # bridge reports ready, never through audible playback.
                    warmup = session.post(URL + "/v1/audio/speech", json={
                        "model": ALIAS, "input": "Ready.", "language": "English",
                        "instructions": DEFAULT_INSTRUCT, "response_format": "wav",
                        "seed": 42, "max_new_tokens": 64,
                    }, timeout=(5, 60))
                    warmup.raise_for_status()
                    return
                time.sleep(0.25)
            raise TimeoutError("Qwen GGUF server did not become ready within 75 seconds")
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            raise


class Qwen3GGUF(Qwen3VoiceDesign):
    def __init__(self, instruct=DEFAULT_INSTRUCT, language="English", moods=True):
        if not instruct or not instruct.strip():
            raise ValueError("VoiceDesign needs a voice description")
        self._instruct = instruct.strip()
        self._language = language.capitalize()
        self.moods_enabled = moods
        self._mood = "neutral"
        self._lock = threading.Lock()
        ensure_server()

    def _generate(self, sentence):
        instruction = self._instruct
        if self.moods_enabled:
            instruction += " " + MOOD_INSTRUCTIONS[self._mood]
        with _session() as session:
            response = session.post(URL + "/v1/audio/speech", json={
                "model": ALIAS, "input": sentence, "language": self._language,
                "instructions": instruction, "response_format": "wav",
                "seed": 42, "max_new_tokens": 768,
            }, timeout=(5, 180))
            response.raise_for_status()
            with wave.open(io.BytesIO(response.content), "rb") as wav:
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 24000):
                    raise ValueError("Expected mono 16-bit 24 kHz WAV from Qwen server")
                frames = wav.readframes(wav.getnframes())
        if not frames:
            raise RuntimeError("Qwen server returned empty audio")
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        return [audio], self.sample_rate


def main():
    import argparse
    import soundfile as sf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="Hello Martin. That sounds rather more like it.")
    parser.add_argument("--instruct", default=DEFAULT_INSTRUCT)
    parser.add_argument("--mood", choices=sorted(MOOD_INSTRUCTIONS), default="contentment")
    parser.add_argument("--out", default="tests/qwen3_gguf_preview.wav")
    args = parser.parse_args()
    engine = Qwen3GGUF(instruct=args.instruct)
    engine.set_mood(args.mood)
    started = time.monotonic()
    audio = np.concatenate(list(engine.stream_sentence(args.text)))
    elapsed = time.monotonic() - started
    seconds = len(audio) / engine.sample_rate
    sf.write(args.out, audio, engine.sample_rate, subtype="PCM_16")
    result = {"text": args.text, "mood": args.mood, "audio_seconds": seconds,
              "synthesis_seconds": elapsed, "rtf": elapsed / seconds,
              "peak": float(np.abs(audio).max()), "path": args.out}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
