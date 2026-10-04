"""Qwen3 1.7B Base Q8 plus a registered reference voice, via qwentts.cpp.

The VoiceDesign model re-derives the speaker from the text description
on every request, so the identity drifts from sentence to sentence.
The Base checkpoint instead clones from a *registered reference voice*:
the server keeps the speaker's latents in memory and every sentence is
synthesised from the same anchor.

References:
  "warm-brit" -- an 18 s take of the warm British female voice recorded
      from the old VoiceDesign server (ref_voices/warm_brit.wav); the
      voice Martin has been hearing, frozen into a fixed speaker.
  "pocket" -- the standard small coding voice (pocket TTS, "anna"),
      captured from the local engine (ref_voices/pocket.wav).
  "p277" -- the old plain pro voice (VCTK p277_023), registered
      x-vector-only when the clip is found in the HF cache.

Mood tags are a silent no-op (the Base prompt builder takes no
instructions), as with the 0.6B clone.
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "tools/qwentts.cpp/build-win/Release/tts-server.exe"
MODEL_DIR = Path.home() / "models/Qwen3-TTS-VoiceDesign-GGUF"
MODEL = MODEL_DIR / "qwen-talker-1.7b-base-Q8_0.gguf"
CODEC = MODEL_DIR / "qwen-tokenizer-12hz-Q8_0.gguf"
URL = "http://127.0.0.1:8095"
ALIAS = "ttstt-qwen3-base-q8"
VOICE = "warm-brit"
REF_DIR = Path(__file__).parent / "ref_voices"
STATE = ROOT / "tests" / "qwen3_gguf_base_server.json"


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


# ------------------------------------------------------------- references
def _as_24k_wav(raw: bytes) -> bytes:
    """Re-encode any WAV as mono 16-bit 24 kHz (the extractor's input)."""
    with wave.open(io.BytesIO(raw), "rb") as src:
        channels = src.getnchannels()
        width = src.getsampwidth()
        rate = src.getframerate()
        frames = src.readframes(src.getnframes())
    if width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV bit depth: {width * 8}")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != 24000:
        n_out = max(1, int(len(samples) * 24000 / rate))
        pos = np.clip((np.arange(n_out) + 0.5) * (len(samples) / n_out) - 0.5,
                      0, len(samples) - 1.0001)
        i0 = pos.astype(int)
        frac = pos - i0
        samples = samples[i0] * (1 - frac) + samples[i0 + 1] * frac
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(24000)
        dst.writeframes(pcm16.tobytes())
    return out.getvalue()


def _p277_ref() -> bytes | None:
    """The old plain pro clip, if the HF cache still holds it."""
    hub = Path.home() / ".cache/huggingface/hub/models--kyutai--tts-voices"
    hits = sorted(hub.glob("snapshots/*/vctk/p277_023.wav")) if hub.is_dir() else []
    return hits[0].read_bytes() if hits else None


def _register(session, name, wav_bytes, ref_text):
    payload = {"name": name, "wav_b64": base64.b64encode(wav_bytes).decode("ascii")}
    if ref_text:
        payload["ref_text"] = ref_text  # enables ICL clone mode
    response = session.post(URL + "/v1/audio/voices", json=payload, timeout=180)
    response.raise_for_status()
    print(f"[qwen3ggufbase] voice '{name}' registered", flush=True)


def _register_references(session):
    """Register the reference voices; idempotent (reuses live entries)."""
    listing = session.get(URL + "/v1/audio/voices", timeout=10)
    listing.raise_for_status()
    present = {v.get("name") for v in listing.json().get("voices", [])}
    if VOICE not in present:
        wav = (REF_DIR / "warm_brit.wav").read_bytes()
        text = (REF_DIR / "warm_brit.txt").read_text(encoding="utf-8").strip()
        _register(session, VOICE, _as_24k_wav(wav), text)
    pocket_wav = REF_DIR / "pocket.wav"
    if pocket_wav.is_file() and "pocket" not in present:
        text = (REF_DIR / "pocket.txt").read_text(encoding="utf-8").strip()
        _register(session, "pocket", _as_24k_wav(pocket_wav.read_bytes()), text)
    p277 = _p277_ref()
    if p277 is not None and "p277" not in present:
        _register(session, "p277", _as_24k_wav(p277), None)  # x-vector only


# ------------------------------------------------------------------ server
def ensure_server():
    """Reuse the base-clone server, or start it hidden on loopback."""
    with _session() as session:
        if _ready(session):
            return
        for path in (SERVER, MODEL, CODEC):
            if not path.is_file():
                raise FileNotFoundError(f"Qwen GGUF base setup is incomplete: {path}")
        logs = ROOT / "tests"
        env = dict(os.environ, GGML_BACKEND="CUDA0")
        args = [str(SERVER), "--model", str(MODEL), "--codec", str(CODEC),
                "--host", "127.0.0.1", "--port", "8095", "--alias", ALIAS,
                "--lang", "English", "--max-batch", "1"]
        with (logs / "qwen3_gguf_base_server.log").open("ab") as out:
            process = subprocess.Popen(
                args, cwd=str(SERVER.parent), env=env, stdin=subprocess.DEVNULL,
                stdout=out, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        STATE.write_text(json.dumps({"pid": process.pid, "model": str(MODEL),
                                     "url": URL, "alias": ALIAS}), encoding="utf-8")
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Qwen GGUF base server exited; "
                                       "see tests/qwen3_gguf_base_server.log")
                if _ready(session):
                    # Registration blocks on the compute worker until the
                    # speaker encoder has run, so it doubles as warmup.
                    _register_references(session)
                    return
                time.sleep(0.25)
            raise TimeoutError("Qwen GGUF base server did not become ready "
                               "within 120 seconds")
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            raise


class Qwen3GGUFBase:
    """1.7B Base talker cloning from a registered reference voice.

    No mood support: the Base prompt builder takes no instructions, so
    mood tags are a silent no-op (as with the 0.6B clone).
    """

    sample_rate = 24000

    def __init__(self, voice: str = VOICE):
        self.voice = voice
        ensure_server()

    def stream_sentence(self, sentence, stop_event=None):
        if stop_event is not None and stop_event.is_set():
            return
        with _session() as session:
            response = session.post(URL + "/v1/audio/speech", json={
                "model": ALIAS, "input": sentence, "language": "English",
                "voice": self.voice, "response_format": "wav",
                "seed": 42, "max_new_tokens": 768,
            }, timeout=(5, 180))
            response.raise_for_status()
            with wave.open(io.BytesIO(response.content), "rb") as wav:
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 24000):
                    raise ValueError("Expected mono 16-bit 24 kHz WAV from Qwen server")
                frames = wav.readframes(wav.getnframes())
        if stop_event is not None and stop_event.is_set():
            return
        if not frames:
            raise RuntimeError("Qwen server returned empty audio")
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        yield audio


def main():
    import argparse
    import soundfile as sf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="Hello. This is the new reference voice.")
    parser.add_argument("--voice", default=VOICE)
    parser.add_argument("--out", default="tests/base_engine_preview.wav")
    args = parser.parse_args()
    engine = Qwen3GGUFBase(voice=args.voice)
    started = time.monotonic()
    audio = np.concatenate(list(engine.stream_sentence(args.text)))
    elapsed = time.monotonic() - started
    seconds = len(audio) / engine.sample_rate
    sf.write(args.out, audio, engine.sample_rate, subtype="PCM_16")
    print(json.dumps({"text": args.text, "voice": args.voice,
                      "audio_seconds": seconds, "synthesis_seconds": elapsed,
                      "rtf": elapsed / seconds, "path": args.out}), flush=True)


if __name__ == "__main__":
    main()
