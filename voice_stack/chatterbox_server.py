"""Chatterbox-Turbo TTS server — the GPU voice (replaces the Qwen3 server).

Spends the exact API the kit's qwen3ggufbase engine expects (tts-server.exe
dialect), so the gate on 8095 and the kit stay untouched:

    GET  /v1/models        -> ids must be exactly ["ttstt-qwen3-base-q8"]
    GET  /v1/audio/voices  -> {"voices": [{"name": ...}, ...]}
    POST /v1/audio/voices  -> {"name", "wav_b64", "ref_text"?} register/replace
    POST /v1/audio/speech  -> OpenAI-ish body; returns mono 16-bit 24 kHz WAV
    GET  /health           -> {"ok": true, ...}

Listens on 127.0.0.1:8097; the TTS gate fronts it on 8095 and holds
/v1/audio/speech until the LLM is idle. Run with the chatterbox venv:

    C:\\Users\\press\\.dsh\\voice-chat\\venv-chatterbox\\Scripts\\python.exe
        voice_stack\\chatterbox_server.py --port 8097
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

ALIAS = "ttstt-qwen3-base-q8"  # shared alias the kit's _ready() expects
REF_DIR = Path(r"C:\Users\press\OneDrive\Projects\DSH_TESTS\dsh-voice-chat\ref_voices")
REF_CACHE_DIR = Path.home() / ".dsh" / "voice-chat" / "chatterbox-refs"
MAX_REF_SECONDS = 12.0  # turbo likes ~10 s prompts

log = logging.getLogger("chatterbox-server")


class Server:
    def __init__(self) -> None:
        self.lock = threading.Lock()          # one generation at a time
        self.voices: dict[str, Path] = {}
        self.peak_alloc_mib = 0
        self.ready = False
        self.load_seconds = None

    # ------------------------------------------------------------ reference
    def _save_ref(self, name: str, wav_bytes: bytes) -> None:
        """Store a registered reference (any WAV) as a <=12 s mono clip."""
        import torchaudio as ta

        wav, sr = ta.load(io.BytesIO(wav_bytes))
        if wav.shape[0] > 1:
            wav = wav[0:1]
        limit = int(MAX_REF_SECONDS * sr)
        if wav.shape[-1] > limit:
            wav = wav[..., :limit]
        REF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = REF_CACHE_DIR / f"{name}.wav"
        ta.save(str(path), wav, sr)
        self.voices[name] = path
        log.info("reference %r stored (%.1f s, %d Hz)", name,
                 wav.shape[-1] / sr, sr)

    def bootstrap_voices(self) -> None:
        """Pre-register the kit's stock voices so the kit skips registration."""
        for name, path in (("pocket", REF_DIR / "pocket.wav"),
                           ("warm-brit", REF_DIR / "warm_brit.wav")):
            if path.is_file():
                self._save_ref(name, path.read_bytes())
            else:
                log.warning("reference %s missing: %s", name, path)

    # ------------------------------------------------------------- synthesis
    def synthesize(self, text: str, voice: str) -> bytes:
        ref = self.voices.get(voice) or self.voices.get("pocket")
        if ref is None:
            raise RuntimeError("no reference voices registered")
        with self.lock:
            t0 = time.time()
            wav = self.model.generate(text, audio_prompt_path=str(ref))
            elapsed = time.time() - t0
        self.peak_alloc_mib = max(
            self.peak_alloc_mib,
            torch.cuda.max_memory_allocated() / 2**20)
        torch.cuda.reset_peak_memory_stats()
        if wav.dim() > 1:
            wav = wav.reshape(-1)  # turbo returns (1, N)
        sr = int(self.model.sr)
        pcm = (wav.clamp(-1.0, 1.0).cpu().numpy() * 32767.0).astype("<i2")
        out = io.BytesIO()
        with wave.open(out, "wb") as dst:
            dst.setnchannels(1)
            dst.setsampwidth(2)
            dst.setframerate(sr)
            dst.writeframes(pcm.tobytes())
        log.info("synth %.3f s in %.1f s (voice=%s, %d chars, peak %d MiB)",
                 pcm.shape[0] / sr, elapsed, voice or "pocket", len(text),
            int(self.peak_alloc_mib))
        return out.getvalue()

    # ------------------------------------------------------------- http
def build_handler(model: Server):
    class Handler(BaseHTTPRequestHandler):
        server_version = "chatterbox-tts/1"

        def log_message(self, fmt, *args):  # keep the log file readable
            log.info("%s %s", self.address_string(), fmt % args)

        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _bytes(self, code: int, data: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                return json.loads(raw or b"{}")
            except Exception:
                return {}

        # ------------------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    {"id": ALIAS, "object": "model", "owned_by": "ttstt"}]})
            elif self.path == "/v1/audio/voices":
                self._json(200, {"voices": [{"name": n} for n in model.voices]})
            elif self.path == "/health":
                free_mib = torch.cuda.mem_get_info()[0] / 2**20 if torch.cuda.is_available() else -1
                self._json(200, {"ok": model.ready, "engine": "chatterbox-turbo",
                                 "voices": sorted(model.voices),
                                 "peak_alloc_mib": int(model.peak_alloc_mib),
                                 "vram_free_mib": int(free_mib)})
            else:
                self._json(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                if self.path == "/v1/audio/voices":
                    body = self._body()
                    name = str(body.get("name") or "").strip()
                    b64 = str(body.get("wav_b64") or "")
                    if not name or not b64:
                        return self._json(400, {"ok": False,
                                                "error": "name + wav_b64 required"})
                    try:
                        model._save_ref(name, base64.b64decode(b64))
                    except Exception as e:
                        return self._json(400, {"ok": False, "error": str(e)})
                    return self._json(200, {"ok": True, "name": name})
                if self.path == "/v1/audio/speech":
                    body = self._body()
                    text = str(body.get("input") or "").strip()
                    if not text:
                        return self._json(400, {"ok": False,
                                                "error": "empty text"})
                    voice = str(body.get("voice") or "pocket")
                    t0 = time.time()
                    try:
                        data = model.synthesize(text, voice)
                    except Exception as e:
                        log.exception("synthesis failed")
                        return self._json(500, {"ok": False, "error": str(e)})
                    log.info("speech %d chars -> %d bytes in %.1f s",
                             len(text), len(data), time.time() - t0)
                    return self._bytes(200, data, "audio/wav")
                return self._json(404, {"ok": False, "error": "not found"})
            except Exception as e:
                log.exception("request failed")
                try:
                    self._json(500, {"ok": False, "error": str(e)})
                except Exception:
                    pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s chatterbox: %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    server = Server()
    log.info("loading Chatterbox-Turbo (this takes a minute) ...")
    t0 = time.time()
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    server.model = ChatterboxTurboTTS.from_pretrained(device="cuda")
    server.load_seconds = time.time() - t0
    server.bootstrap_voices()
    server.ready = True
    log.info("ready in %.0f s (%d reference voices); serving on %s:%d",
             server.load_seconds, len(server.voices), args.host, args.port)

    httpd = ThreadingHTTPServer((args.host, args.port), build_handler(server))
    httpd.serve_forever()


if __name__ == "__main__":
    main()
