"""Generic local TTS server — one HTTP surface for every speech engine.

Text in, audio out, engine-agnostic. The DSH web app (and anything else on
this machine) speaks to this endpoint and never has to know which engine
produces the audio. New engines plug in under voice_stack/tts_engines/
without touching this file.

    python -X utf8 -m voice_stack.tts_server [--port 8188] [--bind 127.0.0.1]

Endpoints (loopback only by design):
    GET  /health   -> {"ok": true, "engines": {name: {loaded, voices, default_voice}}}
    POST /tts      <- {"text": str, "engine"?: str, "voice"?: str}
                      -> audio/wav (PCM16 mono) or a JSON error

The server binds 127.0.0.1 only: it is an unauthenticated synthesis
endpoint, so it must never be reachable off-machine.
"""
from __future__ import annotations

import argparse
import json
import logging
import struct
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s tts-server: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("tts-server")

MAX_TEXT_CHARS = 16_000


def _wav(samples, sample_rate: int) -> bytes:
    """PCM16 mono WAV bytes."""
    import numpy as np
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2").tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE" \
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate,
                                sample_rate * 2, 2, 16) \
        + b"data" + struct.pack("<I", len(pcm))
    return header + pcm


class Handler(BaseHTTPRequestHandler):
    server_version = "dsh-tts/1"

    # ------------------------------------------------------------- helpers
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def log_message(self, fmt, *args):  # keep the console quiet
        log.debug(fmt, *args)

    # ------------------------------------------------------------- routes
    def do_GET(self):  # noqa: N802 (http.server API)
        from . import tts_engines as engines
        if self.path in ("/health", "/engines", "/"):
            catalog = {}
            for eng in engines.engines():
                try:
                    loaded = eng.is_loaded()
                    voices = eng.voices() if loaded else []
                except Exception as e:
                    loaded, voices = False, [f"error: {e}"]
                catalog[eng.name] = {
                    "loaded": loaded,
                    "voices": voices,
                    "default_voice": _default_voice(eng),
                }
            return self._json(200, {"ok": True, "engines": catalog})
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/tts":
            return self._json(404, {"ok": False, "error": "not found"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            text = str(body.get("text") or "").strip()
            engine_name = body.get("engine")
            voice = body.get("voice")
        except Exception as e:
            return self._json(400, {"ok": False, "error": f"bad request: {e}"})
        if not text:
            return self._json(400, {"ok": False, "error": "empty text"})
        if len(text) > MAX_TEXT_CHARS:
            return self._json(400, {"ok": False,
                                    "error": f"text too long (>{MAX_TEXT_CHARS})"})

        from . import tts_engines as engines
        engine = engines.get(engine_name) if engine_name else None
        if engine is None:
            available = [e.name for e in engines.engines()]
            engine = engines.engines()[0] if engines.engines() else None
            if engine is None:
                return self._json(500, {"ok": False,
                                        "error": "no TTS engines available"})
        try:
            engine.load()
            samples, rate = engine.synthesize(text, voice)
        except Exception as e:
            log.exception("synthesis failed (%s)", engine.name)
            return self._json(500, {"ok": False,
                                    "error": f"synthesis failed: {e}"})
        try:
            wav = _wav(samples, rate)
        except Exception as e:
            log.exception("wav encoding failed")
            return self._json(500, {"ok": False, "error": f"wav: {e}"})
        log.info("%s: %d chars -> %d bytes wav (%s, voice=%s)",
                 engine.name, len(text), len(wav), rate, voice or "default")
        self._send(200, wav, "audio/wav")


def _default_voice(engine) -> str:
    getter = getattr(engine, "default_voice", None)
    return getter if isinstance(getter, str) else ""


def _preload() -> None:
    """Warm every engine so the first /tts is a render, not a model load.

    Runs in a background thread: /health is already serving, and a live
    request just waits on the engine's own load guard.
    """
    from . import tts_engines as engines
    for eng in engines.engines():
        try:
            eng.load()
            samples, _ = eng.synthesize("Ready.", None)
            log.info("preloaded %s (%d warm-up samples)", eng.name, samples.size)
        except Exception:
            log.exception("preload failed for engine %s", eng.name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--bind", default="127.0.0.1",
                    help="loopback only on purpose; do not expose this")
    args = ap.parse_args()

    # Importing the package registers all engines (lazy model loads).
    from . import tts_engines  # noqa: F401
    log.info("engines: %s", [e.name for e in tts_engines.engines()])

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    log.info("TTS server on http://%s:%d (loopback)", args.bind, args.port)
    threading.Thread(target=_preload, name="tts-preload", daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
