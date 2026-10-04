"""TTS gate: hold GPU speech synthesis until the 27B LLM is idle.

Sits on 127.0.0.1:8095 in front of the real Qwen3 TTS server (relocated to
127.0.0.1:8097). The dsh-voice-chat kit talks to 8095 exactly as before and
never knows the difference.

Rule (Martin, 03/10): speech synthesis must not start while the LLM is
thinking — both share the one 5090, and synthesis under a decode is what
drops the decode to single digits of tokens per second. Playback of
already-generated audio is unaffected (it never touches the GPU).

    - POST /v1/audio/speech  (and voice registration) waits for LLM idle
    - everything else passes straight through
    - if the LLM is unreachable the gate does not block (no contention then)
    - if the LLM stays busy past MAX_WAIT it synthesises anyway: a late
      voice beats no voice

    python -X utf8 tts_gate.py [--port 8095] [--upstream-port 8097]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "127.0.0.1"
LLM_SLOTS = "http://127.0.0.1:8080/slots"
POLL_SECONDS = 0.5
MAX_WAIT_SECONDS = 20 * 60
GATED_PATHS = ("/v1/audio/speech", "/v1/audio/voices")
LOG_LOCK = threading.Lock()
_LOG_PATH = None


def _log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} tts-gate: {msg}"
    with LOG_LOCK:
        print(line, flush=True)
        if _LOG_PATH:
            try:
                with open(_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass


def llm_busy() -> bool | None:
    """True/False when the LLM server answers; None when unreachable."""
    try:
        with urllib.request.urlopen(LLM_SLOTS, timeout=2) as r:
            slots = json.loads(r.read().decode("utf-8"))
    except Exception:
        return None
    if isinstance(slots, dict):
        slots = [slots]
    return any(isinstance(s, dict) and s.get("is_processing") for s in slots)


def wait_for_llm_idle(who: str) -> None:
    state = llm_busy()
    if state in (False, None):
        return
    started = time.monotonic()
    _log(f"{who}: LLM busy — holding synthesis")
    next_note = started + 15
    while True:
        waited = time.monotonic() - started
        if waited >= MAX_WAIT_SECONDS:
            _log(f"{who}: LLM still busy after {MAX_WAIT_SECONDS // 60} min; synthesising anyway")
            return
        time.sleep(POLL_SECONDS)
        state = llm_busy()
        now = time.monotonic()
        if now >= next_note:
            _log(f"{who}: still waiting for LLM ({waited:.0f}s)")
            next_note = now + 15
        if state in (False, None):
            _log(f"{who}: LLM idle after {waited:.1f}s; synthesising")
            return


class Handler(BaseHTTPRequestHandler):
    server_version = "tts-gate/1"

    def _forward(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        path = self.path
        gated = method == "POST" and path.split("?", 1)[0] in GATED_PATHS
        if gated:
            wait_for_llm_idle(f"{self.client_address[0]}:{self.client_address[1]} {path}")
        req = urllib.request.Request(
            f"http://{UPSTREAM}:{UPSTREAM_PORT}{path}", data=body, method=method)
        for key, value in self.headers.items():
            if key.lower() in ("host", "content-length", "connection"):
                continue
            req.add_header(key, value)
        try:
            with urllib.request.urlopen(req, timeout=600) as up:
                payload = up.read()
                self.send_response(up.status)
                for key, value in up.headers.items():
                    if key.lower() in ("transfer-encoding", "connection", "content-length"):
                        continue
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except urllib.error.HTTPError as e:
            payload = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            err = json.dumps({"ok": False, "error": f"tts upstream unreachable: {e}"}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            _log(f"upstream failure for {path}: {e}")

    def do_GET(self):  # noqa: N802
        self._forward("GET")

    def do_POST(self):  # noqa: N802
        self._forward("POST")

    def log_message(self, fmt, *args):  # keep the console to our own lines
        _log(f"http: {fmt % args}")


def main() -> int:
    global UPSTREAM_PORT, _LOG_PATH
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--upstream-port", type=int, default=8097)
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()
    UPSTREAM_PORT = args.upstream_port
    if args.log_file:
        _LOG_PATH = args.log_file
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    _log(f"gate on 127.0.0.1:{args.port} -> upstream 127.0.0.1:{args.upstream_port} "
         f"(LLM busy-check: {LLM_SLOTS})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
