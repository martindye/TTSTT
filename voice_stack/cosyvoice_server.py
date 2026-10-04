"""CosyVoice3 0.5B TTS server - speaks the tts-server dialect on 8097.

The 04/10 GPU-voice trial (third occupant of the slot after the Qwen3
tts-server.exe and Chatterbox-Turbo). Zero-shot clone of the kit's reference
voices, with natural-language style instructions (instruct2) and fine-grained
in-text control tokens ([breath], [laughter], [sigh], <strong>, CMU phonemes,
pinyin). The /v1 surface is the same dialect the 8095 gate and the kit's
qwen3ggufbase engine expect, under the shared alias.

Extra request field on /v1/audio/speech:
    "style": optional natural-language direction, e.g. "Say this in a very
             happy tone." - applied via inference_instruct2.

Run:  venv-cosyvoice\\python.exe cosyvoice_server.py --port 8097
"""
import argparse
import base64
import io
import json
import os
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

REPO = r"C:\Users\press\.dsh\voice-chat\cosyvoice"
MODEL_DIR = r"C:\Users\press\.dsh\voice-chat\models\cosyvoice3-0.5b"
VOICE_DIR = r"C:\Users\press\.dsh\voice-chat\voices-cosy"
REF_DIR = r"C:\Users\press\OneDrive\Projects\DSH_TESTS\dsh-voice-chat\ref_voices"
ALIAS = "ttstt-qwen3-base-q8"  # shared alias: gate + kit adopt via /v1/models
PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"
BOOTSTRAP_VOICES = (("pocket", "pocket.wav", "pocket.txt"),
                    ("warm-brit", "warm_brit.wav", "warm_brit.txt"))

sys.path.insert(0, REPO)
sys.path.append(os.path.join(REPO, "third_party", "Matcha-TTS"))
sys.path.append(os.path.join(REPO, "stubs"))  # pyworld import stub (no C ext here)


def wav_bytes(pcm_i16: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm_i16.tobytes())
    return buf.getvalue()


class CosyVoiceServer:
    def __init__(self):
        from cosyvoice.cli.cosyvoice import AutoModel
        self.model = AutoModel(model_dir=MODEL_DIR)
        self.rate = int(self.model.sample_rate)
        self.voices = {}  # name -> {"wav": path, "text": ref text}
        self.lock = threading.Lock()
        os.makedirs(VOICE_DIR, exist_ok=True)
        for name, wav_name, txt_name in BOOTSTRAP_VOICES:
            wav = os.path.join(REF_DIR, wav_name)
            if not os.path.exists(wav):
                print(f"[Server] bootstrap ref missing: {wav}", flush=True)
                continue
            ref_text = ""
            txt = os.path.join(REF_DIR, txt_name)
            if os.path.exists(txt):
                with open(txt, encoding="utf-8") as f:
                    ref_text = f.read().strip()
            self.register(name, wav, ref_text)
        print(f"[Server] ready: voices={sorted(self.voices)}", flush=True)

    def register(self, name: str, wav_path: str, ref_text: str) -> None:
        with self.lock:
            # One-time speech-token + speaker-embedding extraction; cached in
            # frontend.spk2info and reused by every zero-shot synthesis.
            self.model.add_zero_shot_spk(PROMPT_PREFIX + ref_text, wav_path, name)
            self.voices[name] = {"wav": wav_path, "text": ref_text}
        print(f"[Server] voice '{name}' registered (T ref={len(ref_text)})", flush=True)

    def synthesize(self, text: str, voice: str, style: str = None) -> bytes:
        v = self.voices.get(voice)
        if v is None:
            raise KeyError(voice)
        chunks = []
        with self.lock:
            if style:
                instruct = "You are a helpful assistant. " + style + "<|endofprompt|>"
                gen = self.model.inference_instruct2(text, instruct, v["wav"], stream=False)
            else:
                gen = self.model.inference_zero_shot(text, "", "", zero_shot_spk_id=voice, stream=False)
            for out in gen:
                chunks.append(out["tts_speech"])
        audio = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
        pcm = torch.clamp(audio, -1.0, 1.0).mul(32767).to(torch.int16).reshape(-1)
        pcm = pcm.numpy()
        return wav_bytes(pcm, self.rate)


def make_handler(server_core: CosyVoiceServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            if self.path in ("/", "/playground"):
                path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "playground.html")
                with open(path, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": ALIAS, "object": "model"}]})
            elif self.path == "/v1/audio/voices":
                self._json(200, {"voices": [{"name": n} for n in server_core.voices]})
            elif self.path == "/health":
                free_mib, total_mib = 0, 0
                try:
                    free, total = torch.cuda.mem_get_info()
                    free_mib, total_mib = free // 2**20, total // 2**20
                except Exception:
                    pass
                self._json(200, {"ok": True, "engine": "cosyvoice3-0.5b",
                                 "voices": sorted(server_core.voices),
                                 "vram_free_mib": free_mib, "vram_total_mib": total_mib})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            try:
                body = self._read_json()
                if self.path == "/v1/audio/voices":
                    name = str(body.get("name", "")).strip()
                    raw = base64.b64decode(body.get("wav_b64", ""))
                    if not name or not raw:
                        return self._json(400, {"error": "name and wav_b64 required"})
                    os.makedirs(VOICE_DIR, exist_ok=True)
                    path = os.path.join(VOICE_DIR, name + ".wav")
                    with open(path, "wb") as f:
                        f.write(raw)
                    ref_text = str(body.get("ref_text", "")).strip()
                    server_core.register(name, path, ref_text)
                    return self._json(200, {"name": name, "status": "registered"})
                if self.path == "/v1/audio/speech":
                    text = str(body.get("input", "")).strip()
                    voice = str(body.get("voice", "pocket"))
                    style = (str(body.get("style", "")).strip() or None)
                    if not text:
                        return self._json(400, {"error": "empty input"})
                    wav = server_core.synthesize(text, voice, style)
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", str(len(wav)))
                    self.end_headers()
                    self.wfile.write(wav)
                    return
                self._json(404, {"error": "not found"})
            except KeyError as e:
                self._json(404, {"error": f"unknown voice: {e.args[0]}"})
            except Exception as e:  # noqa: BLE001 - report any synthesis failure
                import traceback
                traceback.print_exc()
                print(f"[Server] error: {e!r}", flush=True)
                self._json(500, {"error": str(e)})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8097)
    args = ap.parse_args()
    print("[Qwen] cosyvoice_server (CosyVoice3 0.5B)", flush=True)
    core = CosyVoiceServer()
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(core))
    httpd.daemon_threads = True
    print(f"[Server] model {ALIAS} listening on 127.0.0.1:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
