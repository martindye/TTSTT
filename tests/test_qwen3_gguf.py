"""HTTP/WAV contract tests for the complete-sentence GGUF adapter."""
import io
import json
import threading
import unittest
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import numpy as np

from voice_stack import qwen3_gguf_engine as gguf


def make_wav(rate=24000):
    data = io.BytesIO()
    with wave.open(data, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.full(4800, 8192, dtype="<i2").tobytes())
    return data.getvalue()


class GGUFTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.audio = make_wav()
        self.status = 200
        self.model_id = gguf.ALIAS
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": owner.model_id}]}).encode())

            def do_POST(self):
                size = int(self.headers["Content-Length"])
                owner.requests.append(json.loads(self.rfile.read(size)))
                self.send_response(owner.status)
                self.send_header("Content-Type", "audio/wav")
                self.end_headers()
                self.wfile.write(owner.audio)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        patcher = patch.object(gguf, "URL", f"http://127.0.0.1:{self.server.server_port}")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_full_sentence_wav_and_expression(self):
        engine = gguf.Qwen3GGUF(instruct="A British female voice.")
        engine.set_mood("amusement")
        audio = np.concatenate(list(engine.stream_sentence("That worked, rather nicely!")))
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0]["input"], "That worked, rather nicely!")
        self.assertEqual(self.requests[0]["response_format"], "wav")
        self.assertIn("amused", self.requests[0]["instructions"])
        self.assertEqual(self.requests[0]["seed"], 42)
        self.assertEqual(audio.dtype, np.float32)
        np.testing.assert_allclose(audio, 0.25)

    def test_moods_off_uses_only_base_description(self):
        engine = gguf.Qwen3GGUF(instruct="Calm voice.", moods=False)
        self.assertFalse(engine.set_mood("amusement"))
        list(engine.stream_sentence("Hello."))
        self.assertEqual(self.requests[0]["instructions"], "Calm voice.")

    def test_different_server_model_is_not_reused(self):
        self.model_id = "wrong-model"
        with self.assertRaisesRegex(RuntimeError, "different speech model"):
            gguf.Qwen3GGUF()

    def test_server_error_is_surfaced(self):
        engine = gguf.Qwen3GGUF()
        self.status = 500
        with self.assertRaises(gguf.requests.HTTPError):
            list(engine.stream_sentence("Hello."))

    def test_wrong_audio_format_is_rejected(self):
        engine = gguf.Qwen3GGUF()
        self.audio = make_wav(rate=16000)
        with self.assertRaisesRegex(ValueError, "24 kHz"):
            list(engine.stream_sentence("Hello."))

    def test_cancelled_sentence_never_posts(self):
        engine = gguf.Qwen3GGUF()
        cancel = threading.Event()
        cancel.set()
        self.assertEqual(list(engine.stream_sentence("Hello.", cancel)), [])
        self.assertEqual(self.requests, [])

    def test_coding_bridge_keeps_turn_end_buffering(self):
        from types import SimpleNamespace
        from voice_stack.coding_voice import _make_bridge_tts
        args = SimpleNamespace(tts_engine="qwen3gguf", tts_quantize="int4",
                               tts_moods=None, qwen3tts_instruct="British voice.",
                               tts_language="English")
        engine, buffered = _make_bridge_tts(args)
        self.assertIsInstance(engine, gguf.Qwen3GGUF)
        self.assertTrue(buffered)


if __name__ == "__main__":
    unittest.main()
