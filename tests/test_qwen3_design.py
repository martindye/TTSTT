"""Offline adapter and bridge-contract tests; no weights, CUDA or speakers."""
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from voice_stack.qwen3_design_engine import (
    DEFAULT_INSTRUCT, MOOD_INSTRUCTIONS, Qwen3VoiceDesign,
)
from voice_stack.qwen3_tts_engine import Qwen3TTS


class DesignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        (self.path / "config.json").write_text(
            json.dumps({"tts_model_type": "voice_design"}), encoding="utf-8")
        self.model = Mock()
        self.model.generate_voice_design.return_value = ([np.ones(4800)], 24000)
        self.loader = Mock(return_value=self.model)
        self.torch = SimpleNamespace(
            bfloat16="bf16", float32="fp32", device=lambda d: d,
            cuda=SimpleNamespace(is_available=lambda: True,
                                 mem_get_info=lambda d: (8 * 1024**3, 32 * 1024**3)))

    def engine(self, **kwargs):
        with patch.dict("sys.modules", {
            "torch": self.torch,
            "qwen_tts": SimpleNamespace(Qwen3TTSModel=SimpleNamespace(
                from_pretrained=self.loader)),
        }):
            return Qwen3VoiceDesign(model_dir=self.path, **kwargs)

    def test_design_uses_description_and_mood_without_cloning(self):
        engine = self.engine()
        self.assertTrue(engine.set_mood("amusement"))
        chunks = list(engine.stream_sentence("That worked."))
        self.assertEqual(sum(len(c) for c in chunks), 4800)
        self.assertTrue(all(c.dtype == np.float32 and c.ndim == 1 for c in chunks))
        call = self.model.generate_voice_design.call_args.kwargs
        self.assertEqual(call["text"], "That worked.")
        self.assertEqual(call["language"], "English")
        self.assertEqual(call["instruct"], DEFAULT_INSTRUCT + " " + MOOD_INSTRUCTIONS["amusement"])
        self.model.create_voice_clone_prompt.assert_not_called()
        self.assertTrue(self.loader.call_args.kwargs["local_files_only"])
        self.assertEqual(self.loader.call_args.kwargs["attn_implementation"], "sdpa")

    def test_moods_off_preserves_description(self):
        engine = self.engine(moods=False, instruct="A calm voice.")
        self.assertFalse(engine.set_mood("amusement"))
        list(engine.stream_sentence("Hello."))
        self.assertEqual(self.model.generate_voice_design.call_args.kwargs["instruct"], "A calm voice.")

    def test_invalid_mood_does_not_replace_previous(self):
        engine = self.engine()
        engine.set_mood("relief")
        self.assertFalse(engine.set_mood("invalid"))
        self.assertEqual(engine._mood, "relief")

    def test_cancel_before_start_does_no_generation(self):
        engine = self.engine()
        cancel = threading.Event()
        cancel.set()
        self.assertEqual(list(engine.stream_sentence("Hello.", cancel)), [])
        self.model.generate_voice_design.assert_not_called()

    def test_cancel_during_generation_discards_audio(self):
        engine = self.engine()
        cancel = threading.Event()
        def generate(**kwargs):
            cancel.set()
            return [np.ones(4800)], 24000
        self.model.generate_voice_design.side_effect = generate
        self.assertEqual(list(engine.stream_sentence("Hello.", cancel)), [])

    def test_generation_failure_reaches_caller_and_releases_lock(self):
        engine = self.engine()
        self.model.generate_voice_design.side_effect = RuntimeError("test failure")
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            list(engine.stream_sentence("Hello."))
        self.assertTrue(engine._lock.acquire(blocking=False))
        engine._lock.release()

    def test_wrong_sample_rate_fails_instead_of_playing_at_wrong_speed(self):
        engine = self.engine()
        self.model.generate_voice_design.return_value = ([np.ones(4800)], 16000)
        with self.assertRaisesRegex(ValueError, "sample rate"):
            list(engine.stream_sentence("Hello."))

    def test_low_gpu_memory_does_not_attempt_load(self):
        self.torch.cuda.mem_get_info = lambda d: (1024**3, 32 * 1024**3)
        with self.assertRaisesRegex(RuntimeError, "only 1.0 GiB"):
            self.engine()
        self.loader.assert_not_called()

    def test_cpu_preview_loads_without_cuda(self):
        self.torch.cuda.is_available = lambda: False
        self.engine(device="cpu")
        self.assertEqual(self.loader.call_args.kwargs["dtype"], "fp32")

    def test_base_checkpoint_is_rejected(self):
        (self.path / "config.json").write_text('{"tts_model_type":"base"}')
        with self.assertRaisesRegex(ValueError, "not a Qwen VoiceDesign"):
            self.engine()
        self.loader.assert_not_called()

    def test_missing_checkpoint_is_actionable(self):
        with self.assertRaisesRegex(FileNotFoundError, "download_qwen3_design"):
            Qwen3VoiceDesign(model_dir=self.path / "missing")

    def test_existing_clone_path_still_uses_clone_prompt(self):
        engine = Qwen3TTS.__new__(Qwen3TTS)
        engine._lock = threading.Lock()
        engine._language = "English"
        engine._prompt = object()
        engine._model = Mock()
        engine._model.generate_voice_clone.return_value = ([np.ones(4800)], 24000)
        self.assertEqual(len(np.concatenate(list(engine.stream_sentence("Hello.")))), 4800)
        self.assertIs(engine._model.generate_voice_clone.call_args.kwargs["voice_clone_prompt"], engine._prompt)


class BridgeTests(unittest.TestCase):
    def test_coding_bridge_routes_design_and_moods(self):
        from voice_stack.coding_voice import _make_bridge_tts
        args = SimpleNamespace(tts_quantize="int4", tts_engine="qwen3design",
                               tts_moods="off", qwen3tts_model="local-model",
                               qwen3tts_instruct="Warm voice.", tts_language="english",
                               qwen3tts_device="cpu")
        with patch("voice_stack.qwen3_design_engine.Qwen3VoiceDesign") as factory:
            engine, pro = _make_bridge_tts(args)
            self.assertIs(engine, factory.return_value)
            self.assertTrue(pro)
            self.assertFalse(factory.call_args.kwargs["moods"])
            self.assertEqual(factory.call_args.kwargs["instruct"], "Warm voice.")

    def test_dsh_cli_routes_design_options(self):
        from voice_stack.voice_dsh import make_tts_from_args, parse_args
        args = parse_args(["--tts-engine", "qwen3design", "--qwen3tts-instruct",
                           "Warm voice.", "--qwen3tts-device", "cpu"])
        with patch("voice_stack.qwen3_design_engine.Qwen3VoiceDesign") as factory:
            self.assertIs(make_tts_from_args(args), factory.return_value)
            self.assertEqual(factory.call_args.kwargs["instruct"], "Warm voice.")
            self.assertEqual(factory.call_args.kwargs["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
