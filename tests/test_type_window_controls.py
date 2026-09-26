"""Regression checks for mood identity and cancellation, with no audio output."""
import threading
import unittest
from unittest.mock import Mock, patch

from scripts import qwen3_type_window as voice


class ControlsTest(unittest.TestCase):
    def test_design_retains_british_identity(self):
        for mood in ("fear", "sadness", "Whisper gently"):
            self.assertTrue(voice.build_instruction(mood).startswith(voice.DEFAULT_INSTRUCT))

    def test_any_mood_word_passes_to_the_engine(self):
        self.assertEqual(voice.c_emotion("angry"), "anger")
        self.assertIsNone(voice.c_emotion("neutral"))
        self.assertIsNone(voice.c_emotion("   "))
        for mood in ("curiosity", "nostalgia", "comfort"):
            self.assertEqual(voice.c_emotion(mood), mood)

    def test_stop_keeps_worker_busy_until_it_returns(self):
        window = voice.VoiceWindow.__new__(voice.VoiceWindow)
        window.busy = True
        window.cancel = threading.Event()
        window._status = Mock()
        with patch.object(voice.sd, "stop"):
            window.stop()
        self.assertTrue(window.busy)
        self.assertTrue(window.cancel.is_set())

    def test_passage_uses_the_box_seed_and_explicit_emotion(self):
        window = voice.VoiceWindow.__new__(voice.VoiceWindow)
        window.cancel = threading.Event()
        window._ensure_c_engine = Mock()
        window._status = Mock()
        window._finish = Mock()
        with patch.object(voice.requests, "post") as post, \
             patch.object(voice.sf, "read", return_value=([0.0], 24000)), \
             patch.object(voice.sd, "play"), patch.object(voice.sd, "wait"):
            post.return_value.content = b"fake wav"
            window._speak_c("Hello. How are you?", "joy", 42)
        post.assert_called_once()
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["input"], "Hello. How are you?")
        self.assertEqual(body["seed"], 42)
        self.assertEqual(body["emotion"], "joy")

    def test_seed_box_falls_back_to_42(self):
        window = voice.VoiceWindow.__new__(voice.VoiceWindow)
        window.seed = Mock(get=Mock(return_value="   "))
        self.assertEqual(voice.VoiceWindow._seed(window), 42)
        window.seed = Mock(get=Mock(return_value="17"))
        self.assertEqual(voice.VoiceWindow._seed(window), 17)


if __name__ == "__main__":
    unittest.main()
