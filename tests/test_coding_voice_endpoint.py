import queue
import threading
import time
import unittest
from collections import deque
from unittest.mock import Mock

from voice_stack.coding_voice import CodingVoice, FRAME_S, MAX_MAIL_DRAIN_S


class EndpointTest(unittest.TestCase):
    def endpoint(self, pieces, gap=7, age=15):
        cv = CodingVoice.__new__(CodingVoice)
        cv._pieces = pieces
        cv._last_piece = pieces[-1]
        cv._last_word_at = time.time() - gap
        cv._utterance_start = time.time() - age
        cv._utterance_max = 120
        cv._final_silence_s = 3
        cv._timer = None
        cv._stt_queue = queue.Queue()
        cv._arm_timer = Mock()
        cv._do_send = Mock()
        return cv

    def test_sentencepiece_words_do_not_wait_two_minutes(self):
        cv = self.endpoint(['▁So', ',', '▁what', '▁do', '▁you', '▁think'])
        cv._send()
        cv._do_send.assert_called_once()

    def test_long_unpunctuated_utterance_ends_after_pause(self):
        cv = self.endpoint(['▁word'] * 25)
        cv._send()
        cv._do_send.assert_called_once()

    def test_pending_audio_prevents_partial_send_even_at_timeout(self):
        cv = self.endpoint(['▁Not', '▁finished', '.'], age=125)
        for _ in range(4):
            cv._stt_queue.put((None, False))
        cv._send()
        cv._do_send.assert_not_called()

    def test_brief_breath_does_not_send(self):
        cv = self.endpoint(['▁Still', '▁talking', '.'], gap=1)
        cv._send()
        cv._do_send.assert_not_called()

    def test_recovery_replay_finishes_before_sending(self):
        cv = self.endpoint(['▁Not', '▁finished', '.'], age=125)
        cv._stt_replaying = True
        cv._send()
        cv._do_send.assert_not_called()

    def test_mailbox_fits_and_keeps_per_frame_origin(self):
        cv = CodingVoice.__new__(CodingVoice)
        count = int(MAX_MAIL_DRAIN_S / FRAME_S)
        cv._stt_queue = queue.Queue(maxsize=count + 240)
        cv._stt_queue.put(('earlier live', False))
        cv._mailbox = deque(range(count), maxlen=3750)
        cv._mic_lock = threading.Lock()
        cv._drain_mailbox()
        self.assertEqual(cv._stt_queue.get(), ('earlier live', False))
        self.assertEqual([cv._stt_queue.get() for _ in range(count)],
                         [(i, True) for i in range(count)])
        self.assertFalse(cv._mailbox)


if __name__ == '__main__':
    unittest.main()
