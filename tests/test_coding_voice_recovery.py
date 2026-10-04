import threading
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from voice_stack.coding_voice import CodingVoice, FRAME_S, STT_RECOVERY_S
from voice_stack.stt_engine import SttEvent, SttEventKind


class RecoveryTest(unittest.TestCase):
    def bridge(self, stt):
        cv = CodingVoice.__new__(CodingVoice)
        cv.stt = stt
        cv._stop = threading.Event()
        cv._stt_reset_pending = threading.Event()
        cv._stt_unheard = deque(maxlen=round(STT_RECOVERY_S / FRAME_S))
        cv._stt_wordless_frames = 0
        cv._stt_raw_words = 0
        cv._stt_replaying = False
        cv.spoken = SimpleNamespace(busy_audio=False, tts_generating=False)
        cv._on_word = Mock()
        return cv

    def test_stuck_decoder_recovers_and_replays_question_across_pauses(self):
        stuck = [True]
        def feed(frame):
            if not stuck[0] and frame[0] == 1:
                return [SttEvent(SttEventKind.WORD, '▁Hello')]
            return []
        stt = SimpleNamespace(feed_frame=feed, reset=Mock(side_effect=lambda: stuck.__setitem__(0, False)))
        cv = self.bridge(stt)
        for i in range(int(np.ceil(STT_RECOVERY_S / FRAME_S))):
            # A single short question followed by silence, not 90s loud audio.
            pcm = np.full(1920, 1 if i == 20 else 0, dtype=np.float32)
            cv._process_stt_frame(pcm, True)
        stt.reset.assert_called_once()
        cv._on_word.assert_called_once_with('▁Hello', mail=True)

    def test_recognised_words_are_never_replayed(self):
        stt = Mock()
        stt.feed_frame.side_effect = lambda p: [SttEvent(SttEventKind.WORD, '▁Hello')] if p[0] else []
        cv = self.bridge(stt)
        cv._process_stt_frame(np.ones(1920, dtype=np.float32), False)
        for _ in range(int(np.ceil(STT_RECOVERY_S / FRAME_S))):
            cv._process_stt_frame(np.zeros(1920, dtype=np.float32), False)
        cv._on_word.assert_called_once_with('▁Hello', mail=False)

    def test_reply_reset_runs_before_next_microphone_frame(self):
        calls = []
        stt = SimpleNamespace(reset=lambda: calls.append('reset'),
                              feed_frame=lambda p: calls.append('feed') or [])
        cv = self.bridge(stt)
        cv._stt_reset_pending.set()
        cv._process_stt_frame(np.zeros(1920, dtype=np.float32), False)
        self.assertEqual(calls, ['reset', 'feed'])

    def test_nonfinite_microphone_samples_never_reach_decoder(self):
        stt = Mock()
        stt.feed_frame.return_value = []
        cv = self.bridge(stt)
        pcm = np.zeros(1920, dtype=np.float32)
        pcm[:3] = [np.nan, np.inf, -np.inf]
        cv._process_stt_frame(pcm, False)
        self.assertTrue(stt.feed_frame.call_args.args[0].isfinite().all())

    def test_three_reply_cycles_and_replayed_chunks_leave_mic_open(self):
        stt = Mock()
        stt.feed_frame.return_value = []
        cv = self.bridge(stt)
        cv.gw = SimpleNamespace(session_id='test')
        cv._primed_for = 'test'
        cv._chunk_seq = 0
        cv._busy = False
        cv._turn_done = threading.Event()
        cv._drain_mailbox = Mock()
        cv.tts_pro = False
        cv.spoken = Mock(busy_audio=False, tts_generating=False)
        for turn in range(3):
            chunk = dict(type='assistant/chunk', seq=turn*2+1,
                         data=dict(chunk=dict(type='text-delta', text='Ready.')))
            cv._on_event(chunk)
            self.assertTrue(cv._busy)
            cv._on_event(dict(type='turn/end', seq=turn*2+2))
            cv._process_stt_frame(np.zeros(1920, dtype=np.float32), False)
            cv._on_event(chunk)  # replay at stream reconnect
            self.assertFalse(cv._busy)
        self.assertEqual(stt.reset.call_count, 3)
        self.assertEqual(cv._drain_mailbox.call_count, 3)


if __name__ == '__main__':
    unittest.main()
