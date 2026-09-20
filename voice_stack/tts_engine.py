"""Pocket TTS engine (CPU) with streaming sentence output.

Loads the model once, keeps a voice state in memory, and serializes
generation calls (TTSModel is not thread-safe). `stream_sentence()` yields
float32 mono chunks at `sample_rate` (24 kHz) as they are decoded, so the
assistant can start speaking a sentence well before it is fully generated.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger(__name__)


class TTSEngine:
    def __init__(
        self,
        language: str = "english",
        voice: str = "eve",
        quantize: bool = True,
        cpu_threads: int | None = None,
    ):
        import os

        import torch

        # pocket_tts forces torch.set_num_threads(1) on import; restore a
        # sane parallelism for anything else running on CPU (e.g. STT).
        self._threads = cpu_threads or max(2, (os.cpu_count() or 4) // 2)
        self._torch = torch
        self._model = None
        self._voice_state = None
        self._lock = threading.Lock()  # serializes generate_audio_stream calls
        self.sample_rate = 24000

        from pocket_tts import TTSModel  # noqa: F401  (import sets threads=1)

        torch.set_num_threads(self._threads)

        log.info("Loading Pocket TTS (language=%s, quantize=%s) ...", language, quantize)
        self._model = TTSModel.load_model(language=language, quantize=quantize)
        self.sample_rate = int(self._model.sample_rate)
        log.info("Loading voice '%s' ...", voice)
        self._voice_state = self._model.get_state_for_audio_prompt(voice)
        log.info(
            "Pocket TTS ready (sample_rate=%d, threads=%d)",
            self.sample_rate,
            torch.get_num_threads(),
        )

    @property
    def model(self):
        return self._model

    def stream_sentence(self, sentence: str, stop_event: threading.Event):
        """Yield float32 1-D chunks for `sentence` as they become available.

        Stops early (discarding the rest) if `stop_event` is set.
        """
        if not sentence or not sentence.strip():
            return
        with self._lock:
            try:
                for chunk in self._model.generate_audio_stream(
                    self._voice_state, sentence.strip()
                ):
                    if stop_event.is_set():
                        break
                    yield np.asarray(chunk, dtype=np.float32).reshape(-1)
            except Exception:
                log.exception("TTS generation failed for %r", sentence[:80])
