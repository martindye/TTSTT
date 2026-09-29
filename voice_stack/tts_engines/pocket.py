"""Pocket TTS engine — local, CPU-only, no network.

The first engine of the generic TTS server. Loads once (lazy), caches a
voice state per voice, serializes generation (TTSModel is not thread-safe).
"""
from __future__ import annotations

import logging
import os
import threading

log = logging.getLogger(__name__)

# Same roster coding_voice.py falls back to when the pocket_tts metadata is
# unavailable.
_FALLBACK_VOICES = [
    "anna", "vera", "fantine", "eponine", "azelma", "mary",
    "jane", "eve", "cosette", "caro_davy", "alba", "jean",
    "charles", "paul", "george", "michael", "marius",
    "javert", "bill_boerst", "peter_yearsley", "stuart_bell",
]


class PocketEngine:
    name = "pocket"

    def __init__(self, language: str = "english", quantize: bool = True,
                 default_voice: str = "anna"):
        self._language = language
        self._quantize = quantize
        self._default_voice = default_voice
        self._model = None
        self._voice_states: dict[str, object] = {}
        self._voice_names: list[str] | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------- lifecycle
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from pocket_tts import TTSModel  # import sets torch threads to 1

        # pocket_tts forces torch.set_num_threads(1) on import; restore a
        # sane CPU parallelism for generation.
        torch.set_num_threads(max(2, (os.cpu_count() or 4) // 2))
        log.info("pocket: loading model (language=%s, quantize=%s) ...",
                 self._language, self._quantize)
        self._model = TTSModel.load_model(language=self._language,
                                          quantize=self._quantize)
        try:
            from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES as V
            self._voice_names = list(V)
        except Exception:  # metadata moved — fall back to the known roster
            self._voice_names = list(_FALLBACK_VOICES)
        log.info("pocket: model ready (%d voices)", len(self._voice_names))

    # ------------------------------------------------------------------ api
    def voices(self) -> list[str]:
        self.load()
        return list(self._voice_names or _FALLBACK_VOICES)

    def _voice_state(self, voice: str | None):
        self.load()
        key = voice or self._default_voice
        if key not in self._voice_states:
            if self._voice_names and key not in self._voice_names:
                log.warning("pocket: unknown voice %r, trying %r",
                            key, self._default_voice)
                key = self._default_voice
            self._voice_states[key] = self._model.get_state_for_audio_prompt(key)
        return self._voice_states[key]

    def synthesize(self, text: str, voice: str | None = None):
        """Render `text` -> (mono float32 samples in [-1, 1], sample_rate)."""
        text = (text or "").strip()
        if not text:
            raise ValueError("empty text")
        self.load()
        state = self._voice_state(voice)
        with self._lock:
            chunks = list(self._model.generate_audio_stream(state, text))
        import numpy as np
        samples = np.concatenate(
            [np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks])
        return samples, int(self._model.sample_rate)


from . import register  # noqa: E402

register(PocketEngine())
