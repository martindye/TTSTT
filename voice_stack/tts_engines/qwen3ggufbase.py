"""Qwen3 1.7B Base Q8 engine for the generic TTS server (tts_server.py).

Wraps the same registered-reference engine the coding voice used to run
locally (voice_stack.qwen3_gguf_base_engine): the qwentts.cpp server on
127.0.0.1:8095 (shared, started on first use) with the registered
reference voices. Now that the coding bridge plays no local audio, THIS
server owns the voice model, so the replies come out of the device the
user is listening on, not the PC's speakers.

Voices are the registered references:
  "pocket"    the standard small coding voice (default)
  "warm-brit" the warm British female reference
  "p277"      the old plain pro voice (x-vector clone)
"""
from __future__ import annotations

import re
import threading

# The 8095 server caps a generation at max_new_tokens=768 (~60 s of audio
# at 12 Hz), so long text is sent sentence by sentence and the audio is
# concatenated.
_MAX_SENTENCE_CHARS = 280
_SPLIT = re.compile(r'(?<=[.!?])(?=["\')\]]*(?:\s|$))')


def _sentences(text: str, limit: int = _MAX_SENTENCE_CHARS) -> list[str]:
    text = " ".join((text or "").split())
    if not text:
        return []
    out: list[str] = []
    for piece in _SPLIT.split(text):
        piece = piece.strip()
        if not piece:
            continue
        while len(piece) > limit:  # no terminal punctuation: hard cap
            cut = piece.rfind(" ", 1, limit)
            if cut <= 0:
                cut = limit
            out.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            out.append(piece)
    return out


class Qwen3GGUFBaseEngine:
    name = "qwen3ggufbase"
    default_voice = "pocket"
    _VOICES = ("pocket", "warm-brit", "p277")

    def __init__(self) -> None:
        import time as _time

        self._lock = threading.Lock()
        self._impl = None          # Qwen3GGUFBase for the current voice
        self._voice: str | None = None
        self._loaded = False
        # After a failed start (e.g. the GPU is full and the 8095 server
        # cannot come up) fail fast for a while instead of re-attempting a
        # multi-minute launch on every request; the server's engine fallback
        # then serves the voice from the next engine.
        self._cooldown_until = 0.0
        self._now = _time.time

    def _ensure(self, voice: str) -> None:
        if self._loaded and self._voice == voice:
            return
        if self._now() < self._cooldown_until:
            raise RuntimeError(
                "qwen3ggufbase cooling down after a failed start "
                "(GPU busy?)")
        try:
            from ..qwen3_gguf_base_engine import Qwen3GGUFBase

            self._impl = Qwen3GGUFBase(voice=voice)
            self._voice = voice
            self._loaded = True
        except Exception:
            self._cooldown_until = self._now() + 120
            raise

    # ------------------------------------------------------------ lifecycle
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        with self._lock:
            self._ensure(self._voice or self.default_voice)

    # ------------------------------------------------------------------ api
    def voices(self) -> list[str]:
        return list(self._VOICES)

    def synthesize(self, text, voice):
        """Render `text` -> (mono float32 samples in [-1, 1], sample_rate)."""
        text = (text or "").strip()
        if not text:
            raise ValueError("empty text")
        import numpy as np

        want = voice if voice in self._VOICES else self.default_voice
        chunks = []
        with self._lock:  # one voice at a time across concurrent requests
            self._ensure(want)
            impl = self._impl
            for sentence in _sentences(text):
                for audio in impl.stream_sentence(sentence, None):
                    chunks.append(audio)
        if not chunks:
            raise RuntimeError("Qwen3 TTS returned no audio")
        return np.concatenate(chunks), int(impl.sample_rate)


from . import register  # noqa: E402

register(Qwen3GGUFBaseEngine())
