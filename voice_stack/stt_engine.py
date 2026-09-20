"""Kyutai STT 1B streaming speech-to-text (moshi package, PyTorch).

Feeds 80 ms audio frames (1920 samples @ 24 kHz) into the Mimi encoder + LM
and yields text pieces. End-of-utterance is signalled by the model's own
token 0 (the "semantic VAD" from kyutai.org/stt); token 3 is a silence
separator.

Token semantics (see kyutai-labs/moshi rust/moshi-core/src/asr.rs):
  * token 0  -> end of utterance (fires once the trailing silence is absorbed)
  * token 3  -> silence/pad separator
  * other    -> sentencepiece text piece ("▁" marks word starts)

The model has a built-in delay (audio_delay_seconds, 0.5 s for stt-1b):
outputs during the first ~6 frames of a fresh stream are discarded. The
engine is intended to be fed continuously (it streams, like the mic does);
call `reset()` only for recovery or when you genuinely want a clean context.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum

import torch

log = logging.getLogger(__name__)


class SttEventKind(Enum):
    WORD = "word"        # a text piece was produced
    END = "end"          # token 0: the user finished the utterance
    SILENCE = "silence"  # token 3: silence separator


@dataclass
class SttEvent:
    kind: SttEventKind
    piece: str = ""  # text piece for WORD events (may start with a leading space)


class KyutaiSTT:
    """Streaming STT engine. Not thread-safe: feed frames from one thread."""

    def __init__(self, hf_repo: str = "kyutai/stt-1b-en_fr", device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        # torch.compile (Inductor) needs Triton (CUDA) or an MSVC compiler
        # (CPU); this Windows machine has neither, so force the eager path for
        # the rope/rmsnorm kernels. CUDA *graphs* are a separate mechanism and
        # stay enabled, which is where the real speed comes from.
        os.environ.setdefault("NO_TORCH_COMPILE", "1")
        self.sample_rate = 24000
        self.frame_rate = 12.5
        self.frame_size = 1920
        self.delay_tokens = 6
        self._load_all(hf_repo)

    # ------------------------------------------------------------------ load
    def _load_all(self, hf_repo: str):
        from moshi.models import LMGen, loaders

        info = loaders.CheckpointInfo.from_hf_repo(hf_repo)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.mimi = info.get_mimi(device=self.device)
        self.tokenizer = info.get_text_tokenizer()
        self.lm = info.get_moshi(device=self.device, dtype=dtype)
        self.lm_gen = LMGen(self.lm, temp=0.0, temp_text=0.0, use_sampling=False)
        self.stt_config: dict = info.stt_config or {}
        self.sample_rate = int(self.mimi.sample_rate)
        self.frame_rate = float(self.mimi.frame_rate)
        self.frame_size = int(self.sample_rate / self.frame_rate)
        self.delay_tokens = int(
            round(float(self.stt_config.get("audio_delay_seconds", 0.5)) * self.frame_rate)
        )
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        self._primed = False
        self._step_idx = 0
        self.total_frames = 0
        self._warmup()

    def _warmup(self):
        """Run two silent frames so lazy compilation / CUDA-graph capture happens
        now (at load time) instead of on the first real frame, and prime the LM
        the way the first real frame would."""
        zeros = torch.zeros(1, 1, self.frame_size, device=self.device, dtype=torch.float32)
        for _ in range(2):
            codes = self.mimi.encode(zeros)
            if not self._primed:
                # First slice must be stepped twice so it is not replaced by
                # the initial tokens (see the official stt_pytorch.ipynb).
                self.lm_gen.step(codes)
                self._primed = True
            self.lm_gen.step(codes)
        self._step_idx = 2

    # ----------------------------------------------------------------- reset
    def reset(self):
        """Reset LM streaming state (offsets + KV) and the mimi exec mask.

        Note: the mimi SEANet conv buffers retain a few frames of history; that
        is harmless because silence follows between turns. Use for recovery or
        to start from a clean context.
        """
        reset_mask = torch.ones(1, dtype=torch.bool, device=self.device)
        try:
            self.lm_gen.reset_streaming(reset_mask)
        except Exception:
            log.exception("STT LM reset failed; continuing")
        try:
            self.mimi.reset_streaming(reset_mask)
        except Exception:
            log.exception("STT mimi reset failed; continuing")
        self._primed = False
        self._step_idx = 0

    # ------------------------------------------------------------- inference
    def feed_frame(self, pcm: torch.Tensor) -> list[SttEvent]:
        """Process one 80 ms frame of float32 mono PCM (24 kHz).

        `pcm` has exactly `frame_size` samples. Returns a list with at most one
        SttEvent (the model emits at most one text token per frame).
        """
        if pcm.numel() != self.frame_size:
            raise ValueError(f"expected {self.frame_size} samples, got {pcm.numel()}")
        chunk = pcm.view(1, 1, -1).to(device=self.device, dtype=torch.float32)
        codes = self.mimi.encode(chunk)  # [1, K, 1]
        if not self._primed:
            self.lm_gen.step(codes)  # prime (consumed by the initial tokens)
            self._primed = True
        tokens = self.lm_gen.step(codes)
        if tokens is None:
            return []
        tok = int(tokens[0, 0, 0].item())
        self._step_idx += 1
        self.total_frames += 1
        if self._step_idx <= self.delay_tokens:
            return []  # inside the model's built-in delay window
        if tok == 0:
            return [SttEvent(SttEventKind.END)]
        if tok == 3:
            return [SttEvent(SttEventKind.SILENCE)]
        piece = self.tokenizer.id_to_piece(tok)
        return [SttEvent(SttEventKind.WORD, piece=piece)]

    def close(self):
        pass
