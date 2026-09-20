"""Microphone capture and speaker playback built on sounddevice (PortAudio).

- MicCapture: continuous 24 kHz mono float32 capture (Kyutai STT native rate),
  pushed block-by-block to a callback.
- SpeakerPlayback: 24 kHz mono float32 playback (Pocket TTS native rate) with a
  small ring buffer; `flush()` clears pending audio for barge-in.

The PortAudio callbacks run on their own threads; all shared state is guarded
by a lock held only for microseconds.
"""

from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd

# ---------------------------------------------------------------------------
# Microphone
# ---------------------------------------------------------------------------


class MicCapture:
    """Opens an input device and invokes `callback(pcm: np.ndarray float32)`.

    `pcm` has exactly `frame_length` samples (mono). If the system delivers a
    different rate we open the stream at `sample_rate` and rely on the driver
    to convert (standard on Windows/WASAPI).
    """

    def __init__(
        self,
        callback,
        sample_rate: int = 24000,
        frame_ms: int = 80,
        device: int | str | None = None,
    ):
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * frame_ms / 1000)
        self._callback = callback
        self._device = device
        self._stream: sd.RawInputStream | None = None
        self.input_rms = 0.0  # running RMS, for debug/UI

    def start(self):
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_length,
            device=self._device,
            dtype="float32",
            channels=1,
            callback=self._portaudio_cb,
        )
        self._stream.start()

    def _portaudio_cb(self, indata, frames, time_info, status):  # noqa: ARG002
        if status:
            return  # input overflow/underflow: drop block
        audio = np.frombuffer(indata, dtype=np.float32).copy()
        if audio.size:
            self.input_rms = 0.9 * self.input_rms + 0.1 * float(np.sqrt(np.mean(np.square(audio))))
        self._callback(audio)

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None


# ---------------------------------------------------------------------------
# Speakers
# ---------------------------------------------------------------------------


class SpeakerPlayback:
    """Low-latency float32 mono playback.

    `write(audio)` enqueues a 1-D float32 array (or raw float32 bytes); the
    PortAudio callback drains the buffer. `flush()` drops everything queued so
    barge-in is immediate.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        device: int | str | None = None,
        latency: str = "low",
    ):
        self.sample_rate = sample_rate
        self._device = device
        self._latency = latency
        self._lock = threading.Lock()
        self._buffer = bytearray()
        self._stream: sd.RawOutputStream | None = None
        self.pending_seconds = 0.0
        self.dropped_bytes = 0  # bytes discarded by flush() (barge-in accounting)

    def start(self):
        self._stream = sd.RawOutputStream(
            samplerate=self.sample_rate,
            blocksize=960,  # 40 ms at 24 kHz
            device=self._device,
            dtype="float32",
            channels=1,
            latency=self._latency,
            callback=self._portaudio_cb,
        )
        self._stream.start()

    def _portaudio_cb(self, outdata, frames, time_info, status):  # noqa: ARG002
        need = frames * 4
        with self._lock:
            buf = bytes(self._buffer[:need])
            if len(self._buffer) > len(buf):
                del self._buffer[: len(buf)]
            else:
                self._buffer.clear()
        if len(buf) < need:
            buf = buf + b"\x00" * (need - len(buf))
        outdata[:need] = buf
        self.pending_seconds = max(0.0, self.pending_seconds - frames / self.sample_rate)

    def write(self, audio: np.ndarray | bytes):
        """Queue audio for playback. float32 mono ndarray or raw float32 bytes."""
        if isinstance(audio, np.ndarray):
            a = audio.astype(np.float32, copy=False)
            if a.dtype == np.float32:
                np.clip(a, -1.0, 1.0, out=a)
            payload = a.tobytes()
        else:
            payload = audio
        with self._lock:
            self._buffer.extend(payload)
            self.pending_seconds += len(payload) // 4 / self.sample_rate

    def flush(self) -> int:
        """Drop all queued audio; returns the number of samples dropped."""
        with self._lock:
            dropped = len(self._buffer) // 4
            self._buffer.clear()
            self.pending_seconds = 0.0
            self.dropped_bytes += dropped * 4
            return dropped

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
