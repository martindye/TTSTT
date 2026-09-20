"""Conversation orchestrator: mic -> STT -> LLM -> TTS -> speakers.

Two listening modes:

* half-duplex (default, safe with speakers): while the assistant is speaking,
  the mic hears the assistant's own voice coming out of the speakers; those
  transcriptions are ignored so the assistant never "talks to itself".
  The user speaks after the assistant pauses.
* barge-in (``barge_in=True``, use headphones): the user can interrupt the
  assistant mid-answer — N consecutive user words cancel the running LLM/TTS.

Threading model
---------------
* PortAudio input thread : delivers 80 ms mic blocks to the STT queue
* stt worker thread      : runs STT frames, publishes SttEvents
* turn worker thread     : single persistent worker; user utterances are
                            queued; one LLM+TTS pass per utterance
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np
import torch

from .audio_io import MicCapture, SpeakerPlayback
from .llm_client import LLMClient
from .stt_engine import KyutaiSTT, SttEventKind
from .tts_engine import TTSEngine

log = logging.getLogger("voice")

BARGE_MIN_WORDS = 3  # user words required to interrupt the assistant
MAX_HISTORY_MESSAGES = 12
# A word heard while this much assistant audio is still in the playback
# buffer is treated as echo (half-duplex mode).
ECHO_TAIL_SECONDS = 0.15


def split_sentence(buf: str) -> tuple[str, str]:
    """Split `buf` into (complete_sentence, remainder).

    A sentence ends at the first ". ", "! " or "? " (or at a terminal
    [.!?] when nothing else follows). Returns ("", buf) while no boundary
    has been seen yet.
    """
    idx = -1
    for pat in (". ", "! ", "? "):
        i = buf.find(pat)
        if i != -1 and (idx == -1 or i < idx):
            idx = i
    if idx == -1:
        # A terminal mark at the very end of the buffer also ends a sentence.
        i = len(buf)
        while i > 0 and buf[i - 1] not in ".!?":
            i -= 1
        if i > 0 and i == len(buf):
            idx = i - 1
    if idx == -1:
        return "", buf
    return buf[: idx + 1], buf[idx + 1 :].lstrip()


class VoiceAssistant:
    def __init__(
        self,
        llm: LLMClient,
        stt: KyutaiSTT,
        tts: TTSEngine,
        mic_device: int | str | None = None,
        spk_device: int | str | None = None,
        echo_guard_words: int = BARGE_MIN_WORDS,
        barge_in: bool = False,
    ):
        self.llm = llm
        self.stt = stt
        self.tts = tts
        self.barge_in = barge_in

        # Audio plumbing -----------------------------------------------------
        self.speakers = SpeakerPlayback(sample_rate=tts.sample_rate, device=spk_device)
        self.mic = MicCapture(
            callback=self._on_mic_block,
            sample_rate=stt.sample_rate,
            frame_ms=80,  # one STT frame per callback
            device=mic_device,
        )

        # State ---------------------------------------------------------------
        self._stt_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=240)
        self._turn_queue: queue.Queue[str] = queue.Queue()
        self._stt_thread: threading.Thread | None = None
        self._turn_thread: threading.Thread | None = None
        self._history: list[dict] = []
        self._turn_busy = threading.Event()
        self._turn_cancel = threading.Event()
        self._user_words: list[str] = []
        self._user_speaking = False
        self._barge_armed = False
        self._stop = threading.Event()
        self._stale_timer: threading.Timer | None = None
        self.echo_guard_words = echo_guard_words
        self.turn_count = 0
        self.barge_count = 0
        self.echo_dropped = 0  # half-duplex: words ignored as echo

    # ------------------------------------------------------------------ audio
    def _on_mic_block(self, pcm: np.ndarray):
        """PortAudio input thread: enqueue only (STT may be busy)."""
        if self._stt_queue.full():
            try:
                self._stt_queue.get_nowait()  # drop oldest
            except queue.Empty:
                pass
        try:
            self._stt_queue.put_nowait(pcm)
        except queue.Full:
            pass

    def _stt_loop(self):
        while not self._stop.is_set():
            try:
                pcm = self._stt_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._stop.is_set():
                break
            try:
                events = self.stt.feed_frame(torch.from_numpy(pcm))
            except Exception:
                log.exception("STT frame failed")
                continue
            for ev in events:
                self._on_stt_event(ev)

    # ------------------------------------------------------------- STT events
    def _assistant_audible(self) -> bool:
        """True while the assistant may still be heard in the room."""
        return self._turn_busy.is_set() or self.speakers.pending_seconds > ECHO_TAIL_SECONDS

    def _on_stt_event(self, ev):
        if ev.kind is SttEventKind.WORD:
            self._on_word(ev.piece)
        elif ev.kind is SttEventKind.END:
            self._finalize_utterance()
        else:  # SILENCE
            self._arm_stale_timer()

    def _on_word(self, piece: str):
        # Half-duplex: while the assistant is (still) audible, everything the
        # mic transcribes is the assistant's own voice -> ignore it.
        if not self.barge_in and self._assistant_audible():
            self.echo_dropped += 1
            return
        if not self._user_speaking:
            self._user_speaking = True
            self._user_words = []
            self._barge_armed = False
            self._cancel_stale_timer()
        self._user_words.append(piece)
        # Barge-in: enough user words while the assistant responds.
        if (
            self.barge_in
            and self._turn_busy.is_set()
            and not self._barge_armed
            and len(self._user_words) >= self.echo_guard_words
        ):
            self._barge_in()

    def _finalize_utterance(self):
        if not self._user_speaking:
            return
        text = "".join(self._user_words)
        self._user_speaking = False
        self._user_words = []
        self._cancel_stale_timer()
        text = text.replace("▁", " ")
        while "  " in text:
            text = text.replace("  ", " ")
        text = text.strip()
        if not text:
            return
        log.info("USER: %s", text)
        try:
            self._turn_queue.put_nowait(text)
        except queue.Full:
            log.warning("turn queue full, dropping utterance")

    def _arm_stale_timer(self):
        """Silence separator: if no new words arrive within 2 s, finalize."""
        self._cancel_stale_timer()
        # Fallback only: the model's own end-of-utterance token is the primary
        # signal. Long enough that a natural thinking-pause doesn't chop the
        # sentence into separate turns.
        t = threading.Timer(3.5, self._stale_fired)
        t.daemon = True
        t.start()
        self._stale_timer = t

    def _stale_fired(self):
        if self._user_speaking and self._user_words:
            self._finalize_utterance()

    def _cancel_stale_timer(self):
        if self._stale_timer is not None:
            self._stale_timer.cancel()
            self._stale_timer = None

    # ------------------------------------------------------------- turns
    def _turn_loop(self):
        while not self._stop.is_set():
            try:
                text = self._turn_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._stop.is_set():
                break
            self._run_turn(text)

    def _run_turn(self, user_text: str):
        cancel = self._turn_cancel
        cancel.clear()
        self._turn_busy.set()
        self._history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": self.llm.system_prompt}]
        messages.extend(self._history[-MAX_HISTORY_MESSAGES:])
        full: list[str] = []
        buf = ""
        t0 = time.time()
        try:
            for delta in self.llm.stream_chat(messages, cancel):
                if cancel.is_set():
                    break
                full.append(delta)
                buf += delta
                sent, buf = split_sentence(buf)
                while sent:
                    self._speak(sent, cancel)
                    sent, buf = split_sentence(buf)
            if buf and not cancel.is_set():
                self._speak(buf, cancel)
            if not cancel.is_set():
                answer = "".join(full).strip()
                if answer:
                    self._history.append({"role": "assistant", "content": answer})
                    self._history = self._history[-(MAX_HISTORY_MESSAGES + 4):]
        except Exception:
            log.exception("turn failed")
        finally:
            if not self.barge_in:
                # Half-duplex: discard any echo residue captured in the drain
                # window so the next turn starts clean.
                self._user_speaking = False
                self._user_words = []
                self._barge_armed = False
                self._cancel_stale_timer()
            self.turn_count += 1
            log.info(
                "turn %d finished in %.2fs (%d chars, cancelled=%s, echo_dropped=%d)",
                self.turn_count,
                time.time() - t0,
                len("".join(full)),
                cancel.is_set(),
                self.echo_dropped,
            )
            self._turn_busy.clear()

    def _speak(self, sentence: str, cancel: threading.Event):
        log.debug("SPEAK: %s", sentence)
        for chunk in self.tts.stream_sentence(sentence, cancel):
            if cancel.is_set():
                break
            self.speakers.write(chunk)

    def _barge_in(self):
        if not self._turn_busy.is_set():
            return
        self._barge_armed = True
        self.barge_count += 1
        log.info("BARGE-IN: interrupting assistant (user words so far: %d)",
                 len(self._user_words))
        self._turn_cancel.set()
        self.speakers.flush()

    # ------------------------------------------------------------- lifecycle
    def start(self):
        mode = "barge-in" if self.barge_in else "half-duplex"
        self.speakers.start()
        self.stt.reset()
        self._stt_thread = threading.Thread(target=self._stt_loop, daemon=True)
        self._stt_thread.start()
        self._turn_thread = threading.Thread(target=self._turn_loop, daemon=True)
        self._turn_thread.start()
        self.mic.start()
        log.info("Voice assistant ready (%s mode). Speak to begin; Ctrl-C to quit.", mode)

    def stop(self):
        self._stop.set()
        if self._stt_thread:
            self._stt_thread.join(timeout=3)
        deadline = time.time() + 5
        while self._turn_busy.is_set() and time.time() < deadline:
            time.sleep(0.1)
        self.mic.close()
        self.speakers.close()
