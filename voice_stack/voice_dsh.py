"""Voice bridge: mic -> Kyutai STT -> DSH session -> TTS -> speakers.

The "brain" is a real DeepSeek Harness session (the official out-of-process
SDK runtime, see dsh_runtime.py), not a side LLM client:

  * the voice conversation is a full DSH session with a full context window,
    compaction, and durable memory (persisted in .dsh-voice, survives restarts)
  * thinking (reasoning deltas) is captured but never spoken
  * the voice uses the same local llama.cpp model as the web UI

Half-duplex: while the assistant is (or was just) audible, the mic is deaf —
the speakers are in the room and the mic would hear the assistant itself.

Usage:
    python -m voice_stack.voice_dsh                # live: mic -> DSH -> speakers
    python -m voice_stack.voice_dsh --once "..."   # one spoken answer, then exit
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .audio_io import MicCapture, SpeakerPlayback
from .assistant import split_sentence
from .dsh_runtime import DshError, DshRuntime
from .stt_engine import KyutaiSTT, SttEventKind

log = logging.getLogger("voice")

TTSTT_ROOT = Path(__file__).resolve().parents[1]
PROJECTS_DIR = TTSTT_ROOT.parent
DEFAULT_DSH_ROOT = PROJECTS_DIR / "deepseek-harness"
DEFAULT_PERSONA = TTSTT_ROOT / "voice_persona"
DEFAULT_DSH_HOME = TTSTT_ROOT / ".dsh-voice"
DEFAULT_LLM_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "qwen3.8-27b"

# Silence accepted from the user must start this long after the assistant's
# audio finished: STT pipeline + acoustic tail otherwise transcribe the
# assistant's own last words as a new user turn (the echo-runaway bug).
VOICE_TAIL = 1.2  # seconds
# Endpointing is driven by GAPS BETWEEN WORD TOKENS, not by the model's
# END/SILENCE tokens: live testing showed the model emits END after every
# 1-2 words (mid-sentence!) and SILENCE on every idle frame, so neither is a
# usable "done" signal. A 1.2s wordless gap is an utterance boundary.
WORD_GAP_S = 1.2
# After a sentence-final punctuation token (the STT model emits "." / "?" /
# "!" after complete sentences) a much shorter pause is already a confident
# boundary, so the wait shrinks there.
WORD_GAP_SENTENCE_S = 0.6
SENTENCE_END_PUNCT = {".", "!", "?", "…", "。", "！"}
EVENT_STALL_S = 300.0  # no session events for this long -> give up the turn


# ---------------------------------------------------------------------------
# Speaking side: assistant/chunk stream -> sentences -> TTS -> speakers
# ---------------------------------------------------------------------------


class SpokenStream:
    """Consumes DSH ``assistant/chunk`` StreamChunks and speaks text blocks.

    Only blocks of type ``text`` are spoken; ``reasoning`` (thinking) and
    ``tool-call`` blocks are counted/skipped, never spoken.
    """

    def __init__(self, tts, speakers, on_audio=None, cancel=None, muted=False):
        self.tts = tts
        self.speakers = speakers
        self.on_audio = on_audio  # callable() after each audio write
        self.cancel = cancel if cancel is not None else threading.Event()
        # muted=True: run the whole TTS pipeline but write nothing to the
        # speakers (bench mode -- no echo into an open mic, no LLM/GPU idle
        # cost difference). first_audio still marks "TTS produced audio".
        self.muted = muted
        self.block_types: dict[int, str] = {}
        self.bufs: dict[int, str] = {}
        self.reasoning_chars = 0
        self.spoken_chars = 0
        self.first_text_at: float | None = None
        self.first_audio_at: float | None = None
        self.first_reasoning_at: float | None = None

    def feed(self, chunk: dict):
        ctype = chunk.get("type")
        if ctype == "block-start":
            self.block_types[chunk.get("index")] = chunk.get("blockType")
        elif ctype == "text-delta":
            idx = chunk.get("index")
            if self.block_types.get(idx, "text") != "text":
                return
            if self.first_text_at is None:
                self.first_text_at = time.time()
            buf = self.bufs.get(idx, "") + chunk.get("text", "")
            self.bufs[idx] = buf
            sent, buf = split_sentence(buf)
            self.bufs[idx] = buf
            while sent:
                self._speak(sent)
                sent, buf = split_sentence(buf)
                self.bufs[idx] = buf
        elif ctype == "reasoning-delta":
            if self.first_reasoning_at is None:
                self.first_reasoning_at = time.time()
            self.reasoning_chars += len(chunk.get("text", ""))
        elif ctype == "block-end":
            self._flush_index(chunk.get("index"))

    def _flush_index(self, idx):
        rest = self.bufs.pop(idx, "").strip()
        if rest:
            self._speak(rest)

    def flush(self):
        for idx in list(self.bufs):
            self._flush_index(idx)

    def _speak(self, sentence: str):
        if not sentence.strip():
            return
        for audio in self.tts.stream_sentence(sentence, self.cancel):
            if self.cancel.is_set():
                return
            if self.muted:
                if self.first_audio_at is None:
                    self.first_audio_at = time.time()
            else:
                self.speakers.write(audio)
                if self.first_audio_at is None:
                    self.first_audio_at = time.time()
            if self.on_audio:
                self.on_audio()
        self.spoken_chars += len(sentence)


# ---------------------------------------------------------------------------
# The assistant
# ---------------------------------------------------------------------------


class VoiceDshAssistant:
    """Mic -> STT -> DSH session (prompt) -> spoken answer."""

    def __init__(self, stt: KyutaiSTT, tts, runtime: DshRuntime,
                 mic_device=None, spk_device=None, mic_gain: float = 1.0):
        self.stt = stt
        self.tts = tts
        self.runtime = runtime
        self.mic_gain = float(mic_gain)
        self.speakers = SpeakerPlayback(sample_rate=tts.sample_rate, device=spk_device)
        self.mic = MicCapture(
            callback=self._on_mic_block,
            sample_rate=stt.sample_rate,
            frame_ms=80,
            device=mic_device,
        )
        self._stt_queue: "queue.Queue" = queue.Queue(maxsize=240)
        self._turn_queue: "queue.Queue" = queue.Queue()
        self._stt_words = 0
        self._stt_ends = 0
        self._stt_silences = 0
        self._stt_frame_ms = 0.0  # EMA of per-frame processing time
        self._turn_busy = threading.Event()
        self._turn_cancel = threading.Event()
        self._user_words: list[str] = []
        self._user_speaking = False
        self._last_word_piece: str | None = None
        self._last_gap_s: float = WORD_GAP_S
        self._send_timer: threading.Timer | None = None
        self._stop = threading.Event()
        self._audio_end_at = 0.0  # projected time when queued audio ends
        self.turn_count = 0
        self.echo_dropped = 0

    # ------------------------------------------------------------- audio in
    def _on_mic_block(self, pcm):
        if self.mic_gain != 1.0:
            pcm = np.clip(pcm * self.mic_gain, -0.999, 0.999)
        if self._stt_queue.full():
            try:
                self._stt_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._stt_queue.put_nowait(pcm)
        except queue.Full:
            pass

    def _stt_loop(self):
        import torch

        while not self._stop.is_set():
            try:
                pcm = self._stt_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._stop.is_set():
                break
            t_feed = time.perf_counter()
            try:
                events = self.stt.feed_frame(torch.from_numpy(pcm))
            except Exception:
                log.exception("STT frame failed")
                continue
            dt_ms = (time.perf_counter() - t_feed) * 1000.0
            self._stt_frame_ms = 0.9 * self._stt_frame_ms + 0.1 * dt_ms
            for ev in events:
                if ev.kind is SttEventKind.WORD:
                    self._stt_words += 1
                    log.info("stt word %r", ev.piece)
                elif ev.kind is SttEventKind.END:
                    self._stt_ends += 1
                    log.info("stt END-of-utterance")
                else:
                    self._stt_silences += 1
                self._on_stt_event(ev)

    def _assistant_audible(self) -> bool:
        """True while the assistant may still be heard in the room."""
        if self._turn_busy.is_set():
            return True
        return time.time() < self._audio_end_at + VOICE_TAIL

    def _on_stt_event(self, ev):
        if ev.kind is SttEventKind.WORD:
            if self._assistant_audible():
                self.echo_dropped += 1
                return
            if not self._user_speaking:
                self._user_speaking = True
                self._user_words = []
            self._user_words.append(ev.piece)
            self._last_word_piece = ev.piece
            self._arm_send_timer()
        # END / SILENCE are deliberately ignored for endpointing: the model
        # emits END mid-sentence (after 1-2 words) and SILENCE on every idle
        # frame, so only a gap between word tokens marks a real pause.

    def _arm_send_timer(self):
        self._cancel_send_timer()
        # Short gap after sentence-final punctuation, full gap otherwise.
        piece = (self._last_word_piece or "").lstrip("▁").strip()
        gap = WORD_GAP_SENTENCE_S if piece in SENTENCE_END_PUNCT else WORD_GAP_S
        self._last_gap_s = gap
        t = threading.Timer(gap, self._send_utterance)
        t.daemon = True
        t.start()
        self._send_timer = t

    def _cancel_send_timer(self):
        if self._send_timer is not None:
            self._send_timer.cancel()
            self._send_timer = None

    def _send_utterance(self):
        self._send_timer = None
        self._user_speaking = False
        text = "".join(self._user_words)
        self._user_words = []
        text = text.replace("▁", " ")
        while "  " in text:
            text = text.replace("  ", " ")
        text = text.strip()
        if not text:
            return
        log.info("USER: %s  [gap %.1fs]", text, self._last_gap_s)
        try:
            self._turn_queue.put_nowait(text)
        except queue.Full:
            log.warning("turn queue full, dropping utterance")

    # -------------------------------------------------------------- turns
    def _drain_events(self) -> int:
        """Drop pre-turn backlog; return the highest event seq seen."""
        last_seq = -1
        while True:
            try:
                item = self.runtime.events.get_nowait()
            except queue.Empty:
                return last_seq
            if item[0] == "event" and isinstance(item[2], dict):
                s = item[2].get("seq")
                if isinstance(s, int):
                    last_seq = max(last_seq, s)

    def _run_turn(self, text: str):
        self._turn_busy.set()
        self._turn_cancel.clear()
        t0 = time.time()
        spoken: SpokenStream | None = None
        try:
            if not self.runtime.alive:
                log.warning("DSH runtime is down (exit %s); restarting ...",
                            self.runtime.dead_code)
                self.runtime.start()
            min_seq = self._drain_events()
            self.runtime.prompt(text)
            t_prompt = time.time()
            log.info("turn %d: prompt sent in %.2fs", self.turn_count + 1, t_prompt - t0)
            ok, spoken = consume_turn(
                self.runtime, self.tts, self.speakers, self._note_audio,
                min_seq=min_seq, turn_label=f"turn {self.turn_count + 1}",
                cancel=self._turn_cancel,
            )
            if not ok and spoken is None:
                self._say_sorry()
        except DshError as e:
            log.error("turn failed: %s", e)
            self._say_sorry()
        finally:
            self.turn_count += 1
            self._turn_busy.clear()
            self._user_speaking = False
            self._user_words = []
            self._cancel_send_timer()
            dt_first = f"{spoken.first_text_at - t0:.1f}s" if spoken and spoken.first_text_at else "-"
            dt_first_audio = f"{spoken.first_audio_at - t0:.1f}s" if spoken and spoken.first_audio_at else "-"
            dt_first_reason = f"{spoken.first_reasoning_at - t0:.1f}s" if spoken and spoken.first_reasoning_at else "-"
            log.info(
                "turn %d finished in %.1fs (spoken %d chars, thinking %d, "
                "first text %s, first audio %s, first reasoning %s, echo_dropped=%d)",
                self.turn_count, time.time() - t0,
                spoken.spoken_chars if spoken else 0,
                spoken.reasoning_chars if spoken else 0,
                dt_first, dt_first_audio, dt_first_reason, self.echo_dropped,
            )

    def _note_audio(self):
        self._audio_end_at = time.time() + self.speakers.pending_seconds

    def _say_sorry(self):
        try:
            s = SpokenStream(self.tts, self.speakers, self._note_audio)
            s._speak("Something went wrong on my end. Please try again in a moment.")
        except Exception:
            log.exception("failed to speak error message")

    # ------------------------------------------------------------- lifecycle
    def start(self):
        self.speakers.start()
        self.stt.reset()
        threading.Thread(target=self._stt_loop, daemon=True, name="stt").start()
        threading.Thread(target=self._turn_loop, daemon=True, name="turns").start()
        self.mic.start()
        log.info("Voice bridge ready (DSH backend, half-duplex). Speak to begin; Ctrl-C to quit.")

    def _turn_loop(self):
        last_rms_log = 0.0
        while not self._stop.is_set():
            try:
                text = self._turn_queue.get(timeout=0.5)
            except queue.Empty:
                if time.time() - last_rms_log > 10:
                    last_rms_log = time.time()
                    log.info(
                        "mic raw rms=%.5f gain x%.0f | stt ms/frame=%.0f "
                        "backlog=%d frames | words=%d ends=%d sil=%d",
                        self.mic.input_rms, self.mic_gain, self._stt_frame_ms,
                        self._stt_queue.qsize(),
                        self._stt_words, self._stt_ends, self._stt_silences)
                continue
            if self._stop.is_set():
                break
            self._run_turn(text)

    def stop(self):
        self._stop.set()
        deadline = time.time() + 8
        while self._turn_busy.is_set() and time.time() < deadline:
            time.sleep(0.1)
        self.mic.close()
        self.speakers.close()


# ---------------------------------------------------------------------------
# Shared turn consumer (live mode and --once mode)
# ---------------------------------------------------------------------------


def consume_turn(runtime: DshRuntime, tts, speakers, on_audio, min_seq: int = -1,
                 turn_label: str = "turn", cancel: threading.Event | None = None,
                 muted: bool = False):
    """Consume one DSH turn from the runtime event stream, speaking the
    assistant's text blocks. Returns (ok, SpokenStream|None)."""
    spoken = SpokenStream(tts, speakers, on_audio, cancel=cancel, muted=muted)
    ok = False
    deadline = time.time() + EVENT_STALL_S
    while True:
        if time.time() > deadline:
            log.warning("%s: no session events for %.0fs; giving up",
                        turn_label, time.time() - deadline)
            break
        try:
            item = runtime.events.get(timeout=1.0)
        except queue.Empty:
            if not runtime.alive:
                log.error("%s: DSH runtime died mid-turn (exit %s)",
                          turn_label, runtime.dead_code)
                break
            continue
        kind = item[0]
        if kind == "died":
            log.error("%s: DSH runtime died mid-turn (exit %s)", turn_label, item[1])
            break
        if kind == "status":
            continue
        ev = item[2]
        if not isinstance(ev, dict):
            continue
        seq = ev.get("seq")
        if isinstance(seq, int) and seq <= min_seq:
            continue
        etype = ev.get("type")
        if etype == "assistant/chunk":
            chunk = (ev.get("data") or {}).get("chunk") or {}
            if chunk.get("type") == "block-start":
                pass  # handled by feed()
            spoken.feed(chunk)
        elif etype == "tool/call":
            data = ev.get("data") or {}
            log.info("%s: TOOL %s %s", turn_label, data.get("name"),
                     str(data.get("arguments"))[:120])
        elif etype == "turn/end":
            data = ev.get("data") or {}
            spoken.flush()
            reason = data.get("reason")
            if not isinstance(reason, dict) or reason.get("kind") in (
                    "completed", "max-tokens", None):
                ok = True
            else:
                log.warning("%s: turn ended with reason %s", turn_label, reason)
            break
        elif etype in ("session/abort", "turn/error", "error"):
            log.warning("%s: %s: %s", turn_label, etype, str(ev.get("data"))[:200])
            break
    return ok, spoken


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def make_stt(repo: str, device: str) -> KyutaiSTT:
    import torch

    from .stt_engine import KyutaiSTT

    order = {"auto": ["cuda", "cpu"], "cuda": ["cuda"], "cpu": ["cpu"]}[device]
    last_err: Exception | None = None
    for dev in order:
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        try:
            log.info("loading Kyutai STT on %s ...", dev)
            t0 = time.time()
            stt = KyutaiSTT(hf_repo=repo, device=dev)
            log.info("STT ready in %.1fs", time.time() - t0)
            return stt
        except Exception as e:
            last_err = e
            if dev == "cuda" and ("out of memory" in str(e).lower() or "cuda" in str(e).lower()):
                log.warning("GPU STT failed (%s); falling back to CPU", str(e)[:150])
                torch.cuda.empty_cache()
                continue
            raise
    raise RuntimeError(f"could not load STT: {last_err}")


def make_tts(language: str = "english", voice: str = "alba", quantize: bool = True):
    from .tts_engine import TTSEngine

    return TTSEngine(language=language, voice=voice, quantize=quantize)


def make_runtime(args, persona: Path) -> DshRuntime:
    # Fresh session id per process: the DSH SDK creates (never resumes) the
    # session, and creating over a persisted id is an "id collision" error.
    # Within one bridge run the whole conversation shares this one session.
    session_id = "voice-" + time.strftime("%Y%m%d-%H%M%S")
    return DshRuntime(
        dsh_root=str(args.dsh_root),
        dsh_home=str(args.dsh_home),
        model=args.model,
        base_url=args.llm_url,
        persona_cwd=str(persona),
        permission_mode=args.perm,
        session_id=session_id,
        stderr_log=str(TTSTT_ROOT / "tests" / "dsh_runtime.err.log"),
    )


class _MutedSpeakers:
    """No-op speaker sink for --no-speak bench runs (nothing reaches a
    device, so an open mic on the live bridge can't hear the bench)."""
    pending_seconds = 0.0

    def start(self):
        pass

    def close(self):
        pass

    def write(self, audio):
        pass


def run_once(args, text: str) -> int:
    tts = make_tts(args.tts_language, args.tts_voice, args.tts_quantize)
    runtime = make_runtime(args, DEFAULT_PERSONA)
    muted = bool(getattr(args, "no_speak", False))
    if muted:
        speakers = _MutedSpeakers()
    else:
        speakers = SpeakerPlayback(sample_rate=tts.sample_rate, device=args.spk_device)
    speakers.start()
    runtime.start()

    def note_audio():
        pass

    t0 = time.time()
    min_seq = -1
    cancel = threading.Event()
    try:
        runtime.prompt(text)
        ok, spoken = consume_turn(runtime, tts, speakers, note_audio,
                                  min_seq=min_seq, turn_label="once",
                                  cancel=cancel, muted=muted)
        while speakers.pending_seconds > 0.05:
            time.sleep(0.1)
        log.info("ONCE ok=%s spoken=%d thinking=%d", ok,
                 spoken.spoken_chars if spoken else 0,
                 spoken.reasoning_chars if spoken else 0)
        print(
            f"\n--- once --- ok={ok}  total={time.time() - t0:.1f}s  "
            f"first_text={spoken.first_text_at - t0 if spoken and spoken.first_text_at else float('nan'):.1f}s  "
            f"first_audio={spoken.first_audio_at - t0 if spoken and spoken.first_audio_at else float('nan'):.1f}s  "
            f"first_reasoning={spoken.first_reasoning_at - t0 if spoken and spoken.first_reasoning_at else float('nan'):.1f}s  "
            f"spoken={spoken.spoken_chars if spoken else 0} chars "
            f"(thinking {spoken.reasoning_chars if spoken else 0} chars, never spoken)"
        )
        return 0 if ok else 1
    except DshError as e:
        log.error("once failed: %s", e)
        return 1
    finally:
        speakers.close()
        runtime.stop()


def run_live(args) -> int:
    stt = make_stt(args.stt_repo, args.stt_device)
    tts = make_tts(args.tts_language, args.tts_voice, args.tts_quantize)
    runtime = make_runtime(args, DEFAULT_PERSONA)
    runtime.start()

    log.info("mic gain x%.1f", args.mic_gain)
    voice = VoiceDshAssistant(stt, tts, runtime,
                              mic_device=args.mic_device, spk_device=args.spk_device,
                              mic_gain=args.mic_gain)
    voice.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Ctrl-C: shutting down")
    finally:
        voice.stop()
        runtime.stop()
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="voice_stack.voice_dsh", description=__doc__)
    p.add_argument("--once", metavar="TEXT",
                   help="speak one canned utterance, play the spoken answer, exit")
    p.add_argument("--no-speak", dest="no_speak", action="store_true",
                    help="--once only: run the full pipeline but play nothing "
                         "(bench mode: no echo into an open mic)")
    p.add_argument("--mic-device", default=None)
    p.add_argument("--spk-device", default=None)
    p.add_argument("--mic-gain", type=float, default=1.0,
                   help="software gain applied to the mic before STT (default 1.0)")
    p.add_argument("--llm-url", default=DEFAULT_LLM_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dsh-root", default=str(DEFAULT_DSH_ROOT))
    p.add_argument("--dsh-home", default=str(DEFAULT_DSH_HOME))
    p.add_argument("--persona", default=None,
                   help="persona workspace dir (AGENTS.md); default voice_persona/")
    p.add_argument("--perm", default="workspace-write",
                   help="DSH permission mode for the voice agent")
    p.add_argument("--stt-repo", default="kyutai/stt-1b-en_fr")
    p.add_argument("--stt-device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--tts-language", default="english")
    p.add_argument("--tts-voice", default="eve")
    p.add_argument("--tts-quantize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("urllib3", "requests", "huggingface_hub", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.once:
        return run_once(args, args.once)
    return run_live(args)


if __name__ == "__main__":
    sys.exit(main())
