"""Coding Voice: speak into the open coding chat; that session's visible
assistant output is spoken back (thinking/tool activity is never spoken).

    mic -> STT -> endpointing -> gateway session/prompt (appears in the chat)
    session/follow stream -> assistant text -> sentence-streamed TTS -> speakers

The follow stream is cycled every ~50 s (the gateway drops idle mux
connections ~75 s) and every (re)open replays the snapshot, so a dropped
connection never loses the tail of a reply or the turn/end that releases
the mic.

Run:
    python -X utf8 -m voice_stack.coding_voice [--gui http://127.0.0.1:3080]
        [--session SESSION_ID] [--gui-home C:\\Users\\press\\.dsh]
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
from collections import deque
import re
import sys
import threading
import time
import uuid
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from .stt_engine import KyutaiSTT, SttEventKind  # noqa: E402
from .tts_engine import TTSEngine  # noqa: E402
from .audio_io import MicCapture, SpeakerPlayback  # noqa: E402
from .assistant import split_sentence  # noqa: E402
from .voice_dsh import make_stt, make_tts  # noqa: E402
from .handoff import (  # noqa: E402
    find_target_session,
    load_browser_secret,
    mint_cookie,
    sessions_dir_for,
)

LOG_PREFIX = "coding-voice:"

# Word-gap endpointing (same shape as voice_dsh): after the last STT word wait
# this long before sending; sentence-final punctuation halves the wait.
WORD_GAP_S = 1.2
WORD_GAP_SENTENCE_S = 0.6
SENTENCE_END_PUNCT = {".", "!", "?", "\u2026", "\u3002", "\uff01", "\uff1f"}

# Half-duplex: ignore the mic this long after our own audio stops.
VOICE_TAIL_S = 1.0

# Cycle the follow stream this often (server idle-closes at ~75 s).
STREAM_CYCLE_S = 50.0
# Floor (mic-closed) safety releases.
FLOOR_IDLE_S = 2.5      # audio drained + no chunks for this long
FLOOR_MAX_S = 45.0      # absolute cap

FLOOR_LOG_ONCE = True


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {LOG_PREFIX} {msg}", flush=True)


def _norm(s: str) -> str:
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# Speaking side: session chunk stream -> sentences -> TTS, with resume.
# ---------------------------------------------------------------------------

class SpokenTurn:
    """Speaks the assistant's visible TEXT of the current turn, sentence by
    sentence, and remembers exactly what it spoke so a snapshot replay can
    resume without double-speaking. Reasoning/tool blocks are ignored."""

    def __init__(self, tts, speakers, on_audio, cancel: threading.Event):
        self.tts = tts
        self.speakers = speakers
        self.on_audio = on_audio
        self.cancel = cancel
        self.reset_turn()

    def reset_turn(self) -> None:
        self.spoken_text = ""
        self.buf = ""
        self.block_type: dict[int, str] = {}
        self._paused = False
        self._pending: list[str] = []

    def set_paused(self, paused: bool) -> None:
        """While paused (agent busy) sentences are buffered, not spoken; on
        un-pause the buffer is played back (in its own thread, so the event
        loop is never blocked)."""
        self._paused = paused
        if not paused and self._pending:
            threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        while self._pending and not self.cancel.is_set():
            self._speak(self._pending.pop(0))

    MAX_PENDING = 20  # ~2 min of speech; drop oldest beyond that

    def speak_now(self, sentence: str) -> None:
        """Speak immediately even while paused (canned lines)."""
        was_paused = self._paused
        self._paused = False
        try:
            self._speak(sentence)
        finally:
            self._paused = was_paused

    # -- live chunk stream (StreamChunk protocol, as in assistant/chunk) ----
    def feed(self, chunk: dict) -> None:
        t = chunk.get("type")
        idx = chunk.get("index", 0)
        if t == "block-start":
            self.block_type[idx] = chunk.get("blockType", "text")
        elif t == "text-delta":
            if self.block_type.get(idx, "text") != "text":
                return
            self.buf += chunk.get("text", "")
            self._pump()
        elif t == "block-end":
            if self.block_type.get(idx, "text") == "text" and self.buf.strip():
                self._speak(self.buf.strip())
                self.buf = ""

    def _pump(self) -> None:
        while True:
            sent, self.buf = split_sentence(self.buf)
            if not sent:
                return
            self._speak(sent)

    def _speak(self, sentence: str) -> None:
        sentence = sentence.strip()
        if not sentence or self.cancel.is_set():
            return
        if self._paused:
            # Agent is busy: buffer the text, skip TTS (no CPU burn, no
            # audio), and catch up when the turn ends. Count it in
            # spoken_text so snapshot replay can't speak it a second time.
            self._pending.append(sentence)
            if len(self._pending) > self.MAX_PENDING:
                self._pending.pop(0)
            self.spoken_text += sentence + " "
            return
        try:
            for audio in self.tts.stream_sentence(sentence, self.cancel):
                if self.cancel.is_set():
                    return
                self.speakers.write(audio)
                if self.on_audio:
                    self.on_audio()
        except Exception as e:
            log(f"TTS failed: {e}")
        self.spoken_text += sentence + " "

    # -- snapshot replay -----------------------------------------------------
    def resume_with(self, full_text: str) -> None:
        """Speak the part of `full_text` not yet spoken (word-overlap dedup)."""
        words = (full_text or "").split()
        n = len(self.spoken_text.split())
        if n >= len(words):
            return
        rest = " ".join(words[n:])
        if rest.strip():
            log(f"resuming reply, {len(words) - n} words left")
            self.buf = rest
            self._pump()
            # Whatever _pump left is a trailing fragment; keep it buffered.


# ---------------------------------------------------------------------------
# Gateway session client: inject prompts + follow the session event stream.
# ---------------------------------------------------------------------------

class GatewaySession:
    """One DSH GUI session reached through the loopback gateway."""

    def __init__(self, gui_url: str, session_id: str, authority: str, secret: bytes):
        self.gui_url = gui_url.rstrip("/")
        self.session_id = session_id
        self.cookie = mint_cookie(authority, secret)

    def inject(self, text: str) -> None:
        """Send a user utterance into the session (mode 'queue')."""
        import requests

        body = {
            "type": "client-request",
            "rpcId": str(uuid.uuid4()),
            "method": "session/prompt",
            "payload": {"args": {"request": {
                "requestId": str(uuid.uuid4()),
                "sessionId": self.session_id,
                "mode": "queue",
                "content": [{"type": "text", "text": text}],
            }}},
        }
        resp = requests.post(
            self.gui_url + "/api/session/prompt",
            headers={"Cookie": self.cookie},
            json=body,
            timeout=15,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"session/prompt HTTP {resp.status_code}")
        data = resp.json()
        result = data.get("result", data)
        if not (isinstance(result, dict) and result.get("ok")):
            raise RuntimeError(f"session/prompt not accepted: {str(data)[:200]}")

    async def follow(self, on_event, on_snapshot, stop: threading.Event) -> None:
        """Follow the session's event stream, cycling the connection every
        STREAM_CYCLE_S. on_event(ev) per live session event; on_snapshot(snap)
        per opening snapshot frame. Reconnects with backoff until stop.
        """
        import websockets

        delay = 1.0
        while not stop.is_set():
            opened_at = 0.0
            try:
                async with websockets.connect(
                    self.gui_url.replace("http://", "ws://") + "/api/remote.mux",
                    additional_headers={"Cookie": self.cookie},
                    max_size=200 * 1024 * 1024,
                    ping_interval=15,
                    ping_timeout=15,
                    close_timeout=5,
                ) as ws:
                    stream_id = str(uuid.uuid4())
                    await ws.send(json.dumps({
                        "type": "open",
                        "streamId": stream_id,
                        "endpoint": "session/follow",
                        "payload": {"args": {"request": {
                            "address": {"kind": "session", "sessionId": self.session_id},
                            "maxMessages": 1,
                        }}},
                    }))
                    opened_at = time.time()
                    log("follow stream open")
                    delay = 1.0
                    while not stop.is_set():
                        if time.time() - opened_at > STREAM_CYCLE_S:
                            break  # cycle ahead of the server's idle close
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue
                        try:
                            frame = json.loads(raw)
                        except Exception:
                            continue
                        entry = frame.get("value") if isinstance(frame, dict) else frame
                        if not isinstance(entry, dict):
                            continue
                        et = entry.get("type")
                        if et == "snapshot":
                            on_snapshot(entry)
                        elif et == "event":
                            ev = entry.get("event")
                            if isinstance(ev, dict):
                                on_event(ev)
            except Exception as e:
                if stop.is_set():
                    return
                log(f"follow stream error: {type(e).__name__}: {e}; retry in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 5.0)
            if stop.is_set():
                return
            log("follow stream cycling")


# ---------------------------------------------------------------------------
# Main object
# ---------------------------------------------------------------------------

class CodingVoice:
    """Mic -> STT -> gateway inject; session stream -> spoken reply -> TTS."""

    def __init__(self, stt, tts, gw: GatewaySession,
                 mic_device=None, spk_device=None, utterance_max=120.0):
        self.stt = stt
        self.tts = tts
        self.gw = gw
        self._utterance_max = utterance_max  # s; long-utterance safety cap
        self.speakers = SpeakerPlayback(sample_rate=tts.sample_rate,
                                        device=spk_device)
        self.mic = MicCapture(callback=self._on_mic_block,
                              sample_rate=stt.sample_rate, frame_ms=80,
                              device=mic_device)
        self._stop = threading.Event()
        self._cancel = threading.Event()
        self._audio_end_at = 0.0
        # Snapshot-replay watermarks (session-global seq space, never reset):
        # messages handled live are remembered by seq so a 50 s cycle never
        # re-speaks them; in-flight chunk rows are fed only once, by seq.
        self._seen_msg_seqs: set[int] = set()
        self._msg_seq = 0
        self._chunk_seq = 0
        self._primed = False
        self.spoken = SpokenTurn(tts, self.speakers, self._note_audio, self._cancel)
        self._stt_queue: queue.Queue = queue.Queue(maxsize=240)
        self._pieces: list[str] = []
        self._last_piece = ""
        self._timer: threading.Timer | None = None
        # Floor (half-duplex): held from inject until the reply is done.
        self._floor = threading.Event()
        self._turn_done = threading.Event()
        self._floor_since = 0.0
        self._last_chunk_at = 0.0
        self._inject_at_ms = 0
        # Endpointing: when the current utterance started + keep-open state.
        self._utterance_start = 0.0
        # Agent-busy: while a turn of THIS session is running, the STT model
        # is OFF (zero GPU for the agent's thinking), TTS is off, and the mic
        # goes to a bounded mailbox that is transcribed right after the turn.
        self._busy = False
        self._busy_notified = False
        self._mailbox: deque = deque(maxlen=3750)  # 5 min of 80 ms frames
        self._mail_left = 0  # queued frames still to transcribe as mailbox
        self._mail_skip = 0  # stale (pre-busy) frames ahead of the mailbox

    # ------------------------------------------------------------ echo guard
    def _note_audio(self) -> None:
        self._audio_end_at = time.time() + self.speakers.pending_seconds

    def _audible(self) -> bool:
        if self._floor.is_set():
            return True
        return time.time() < self._audio_end_at + VOICE_TAIL_S

    # --------------------------------------------------------------- mic in
    def _on_mic_block(self, pcm):
        if self._busy:
            # Keep the GPU 100% free for the agent's thinking: no STT feed,
            # hold the audio (5 min ring) and transcribe it when the turn
            # ends, so nothing the user said is lost.
            self._mailbox.append(pcm)
            return
        if self._stt_queue.full():
            try:
                self._stt_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._stt_queue.put_nowait(pcm)
        except queue.Full:
            pass

    def _drain_mailbox(self) -> None:
        """Feed the buffered (busy-period) mic audio into the STT queue."""
        n = len(self._mailbox)
        if not n:
            return
        for pcm in self._mailbox:
            if self._stt_queue.full():
                try:
                    self._stt_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._stt_queue.put_nowait(pcm)
            except queue.Full:
                break
        self._mailbox.clear()
        self._mail_skip = self._stt_queue.qsize()  # stale frames first
        self._mail_left = n
        log(f"mailbox: transcribing {n * 0.08:.0f}s of buffered speech")

    def _stt_loop(self):
        import torch

        while not self._stop.is_set():
            try:
                pcm = self._stt_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._stop.is_set():
                break
            is_mail = False
            if self._mail_skip > 0:
                self._mail_skip -= 1
            elif self._mail_left > 0:
                self._mail_left -= 1
                is_mail = True
            try:
                events = self.stt.feed_frame(torch.from_numpy(pcm))
            except Exception as e:
                log(f"STT frame failed: {e}")
                continue
            for ev in events:
                if ev.kind is SttEventKind.WORD:
                    self._on_word(ev.piece, mail=is_mail)
            # END / SILENCE deliberately ignored (word-gap endpointing).

    # ------------------------------------------------------------- words
    def _on_word(self, piece: str, mail: bool = False) -> None:
        if not piece.strip():
            return
        # Mailbox words were captured while the reply was not playing, so
        # they bypass the echo gate; live words are still gated.
        if self._audible() and not mail:
            return
        if not self._pieces:
            self._utterance_start = time.time()
        self._pieces.append(piece)
        self._last_piece = piece
        self._arm_timer()

    def _arm_timer(self) -> None:
        if self._timer:
            self._timer.cancel()
        piece = (self._last_piece or "").lstrip("\u2581").strip()
        gap = (WORD_GAP_SENTENCE_S if piece in SENTENCE_END_PUNCT
               else WORD_GAP_S)
        self._timer = threading.Timer(gap, self._send)
        self._timer.daemon = True
        self._timer.start()

    def _send(self) -> None:
        """Fired after a word gap. Send now if the utterance looks complete
        or has hit the max length; otherwise keep the door open — the user
        may just be pausing mid-thought."""
        if self._timer:
            self._timer.cancel()
            self._timer = None
        if not self._pieces:
            return
        text = "".join(self._pieces)
        words = len(text.split())
        last = (self._last_piece or "").lstrip("\u2581").strip()
        final = last in SENTENCE_END_PUNCT
        age = time.time() - self._utterance_start
        if final and words <= 14:
            pass  # complete short utterance ("what's the weather?")
        elif not final and words <= 8:
            pass  # short command without punctuation ("fix the login bug")
        elif age >= self._utterance_max:
            log(f"utterance max ({self._utterance_max:.0f}s) reached, sending")
        else:
            # Incomplete or long: hold the door open, recheck after the gap.
            self._arm_timer()
            return
        self._do_send()

    def _do_send(self) -> None:
        text = "".join(self._pieces)
        self._pieces = []
        self._last_piece = ""
        self._utterance_start = 0.0
        text = text.replace("\u2581", " ")
        while "  " in text:
            text = text.replace("  ", " ")
        text = text.strip()
        if not text:
            return
        log(f"USER: {text}")
        self._inject_at_ms = int(time.time() * 1000)
        self.spoken.reset_turn()
        self._turn_done.clear()
        self._floor.set()
        self._floor_since = time.time()
        if self._busy and not self._busy_notified:
            self._busy_notified = True
            self.spoken.speak_now("I'm still working on it. One moment.")
        try:
            self.gw.inject(text)
        except Exception as e:
            self._floor.clear()
            log(f"inject failed: {e}")

    # --------------------------------------------------- session stream in
    def _set_busy(self, busy: bool) -> None:
        if busy == self._busy:
            return
        self._busy = busy
        if busy:
            self._busy_notified = False
            self.spoken.set_paused(True)
            log("agent busy — STT off (GPU free), mic to mailbox, "
                "reply buffered")
        else:
            self.spoken.set_paused(False)
            self._drain_mailbox()
            log("agent done — speaking buffered reply")

    def _on_event(self, ev: dict) -> None:
        t = ev.get("type")
        if t == "assistant/chunk":
            self._set_busy(True)
            chunk = (ev.get("data") or {}).get("chunk")
            if isinstance(chunk, dict):
                self._last_chunk_at = time.time()
                self.spoken.feed(chunk)
            seq = ev.get("seq") or 0
            if isinstance(seq, int) and seq > self._chunk_seq:
                self._chunk_seq = seq
        elif t == "assistant/message":
            # Remember messages fully streamed live so snapshot cycles never
            # speak them again.
            seq = ev.get("seq")
            if isinstance(seq, int):
                self._seen_msg_seqs.add(seq)
        elif t == "turn/end":
            self._turn_done.set()
            self._set_busy(False)

    def _on_snapshot(self, snap: dict) -> None:
        """Replay what the live stream missed — and only that.

        Every record carries a session-global seq; watermarks make replay
        idempotent across the 50 s stream cycles, so nothing is ever spoken
        twice.
        """
        records = snap.get("records") or []
        last_msg = None
        max_seq = 0
        for rec in records:
            if rec.get("type") != "event":
                continue
            ev = rec.get("event") or {}
            seq = ev.get("seq") or 0
            if isinstance(seq, int):
                max_seq = max(max_seq, seq)
                if ev.get("type") == "assistant/message":
                    last_msg = ev

        if not self._primed:
            # First snapshot: remember where history ends, speak nothing.
            self._primed = True
            if last_msg is not None:
                self._msg_seq = last_msg.get("seq") or 0
                self._seen_msg_seqs.add(self._msg_seq)
            self._chunk_seq = max(self._chunk_seq, max_seq)
            return

        # A message that committed while we were away: speak it once.
        # resume_with() additionally word-dedupes against what we already
        # said (covers messages streamed live before the event was seen).
        if last_msg is not None:
            seq = last_msg.get("seq") or 0
            self._msg_seq = max(self._msg_seq, seq)
            if seq not in self._seen_msg_seqs:
                self._seen_msg_seqs.add(seq)
                if len(self._seen_msg_seqs) > 1000:
                    self._seen_msg_seqs = set(list(self._seen_msg_seqs)[-200:])
                ts = last_msg.get("time") or 0
                now_ms = self._inject_at_ms
                if not now_ms or ts >= now_ms - 5000:
                    msg = (last_msg.get("data") or {}).get("message") or {}
                    content = msg.get("content") or []
                    text = "".join(b.get("text", "") for b in content
                                   if isinstance(b, dict)
                                   and b.get("type") == "text")
                    if text.strip():
                        self.spoken.resume_with(text)

        # Turn still in progress: feed in-flight text rows we have not
        # consumed yet (by seq), so cycles can't double-speak.
        fed = 0
        for rec in records:
            if rec.get("type") == "event":
                ev = rec.get("event") or {}
                if ev.get("type") == "assistant/chunk":
                    seq = ev.get("seq") or 0
                    if seq > self._chunk_seq:
                        self._chunk_seq = seq
                        chunk = (ev.get("data") or {}).get("chunk")
                        if isinstance(chunk, dict):
                            self.spoken.feed(chunk)
                            fed += 1
            elif rec.get("type") == "chunks":
                ev = rec.get("event") or {}
                et = ev.get("type") or ""
                seq = ev.get("seq") or 0
                if et == "chunkrow/text-chunks" and seq > self._chunk_seq:
                    self._chunk_seq = seq
                    self._set_busy(True)  # a turn is in flight
                    data = ev.get("data") or {}
                    if isinstance(data, dict):
                        idx = data.get("index", 0)
                        self.spoken.block_type.setdefault(idx, "text")
                        for part in data.get("texts") or []:
                            if part:
                                self.spoken.buf += part
                        self.spoken._pump()
                        fed += 1
            # reasoning/tool rows: deliberately skipped (never spoken)
        if fed:
            log(f"snapshot replay: {fed} chunk(s)")

    # ----------------------------------------------------------- floor loop
    def _floor_loop(self) -> None:
        """Release the mic once the reply is over; never latch forever."""
        while not self._stop.is_set():
            if self._floor.is_set():
                now = time.time()
                drained = self.speakers.pending_seconds <= 0.05
                if self._turn_done.is_set() and drained:
                    time.sleep(0.5)  # let final TTS flush
                    if self.speakers.pending_seconds <= 0.05:
                        log("floor: reply done, mic open")
                        self._floor.clear()
                elif (drained
                      and now - self._last_chunk_at > FLOOR_IDLE_S
                      and now - self._floor_since > 5.0):
                    log("floor: idle with no turn/end, releasing (safety)")
                    self._floor.clear()
                elif now - self._floor_since > FLOOR_MAX_S:
                    log("floor: max hold reached, releasing (safety)")
                    self._floor.clear()
            time.sleep(0.1)

    # -------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self.speakers.start()
        self.mic.start()
        threading.Thread(target=self._stt_loop, daemon=True).start()
        threading.Thread(target=self._floor_loop, daemon=True).start()

        def _ws():
            asyncio.run(self.gw.follow(self._on_event, self._on_snapshot,
                                       self._stop))

        threading.Thread(target=_ws, daemon=True).start()
        log(f"ready — speaking into session {self.gw.session_id}; "
            "its visible replies will be spoken")

    def stop(self) -> None:
        self._stop.set()
        self._cancel.set()
        self.stt.close()
        self.mic.close()
        self.speakers.close()


def _pick_session(args) -> str:
    sdir = sessions_dir_for(Path(args.gui_home))
    if sdir is None:
        raise SystemExit(f"no sessions dir under {args.gui_home}; pass --session")
    hit = find_target_session(sdir, args.session)
    if hit is None:
        raise SystemExit("no TTSTT sessions found under the GUI home; pass --session")
    log(f"target session: {hit}")
    return hit


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Coding Voice: speak into the coding chat")
    ap.add_argument("--gui", default="http://127.0.0.1:3080")
    ap.add_argument("--session", default=None,
                    help="session id (default: newest TTSTT session)")
    ap.add_argument("--gui-home", default=r"C:\Users\press\.dsh")
    ap.add_argument("--tts-voice", default="eve")
    ap.add_argument("--tts-language", default="english")
    ap.add_argument("--tts-quantize", default="int4", choices=["int4", "none"])
    ap.add_argument("--stt-repo", default="kyutai/stt-1b-en_fr")
    ap.add_argument("--stt-device", default="cuda",
                    choices=["auto", "cuda", "cpu"])
    ap.add_argument("--spk-device", type=int, default=None)
    ap.add_argument("--mic-device", type=int, default=None)
    ap.add_argument("--utterance-max", type=float, default=120.0,
                    help="max seconds a spoken utterance may run before it "
                         "is sent anyway (default 120)")
    ap.add_argument("--list-voices", action="store_true",
                    help="print the available TTS voices and exit")
    args = ap.parse_args()

    if args.list_voices:
        try:
            from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES as V
            names = list(V)
        except Exception:
            names = ["anna", "vera", "fantine", "eponine", "azelma", "mary",
                     "jane", "eve", "cosette", "caro_davy", "alba", "jean",
                     "charles", "paul", "george", "michael", "marius",
                     "javert", "bill_boerst", "peter_yearsley",
                     "stuart_bell"]
        print("Available --tts-voice names:")
        for n in names:
            print(f"  {n}")
        return 0

    gui = args.gui.rstrip("/")
    authority = re.sub(r"^https?://", "", gui).split("/")[0]

    secret = load_browser_secret(Path(args.gui_home))
    if secret is None:
        raise SystemExit(f"cannot read browser secret from {args.gui_home}")
    session_id = _pick_session(args)
    gw = GatewaySession(gui, session_id, authority, secret)
    log(f"GUI: {gui}  session: {session_id}")

    stt = make_stt(args.stt_repo, args.stt_device)
    tts = make_tts(args.tts_language, args.tts_voice,
                   args.tts_quantize == "int4")

    cv = CodingVoice(stt, tts, gw,
                     mic_device=args.mic_device, spk_device=args.spk_device,
                     utterance_max=args.utterance_max)
    cv.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cv.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
