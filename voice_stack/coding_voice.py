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
        [--session SESSION_ID] [--workspace C:\\...\\SomeWorkspace]
        [--gui-home DSH_HOME]

--workspace points the voice at another DSH workspace's newest session
(default: the TTSTT workspace, i.e. this coding chat).

Auto-follow: unless the target is pinned with --session, the bridge keeps
re-resolving the newest-WRITTEN session of the target workspace. A session
file's mtime moves when the session receives activity, so this tracks the
window the user is talking to: a newly opened window becomes newest as soon
as it exists, and an active reply keeps its window newest while the agent
works. Switches are debounced over two polls and never happen mid-turn.
A window that is merely looked at (nothing landed in it yet) is picked up
the moment anything lands in it.
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
    load_browser_secret,
    mint_cookie,
    sessions_dir_for,
)

LOG_PREFIX = "coding-voice:"

# Word-gap endpointing (same shape as voice_dsh): after the last STT word wait
# this long before checking; sentence-final punctuation halves the wait.
WORD_GAP_S = 1.2
WORD_GAP_SENTENCE_S = 0.6
# An unpunctuated fragment is only sent after this much continuous silence
# (it may still be the lead-in to a longer utterance). Punctuated sentences
# send immediately; long ones after LONG_FINAL_SILENCE_S of real silence.
COMMAND_SILENCE_S = 6.0
LONG_FINAL_SILENCE_S = 3.0
SENTENCE_END_PUNCT = {".", "!", "?", "\u2026", "\u3002", "\uff01", "\uff1f"}

# STT model choices ("--stt-model"). The 2.6B model is English-only and the
# most accurate Kyutai release, at the cost of ~2 s more delay and ~7 GB of
# VRAM (weights + mimi + KV).
STT_MODEL_REPOS = {
    "fast": "kyutai/stt-1b-en_fr",
    "accurate": "kyutai/stt-2.6b-en",
}
# Free VRAM needed before the accurate model is even attempted.
ACCURATE_MIN_FREE_VRAM = 8 * 1024**3


def _free_vram_bytes() -> int | None:
    """Free GPU memory in bytes, via nvidia-smi (no CUDA context is created,
    so this never blocks the way a torch call would on a busy GPU).
    Returns None when no NVIDIA GPU / nvidia-smi is available."""
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0]) * 1024 * 1024
    except Exception:
        return None

# Half-duplex: ignore the mic this long after our own audio stops.
VOICE_TAIL_S = 1.0

# At turn end at most this much of the mailbox is transcribed (anything
# older is dropped). 3 min (Martin, Oct 2026): long turns can no longer
# eat the speaker's words. The ring itself holds 5 min.
MAX_MAIL_DRAIN_S = 180.0
FRAME_S = 0.08

# Cycle the follow stream this often (server idle-closes at ~75 s).
STREAM_CYCLE_S = 50.0
# Floor (mic-closed) safety releases.
FLOOR_IDLE_S = 2.5      # audio drained + no chunks for this long
FLOOR_MAX_S = 240.0     # absolute backstop; normal release is turn-end +
                       # drained audio, so this only bites when the audio
                       # pipeline is stuck. Long spoken replies (2-3 min)
                       # must not be cut off mid-sentence.

FLOOR_LOG_ONCE = True


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {LOG_PREFIX} {msg}", flush=True)


def _make_bridge_tts(args):
    """Build the bridge's TTS engine. Returns (tts, tts_pro) where tts_pro
    says whether the LOADED model is the big one — the pause logic below
    keys off the loaded model, not the requested engine, so a pocket
    fallback also speaks live.

    pocket (default): the small local model, behaviour unchanged.
    kyutai16b: the big Kyutai 1.6B model (much better voice, ~1.3 s of
    startup latency per sentence from the audio delay). With moods on
    (the default for this engine) the voice is the EARS p003 speaker and
    each reply is spoken in the mood that fits how the turn went; with
    moods off the voice is --tts-voice, defaulting to the Liverpool cut
    vctk/p277_023. If the big model fails to load for any reason, fall
    back to the pocket model so the bridge still starts.
    """
    quantize = args.tts_quantize != "none"
    moods_choice = getattr(args, "tts_moods", None)
    if args.tts_engine != "kyutai16b":
        if moods_choice is not None:
            log("TTS: --tts-moods needs the Pro engine; "
                "pocket ignores it")
        return make_tts(args.tts_language, args.tts_voice, quantize), False
    moods_on = (moods_choice or "auto") == "auto"
    if moods_on:
        if args.tts_voice and args.tts_voice != "anna":
            log("TTS: moods on, so --tts-voice "
                f"{args.tts_voice} is ignored (the mood voice is the "
                "EARS p003 set)")
        pro_voice = "ears/p003 (23 moods, preloaded at start)"
    else:
        pro_voice = (args.tts_voice if args.tts_voice and args.tts_voice != "anna"
                     else "vctk/p277_023.wav")
    log(f"TTS: loading the big model (kyutai16b, voice={pro_voice}, "
        f"quantized={quantize}) - first load can take a minute...")
    try:
        from .kyutai_tts_engine import KyutaiTTS16B
        if moods_on:
            tts = KyutaiTTS16B(quantize=quantize, moods=True)
        else:
            tts = KyutaiTTS16B(voice=pro_voice, quantize=quantize)
        log(f"TTS: big model ready (kyutai 1.6b, {tts.sample_rate} Hz, "
            f"int8={quantize})")
        return tts, True
    except Exception:
        import traceback
        log("TTS: big model failed to load - "
            "falling back to the pocket TTS")
        traceback.print_exc()
        return make_tts(args.tts_language, args.tts_voice, quantize), False


def _norm(s: str) -> str:
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# Mood picking (Pro engine, moods on): how the turn went -> which of the
# speaker's moods speaks the reply. The reply's own words are the only
# signal the bridge can trust (tool results and reasoning are never
# visible to it), so this is plain word scanning over the text that is
# about to be spoken. Order: distress first (a crash is never buried),
# then the win before the failure it fixed ("the error is fixed" is
# good news), then the plain failures. The word tables are the
# discretionary part of the design - retune freely.
# ---------------------------------------------------------------------------

_MOOD_DISTRESS = ("crash", "broken", "gave up", "give up", "no luck",
                  "stuck", "blocked", "wrecked", "disaster", "ruined")
_MOOD_WIN = ("fixed", "works", "working", "worked", "passed", "passing",
             "done", "success", "succeeded", "resolved", "solved",
             "deployed", "shipped", "sorted", "all good", "ready to go")
_MOOD_FAIL = ("fail", "error", "didn't work", "doesn't work",
              "does not work", "still not", "regression", "can't",
              "cannot", "nope")
_MOOD_PUZZLE = ("not sure", "unclear", "puzzl", "mystery", "no idea",
                "baffl")
_MOOD_FUNNY = ("funny", "hilarious", "bizarre", "weird", "haha")
_MOOD_PROBE = ("looked", "looking", "tried", "trying", "checked",
               "investigat", "digging")


def pick_mood(text: str) -> str:
    """Pick the speaking mood for a turn from the turn's own words.

    Returns one of the EARS mood names (see kyutai_tts_engine.MOODS);
    'neutral' when nothing matches. Deliberately conservative: most
    turns should land on neutral or contentment, and the strong moods
    need strong words.
    """
    t = " " + re.sub(r"[^a-z']+", " ", (text or "").lower()) + " "
    # Negated signals are the opposite signal: "no errors" is good news,
    # "not working" is a failure — strip the negative forms first so
    # "working" and "done" cannot leak through them.
    t = re.sub(r"\b(not working|not done|no crashes?|no errors?|"
               r"no fails?ures?)\b", " ", t)
    if any(w in t for w in _MOOD_DISTRESS):
        return "distress"
    if any(w in t for w in _MOOD_WIN):
        return "pride" if " finally " in t else "contentment"
    if any(w in t for w in _MOOD_FAIL):
        return "disappointment"
    if any(w in t for w in _MOOD_PUZZLE):
        return "confusion"
    if any(w in t for w in _MOOD_FUNNY):
        return "amusement"
    if len(t) > 500 and any(w in t for w in _MOOD_PROBE):
        return "interest"
    return "neutral"


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
        self._tts_gen = 0  # sentences currently mid-generation (see _speak)
        self._gen_lock = threading.Lock()
        self.reset_turn()

    def reset_turn(self) -> None:
        # An interruption (new user utterance) must not eat the previous
        # reply's unspoken tail: flush whatever is still buffered through
        # the TTS queue before clearing the turn state. (getattr: the first
        # call comes from __init__, before these attributes exist — nothing
        # to flush then.)
        if getattr(self, "buf", "").strip():
            try:
                self._pump()
                if self.buf.strip():
                    self._speak(self.buf.strip())
            except Exception as e:  # noqa: BLE001
                log(f"reset_turn flush failed: {type(e).__name__}: {e}")
        self.spoken_text = ""
        self.buf = ""
        self.block_type: dict[int, str] = {}
        self._paused = False
        self._pending: list[str] = []
        self._draining = False

    @property
    def busy_audio(self) -> bool:
        """True while reply audio is still owed (buffered, draining, or in
        the speaker queue). The mic must not be released while this holds,
        otherwise the bridge hears and transcribes its own voice."""
        return self._draining or bool(self._pending)

    @property
    def pending_text(self) -> str:
        """The buffered, not-yet-spoken reply (Pro mode holds the whole
        turn here until the end). Used to pick the reply's mood before
        the drain starts, so the whole reply comes out in one mood."""
        return " ".join(self._pending)

    @property
    def tts_generating(self) -> bool:
        """True while the TTS model is actually generating a sentence.
        STT inference must not run then: a second model on the GPU
        corrupts the in-flight CUDA graph capture ("operation not
        permitted when stream is capturing"), and the sticky capture
        error silences every later sentence in the process."""
        return self._tts_gen > 0

    def set_paused(self, paused: bool) -> None:
        """While paused, sentences are buffered, not spoken; on un-pause the
        buffer is played back (in its own thread, so the event loop is never
        blocked). Only the big (Pro) TTS model is ever paused: it cannot
        share the GPU with the agent's LLM, so its reply is spoken in full
        only after the agent has finished. The pocket model is small
        enough to coexist with the LLM and speaks live, unpaused."""
        self._paused = paused
        if not paused and self._pending:
            self._draining = True
            threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        try:
            while self._pending and not self.cancel.is_set():
                self._speak(self._pending.pop(0))
            # Hold "busy" until the speakers are actually empty.
            while not self.cancel.is_set() \
                    and self.speakers.pending_seconds > 0.05:
                time.sleep(0.1)
        finally:
            self._draining = False

    MAX_PENDING = 500  # text only (cheap) — a runaway safety valve, not a
                       # speech-length limit; drop oldest beyond that

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
        # Raw text straight to the model: Martin judged its own number-
        # reading fine (23/09). text_norm.py stays on disk as a dormant
        # fallback should raw digits ever misbehave again.
        log(f"tts: {sentence[:60]!r}")
        with self._gen_lock:
            self._tts_gen += 1
        try:
            import numpy as _np
            peak, n = 0.0, 0
            for audio in self.tts.stream_sentence(sentence, self.cancel):
                if self.cancel.is_set():
                    break
                if isinstance(audio, (bytes, bytearray)):
                    a = _np.frombuffer(audio, dtype=_np.float32)
                else:
                    a = _np.asarray(audio, dtype=_np.float32)
                if a.size:
                    p = float(_np.abs(a).max())
                    if p > peak:
                        peak = p
                    n += 1
                self.speakers.write(audio)
                if self.on_audio:
                    self.on_audio()
            log(f"tts audio: {n} chunk(s), peak={peak:.3f}")
        except Exception as e:
            log(f"TTS failed: {e}")
        finally:
            with self._gen_lock:
                self._tts_gen -= 1
        self.spoken_text += sentence + " "

    # -- snapshot replay -----------------------------------------------------
    def resume_with(self, full_text: str) -> None:
        """Speak the part of `full_text` not yet spoken.

        The live stream speaks a message in order from its start, so what we
        have already said of THIS message is a prefix of it, sitting at the
        tail of spoken_text (spoken_text also holds words of earlier
        messages of the turn — those must not count). Align on the longest
        such prefix; if there is no head overlap at all (we joined
        mid-message, or spoken_text belongs to other messages) speak the
        whole message: an echo is acceptable, a dropped tail is not.
        """
        words = (full_text or "").split()
        if not words:
            return
        spoken = self.spoken_text.split()
        m, s = len(words), len(spoken)
        k = 0
        if s:
            for kk in range(min(m, s), 0, -1):
                if spoken[s - kk:] == words[:kk]:
                    k = kk
                    break
        if k >= m:
            return  # fully spoken already
        rest = " ".join(words[k:])
        if rest.strip():
            log(f"resuming reply, {m - k} words left")
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
        self.authority = authority
        self.secret = secret

    def _cookie(self) -> str:
        """A freshly signed cookie, minted per use. Cookies expire (2 h
        lifetime) and this bridge runs for days — minting is a local HMAC,
        so freshness is free and the gateway can never see a stale one."""
        return mint_cookie(self.authority, self.secret)

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
            headers={"Cookie": self._cookie()},
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

        The target may change underneath (CodingVoice.retarget, auto-follow):
        each open follows the session id current AT OPEN TIME, and the read
        loop drops the stream as soon as it sees the id move, so a retarget
        reopens within one second instead of waiting for the 50 s cycle.
        """
        import websockets

        delay = 1.0
        while not stop.is_set():
            opened_at = 0.0
            sid = self.session_id  # auto-follow may move this; open for sid
            try:
                async with websockets.connect(
                    self.gui_url.replace("http://", "ws://") + "/api/remote.mux",
                    additional_headers={"Cookie": self._cookie()},
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
                            "address": {"kind": "session", "sessionId": sid},
                            # A wide window so a missed turn/end is visible in
                            # the snapshot even with events after it (the
                            # recovery scan in _on_snapshot needs the headroom).
                            "maxMessages": 200,
                        }}},
                    }))
                    opened_at = time.time()
                    log("follow stream open")
                    delay = 1.0
                    while not stop.is_set():
                        if self.session_id != sid:
                            break  # retargeted: reopen for the new session
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
                import traceback
                for line in traceback.format_exc().strip().splitlines()[-6:]:
                    log(f"    tb: {line}")
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
                 mic_device=None, spk_device=None, utterance_max=120.0,
                 tts_pro: bool = False):
        self.stt = stt
        self.tts = tts
        # True while the LOADED TTS engine is the big GPU model: it cannot
        # share the GPU with the agent's LLM, so the reply is buffered and
        # spoken at turn end. The pocket model coexists with the LLM and
        # speaks live during the turn (the original behaviour).
        self.tts_pro = tts_pro
        # Last mood actually applied to the Pro engine (moods on only).
        self._mood = "neutral"
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
        # Snapshot-replay watermarks (session-global seq space). They are
        # PER-SESSION: _primed_for names the session they belong to, and a
        # retarget re-arms them on the new session's first snapshot, so a
        # 50 s cycle never re-speaks and a switch never crosses streams.
        self._seen_msg_seqs: set[int] = set()
        self._msg_seq = 0
        self._chunk_seq = 0
        self._primed = False
        self._primed_for: str | None = None
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
        self._last_word_at = 0.0
        # Agent-busy: while a turn of THIS session is running, the STT model
        # is OFF (zero GPU for the agent's thinking); the Pro TTS buffers
        # the reply and speaks it at turn end, the pocket TTS speaks it
        # live; the mic goes to a bounded mailbox transcribed after the
        # turn.
        self._busy = False
        self._busy_notified = False
        # Seq (session-global) of the first chunk that armed _busy. A
        # snapshot showing turn/end after that seq means we missed the live
        # turn/end (stream gap) and must release — see _on_snapshot.
        self._busy_since_seq = 0
        self._mailbox: deque = deque(maxlen=3750)  # 5 min of 80 ms frames
        self._mail_left = 0  # queued frames still to transcribe as mailbox
        self._mail_skip = 0  # stale (pre-busy) frames ahead of the mailbox

    # ------------------------------------------------------------ echo guard
    def _note_audio(self) -> None:
        self._audio_end_at = time.time() + self.speakers.pending_seconds

    def _audible(self) -> bool:
        # Echo guard only: drop words while MY voice is (just was) in the
        # air. Do NOT gate on the floor — the user may still be talking
        # right after one of their sentences was sent, and that speech is
        # legitimate input, not echo. (The mic-level _speaking() drop is
        # the primary echo defence; this tail catches stragglers.)
        return time.time() < self._audio_end_at + VOICE_TAIL_S

    # --------------------------------------------------------------- mic in
    def _speaking(self) -> bool:
        """True while our own audio is actually in the air (speaker queue
        non-empty, or still queued). Buffered-but-unsent reply text does
        NOT count: it is not in the room yet, so the user's speech must
        still reach the mailbox."""
        return (self.speakers.pending_seconds > 0.05
                or time.time() < self._audio_end_at)

    # --------------------------------------------------------------- mic in
    def _on_mic_block(self, pcm):
        # 1) Our own audio in the air (reply drain, canned lines): drop.
        #    Checked BEFORE the mailbox, because that audio must never be
        #    buffered — it would be transcribed at turn end and injected
        #    back as 'user speech' (the self-echo loop).
        if self._speaking():
            return
        if self._busy or self.spoken.busy_audio:
            # Keep the GPU 100% free for the agent's thinking and for the
            # reply's TTS drain: no STT feed, hold the user's audio (ring
            # buffer) and transcribe it when the window closes, so nothing
            # they said is lost. (busy_audio = reply audio buffered or
            # draining: an STT frame here would collide with the TTS
            # model's CUDA graph capture and silence the whole reply.
            # Note the _speaking() check above still drops frames while
            # our own reply audio is in the air — echo, not the user.)
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
        """Feed the buffered (busy-period) mic audio into the STT queue.
        Only the most recent MAX_MAIL_DRAIN_S is transcribed: by the time a
        long turn is over, older buffered audio is stale, and replaying
        minutes of it is what made the bridge feel like a loop."""
        n = len(self._mailbox)
        if not n:
            return
        keep = int(MAX_MAIL_DRAIN_S / FRAME_S)
        if n > keep:
            # deque has no slice del — pop the stale head one frame at a time.
            for _ in range(n - keep):
                self._mailbox.popleft()
            n = keep
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
            # STT and TTS must never run on the GPU at the same time: an
            # STT frame while the TTS model is mid-generation corrupts
            # its CUDA graph capture ("operation not permitted when
            # stream is capturing"), and the sticky capture error then
            # silences every later sentence in the process. Hold the
            # frame (and whatever is queued behind it) until the reply
            # audio and any canned line are done.
            while (self.spoken.busy_audio or self.spoken.tts_generating
                   ) and not self._stop.is_set():
                time.sleep(0.2)
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
        self._last_word_at = time.time()
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
        silence = time.time() - self._last_word_at
        age = time.time() - self._utterance_start
        if final and words <= 14:
            pass  # complete short sentence ("what's the weather?")
        elif final and silence >= LONG_FINAL_SILENCE_S:
            pass  # long sentence that ended properly, and a real pause
        elif (not final) and 3 <= words <= 12 and silence >= COMMAND_SILENCE_S:
            # A short command the user has really stopped speaking
            # ("fix the login bug"). No-punctuation fragments shorter than
            # 3 words are never sent this way — they are lead-ins
            # ("so, can ...") that belong to a longer utterance.
            pass
        elif age >= self._utterance_max:
            log(f"utterance max ({self._utterance_max:.0f}s) reached, sending")
        else:
            # Hold the door open: the user may just be pausing mid-sentence
            # or mid-utterance. Recheck after the next gap.
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
        # The gateway occasionally rejects a valid cookie for a few seconds
        # (transient 401s / dropped upgrades observed in the wild), so one
        # failed delivery is retried before the utterance is declared lost.
        err: Exception | None = None
        for attempt in (0, 1, 2):
            if attempt:
                time.sleep(2.0 * attempt)
                log(f"inject retry {attempt}: previous: {err!r}")
            try:
                self.gw.inject(text)
                err = None
                break
            except Exception as e:  # noqa: BLE001
                err = e
        if err is not None:
            self._floor.clear()
            log(f"inject failed after retries: {err}")

    # ---------------------------------------------------------- auto-follow
    def retarget(self, sid: str) -> bool:
        """Point the bridge at a different session (auto-follow).

        Refused while a turn of the current session is in flight — the
        caller (SessionFollower) retries on its next poll. The follow stream
        notices the id change on its own and reopens for the new session
        (<= 1 s); that session's first snapshot re-arms the watermarks, and
        any leftover frame from the old session is dropped by the
        _primed_for / snapshot-header checks. Never double-speaks, never
        crosses streams.
        """
        if sid == self.gw.session_id or self._busy:
            return False
        log(f"retarget: {self.gw.session_id} -> {sid}")
        self.gw.session_id = sid
        # Drop per-session state so the old session can't contaminate the
        # new one: its spoken words would inflate resume_with's dedup count
        # (silently truncating the new session's first replies), and a stale
        # _turn_done / busy watermark would misfire the snapshot recovery.
        self.spoken.reset_turn()
        self._turn_done.clear()
        return True

    # --------------------------------------------------- session stream in
    def _set_busy(self, busy: bool) -> None:
        if busy == self._busy:
            return
        self._busy = busy
        if busy:
            self._busy_notified = False
            if self.tts_pro:
                # Big TTS model: PAUSED for the whole turn. The reply text
                # is buffered (no GPU, no audio) and spoken in full once
                # the turn ends — the big model and the agent's LLM never
                # share the GPU in the same moment.
                self.spoken.set_paused(True)
                log("agent busy — STT off (GPU free), mic to mailbox, "
                    "reply buffered, spoken at turn end")
            else:
                # Pocket TTS: small enough to coexist with the agent's
                # LLM, so the reply is spoken live while the agent works.
                # STT stays OFF (GPU for the agent) and the mic feeds the
                # mailbox; while the reply audio is in the air the mic
                # frames are dropped (echo guard), and in the gaps the user
                # can talk into the mailbox.
                log("agent busy — STT off (GPU free), mic to mailbox, "
                    "reply spoken live")
        else:
            self._busy_since_seq = 0
            self._apply_mood()
            self.spoken.set_paused(False)
            if self.tts_pro:
                def _deferred_mailbox():
                    # Transcribe the mailbox only AFTER the reply audio has
                    # finished: STT and TTS must never share the GPU (their
                    # CUDA streams collide mid graph-capture — see the
                    # 09:53 "operation not permitted when stream is capturing"
                    # hit), and by the time a long reply has played, older
                    # mailbox audio is stale anyway.
                    try:
                        while (self.spoken.busy_audio
                               or self.speakers.pending_seconds > 0.05):
                            if self._stop.is_set():
                                return
                            time.sleep(0.2)
                        self._drain_mailbox()
                    except Exception as e:  # noqa: BLE001
                        log(f"mailbox drain failed: {type(e).__name__}: {e}")

                threading.Thread(target=_deferred_mailbox,
                                 daemon=True).start()
            else:
                # The reply was spoken live, so drain right away. A drain
                # failure must never leave the stream dead or the busy
                # state stuck: log and continue, the mic hands back either
                # way. The STT worker still holds its frames while the TTS
                # model is mid-sentence (tts_generating gate), so this
                # never collides with a live sentence.
                try:
                    self._drain_mailbox()
                except Exception as e:  # noqa: BLE001
                    log(f"mailbox drain failed: {type(e).__name__}: {e}")
            log("agent done — mic returns after reply audio")

    def _apply_mood(self) -> None:
        """Moods on (Pro engine only): pick this reply's mood from the
        reply's own words and switch the voice BEFORE the buffered reply
        drains, so the whole reply comes out in one mood. The mood
        carries over otherwise: an unknown or not-yet-loaded mood is a
        no-op, and the engine simply keeps its current condition set.
        Pocket has no moods (no set_mood), so this is a no-op there.
        """
        set_mood = getattr(self.tts, "set_mood", None)
        if not (self.tts_pro and set_mood):
            return
        text = self.spoken.pending_text or self.spoken.spoken_text
        mood = pick_mood(text)
        if set_mood(mood):
            if mood != self._mood:
                log(f"mood: {self._mood} -> {mood}")
            self._mood = mood
        elif mood != "neutral":
            log(f"mood: want {mood} but it is not loaded yet; "
                f"keeping {self._mood}")

    def _on_event(self, ev: dict) -> None:
        # Drop leftovers from a session we no longer follow (the follow
        # stream can deliver one or two frames of the old session before it
        # notices the retarget; _primed_for lags the switch until the new
        # session's first snapshot re-arms it).
        if self._primed_for != self.gw.session_id:
            return
        t = ev.get("type")
        if t == "assistant/chunk":
            if not self._busy:
                self._busy_since_seq = ev.get("seq") or 0
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

        header = snap.get("header") or {}
        snap_sid = header.get("id") if isinstance(header, dict) else None
        if snap_sid and snap_sid != self.gw.session_id:
            return  # leftover from a stream of a session we no longer follow

        if self._primed_for != self.gw.session_id:
            # First snapshot of THIS session — a fresh start, or right after
            # a retarget: re-baseline the watermarks on this session's
            # history, speak nothing of it. (Per-session seq spaces make
            # carrying the old session's watermarks over meaningless.)
            self._primed = True
            self._primed_for = self.gw.session_id
            if last_msg is not None:
                self._msg_seq = last_msg.get("seq") or 0
                self._seen_msg_seqs = {self._msg_seq}
            self._chunk_seq = max_seq
            return

        # Recovery: the live stream can gap (error, 50 s cycle, retarget)
        # and drop the turn/end that releases the busy state. If this
        # snapshot already contains a turn/end after the event that armed
        # us, the turn is over — release now instead of waiting for a
        # turn/end that will never come.
        if self._busy and self._busy_since_seq:
            for rec in records:
                if rec.get("type") != "event":
                    continue
                ev = rec.get("event") or {}
                if ev.get("type") != "turn/end":
                    continue
                s = ev.get("seq") or 0
                if isinstance(s, int) and s > self._busy_since_seq:
                    log("snapshot: missed turn/end — releasing (recovery)")
                    self._turn_done.set()
                    self._set_busy(False)
                    break

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
                        # A new chunk means a turn is in flight: re-arm busy
                        # exactly like the live path does, so a misfired
                        # release can't leave the mic live mid-reply.
                        if not self._busy:
                            self._busy_since_seq = seq
                            self._set_busy(True)
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
                    if not self._busy:
                        self._busy_since_seq = seq
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
                drained = (self.speakers.pending_seconds <= 0.05
                           and not self.spoken.busy_audio)
                if self._turn_done.is_set() and drained:
                    time.sleep(0.5)  # let final TTS flush
                    drained = (self.speakers.pending_seconds <= 0.05
                               and not self.spoken.busy_audio)
                    if drained:
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


def _project_key(cwd: str) -> str:
    """DSH's projectKey (session-persistence-jsonl format.ts), verbatim:
    filesystem/drive separators collapse to '-', [A-Za-z0-9._-] are kept,
    every other character (including '~') becomes ~XXXX hex, leading
    hyphens are stripped, and the slug is capped at 251 chars."""
    readable = []
    sep_run = False
    for ch in cwd:
        if ch in "/\\:":
            if not sep_run:
                readable.append("-")
            sep_run = True
        elif ch != "~" and (ch.isascii() and ch.isalnum() or ch in "._-"):
            readable.append(ch)
            sep_run = False
        else:
            readable.append("~%04X" % ord(ch))
            sep_run = False
    slug = "".join(readable).lstrip("-") or "root"
    return "--" + slug[:251] + "--"


def _workspace_sessions_dir(gui_home: str, workspace: str) -> Path | None:
    """Sessions dir for an arbitrary workspace root, under gui_home."""
    base = Path(gui_home) / "sessions"
    cand = base / _project_key(os.path.normpath(workspace))
    return cand if cand.is_dir() else None


def newest_session(sdir: Path) -> str | None:
    """Newest session-<...> directory in a workspace session store.

    Non-session artifacts sharing the store (e.g. handoff-<ts> task folders
    from the voice handoff) are ignored: auto-follow should track the
    user's chat windows, not one-shot tasks.

    Ties (filesystem timestamp granularity) break deterministically by
    session id; in practice real activity is far more than a granularity
    tick apart, so this only matters for synthetic same-instant writes.
    """
    best: tuple[float, str] | None = None
    try:
        entries = list(sdir.iterdir())
    except OSError:
        return None
    for sd in entries:
        if not sd.name.startswith("session-"):
            continue
        try:
            if not sd.is_dir():
                continue
        except OSError:
            continue
        mt = 0.0
        try:
            for f in sd.iterdir():
                if f.is_file():
                    mt = max(mt, f.stat().st_mtime)
        except OSError:
            pass
        key = (mt, sd.name)
        if best is None or key > best:
            best = key
    return best[1] if best else None


class SessionFollower(threading.Thread):
    """Auto-follow: keep the bridge pointed at the newest-WRITTEN session of
    the target workspace.

    A session file's mtime moves only when the session receives activity, so
    'newest written' tracks the window the user is talking to: a freshly
    opened window becomes newest as soon as it exists, and an active reply
    keeps its window newest while the agent works. A window that is merely
    looked at (nothing landed in it yet) is picked up the moment anything
    lands in it — a true focus signal would need a browser-side heartbeat,
    which the DSH gateway does not expose.

    A switch needs two consecutive agreeing polls (debounce against an mtime
    blip) and is refused by retarget() mid-turn (retried on the next poll).
    """

    def __init__(self, sdir: Path, cv: "CodingVoice",
                 interval: float = 3.0, stable_polls: int = 2):
        super().__init__(daemon=True, name="session-follower")
        self._sdir = sdir
        self._cv = cv
        self._interval = interval
        self._stable = stable_polls
        self._stop_ev = threading.Event()

    def stop(self) -> None:
        self._stop_ev.set()

    def run(self) -> None:
        polls = 0
        while not self._stop_ev.is_set():
            if self._stop_ev.wait(self._interval):
                return
            try:
                hit = newest_session(self._sdir)
            except OSError:
                continue
            current = self._cv.gw.session_id
            if hit and hit != current:
                polls += 1
                if polls >= self._stable:
                    if self._cv.retarget(hit):
                        log(f"auto-follow: speaking into {hit}")
                        polls = 0
                    # else: mid-turn — keep the count, retry next poll
            else:
                polls = 0


def _resolve_target(args) -> tuple[Path, str]:
    """(sessions dir, target session id) for the CLI args.

    The dir is what SessionFollower re-scans when auto-follow is on; the id
    is what the bridge starts on. An explicit --session pins the target and
    disables auto-follow (main() checks args.session).
    """
    if args.workspace:
        sdir = _workspace_sessions_dir(args.gui_home,
                                       os.path.normpath(args.workspace))
        if sdir is None:
            raise SystemExit(
                f"no sessions found for workspace {args.workspace!r} under "
                f"{args.gui_home}/sessions; check the path (or use --session)")
    else:
        sdir = sessions_dir_for(Path(args.gui_home))
        if sdir is None:
            raise SystemExit(f"no sessions dir under {args.gui_home}; "
                             "pass --session or --workspace")
    if args.session:
        hit = args.session
    else:
        hit = newest_session(sdir)
        if hit is None:
            raise SystemExit("no sessions found for that workspace; "
                              "pass --session")
    log(f"target session: {hit}")
    return sdir, hit


def main() -> int:
    import argparse
    import atexit
    import faulthandler

    # Death forensics: the bridge has died once with no trace at all.
    # faulthandler dumps all threads on fatal signals (segfault & co);
    # atexit marks a clean exit; the thread hook catches runaway threads
    # whose exception would otherwise vanish into a daemon thread.
    faulthandler.enable()
    atexit.register(lambda: log("process exiting (clean exit path)"))
    threading.excepthook = lambda info: log(
        f"UNCAUGHT exception in thread {info.thread.name}: "
        f"{info.exc_type.__name__}: {info.exc_value}")

    ap = argparse.ArgumentParser(description="Coding Voice: speak into the coding chat")
    ap.add_argument("--gui", default="http://127.0.0.1:3080")
    ap.add_argument("--session", default=None,
                    help="session id (default: newest session of the target "
                         "workspace)")
    ap.add_argument("--workspace", default=None,
                    help="workspace root whose newest session is the target "
                         "(default: the TTSTT workspace)")
    ap.add_argument("--gui-home", default=None,
                    help="DSH home (default: $DSH_HOME or ~/.dsh)")
    ap.add_argument("--tts-engine", default="pocket",
                    choices=["pocket", "kyutai16b"],
                    help="TTS backend: pocket = the small local model "
                         "(default, unchanged); kyutai16b = the big Kyutai "
                         "TTS 1.6B model, int8-quantised on the GPU "
                         "(~2 GB extra VRAM, slower to load, much better "
                         "voice). English only.")
    ap.add_argument("--tts-voice", default="anna")
    ap.add_argument("--tts-language", default="english")
    ap.add_argument("--tts-quantize", default="int4",
                    choices=["int4", "int8", "none"],
                    help="pocket: int4 (default) or none; kyutai16b: int8 "
                         "(default; 'int4' is accepted and means int8) or "
                         "none (full precision, ~3.7 GB VRAM)")
    ap.add_argument("--tts-moods", default=None, choices=["auto", "off"],
                    help="Pro engine only. auto (default) = each reply is "
                         "spoken in the EARS p003 mood that fits how the "
                         "turn went (23 moods, preloaded at start, neutral "
                         "fallback); off = one fixed voice: --tts-voice, "
                         "defaulting to the Liverpool cut vctk/p277_023")
    ap.add_argument("--stt-model", default="fast",
                    choices=sorted(STT_MODEL_REPOS),
                    help="STT model. fast = stt-1b-en_fr, ~0.5 s delay "
                         "(default). accurate = stt-2.6b-en, noticeably more "
                         "accurate but ~2.5 s delay and ~7 GB of free VRAM; "
                         "first use downloads the model.")
    ap.add_argument("--stt-repo", default=None,
                    help="override: exact Hugging Face repo for the STT "
                         "model (beats --stt-model)")
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
    if args.gui_home is None:
        args.gui_home = str(Path(os.environ.get("DSH_HOME")
                                 or (Path.home() / ".dsh")))

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
    sdir, session_id = _resolve_target(args)
    gw = GatewaySession(gui, session_id, authority, secret)
    log(f"GUI: {gui}  session: {session_id}")

    stt_repo = args.stt_repo or STT_MODEL_REPOS[args.stt_model]
    log(f"STT model: {args.stt_model} ({stt_repo})")
    if args.stt_model == "accurate" and not args.stt_repo \
            and args.stt_device in ("cuda", "auto"):
        free = _free_vram_bytes()
        if free is not None and free < ACCURATE_MIN_FREE_VRAM:
            raise SystemExit(
                f"the 'accurate' STT model (stt-2.6b-en) needs ~7 GB of "
                f"free VRAM but only {free / 1024 / 1024:.0f} MB is free "
                "right now. Free up the GPU (e.g. stop the big LLM "
                "server) or start with --stt-model fast.")
    stt = make_stt(stt_repo, args.stt_device)
    tts, tts_pro = _make_bridge_tts(args)

    cv = CodingVoice(stt, tts, gw,
                     mic_device=args.mic_device, spk_device=args.spk_device,
                     utterance_max=args.utterance_max, tts_pro=tts_pro)
    cv.start()
    follower = None
    if args.session is None:
        follower = SessionFollower(sdir, cv)
        follower.start()
        log(f"auto-follow on: newest session of {sdir.name} "
            "(re-resolved as windows change; --session pins)")
    try:
        # Keep the log mtime fresh so an outside watcher (ensure_coding_voice)
        # can tell a live-but-idle bridge from a dead one without asking the
        # process itself: "log written within ~90 s" == alive.
        last_hb = 0.0
        while True:
            time.sleep(1)
            now = time.time()
            if now - last_hb >= 30:
                last_hb = now
                log("heartbeat")
    except KeyboardInterrupt:
        log("KeyboardInterrupt - console ctrl-C/close event, exiting")
    finally:
        if follower is not None:
            follower.stop()
        cv.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
