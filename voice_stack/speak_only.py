"""Speak-only voice bridge: TTS without ears.

Speaks the agent's visible replies with the POCKET TTS model, which runs on
the CPU. No microphone, no STT model, zero GPU — so it coexists with the big
LLM even at 350k context, where the full coding voice (Kyutai STT on the
GPU) does not fit.

    python -X utf8 -m voice_stack.speak_only --workspace 'C:\\...\\DSH_TESTS'

New file — the full coding voice (coding_voice.py) is untouched. Reuses from
it: GatewaySession (the DSH session stream client), SpokenTurn (sentence ->
TTS -> speakers), _make_bridge_tts, SessionFollower (auto-follow), and the
session-workspace helpers.

The event handling below is the text half of CodingVoice's stream handling:
the same watermarks, the same dedup rules, minus everything that exists only
for the microphone (floor, mailbox, endpointing, moods for Pro).
"""

from __future__ import annotations

import argparse
import asyncio
import faulthandler
import os
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from .audio_io import SpeakerPlayback
from .coding_voice import (
    GatewaySession,
    SessionFollower,
    SpokenTurn,
    _make_bridge_tts,
    _resolve_target,
    log,
)
from .handoff import load_browser_secret


class SpeakOnly:
    """Session stream -> visible assistant text -> pocket TTS -> speakers.

    Duck-types the two members SessionFollower needs from a CodingVoice
    (``gw.session_id`` and ``retarget()``) so auto-follow works unchanged.
    """

    def __init__(self, tts, gw: GatewaySession, spk_device=None):
        self.tts = tts
        self.gw = gw
        self.speakers = SpeakerPlayback(sample_rate=tts.sample_rate,
                                        device=spk_device)
        self._stop = threading.Event()
        self._cancel = threading.Event()
        self.spoken = SpokenTurn(tts, self.speakers, None, self._cancel)
        # Per-session watermarks (session-global seq space); the first
        # snapshot of a session re-baselines them (see _on_snapshot).
        self._primed_for: str | None = None
        self._chunk_seq = 0
        self._seen_msg_seqs: set[int] = set()

    # ---------------------------------------------------------- auto-follow
    def retarget(self, sid: str) -> bool:
        """Point at a different session (auto-follow). The follow stream
        notices the id change on its own and reopens for the new session;
        its first snapshot re-arms the watermarks, so nothing from the old
        session leaks in and nothing is spoken twice."""
        if sid == self.gw.session_id:
            return False
        if self.spoken.busy_audio or self.speakers.pending_seconds > 0.05:
            return False  # mid-reply: SessionFollower retries next poll
        log(f"retarget: {self.gw.session_id} -> {sid}")
        self.gw.session_id = sid
        self._chunk_seq = 0
        self._seen_msg_seqs = set()
        self._primed_for = None
        self.spoken.reset_turn()
        return True

    # --------------------------------------------------- session stream in
    def _on_event(self, ev: dict) -> None:
        # Drop leftovers from a session we no longer follow (the stream can
        # deliver frames of the old session before it notices the retarget;
        # the new session's first snapshot re-arms _primed_for).
        if self._primed_for != self.gw.session_id:
            return
        t = ev.get("type")
        if t == "assistant/chunk":
            seq = ev.get("seq")
            if isinstance(seq, int) and seq <= self._chunk_seq:
                # A stream cycle can re-deliver (or reorder) tail chunks of
                # an already-spoken turn; the watermark covers it — drop.
                return
            if isinstance(seq, int) and seq > self._chunk_seq:
                self._chunk_seq = seq
            chunk = (ev.get("data") or {}).get("chunk")
            if isinstance(chunk, dict):
                self.spoken.feed(chunk)
        elif t == "assistant/message":
            # Remember messages fully streamed live so snapshot cycles never
            # speak them again.
            seq = ev.get("seq")
            if isinstance(seq, int):
                self._seen_msg_seqs.add(seq)
                if len(self._seen_msg_seqs) > 1000:
                    self._seen_msg_seqs = set(
                        list(self._seen_msg_seqs)[-200:])
        # turn/end: nothing to release — there is no mic to open.

    def _on_snapshot(self, snap: dict) -> None:
        """Replay what the live stream missed — and only that."""
        records = snap.get("records") or []
        last_msg = None
        last_end = 0
        max_seq = 0
        for rec in records:
            if rec.get("type") != "event":
                continue
            ev = rec.get("event") or {}
            seq = ev.get("seq") or 0
            if not isinstance(seq, int):
                continue
            max_seq = max(max_seq, seq)
            et = ev.get("type")
            if et == "assistant/message":
                last_msg = ev
            elif et == "turn/end":
                last_end = max(last_end, seq)

        header = snap.get("header") or {}
        snap_sid = header.get("id") if isinstance(header, dict) else None
        if snap_sid and snap_sid != self.gw.session_id:
            return  # leftover from a session we no longer follow

        if self._primed_for != self.gw.session_id:
            # First snapshot of THIS session — fresh start, or right after a
            # retarget: re-baseline the watermarks on this session's history
            # and speak none of it.
            self._primed_for = self.gw.session_id
            self._chunk_seq = max_seq
            if last_msg is not None:
                self._seen_msg_seqs = {last_msg.get("seq") or 0}
            return

        # A message that committed while we were away: speak it once.
        # resume_with() additionally word-dedupes against what we already
        # said (covers messages streamed live before the event was seen).
        if last_msg is not None:
            seq = last_msg.get("seq") or 0
            if seq not in self._seen_msg_seqs:
                self._seen_msg_seqs.add(seq)
                msg = (last_msg.get("data") or {}).get("message") or {}
                content = msg.get("content") or []
                text = "".join(b.get("text", "") for b in content
                               if isinstance(b, dict)
                               and b.get("type") == "text")
                if text.strip():
                    self.spoken.resume_with(text)

        # A turn still in flight: feed chunk rows we have not consumed yet.
        fed = 0
        for rec in records:
            if rec.get("type") != "event":
                continue
            ev = rec.get("event") or {}
            if ev.get("type") != "assistant/chunk":
                continue
            seq = ev.get("seq") or 0
            if seq <= self._chunk_seq:
                continue
            self._chunk_seq = seq
            if seq <= last_end:
                # The turn already ended in this snapshot: the committed
                # message (spoken above via resume_with) owns its text.
                continue
            chunk = (ev.get("data") or {}).get("chunk")
            if isinstance(chunk, dict):
                self.spoken.feed(chunk)
                fed += 1
        if fed:
            log(f"snapshot replay: {fed} chunk(s)")

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self.speakers.start()

        def _ws():
            asyncio.run(self.gw.follow(self._on_event, self._on_snapshot,
                                       self._stop))

        threading.Thread(target=_ws, daemon=True, name="follow").start()
        log(f"ready - speaking into session {self.gw.session_id}")

    def stop(self) -> None:
        self._stop.set()
        self._cancel.set()
        self.speakers.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    # Death forensics, same as the full bridge.
    faulthandler.enable()
    threading.excepthook = lambda info: log(
        f"UNCAUGHT exception in thread {info.thread.name}: "
        f"{info.exc_type.__name__}: {info.exc_value}")

    ap = argparse.ArgumentParser(
        description="Speak-only voice bridge (pocket TTS on the CPU; "
                    "no mic, no STT)")
    ap.add_argument("--gui", default="http://127.0.0.1:3080")
    ap.add_argument("--session", default=None,
                    help="session id (default: newest session of the "
                         "target workspace; auto-follow then tracks it)")
    ap.add_argument("--workspace", default=None,
                    help="workspace root whose newest session is the target")
    ap.add_argument("--gui-home", default=None,
                    help="DSH home (default: $DSH_HOME or ~/.dsh)")
    ap.add_argument("--tts-voice", default="anna")
    ap.add_argument("--tts-language", default="english")
    ap.add_argument("--tts-quantize", default="int4",
                    choices=["int4", "none"])
    ap.add_argument("--spk-device", type=int, default=None)
    args = ap.parse_args()
    if args.gui_home is None:
        args.gui_home = str(Path(os.environ.get("DSH_HOME")
                                 or (Path.home() / ".dsh")))

    gui = args.gui.rstrip("/")
    authority = re.sub(r"^https?://", "", gui).split("/")[0]

    secret = load_browser_secret(Path(args.gui_home))
    if secret is None:
        raise SystemExit(f"cannot read browser secret from {args.gui_home}")
    sdir, session_id = _resolve_target(args)
    gw = GatewaySession(gui, session_id, authority, secret)
    log(f"GUI: {gui}  session: {session_id}")

    # Pocket TTS only — the whole point of this mode is CPU-only audio.
    tts_args = SimpleNamespace(
        tts_engine="pocket",
        tts_language=args.tts_language,
        tts_voice=args.tts_voice,
        tts_quantize=args.tts_quantize,
        tts_moods=None,
        qwen3tts_model=None,
        qwen3tts_ref=None,
        qwen3tts_instruct=None,
        qwen3tts_device="cuda:0",
    )
    tts, tts_pro = _make_bridge_tts(tts_args)
    if tts_pro:
        raise SystemExit("speak_only must never load the Pro engine")
    log(f"TTS ready ({tts.sample_rate} Hz, pocket/CPU)")

    bridge = SpeakOnly(tts, gw, spk_device=args.spk_device)
    bridge.start()
    follower = None
    if args.session is None:
        follower = SessionFollower(sdir, bridge)
        follower.start()
        log(f"auto-follow on: newest session of {sdir.name} "
            "(re-resolved as windows change; --session pins)")
    try:
        last_hb = 0.0
        while True:
            time.sleep(1)
            now = time.time()
            if now - last_hb >= 30:
                last_hb = now
                log("heartbeat")
    except KeyboardInterrupt:
        log("KeyboardInterrupt - exiting")
    finally:
        if follower is not None:
            follower.stop()
        bridge.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
