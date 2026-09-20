"""Regression tests for coding-voice auto-follow (session retargeting).

Runs against the live DSH gateway on 127.0.0.1:3080 (must be up, with a
valid browser secret in C:\\Users\\press\\.dsh\\.credentials.yaml). No mic,
no GPU, no audio — gateway/client logic only.

    python -X utf8 tests\\coding_voice_retarget_test.py

Exits 0 when every part passes.
"""
from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from voice_stack.coding_voice import (  # noqa: E402
    CodingVoice,
    GatewaySession,
    SessionFollower,
    newest_session,
)
from voice_stack.handoff import load_browser_secret, sessions_dir_for  # noqa: E402

GUI = "http://127.0.0.1:3080"
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def all_sessions() -> list[str]:
    """TTSTT workspace session ids, newest written first."""
    sdir = sessions_dir_for(Path(r"C:\Users\press\.dsh"))
    assert sdir is not None, "no TTSTT sessions dir under .dsh"
    out = []
    for sd in sdir.iterdir():
        if not (sd.name.startswith("session-") and sd.is_dir()):
            continue
        mt = 0.0
        for f in sd.iterdir():
            if f.is_file():
                mt = max(mt, f.stat().st_mtime)
        out.append((mt, sd.name))
    out.sort(reverse=True)
    return [n for _, n in out]


# ---------------------------------------------------------------------------
# Part 1: newest_session() — the mtime heuristic, filtered to session-*
# ---------------------------------------------------------------------------
def test_newest_session() -> None:
    print("part 1: newest_session()")
    tmp = Path(tempfile.mkdtemp(prefix="cvtest-"))
    try:
        (tmp / "session-aaa").mkdir()
        (tmp / "session-aaa" / "session.jsonl.zstd").write_text("a")
        (tmp / "session-bbb").mkdir()
        (tmp / "session-bbb" / "session.jsonl.zstd").write_text("b")
        (tmp / "handoff-20260101-000000").mkdir()  # must be ignored
        (tmp / "handoff-20260101-000000" / "session.jsonl.zstd").write_text("x")
        time.sleep(0.05)
        (tmp / "session-bbb" / "session.jsonl.zstd").touch()  # newest

        got = newest_session(tmp)
        check("picks the newest-written session", got == "session-bbb",
              f"got {got!r}")
        # A fresh handoff artifact must not steal the target.
        (tmp / "handoff-20260101-000001").mkdir()
        (tmp / "handoff-20260101-000001" / "session.jsonl.zstd").write_text("y")
        got2 = newest_session(tmp)
        check("ignores handoff-* folders", got2 == "session-bbb",
              f"got {got2!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Part 2: SessionFollower — debounce, busy gate, retry after unblock
# ---------------------------------------------------------------------------
def _fake_cv(session_id: str, busy: list[bool]):
    state = {"id": session_id}

    def retarget(sid: str) -> bool:
        if busy[0]:
            return False
        state["id"] = sid
        return True

    cv = SimpleNamespace(gw=SimpleNamespace(session_id=property(
        lambda self: state["id"])))
    # gw.session_id must track state["id"]; rewrap:
    cv.gw = SimpleNamespace(**{})
    class _GW:
        @property
        def session_id(self) -> str:
            return state["id"]
        @session_id.setter
        def session_id(self, v: str) -> None:
            state["id"] = v
    cv.gw = _GW()
    cv.retarget = retarget
    cv.calls: list[str] = []
    real = retarget

    def recording(sid: str) -> bool:
        ok = real(sid)
        if ok:
            cv.calls.append(sid)
        return ok
    cv.retarget = recording
    return cv


def test_follower() -> None:
    print("part 2: SessionFollower")
    tmp = Path(tempfile.mkdtemp(prefix="cvtest-"))
    try:
        (tmp / "session-aaa").mkdir()
        (tmp / "session-aaa" / "session.jsonl.zstd").write_text("a")
        (tmp / "session-bbb").mkdir()
        (tmp / "session-bbb" / "session.jsonl.zstd").write_text("b")
        time.sleep(0.05)
        (tmp / "session-bbb" / "session.jsonl.zstd").touch()  # real gap

        # Debounce: two agreeing polls are required before the switch.
        cv = _fake_cv("session-aaa", [False])
        fol = SessionFollower(tmp, cv, interval=0.1, stable_polls=2)
        fol.start()
        time.sleep(0.8)  # ~8 polls
        fol.stop()
        fol.join(timeout=2)
        check("switches after 2 agreeing polls",
              cv.calls == ["session-bbb"], f"calls={cv.calls}")

        # Busy: refused every poll while busy, switches once unblocked.
        busy = [True]
        cv2 = _fake_cv("session-aaa", busy)
        fol2 = SessionFollower(tmp, cv2, interval=0.1, stable_polls=1)
        fol2.start()
        time.sleep(0.35)
        n_while_busy = len(cv2.calls)
        busy[0] = False
        time.sleep(0.35)
        fol2.stop()
        fol2.join(timeout=2)
        check("refused while busy", n_while_busy == 0,
              f"calls while busy={cv2.calls}")
        check("switches once unblocked", cv2.calls == ["session-bbb"],
              f"calls={cv2.calls}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Part 3: CodingVoice retarget + event/snapshot guards (no audio hardware)
# ---------------------------------------------------------------------------
def _bare_cv(session_id: str) -> CodingVoice:
    """A CodingVoice with only the state the retarget paths touch — no mic,
    no speakers, no GPU."""
    cv = CodingVoice.__new__(CodingVoice)
    cv.gw = SimpleNamespace(session_id=session_id)
    cv._seen_msg_seqs: set[int] = set()
    cv._msg_seq = 0
    cv._chunk_seq = 0
    cv._primed = False
    cv._primed_for = None
    cv._busy = False
    cv._turn_done = threading.Event()
    spoken = SimpleNamespace(
        feed=lambda chunk: spoken.feeded.append(chunk),
        resume_with=lambda text: spoken.texts.append(text),
        set_paused=lambda paused: None,
        feeded=[], texts=[],
    )
    cv.spoken = spoken
    return cv


def _snap(sid: str, seq: int = 5) -> dict:
    return {
        "type": "snapshot",
        "header": {"id": sid, "version": 0},
        "cursor": seq,
        "records": [
            {"type": "event", "event": {
                "type": "assistant/message", "seq": seq, "time": 0,
                "data": {"message": {"content": []}}}},
        ],
        "hasMore": False,
    }


def test_guards() -> None:
    print("part 3: retarget + stream guards")
    A, B = "session-aaa", "session-bbb"
    cv = _bare_cv(A)
    cv._busy = True
    check("retarget refused while busy", cv.retarget("session-xyz") is False)
    cv._busy = False
    check("retarget moves the target",
          cv.retarget(B) is True and cv.gw.session_id == B)
    check("retarget refuses no-op", cv.retarget(B) is False)

    # A stale event from the OLD session (A) must be dropped while we
    # follow B (primed_for is still un-armed until B's snapshot arrives).
    cv._on_event({"type": "assistant/chunk", "seq": 99,
                  "data": {"chunk": {"type": "text-delta", "text": "old"}}})
    check("stale event of old session dropped", cv.spoken.feeded == [])

    # A stale SNAPSHOT of session A (leftover of its stream) is dropped.
    cv._on_snapshot(_snap(A, seq=50))
    check("stale snapshot of old session dropped", cv._primed_for is None)

    # B's first snapshot primes the watermarks.
    cv._on_snapshot(_snap(B, seq=7))
    check("new session snapshot primes watermarks",
          cv._primed_for == B and cv._chunk_seq == 7
          and cv._seen_msg_seqs == {7})

    # A live event of B now passes the guard.
    cv._on_event({"type": "assistant/chunk", "seq": 8,
                  "data": {"chunk": {"type": "text-delta", "text": "hi"}}})
    check("live event of new session accepted",
          any("hi" in str(c) for c in cv.spoken.feeded))

    # A late snapshot of A (its 50 s cycle) must not re-prime us onto A.
    cv._on_snapshot(_snap(A, seq=50))
    check("late snapshot of old session ignored",
          cv._primed_for == B and cv._chunk_seq == 8
          and cv._seen_msg_seqs == {7})


# ---------------------------------------------------------------------------
# Part 4: GatewaySession.follow — a retarget reopens the stream promptly
# ---------------------------------------------------------------------------
def test_follow_retarget() -> None:
    print("part 4: GatewaySession.follow retarget (live gateway)")
    sessions = all_sessions()
    if len(sessions) < 2:
        print(f"  [SKIP] need two TTSTT sessions (have {len(sessions)})")
        return
    newest, other = sessions[0], sessions[-1]
    secret = load_browser_secret(Path(r"C:\Users\press\.dsh"))
    assert secret is not None, "no browser secret in .dsh"
    gw = GatewaySession(GUI, other, "127.0.0.1:3080", secret)

    events: list[str] = []
    stop = threading.Event()

    def runner() -> None:
        def on_event(ev):
            events.append(ev.get("type", "?"))

        def on_snapshot(snap):
            sid = (snap.get("header") or {}).get("id")
            events.append("snap:" + str(sid))

        asyncio.run(gw.follow(on_event, on_snapshot, stop))

    t = threading.Thread(target=runner, daemon=True)
    t.start()

    def wait_snap(sid: str, timeout: float) -> bool:
        marker = f"snap:{sid}"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if marker in events:
                return True
            time.sleep(0.05)
        return False

    t0 = time.time()
    ok1 = wait_snap(other, 15)
    check("first stream opened", ok1)
    if not ok1:
        stop.set()
        t.join(timeout=5)
        return
    open1_at = time.time()

    gw.session_id = newest  # what CodingVoice.retarget() does
    t_change = time.time()
    ok2 = wait_snap(newest, 10)
    check("retarget reopens for the new session", ok2)
    if ok2:
        check("reopen was prompt (< 4 s), not the 50 s cycle",
              time.time() - t_change < 4.0,
              f"{time.time() - t_change:.1f}s")
    stop.set()
    t.join(timeout=10)


def main() -> int:
    for part in (test_newest_session, test_follower, test_guards,
                 test_follow_retarget):
        try:
            part()
        except Exception:
            traceback.print_exc()
            FAILS.append(part.__name__ + " (exception)")
        print()
    if FAILS:
        print(f"RESULT: {len(FAILS)} FAILURE(S): {FAILS}")
        return 1
    print("RESULT: all parts passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
