"""Probe: list turn/* + user/message events with timestamps from the
session journal (via a one-shot follow snapshot)."""
import asyncio
import datetime
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from voice_stack.handoff import load_browser_secret, mint_cookie  # noqa: E402

SID = "session-cbca2f75-7a57-4c7f-9f94-ea6a99bb0f2b"


def ts(ms):
    return datetime.datetime.fromtimestamp(ms / 1000).strftime("%H:%M:%S") \
        if ms else "?"


async def main():
    import websockets

    secret = load_browser_secret(Path(r"C:\Users\press\.dsh"))
    cookie = mint_cookie("127.0.0.1:3080", secret)
    async with websockets.connect(
        "ws://127.0.0.1:3080/api/remote.mux",
        additional_headers={"Cookie": cookie},
        max_size=200 * 1024 * 1024,
        ping_interval=15,
    ) as ws:
        await ws.send(json.dumps({
            "type": "open",
            "streamId": str(uuid.uuid4()),
            "endpoint": "session/follow",
            "payload": {"args": {"request": {
                "address": {"kind": "session", "sessionId": SID},
                "maxMessages": 2000,
            }}},
        }))
        raw = await asyncio.wait_for(ws.recv(), timeout=20)
        frame = json.loads(raw)
        entry = frame.get("value", frame)
        records = entry.get("records") or []
        for r in records:
            if r.get("type") != "event":
                continue
            ev = r.get("event") or {}
            t = ev.get("type")
            if t not in ("turn/start", "turn/end", "user/message",
                         "step/start", "step/end"):
                continue
            extra = ""
            if t == "user/message":
                data = ev.get("data") or {}
                content = data.get("content") or []
                txt = " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text")
                extra = " text=" + repr(txt[:50])
            print(f"{ts(ev.get('time'))}  {t:12s} seq={ev.get('seq')}{extra}")


asyncio.run(main())
