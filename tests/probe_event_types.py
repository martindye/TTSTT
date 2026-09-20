"""Probe: open a follow stream with a large history window and list the
distinct event types (plus the last turn-ish events) in the snapshot."""
import asyncio
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from voice_stack.handoff import load_browser_secret, mint_cookie  # noqa: E402

SID = "session-cbca2f75-7a57-4c7f-9f94-ea6a99bb0f2b"
MAXMSG = int(sys.argv[1]) if len(sys.argv) > 1 else 500


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
                "maxMessages": MAXMSG,
            }}},
        }))
        raw = await asyncio.wait_for(ws.recv(), timeout=15)
        frame = json.loads(raw)
        entry = frame.get("value", frame)
        records = entry.get("records") or []
        print(f"cursor={entry.get('cursor')} hasMore={entry.get('hasMore')} "
              f"records={len(records)}")
        types: dict = {}
        for r in records:
            if r.get("type") != "event":
                continue
            ev = r.get("event") or {}
            t = ev.get("type")
            types[t] = types.get(t, 0) + 1
        for t, n in sorted(types.items(), key=lambda kv: -kv[1]):
            print(f"  {n:5d}  {t}")
        print("--- last 12 events ---")
        events = [(r.get("event") or {}) for r in records
                  if r.get("type") == "event"]
        for ev in events[-12:]:
            print(f"  seq={ev.get('seq')} t={ev.get('type')} "
                  f"time={ev.get('time')}")


asyncio.run(main())
