"""One-shot probe: open a session/follow stream and dump the snapshot frame
shape (top-level keys, header fields) so retargeting code can rely on them."""
import asyncio
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from voice_stack.handoff import load_browser_secret, mint_cookie  # noqa: E402

GUI = "http://127.0.0.1:3080"
SID = sys.argv[1] if len(sys.argv) > 1 else "session-cbca2f75-7a57-4c7f-9f94-ea6a99bb0f2b"


async def main():
    import websockets

    secret = load_browser_secret(Path(r"C:\Users\press\.dsh"))
    cookie = mint_cookie("127.0.0.1:3080", secret)
    stream_id = str(uuid.uuid4())
    async with websockets.connect(
        "ws://127.0.0.1:3080/api/remote.mux",
        additional_headers={"Cookie": cookie},
        max_size=200 * 1024 * 1024,
        ping_interval=15,
    ) as ws:
        await ws.send(json.dumps({
            "type": "open",
            "streamId": stream_id,
            "endpoint": "session/follow",
            "payload": {"args": {"request": {
                "address": {"kind": "session", "sessionId": SID},
                "maxMessages": 1,
            }}},
        }))
        for _ in range(3):
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            frame = json.loads(raw)
            entry = frame.get("value") if isinstance(frame, dict) else frame
            if not isinstance(entry, dict):
                print("frame (non-dict):", str(frame)[:300])
                continue
            et = entry.get("type")
            if et == "snapshot":
                print("snapshot top-level keys:", sorted(entry.keys()))
                hdr = entry.get("header")
                print("header:", json.dumps(hdr)[:500])
                recs = entry.get("records") or []
                print("records:", len(recs))
                if recs:
                    r0 = recs[0]
                    print("first record keys:", sorted(r0.keys()), "type:", r0.get("type"))
                # Show one event record's event keys if present
                for r in recs[:5]:
                    ev = r.get("event")
                    if isinstance(ev, dict):
                        print("event keys:", sorted(ev.keys()), "type:", ev.get("type"))
                        break
                return
            else:
                print("frame type:", et, str(entry)[:200])
                return


asyncio.run(main())
