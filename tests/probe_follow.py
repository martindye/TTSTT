"""Probe: open a session/follow WS stream on the GUI gateway for one session
and print the shape of the frames it receives. One-shot diagnostic."""
import asyncio
import base64
import hashlib
import hmac
import json
import sys
import time
import uuid

import yaml  # noqa: F401  (availability probe)
import websockets

GUI = "http://127.0.0.1:3080"
WS_URI = "ws://127.0.0.1:3080/api/remote.mux"
SID = sys.argv[1] if len(sys.argv) > 1 else "session-d3ab4151-4238-4d8e-a05c-b6dbf5fd7990"
DURATION_S = int(sys.argv[2]) if len(sys.argv) > 2 else 45


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def load_secret():
    doc = yaml.safe_load(open(r"C:\Users\press\.dsh\.credentials.yaml", encoding="utf-8"))
    sec = doc["records"]["client-connection/browser-session"]["payload"]["secret"]
    return base64.urlsafe_b64decode(sec + "=" * (-len(sec) % 4))


def cookie():
    authority = "127.0.0.1:3080"
    name = "dsh-auth-" + b64url(hashlib.sha256(authority.encode()).digest())
    now = int(time.time() * 1000)
    payload = {"version": 1, "authority": authority, "issuedAt": now, "expiresAt": now + 3600_000}
    body = b64url(json.dumps(payload, separators=(",", ":")).encode())
    sig = b64url(hmac.new(load_secret(), body.encode(), hashlib.sha256).digest())
    return f"{name}=v1.{body}.{sig}"


def describe(frame):
    if not isinstance(frame, dict):
        return f"unparsed: {str(frame)[:120]}"
    ftype = frame.get("type")
    if ftype in ("item", "end", "error"):
        v = frame.get("value")
        if isinstance(v, dict) and v.get("type") == "snapshot":
            recs = v.get("records") or []
            kinds = {}
            for r in recs[-10:]:
                t = r.get("type")
                if t == "event":
                    t += "/" + str((r.get("event") or {}).get("type"))
                kinds[t] = kinds.get(t, 0) + 1
            return f"snapshot cursor={v.get('cursor')} hasMore={v.get('hasMore')} last-records={kinds}"
        # live follow frames: either the mux value or the frame itself is
        # {type:'event', event:{type,seq,data}}
        cand = v if isinstance(v, dict) else frame
        if isinstance(cand, dict) and cand.get("type") == "event" and isinstance(cand.get("event"), dict):
            ev = cand["event"]
            extra = ""
            if ev.get("type") == "assistant/chunk":
                chunk = ((ev.get("data") or {}).get("chunk")) or {}
                extra = f" chunk.type={chunk.get('type')} idx={chunk.get('index')}"
            return f"event seq={ev.get('seq')} type={ev.get('type')}{extra}"
        if isinstance(v, dict) and v.get("type") == "event" and isinstance(v.get("event"), dict):
            ev = v["event"]
            extra = ""
            if ev.get("type") == "assistant/chunk":
                chunk = ((ev.get("data") or {}).get("chunk")) or {}
                extra = f" chunk.type={chunk.get('type')} idx={chunk.get('index')}"
            return f"value-event seq={ev.get('seq')} type={ev.get('type')}{extra}"
        return f"{ftype}: {json.dumps(frame)[:160]}"
    return f"frame {json.dumps(frame)[:160]}"


async def main():
    ws = await websockets.connect(WS_URI, additional_headers={"Cookie": cookie()},
                                  max_size=200 * 1024 * 1024)
    stream_id = str(uuid.uuid4())
    await ws.send(json.dumps({
        "type": "open",
        "streamId": stream_id,
        "endpoint": "session/follow",
        "payload": {"args": {"request": {
            "address": {"kind": "session", "sessionId": SID},
        }}},
    }))
    print(f"opened follow for {SID}")
    t0 = time.time()
    n = 0
    while time.time() - t0 < DURATION_S:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=DURATION_S)
        except asyncio.TimeoutError:
            print(f"[{time.time()-t0:.0f}s] (quiet {DURATION_S}s, no frames)")
            break
        n += 1
        try:
            frame = json.loads(raw)
            print(f"[{time.time()-t0:5.1f}s] #{n} {describe(frame)}")
        except Exception as e:
            print(f"[{time.time()-t0:5.1f}s] #{n} unparseable: {e} {str(raw)[:120]}")
        if n > 200:
            print("... (capped at 200 frames)")
            break
    await ws.close()
    print(f"done, {n} frames in {time.time()-t0:.0f}s")


asyncio.run(main())
