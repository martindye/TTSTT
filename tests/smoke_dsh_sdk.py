"""Smoke test: spawn the DSH SDK runtime and round-trip one prompt.

Validates the voice-bridge foundation:
  node --import tsx/esm apps/cli/src/bin.ts --profile sdk
  JSON-RPC over stdio: initialize -> session/prompt -> session.event stream
"""
import json
import os
import subprocess
import sys
import tempfile

REPO = r"C:\Users\press\OneDrive\Projects\deepseek-harness"
BIN = os.path.join(REPO, "apps", "cli", "src", "bin.ts")
MODEL_URL = "http://127.0.0.1:8080/v1"

home = os.path.join(tempfile.gettempdir(), "dsh-voice-smoke")
os.makedirs(home, exist_ok=True)
os.makedirs(os.path.join(home, "person"), exist_ok=True)

env = dict(os.environ)
env.update({
    "DSH_HOME": home,
    "DSH_PERMISSION_MODE": "workspace-write",
    "DSH_TELEMETRY_DISABLED": "1",
    "DEEPSEEK_API_KEY": "local-llama-cpp",
    "DEEPSEEK_BASE_URL": MODEL_URL,
})

proc = subprocess.Popen(
    ["node", "--import", "tsx/esm", BIN, "--profile", "sdk"],
    cwd=REPO, env=env,
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, bufsize=1,
)

def send(obj):
    proc.stdin.write(json.dumps(obj) + "\n")
    proc.stdin.flush()

send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "cwd": REPO, "provider": "deepseek-official", "model": "qwen3.8-27b"}})

import time
t0 = time.time()
text_parts = []
saw_init = False
turn_end = None
deadline = time.time() + 180

try:
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            print("NON-JSON:", line[:200]); continue
        if msg.get("id") == 1:
            print(f"[{time.time()-t0:.1f}s] initialize result:", json.dumps(msg.get("result", msg.get("error")))[:200])
            saw_init = True
            send({"jsonrpc": "2.0", "id": 2, "method": "session/prompt", "params": {
                "sessionId": "voice-smoke",
                "contentBlocks": [{"type": "text", "text": "Say hello in exactly three words. Do not use any tools."}],
            }})
            continue
        if msg.get("id") == 2:
            print(f"[{time.time()-t0:.1f}s] prompt receipt:", json.dumps(msg.get("result", msg.get("error")))[:200])
            continue
        method = msg.get("method")
        if method == "session.event":
            ev = msg["params"]["event"]
            etype = ev.get("type")
            if etype == "assistant/chunk":
                ch = ev["data"]["chunk"]
                if ch.get("type") == "text-delta":
                    text_parts.append(ch["text"])
                elif ch.get("type") == "reasoning-delta":
                    pass  # counted, not spoken
            elif etype in ("assistant/message", "turn/end", "turn/start", "user/message", "error", "turn/error"):
                print(f"[{time.time()-t0:.1f}s] EVENT {etype}: {json.dumps(ev.get('data', {}))[:200]}")
                if etype == "turn/end":
                    turn_end = ev["data"]
                    break
            else:
                print(f"[{time.time()-t0:.1f}s] event: {etype}")
        elif method == "session.status":
            print(f"[{time.time()-t0:.1f}s] status: {msg['params'].get('status')}")
        else:
            print(f"[{time.time()-t0:.1f}s] notif {method}: {json.dumps(msg.get('params'))[:120]}")
finally:
    try:
        send({"jsonrpc": "2.0", "id": 99, "method": "shutdown", "params": None})
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    err = proc.stderr.read() if proc.stderr else ""
    if err:
        print("--- stderr tail ---")
        print("\n".join(err.splitlines()[-15:]))

print(f"\nSPOKEN TEXT: {text_parts!r}")
print("TURN_END:", turn_end)
print("OK" if saw_init and text_parts and turn_end else "FAILED")
