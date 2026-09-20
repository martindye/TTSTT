"""Probe the llama.cpp server for thinking-control fields (one-shot)."""
import json
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
BASE = {
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Say hi in one word."}],
    "max_tokens": 30,
    "stream": False,
}


def probe(name, extra):
    body = dict(BASE)
    body.update(extra)
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        detail = getattr(e, "read", lambda: b"")()
        print(f"{name}: ERROR {e} {detail[:300]!r}")
        return
    dt = time.time() - t0
    msg = data["choices"][0]["message"]
    usage = data.get("usage", {})
    content = (msg.get("content") or "").strip()
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    print(f"{name}: {dt:.1f}s content={content[:60]!r} reasoning_len={len(reasoning)} "
          f"usage={json.dumps(usage)}")


probe("baseline       ", {})
probe("thinking=off   ", {"thinking": {"type": "disabled"}})
probe("effort=low     ", {"reasoning_effort": "low"})
probe("effort=off     ", {"reasoning_effort": "off"})
probe("chatkw-nothink ", {"chat_template_kwargs": {"enable_thinking": False}})
