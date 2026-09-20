"""Probe the Qwen LLM: is it thinking? Can we disable it via chat_template_kwargs?"""
import json
import time

import requests

BASE = "http://127.0.0.1:8080/v1"

def ask(extra: dict, label: str) -> None:
    payload = {
        "model": "qwen3.8-27b",
        "messages": [
            {"role": "system", "content": "You are a helpful voice assistant. Answer briefly in plain spoken language."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "max_tokens": 300,
        "temperature": 0.7,
        "stream": False,
    }
    payload.update(extra)
    t0 = time.time()
    r = requests.post(f"{BASE}/chat/completions", json=payload, timeout=120)
    dt = time.time() - t0
    if r.status_code != 200:
        print(f"[{label}] HTTP {r.status_code}: {r.text[:300]}")
        return
    obj = r.json()
    msg = obj["choices"][0]["message"]
    usage = obj.get("usage", {})
    print(f"[{label}] {dt:.2f}s  usage={usage.get('completion_tokens')} tokens")
    if msg.get("reasoning_content"):
        print(f"  REASONING ({len(msg['reasoning_content'])} chars): {msg['reasoning_content'][:150]}...")
    print(f"  CONTENT: {msg.get('content')!r}")

ask({}, "default")
ask({"chat_template_kwargs": {"enable_thinking": False}}, "enable_thinking=False")
ask({"chat_template_kwargs": {"thinking": False}}, "thinking=False")
ask({"chat_template_kwargs": {"think": False}}, "think=False")
