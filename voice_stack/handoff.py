"""Hand a coding task to a DSH coding session in the TTSTT workspace.

Two delivery modes:

1. GATEWAY mode (default). If the DSH web GUI (``dsh web``, default
   http://127.0.0.1:3080) is running, the task is queued straight into the
   most recently active TTSTT session -- the open coding chat when the user
   has one open, otherwise the last written one. It lands as a normal user
   message (clearly labelled) and that session's agent works it. The gateway
   resumes a cold session automatically, so this works whether or not the
   session is open in the browser.

   Gateway auth uses the same scheme the browser uses: the browser-session
   secret from ``<dsh-home>/.credentials.yaml`` mints the signed
   ``dsh-auth-*`` cookie (loopback-only gateway; this is the local user's own
   machine and their own gateway).

2. SPAWN mode (fallback). If the gateway is unreachable (or ``--spawn``),
   a fresh DSH SDK session (``handoff-<ts>``) runs the task standalone and
   appears in the GUI session list when the GUI is open.

Usage (from the TTSTT folder):
    python -X utf8 -m voice_stack.handoff "fix the flaky endpointing test"
    python -X utf8 -m voice_stack.handoff --spawn "work without the GUI"
    python -X utf8 -m voice_stack.handoff --session <id> "target this session"

stdout protocol (the voice agent reads these lines):
    HANDOFF RESULT: injected | spawned | failed | timeout
    HANDOFF SUMMARY: <one or two sentences for the voice to speak>
Exit code: 0 on injected/spawned-ok, 1 otherwise.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import logging
import os
import queue
import re
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

TTSTT_ROOT = Path(__file__).resolve().parents[1]
PROJECTS_DIR = TTSTT_ROOT.parent
DEFAULT_DSH_ROOT = PROJECTS_DIR / "deepseek-harness"
DEFAULT_GUI_HOME = Path.home() / ".dsh"
DEFAULT_GUI_URL = "http://127.0.0.1:3080"
DEFAULT_LLM_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "qwen3.8-27b"
COOKIE_LIFETIME_MS = 2 * 3600 * 1000  # stays well under the gateway's max age

HANDOFF_BANNER = (
    "[VOICE-HANDOFF] The user's voice assistant handed this task to you "
    "(the coding agent), on the user's behalf. Do it now, then report "
    "what you did in a short summary.\n\n"
)

log = logging.getLogger("handoff")


# ---------------------------------------------------------------------------
# Target session discovery ("currently open" ~= most recently written)
# ---------------------------------------------------------------------------

def sessions_dir_for(gui_home: Path) -> Path | None:
    base = gui_home / "sessions"
    if not base.is_dir():
        return None
    encoded = "--" + re.sub(r"[^A-Za-z0-9]+", "-", str(TTSTT_ROOT)) + "--"
    direct = base / encoded
    if direct.is_dir():
        return direct
    for cand in base.iterdir():
        if cand.is_dir() and TTSTT_ROOT.name in cand.name:
            return cand
    return None


def find_target_session(sdir: Path, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    newest_id: str | None = None
    newest_mt = -1.0
    for sd in sdir.iterdir():
        if not sd.is_dir():
            continue
        mt = 0.0
        for f in sd.iterdir():
            try:
                if f.is_file():
                    mt = max(mt, f.stat().st_mtime)
            except OSError:
                pass
        if mt > newest_mt:
            newest_mt, newest_id = mt, sd.name
    return newest_id


# ---------------------------------------------------------------------------
# Gateway client (cookie minted from the local credentials store)
# ---------------------------------------------------------------------------

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def load_browser_secret(gui_home: Path) -> bytes | None:
    """Read the client-connection/browser-session signing secret."""
    try:
        import yaml
    except ImportError:
        return None
    p = gui_home / ".credentials.yaml"
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:
        log.exception("cannot read %s", p)
        return None
    rec = ((doc or {}).get("records") or {}).get("client-connection/browser-session") or {}
    secret_b64 = (rec.get("payload") or {}).get("secret")
    if not isinstance(secret_b64, str) or not secret_b64:
        return None
    try:
        return base64.urlsafe_b64decode(secret_b64 + "=" * (-len(secret_b64) % 4))
    except Exception:
        log.exception("bad browser-session secret in %s", p)
        return None


def mint_cookie(authority: str, secret: bytes) -> str:
    """The exact cookie the browser receives: v1.<b64url(json)>.<hmac-sha256>."""
    name = "dsh-auth-" + _b64url(hashlib.sha256(authority.encode()).digest())
    now = int(time.time() * 1000)
    payload = {"version": 1, "authority": authority,
               "issuedAt": now, "expiresAt": now + COOKIE_LIFETIME_MS}
    body = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64url(hmac.new(secret, body.encode(), hashlib.sha256).digest())
    return f"{name}=v1.{body}.{sig}"


def inject_via_gateway(gui_url: str, target: str, task: str, secret: bytes) -> tuple[bool, str]:
    """Queue the task into `target` through the GUI gateway. (ok, detail)"""
    import requests

    u = urlparse(gui_url)
    authority = f"{u.hostname}:{u.port}" if u.port else (u.hostname or "")
    cookie = mint_cookie(authority, secret)
    wrapped = HANDOFF_BANNER + task
    body = {
        "type": "client-request",
        "rpcId": str(uuid.uuid4()),
        "method": "session/prompt",
        "payload": {"args": {"request": {
            "requestId": str(uuid.uuid4()),
            "sessionId": target,
            "mode": "queue",
            "content": [{"type": "text", "text": wrapped}],
        }}},
    }
    try:
        r = requests.post(f"{gui_url.rstrip('/')}/api/session/prompt",
                          json=body,
                          headers={"Cookie": cookie, "Content-Type": "application/json"},
                          timeout=15)
    except Exception as e:
        return False, f"gateway unreachable ({e.__class__.__name__}: {e})"
    if r.status_code in (401, 403):
        return False, f"gateway rejected our cookie (HTTP {r.status_code}) - is the DSH web server the one on {gui_url}?"
    if r.status_code != 200:
        return False, f"gateway HTTP {r.status_code}: {r.text[:200]}"
    try:
        result = r.json().get("result") or {}
    except ValueError:
        return False, f"gateway returned non-JSON: {r.text[:200]}"
    if result.get("ok"):
        return True, f"queued in session {target}"
    err = result.get("error") or {}
    return False, f"gateway error: {err.get('code', '?')}: {err.get('message', '?')[:200]}"


# ---------------------------------------------------------------------------
# Spawn mode (fallback): fresh SDK session, same as before
# ---------------------------------------------------------------------------

def run_spawn(task: str, perm: str, timeout_s: float, dsh_home: str,
              dsh_root: str, model: str, base_url: str) -> int:
    from .dsh_runtime import DshError, DshRuntime

    runtime = DshRuntime(
        dsh_root=dsh_root,
        dsh_home=dsh_home,
        model=model,
        base_url=base_url,
        persona_cwd=str(TTSTT_ROOT),
        permission_mode=perm,
        session_id=f"handoff-{time.strftime('%Y%m%d-%H%M%S')}",
        stderr_log=str(TTSTT_ROOT / "tests" / "dsh_handoff.err.log"),
    )
    print("handoff: starting DSH coding session in TTSTT (spawn mode)", flush=True)
    t0 = time.time()
    try:
        runtime.start()
    except DshError as e:
        print(f"HANDOFF RESULT: failed\nHANDOFF SUMMARY: runtime failed to start: {e}")
        return 1

    ok = False
    text_blocks: dict[int, list[str]] = {}
    deadline = time.time() + timeout_s
    last_beat = time.time()
    summary = ""
    try:
        runtime.prompt(task)
        print(f"handoff: task sent in {time.time() - t0:.1f}s; working ...", flush=True)
        while True:
            if time.time() > deadline:
                print("HANDOFF RESULT: timeout "
                      f"(gave up after {timeout_s / 60:.0f} min; the session is still in "
                      "your GUI session list - open it to continue)", flush=True)
                return 1
            try:
                item = runtime.events.get(timeout=1.0)
            except queue.Empty:
                continue
            kind = item[0]
            if kind == "died":
                print(f"HANDOFF RESULT: runtime-died (code {item[1]})")
                return 1
            if kind == "status" or not isinstance(item[2], dict):
                continue
            ev = item[2]
            etype = ev.get("type")
            if etype == "assistant/chunk":
                chunk = (ev.get("data") or {}).get("chunk") or {}
                ctype = chunk.get("type")
                if ctype == "block-start":
                    if chunk.get("blockType") == "text":
                        text_blocks.setdefault(chunk.get("index"), [])
                elif ctype == "text-delta":
                    idx = chunk.get("index")
                    if idx in text_blocks:
                        text_blocks[idx].append(chunk.get("text", ""))
            elif etype == "tool/call":
                data = ev.get("data") or {}
                print(f"  [tool] {data.get('name')} {str(data.get('arguments'))[:120]}",
                      flush=True)
                last_beat = time.time()
            elif time.time() - last_beat > 60:
                last_beat = time.time()
                print(f"  ... still working ({(time.time() - t0) / 60:.0f} min)", flush=True)
            elif etype == "turn/end":
                reason = (ev.get("data") or {}).get("reason") or {}
                ok = reason.get("kind") in ("completed", "max-tokens", None)
                break
            elif etype in ("session/abort", "turn/error", "error"):
                print(f"HANDOFF RESULT: failed\nHANDOFF SUMMARY: {str(ev.get('data'))[:300]}")
                return 1
        for idx in sorted(text_blocks, reverse=True):
            text = "".join(text_blocks[idx]).strip()
            if text:
                summary = text
                break
        status = "ok" if ok else "failed"
        print(f"HANDOFF RESULT: spawned {status}  ({time.time() - t0:.0f}s)")
        print("HANDOFF SUMMARY: " + (summary[:1500] or "(no text)"))
        return 0 if ok else 1
    finally:
        runtime.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="voice_stack.handoff",
                                description="Hand a task to a DSH coding agent in TTSTT.")
    p.add_argument("task", nargs="+", help="the task to hand off")
    p.add_argument("--session", default=None, help="explicit target session id")
    p.add_argument("--gui", default=DEFAULT_GUI_URL,
                   help="GUI gateway base URL (default %(default)s)")
    p.add_argument("--gui-home", default=str(DEFAULT_GUI_HOME),
                   help="DSH home of the GUI gateway (default: ~/.dsh)")
    p.add_argument("--spawn", action="store_true",
                   help="skip the gateway; always spawn a fresh handoff session")
    p.add_argument("--perm", default="danger-full-access",
                   help="spawn mode: permission mode for the coding session")
    p.add_argument("--timeout", type=float, default=900.0,
                   help="spawn mode: seconds to wait for the turn (default 900)")
    p.add_argument("--dsh-home", default=str(DEFAULT_GUI_HOME))
    p.add_argument("--dsh-root", default=str(DEFAULT_DSH_ROOT))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--llm-url", default=DEFAULT_LLM_URL)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    task = " ".join(args.task).strip()
    if not task:
        print("HANDOFF RESULT: failed\nHANDOFF SUMMARY: empty task")
        return 1

    # 1) target session: explicit, else the most recently written TTSTT session
    gui_home = Path(args.gui_home)
    sdir = sessions_dir_for(gui_home)
    target = None
    if sdir is not None:
        target = find_target_session(sdir, args.session)
    if args.session:
        target = args.session

    # 2) gateway mode
    if target and not args.spawn:
        secret = load_browser_secret(gui_home)
        if secret is None:
            print("handoff: no browser-session secret in "
                  f"{gui_home / '.credentials.yaml'}; falling back to spawn mode")
        else:
            ok, detail = inject_via_gateway(args.gui, target, task, secret)
            if ok:
                print(f"handoff: {detail}")
                print("HANDOFF RESULT: injected")
                print("HANDOFF SUMMARY: I handed that to your coding agent. "
                      "It is in the coding chat now and will work on it there "
                      "while you watch.")
                return 0
            print(f"handoff: gateway injection failed ({detail}); falling back to spawn mode")

    # 3) spawn mode (fallback or forced)
    return run_spawn(task, args.perm, args.timeout, args.dsh_home, str(args.dsh_root),
                     args.model, args.llm_url)


if __name__ == "__main__":
    import sys
    sys.exit(main())
