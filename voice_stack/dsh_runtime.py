"""JSON-RPC client for the DeepSeek Harness SDK runtime.

Spawns a separate DSH process (the official "out-of-process harness SDK"):

    node --import tsx/esm <dsh_root>/apps/cli/src/bin.ts --profile sdk

and speaks its newline-delimited JSON-RPC protocol over stdio:

    -> initialize       {cwd, provider, model}
    -> session/prompt   {sessionId, contentBlocks: [{type: "text", text}]}
    -> shutdown
    <- notifications:
         session.event   {sessionId, event}   (every session-log event, live)
         session.status  {sessionId, status}  (idle | running)

The voice session lives in its own DSH_HOME, so it has a full DSH context
window, compaction, and durable memory across restarts (the session is
persisted in DSH_HOME/sessions and reloaded by session id).

The model behind it is the local llama.cpp server (OpenAI-compatible),
configured through DEEPSEEK_BASE_URL / DEEPSEEK_API_KEY, same as the web UI.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import queue
import subprocess
import threading
import time

log = logging.getLogger("voice.dsh")


class DshError(RuntimeError):
    """The DSH runtime failed (spawn, handshake, or protocol error)."""


class DshRuntime:
    def __init__(
        self,
        dsh_root: str,
        dsh_home: str,
        *,
        model: str = "qwen3.8-27b",
        base_url: str = "http://127.0.0.1:8080/v1",
        persona_cwd: str | None = None,
        permission_mode: str = "workspace-write",
        session_id: str = "voice",
        node: str = "node",
        startup_timeout: float = 240.0,
        stderr_log: str | None = None,
    ):
        self.dsh_root = os.path.abspath(dsh_root)
        self.dsh_home = os.path.abspath(dsh_home)
        self.model = model
        self.base_url = base_url
        self.persona_cwd = os.path.abspath(persona_cwd) if persona_cwd else self.dsh_root
        self.permission_mode = permission_mode
        self.session_id = session_id
        self.node = node
        self.startup_timeout = startup_timeout
        self.stderr_log = stderr_log

        self.bin = os.path.join(self.dsh_root, "apps", "cli", "src", "bin.ts")
        self._ids = itertools.count(1)
        self._pending: dict[int, dict] = {}
        self._stdin_lock = threading.Lock()
        self.events: "queue.Queue[tuple]" = queue.Queue()
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._stderr_file = None
        self._dead_code: int | None = None

    # ------------------------------------------------------------------ state
    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None and self._dead_code is None

    @property
    def dead_code(self):
        return self._dead_code

    # ------------------------------------------------------------------- api
    def start(self):
        """Spawn the runtime and complete the initialize handshake."""
        if self.alive:
            return
        if not os.path.isfile(self.bin):
            raise DshError(f"DSH checkout not found at {self.dsh_root} (missing apps/cli/src/bin.ts)")
        os.makedirs(self.dsh_home, exist_ok=True)
        env = dict(os.environ)
        env.update(
            {
                "DSH_HOME": self.dsh_home,
                "DSH_PERMISSION_MODE": self.permission_mode,
                "DSH_TELEMETRY_DISABLED": "1",
                "DEEPSEEK_API_KEY": "local-llama-cpp",
                "DEEPSEEK_BASE_URL": self.base_url,
            }
        )
        self._dead_code = None
        if self.stderr_log:
            self._stderr_file = open(self.stderr_log, "ab", buffering=0)
        cmd = [self.node, "--import", "tsx/esm", self.bin, "--profile", "sdk"]
        log.info("spawning DSH SDK runtime (session=%s, cwd=%s, home=%s, model=%s)",
                 self.session_id, self.persona_cwd, self.dsh_home, self.model)
        t0 = time.time()
        self._proc = subprocess.Popen(
            cmd,
            cwd=self.dsh_root,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file if self._stderr_file else subprocess.DEVNULL,
        )
        self._reader = threading.Thread(target=self._read_loop, name="dsh-stdout", daemon=True)
        self._reader.start()
        try:
            self.call("initialize", {
                "cwd": self.persona_cwd,
                "provider": "deepseek-official",
                "model": self.model,
            }, timeout=self.startup_timeout)
        except Exception:
            self.kill()
            raise
        log.info("DSH runtime ready in %.1fs (pid %s)", time.time() - t0, self._proc.pid)

    def prompt(self, text: str) -> dict:
        """Send one user utterance to the voice session; returns the receipt."""
        result = self.call("session/prompt", {
            "sessionId": self.session_id,
            "contentBlocks": [{"type": "text", "text": text}],
        }, timeout=60)
        if "error" in result:
            raise DshError(f"session/prompt rejected: {result['error']}")
        return result.get("result", {})

    def stop(self):
        """Politely shut down (best effort); the child is reaped either way."""
        if not self.alive:
            self._reap()
            return
        try:
            self.call("shutdown", None, timeout=15)
        except Exception:
            pass
        try:
            self._proc.wait(timeout=10)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._reap()

    def kill(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._reap()

    def _reap(self):
        if self._proc is not None:
            try:
                self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None
        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except Exception:
                pass
            self._stderr_file = None

    # -------------------------------------------------------------- protocol
    def call(self, method: str, params, timeout: float = 60.0) -> dict:
        if self._proc is None or self._proc.poll() is not None:
            raise DshError("DSH runtime is not running")
        rid = next(self._ids)
        box = {"done": threading.Event(), "msg": None}
        self._pending[rid] = box
        try:
            if params is None:
                body = {"jsonrpc": "2.0", "id": rid, "method": method}
            else:
                body = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
            line = (json.dumps(body, ensure_ascii=False) + "\n").encode("utf-8")
            with self._stdin_lock:
                self._proc.stdin.write(line)
                self._proc.stdin.flush()
        except Exception as e:
            self._pending.pop(rid, None)
            raise DshError(f"write to DSH runtime failed: {e}") from e
        if not box["done"].wait(timeout):
            self._pending.pop(rid, None)
            raise DshError(f"timeout ({timeout:.0f}s) waiting for {method!r} response")
        self._pending.pop(rid, None)
        return box["msg"]

    def _read_loop(self):
        assert self._proc is not None and self._proc.stdout is not None
        for raw in self._proc.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                log.warning("non-JSON line from DSH runtime: %.200s", line)
                continue
            if isinstance(msg, dict) and "id" in msg and ("result" in msg or "error" in msg):
                box = self._pending.pop(msg["id"], None)
                if box is not None:
                    box["msg"] = msg
                    box["done"].set()
                else:
                    log.debug("late response for %s", msg.get("id"))
                continue
            method = msg.get("method") if isinstance(msg, dict) else None
            if method == "session.event":
                params = msg.get("params", {})
                self.events.put(("event", params.get("sessionId"), params.get("event")))
            elif method == "session.status":
                self.events.put(("status", msg.get("params", {})))
            elif method:
                log.debug("DSH runtime notification %s", method)
        code = self._proc.wait()
        log.warning("DSH runtime process exited (code %s)", code)
        self.events.put(("died", code))
