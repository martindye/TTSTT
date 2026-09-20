"""OpenAI-compatible streaming chat client for the local llama.cpp server.

Targets the existing Qwen/DeepSeek harness (http://127.0.0.1:8080/v1 by
default). Supports cancellation (for barge-in) and strips reasoning/thinking
content so only spoken text is returned.
"""

from __future__ import annotations

import json
import logging
import threading

import requests

log = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a warm, concise voice assistant. Answer in plain spoken language: "
    "short sentences, no markdown, no lists, no emojis, no code blocks. "
    "Keep answers to one or two sentences unless the user asks for detail. "
    "If you do not know something, say so briefly."
)


class LLMClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "qwen3.8-27b",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 400,
        timeout: float = 180.0,
        disable_thinking: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        # Qwen3-style thinking models: turn off internal reasoning via the
        # llama.cpp chat template (measured ~3.6x faster, and the model reads
        # answers out loud so reasoning tokens are wasted latency anyway).
        # Unknown keys are ignored by templates that don't support them.
        self.disable_thinking = disable_thinking
        self._session = requests.Session()

    def health(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}/models", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def stream_chat(self, messages: list[dict], cancel_event: threading.Event):
        """Yield text deltas (content only, reasoning stripped).

        Stops early when `cancel_event` is set.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.disable_thinking:
            # llama.cpp chat-template flag; ignored by non-thinking templates.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            with self._session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                stream=True,
                timeout=(10.0, self.timeout),
            ) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:300]}")
                for line in r.iter_lines(decode_unicode=True):
                    if cancel_event.is_set():
                        log.info("LLM stream cancelled by caller")
                        return
                    if not line:
                        continue
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield content
                    # delta.get("reasoning_content") is intentionally dropped
        except requests.RequestException as e:
            raise RuntimeError(f"LLM request failed: {e}") from e
