"""Test the local GPU Q8 build of Qwen3-TTS-12Hz-1.7B-VoiceDesign.

Sends the complete passage with the user instruction unchanged and an explicit seed. No sentence splitting or streaming.
The window preserves the draft and settings on close. It verifies the serving
model alias before sending speech. VoiceDesign designs voices; it does not clone
an audio reference. Older engine helpers remain for compatibility only.
"""
from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests
import sounddevice as sd
import soundfile as sf
import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voice_stack.qwen3_gguf_engine import (  # noqa: E402
    ALIAS as DESIGN_ALIAS,
    URL,
    ensure_server as ensure_design_server,
)
from voice_stack.qwen3_gguf_base_engine import (  # noqa: E402
    ALIAS as BASE_ALIAS,
    ensure_server as ensure_base_server,
)
from voice_stack.qwen3_design_engine import DEFAULT_INSTRUCT  # noqa: E402

SPEECH_URL = URL + "/v1/audio/speech"
C_URL = "http://127.0.0.1:8096"
C_SPEECH_URL = C_URL + "/v1/audio/speech"
STOP_SCRIPT = ROOT / "scripts" / "stop_qwen3_gguf.ps1"
BASE_STATE = ROOT / "tests" / "qwen3_gguf_base_server.json"
DESIGN_STATE = ROOT / "tests" / "qwen3_gguf_server.json"

DESIGNED = "Qwen3 1.7B VoiceDesign Q8"
C_EMOTING = "British female (emoting clone)"
WINDOW_STATE = ROOT / "tests" / "voice_window_state.json"
REFERENCE_AUDIO = ROOT / "voice_stack" / "ref_voices" / "british_p225.wav"

# voice label -> (engine, voice id or None). The designed voice is the
# window's standard: it is first in the list and the default selection.
VOICES = {
    DESIGNED: ("design", None),
}

def build_instruction(emotion: str) -> str:
    """Forward the user's instruction verbatim to VoiceDesign."""
    return emotion


# Names the C engine accepts directly, so free text in the emotion box
# ("angry", "happy", ...) works as-is.
C_ALIASES = {
    "sad": "sad", "sadness": "sad",
    "joy": "joy", "happy": "joy", "joyful": "joy",
    "anger": "anger", "angry": "anger", "rage": "anger",
    "fear": "fear", "afraid": "fear",
    "disgust": "disgust", "disgusted": "disgust",
    "surprise": "surprise", "surprised": "surprise",
}


def c_emotion(emotion: str) -> str | None:
    """Emotion box content -> the emotion to send to the C engine.

    The six base emotions are voiced on the engine; anything else is
    passed through and the engine falls back to plain speech for it.
    """
    em = emotion.strip().lower()
    if not em or em == "neutral":
        return None
    return C_ALIASES.get(em, em)


def server_ids():
    """Return the model ids served on port 8095, or None if nothing answers."""
    try:
        r = requests.get(URL + "/v1/models", timeout=1)
        if r.status_code == 200:
            return [m.get("id") for m in r.json().get("data", [])]
    except requests.RequestException:
        pass
    return None


def split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?…])\s+|\n+", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


class VoiceWindow:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("TTSTT - Qwen3 VoiceDesign (revision 6)")

        self.busy = False
        self.closed = False
        self.cancel = threading.Event()
        self.started_engine = None  # engine this window started, if any

        top = tk.Frame(root)
        top.pack(fill="x", padx=10, pady=(10, 2))
        tk.Label(top, text="Voice:").pack(side="left")
        self.voice_combo = ttk.Combobox(
            top, state="readonly", width=30, values=list(VOICES))
        self.voice_combo.set(DESIGNED)
        self.voice_combo.pack(side="left", padx=(6, 14), ipady=2)
        self.voice_combo.bind("<<ComboboxSelected>>",
                              lambda _e: self._voice_changed())
        tk.Label(top, text="Voice / emotion instruction:").pack(side="left")
        self.emotion = tk.Entry(top, width=20)
        self.emotion.pack(side="left", padx=(4, 0), ipady=2)
        self.emotion.config(state="normal")
        tk.Label(top, text="Seed:").pack(side="left", padx=(14, 0))
        self.seed = tk.Entry(top, width=8)
        self.seed.insert(0, "42")
        self.seed.pack(side="left", padx=(4, 0), ipady=2)
        hint = (tk.Label(root, fg="gray", wraplength=740,
                         text="Your instruction is sent unchanged to VoiceDesign. The full passage is generated together."))
        hint.pack(anchor="w", padx=14)
        self.text = tk.Text(root, wrap="word", height=7, font=("Segoe UI", 12))
        self.text.pack(fill="both", expand=True, padx=10, pady=(4, 4))
        self.text.insert("1.0", "Type something, then press Speak.")

        bottom = tk.Frame(root)
        bottom.pack(fill="x", padx=10, pady=(4, 10))
        self.speak_btn = tk.Button(bottom, text="Speak   (Ctrl+Enter)",
                                  command=self.speak)
        self.speak_btn.pack(side="left")
        self.stop_btn = tk.Button(bottom, text="Stop", command=self.stop,
                                  state="disabled")
        self.stop_btn.pack(side="left", padx=(8, 0))
        self.status_var = tk.StringVar(
            value="Ready - the voice server loads on the first Speak (about a minute).")
        tk.Label(root, textvariable=self.status_var, wraplength=740).pack(fill="x", padx=10, pady=4)

        root.bind("<Control-Return>", lambda _e: self.speak())
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        try:
            state = json.loads(WINDOW_STATE.read_text(encoding="utf-8"))
            self.text.delete("1.0", "end")
            self.text.insert("1.0", state["text"])
            self.emotion.insert(0, state.get("emotion", ""))
            self.seed.delete(0, "end")
            self.seed.insert(0, state.get("seed", "42"))
            self.voice_combo.set(state.get("voice") if state.get("voice") in VOICES else DESIGNED)
            self._voice_changed()
        except (OSError, ValueError, KeyError):
            pass

        root.update_idletasks()
        x = max(0, (root.winfo_screenwidth() - 780) // 2)
        y = max(0, (root.winfo_screenheight() - 400) // 3)
        root.geometry(f"780x400+{x}+{y}")
        root.lift()
        root.focus_force()
        try:
            root.attributes("-topmost", True)
            root.after(2000, lambda: root.attributes("-topmost", False))
        except tk.TclError:
            pass

    # -- status helpers ----------------------------------------------------
    def _status(self, message: str) -> None:
        if not self.closed:
            self.root.after(0, lambda m=message: self.status_var.set(m))

    # -- voice selection ----------------------------------------------------
    def _voice_changed(self) -> None:
        label = self.voice_combo.get()
        takes_emotion = label == DESIGNED or label == C_EMOTING
        self.emotion.config(state="normal" if takes_emotion else "disabled")

    # -- speaking -----------------------------------------------------------
    def play_reference(self) -> None:
        if self.busy:
            return
        audio, rate = sf.read(REFERENCE_AUDIO, dtype="float32")
        sd.play(audio, rate)
        self.stop_btn.config(state="normal")
        self._status("Original British recording (not synthesised).")

    def _save_state(self) -> None:
        WINDOW_STATE.parent.mkdir(exist_ok=True)
        WINDOW_STATE.write_text(json.dumps({"text": self.text.get("1.0", "end-1c"),
            "voice": self.voice_combo.get(), "emotion": self.emotion.get(),
            "seed": self.seed.get()}, ensure_ascii=False), encoding="utf-8")

    def _seed(self) -> int:
        """Reject the random-seed sentinel and accidental invalid input."""
        seed = int(self.seed.get().strip() or "42")
        if not 0 <= seed <= 4294967295:
            raise ValueError("Seed must be between 0 and 4294967295; use 42 for repeatable tests.")
        return seed

    def speak(self, text_override: str | None = None) -> None:
        if self.busy:
            return
        text = (text_override if text_override is not None else self.text.get("1.0", "end-1c")).strip()
        if not text:
            self._status("Type something first.")
            return
        label = self.voice_combo.get()
        engine, voice = VOICES.get(label, (None, None))
        if not engine:
            self._status("Pick a voice first.")
            return
        emotion = (self.emotion.get()
                   if engine in ("design", "c") else "")
        if not emotion.strip():
            self._status("Enter a voice or emotion instruction first.")
            return
        try:
            seed = self._seed()
        except ValueError:
            self._status("Enter a seed from 0 to 4294967295. Use 42 for repeatable tests.")
            return
        self.busy = True
        self._save_state()
        sd.stop()
        self.cancel.clear()
        self.speak_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.voice_combo.config(state="disabled")
        self.emotion.config(state="disabled")
        self.seed.config(state="disabled")
        threading.Thread(target=self._speak,
                         args=(text, engine, voice, emotion, seed),
                         daemon=True).start()

    def stop(self) -> None:
        self.cancel.set()
        try:
            sd.stop()
        except Exception:
            pass
        if self.busy:
            self._status("Stopped playback; waiting for the current generation to finish.")
        else:
            self.stop_btn.config(state="disabled")
            self._status("Stopped.")

    def _speak(self, text: str, engine: str, voice: str, emotion: str,
               seed: int) -> None:
        try:
            if engine == "c":
                self._speak_c(text, emotion, seed)
                return
            self._ensure_engine(engine)
            if self.cancel.is_set():
                return
            instruction = (build_instruction(emotion)
                           if engine == "design" else None)
            sentences = [text]
            total = len(sentences)
            for i, sentence in enumerate(sentences, 1):
                if self.cancel.is_set():
                    break
                self._status(f"Generating VoiceDesign; seed {seed}; emotion: {emotion or 'neutral'}...")
                started = time.perf_counter()
                payload = {
                    "model": (DESIGN_ALIAS if engine == "design"
                              else BASE_ALIAS),
                    "input": sentence,
                    "language": "English",
                    "response_format": "wav",
                    "seed": seed,
                    "max_new_tokens": 768,
                }
                if engine == "design":
                    payload["instructions"] = instruction
                else:
                    payload["voice"] = voice
                r = requests.post(SPEECH_URL, json=payload,
                                  timeout=(5, 180))
                r.raise_for_status()
                elapsed = time.perf_counter() - started
                (ROOT / "tests" / "voice_window_last_request.json").write_text(
                    json.dumps({"url": SPEECH_URL, "payload": payload,
                                "generation_seconds": elapsed}, indent=2), encoding="utf-8")
                audio, rate = sf.read(io.BytesIO(r.content), dtype="float32")
                if self.cancel.is_set():
                    break
                sd.play(audio, rate)
                sd.wait()
            if not self.cancel.is_set():
                self._finish(f"Done: VoiceDesign; seed {seed}; generated in {elapsed:.1f}s.")
        except Exception as exc:
            self._status(f"Error while speaking: {exc}")
        finally:
            if self.cancel.is_set():
                self._status("Stopped.")
            self._reset()

    def _finish(self, message: str) -> None:
        self._status(message)

    # -- C engine (port 8096, WSL) -------------------------------------------
    def _speak_c(self, text: str, emotion: str, seed: int) -> None:
        """Speak a complete passage with the server's fixed British reference."""
        self._ensure_c_engine()
        if self.cancel.is_set():
            return
        em = c_emotion(emotion)
        # Keep context across sentences; the user explicitly prefers no streaming.
        sentences = [text]
        total = len(sentences)
        for i, sentence in enumerate(sentences, 1):
            if self.cancel.is_set():
                break
            self._status(f"Generating British female, {em or 'neutral'}...")
            started = time.perf_counter()
            payload = {
                "model": "qwen3-tts-0.6b",
                "input": sentence,
                "language": "English",
                "response_format": "wav",
                "seed": seed,
            }
            if em:
                payload["emotion"] = em
            r = requests.post(C_SPEECH_URL, json=payload, timeout=(10, 900))
            r.raise_for_status()
            audio, rate = sf.read(io.BytesIO(r.content), dtype="float32")
            if self.cancel.is_set():
                break
            elapsed = time.perf_counter() - started
            self._status(f"Playing British female ({em or 'neutral'}); generated in {elapsed:.1f}s.")
            sd.play(audio, rate)
            sd.wait()
        if not self.cancel.is_set():
            self._finish(f"Done: {em or 'neutral'}; generation {elapsed:.1f}s.")

    def _ensure_c_engine(self) -> None:
        """Make sure the C engine answers on 8096; wake it via WSL if not."""
        if self._c_engine_up():
            return
        self._status("Waking the emoting engine - can take a couple of "
                     "minutes first...")
        try:
            subprocess.Popen(
                ["wsl.exe", "-d", "Ubuntu-24.04", "-u", "press", "bash",
                 "/mnt/" + ROOT.drive[0].lower() + ROOT.as_posix()[2:]
                 + "/scripts/start_c_voice.sh"],
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL)
        except OSError as exc:
            raise RuntimeError(
                f"Could not wake WSL for the emoting engine: {exc}") from exc
        deadline = time.time() + 300
        while time.time() < deadline:
            if self.cancel.is_set():
                return
            if self._c_engine_up():
                return
            time.sleep(2)
        raise RuntimeError(
            "The emoting engine did not answer on port 8096 - check the "
            "c-voice log inside WSL")

    @staticmethod
    def _c_engine_up() -> bool:
        try:
            r = requests.get(C_URL + "/v1/health", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def _reset(self) -> None:
        if self.closed:
            return
        def reset():
            self.busy = False
            self.speak_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.voice_combo.config(state="readonly")
            self.seed.config(state="normal")
            self._voice_changed()
        self.root.after(0, reset)

    # -- engine management ---------------------------------------------------
    def _ensure_engine(self, engine: str) -> None:
        """Make sure the right server runs on 8095, switching if we must."""
        wanted = DESIGN_ALIAS if engine == "design" else BASE_ALIAS
        ids = server_ids()
        if ids == [wanted]:
            return
        if ids is None:
            name = "designed-voice" if engine == "design" else "clone"
            self._status(f"Loading the {name} server - about a minute...")
            self._start(engine)
            return
        alias0 = ids[0] if ids else None
        current = ("design" if alias0 == DESIGN_ALIAS
                   else "base" if alias0 == BASE_ALIAS else None)
        if current == engine:
            return
        if current is None or self.started_engine != current:
            raise RuntimeError(
                "Port 8095 is busy with another Qwen speech server that this "
                "window did not start. Close or stop it first.")
        name = "designed-voice" if engine == "design" else "clone"
        self._status(f"Switching to the {name} engine - about a minute...")
        self._stop_engine(current)
        self._start(engine)

    def _start(self, engine: str) -> None:
        if engine == "design":
            ensure_design_server()
        else:
            ensure_base_server()
        self.started_engine = engine

    def _stop_engine(self, engine: str, wait: bool = True) -> None:
        state = BASE_STATE if engine == "base" else DESIGN_STATE
        try:
            subprocess.Popen(
                ["powershell", "-NoProfile", "-File", str(STOP_SCRIPT),
                 "-StatePath", str(state)],
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError:
            return
        if not wait:
            return
        for _ in range(60):
            if server_ids() is None:
                return
            time.sleep(0.5)

    # -- shutdown ------------------------------------------------------------
    def on_close(self) -> None:
        self._save_state()
        self.closed = True
        if self.busy:
            self.cancel.set()
            try:
                sd.stop()
            except Exception:
                pass
        if self.started_engine:
            self._stop_engine(self.started_engine, wait=False)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    VoiceWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
