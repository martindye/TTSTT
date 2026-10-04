"""Qwen3-TTS 1.7B VoiceDesign adapter for the DeepSeek voice bridges.

Uses the installed qwen-tts API, which returns complete sentences, not
incremental audio. This adds expression, not a promise of lower latency.
Models must be downloaded explicitly; bridge startup stays offline.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import threading

from .qwen3_tts_engine import Qwen3TTS

log = logging.getLogger(__name__)
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_MODEL_DIR = Path.home() / "models" / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_INSTRUCT = (
    "A warm British English female voice, with a natural conversational rhythm, "
    "clear diction and a lightly playful personality. Speak at a brisk but "
    "comfortable pace, with varied intonation and short natural pauses."
)
# Keep the identity instruction fixed; append delivery only for each mood.
MOOD_INSTRUCTIONS = {
    "neutral": "Speak naturally and conversationally.",
    "adoration": "Speak with gentle affection.",
    "amazement": "Sound pleasantly surprised.",
    "amusement": "Sound amused, with a light smile in the voice.",
    "anger": "Sound firm and irritated, without shouting.",
    "confusion": "Sound thoughtfully puzzled.",
    "contentment": "Sound warmly pleased and satisfied.",
    "cuteness": "Sound sweet and lightly playful.",
    "desire": "Sound eager and hopeful.",
    "disappointment": "Sound mildly disappointed.",
    "disgust": "Sound disapproving and put off.",
    "distress": "Sound concerned and urgent, but clear.",
    "embarassment": "Sound a little sheepish.",
    "extasy": "Sound delighted and excited.",
    "fear": "Sound nervous and cautious.",
    "guilt": "Sound sincerely apologetic.",
    "interest": "Sound curious and engaged.",
    "pain": "Sound uncomfortable and subdued.",
    "pride": "Sound quietly proud and confident.",
    "realization": "Sound as though something has just clicked.",
    "relief": "Sound relieved, with relaxed delivery.",
    "sadness": "Sound gentle and sad.",
    "serenity": "Sound calm and reassuring.",
}


class Qwen3VoiceDesign(Qwen3TTS):
    """Same audio interface as the clone engine, with natural-language moods."""

    def __init__(self, model_dir=None, instruct=DEFAULT_INSTRUCT,
                 language="English", moods=True, device="cuda:0"):
        model_path = Path(model_dir or DEFAULT_MODEL_DIR).expanduser().resolve()
        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"VoiceDesign is not downloaded at {model_path}. "
                "Run python scripts/download_qwen3_design.py --download first.")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("tts_model_type") != "voice_design":
            raise ValueError(f"{model_path} is not a Qwen VoiceDesign checkpoint")
        if not instruct or not instruct.strip():
            raise ValueError("VoiceDesign needs a non-empty voice description")

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from qwen_tts import Qwen3TTSModel

        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("VoiceDesign requested CUDA, but CUDA is unavailable")
            free, _ = torch.cuda.mem_get_info(torch.device(device))
            # Conservative headroom: 4.52 GB checkpoint plus activations/cache.
            # The actual peak needs a local benchmark before live activation.
            if free < 6 * 1024**3:
                raise RuntimeError(
                    f"VoiceDesign needs about 6 GiB free for this bf16 setup; "
                    f"only {free / 1024**3:.1f} GiB is free. "
                    "Free GPU memory before starting the bridge, or use "
                    "--qwen3tts-device cpu for a slow preview.")
        self._model_dir = str(model_path)
        self._language = language.capitalize()
        self._instruct = instruct.strip()
        self.moods_enabled = moods
        self._mood = "neutral"
        self._lock = threading.Lock()
        log.info("loading VoiceDesign from %s on %s (sdpa)", model_path, device)
        self._model = Qwen3TTSModel.from_pretrained(
            str(model_path), device_map=device,
            dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            attn_implementation="sdpa", local_files_only=True)

    def set_mood(self, mood: str) -> bool:
        if not self.moods_enabled or mood not in MOOD_INSTRUCTIONS:
            return False
        with self._lock:
            self._mood = mood
        return True

    def _generate(self, sentence: str):
        instruction = self._instruct
        if self.moods_enabled:
            instruction += " " + MOOD_INSTRUCTIONS[self._mood]
        wavs, sr = self._model.generate_voice_design(
            text=sentence, language=self._language, instruct=instruction)
        if int(sr) != self.sample_rate:
            raise ValueError(f"VoiceDesign returned unexpected sample rate: {sr}")
        return wavs, sr


def add_design_arguments(parser):
    parser.add_argument("--qwen3tts-instruct", default=DEFAULT_INSTRUCT,
                        help="qwen3design/qwen3gguf: voice identity and delivery description")
    parser.add_argument("--qwen3tts-device", default="cuda:0",
                        choices=["cuda:0", "cpu"],
                        help="qwen3design device; CPU is for slow previews")


def main():
    import argparse
    import time
    import numpy as np
    import soundfile as sf

    parser = argparse.ArgumentParser(description=__doc__)
    add_design_arguments(parser)
    parser.add_argument("--model", default=None)
    parser.add_argument("--mood", choices=sorted(MOOD_INSTRUCTIONS), default="interest")
    parser.add_argument("--text", default="Hello Martin. Shall we give this a little more personality?")
    parser.add_argument("--out", default="qwen3_design_preview.wav")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    engine = Qwen3VoiceDesign(model_dir=args.model,
                             instruct=args.qwen3tts_instruct,
                             device=args.qwen3tts_device)
    engine.set_mood(args.mood)
    start = time.monotonic()
    chunks = list(engine.stream_sentence(args.text))
    elapsed = time.monotonic() - start
    if not chunks:
        raise RuntimeError("VoiceDesign returned no audio")
    audio = np.concatenate(chunks)
    sf.write(args.out, audio, engine.sample_rate, subtype="PCM_16")
    duration = len(audio) / engine.sample_rate
    print(f"Wrote {args.out}: {duration:.2f}s audio, {elapsed:.2f}s synthesis, "
          f"RTF {elapsed / duration:.2f}. First audio waits for synthesis.")


if __name__ == "__main__":
    main()
