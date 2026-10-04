"""Benchmark whole-sentence GPU Q8 speech with optional local STT roundtrip."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from voice_stack.qwen3_gguf_engine import Qwen3GGUF


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcribe", action="store_true")
    args = parser.parse_args()
    stt = None
    if args.transcribe:
        import torch
        from voice_stack.stt_engine import KyutaiSTT, SttEventKind
        print("Loading the microphone model for a realistic memory test...", flush=True)
        stt = KyutaiSTT(device="cuda")
    engine = Qwen3GGUF()
    cases = [
        ("neutral", "Done. The bridge is live again with the fixes."),
        ("amusement", "Well, that was a rather roundabout way of saying hello."),
        ("interest", "Hello Martin. Shall we give this a little more personality?"),
    ]
    results = []
    for index, (mood, text) in enumerate(cases):
        engine.set_mood(mood)
        start = time.monotonic()
        audio = np.concatenate(list(engine.stream_sentence(text)))
        elapsed = time.monotonic() - start
        duration = len(audio) / engine.sample_rate
        out = ROOT / "tests" / f"qwen3_gguf_{index}_{mood}.wav"
        sf.write(out, audio, engine.sample_rate, subtype="PCM_16")
        result = {"mood": mood, "text": text, "audio_seconds": duration,
                  "synthesis_seconds": elapsed, "rtf": elapsed / duration,
                  "peak": float(np.abs(audio).max()), "path": str(out)}
        if stt:
            with torch.inference_mode():
                stt.reset()
            # Leading silence gives the decoder its initial delay window.
            padded = np.concatenate([np.zeros(24000, dtype=np.float32), audio,
                                     np.zeros(48000, dtype=np.float32)])
            padded = np.pad(padded, (0, (-len(padded)) % stt.frame_size))
            pieces = []
            with torch.inference_mode():
                for offset in range(0, len(padded), stt.frame_size):
                    frame = torch.from_numpy(padded[offset:offset + stt.frame_size])
                    pieces.extend(event.piece for event in stt.feed_frame(frame)
                                  if event.kind == SttEventKind.WORD)
            result["transcript"] = "".join(pieces).replace("\u2581", " ").strip()
        result["gpu_memory"] = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader"
        ], text=True).strip()
        results.append(result)
        print(json.dumps(result), flush=True)
        (ROOT / "tests/qwen3_gguf_benchmark.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
