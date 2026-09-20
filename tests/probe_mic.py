"""Mic + STT diagnostic probe.

Runs for ~30s: prints per-second input RMS, and (if the STT model loads in
time) the words it transcribes. Run it, then SPEAK a normal test sentence.
Layers this isolates:
  - RMS ~0 the whole time        -> mic is silent (device/driver/mute)
  - RMS healthy, no words        -> STT problem
  - RMS + words both fine        -> problem is in the bridge, not the audio
"""
import logging
import os
import sys
import time

import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROBE_S = 45
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("probe")

def main():
    import sounddevice

    print("== devices ==")
    print(sd.query_devices(), flush=True)
    try:
        di, do = sd.default.device
        print(f"== default input={di} output={do} ==", flush=True)
    except Exception as e:
        print("no default device:", e, flush=True)

    stt = None
    t0 = time.time()
    import torch
    for dev in (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]):
        try:
            from voice_stack.stt_engine import KyutaiSTT
            stt = KyutaiSTT(hf_repo="kyutai/stt-1b-en_fr", device=dev)
            print(f"STT ready in {time.time() - t0:.1f}s ({dev})", flush=True)
            break
        except Exception as e:
            print(f"STT load on {dev} failed: {str(e)[:120]}", flush=True)
            if dev == "cuda":
                import torch as _t
                _t.cuda.empty_cache()
    if stt is None:
        print("STT unavailable - probe will report RMS only", flush=True)

    state = {"rms": 0.0, "last_mark": 0.0, "words": []}

    def on_audio(indata, frames, time_info, status):
        pcm = np.frombuffer(indata, dtype=np.float32)
        state["rms"] = 0.9 * state["rms"] + 0.1 * float(np.sqrt(np.mean(np.square(pcm))))
        now = time.time()
        if now - state["last_mark"] >= 1.0:
            state["last_mark"] = now
            print(f"t={now - t0:4.1f}s  rms={state['rms']:.5f}", flush=True)
        if stt is not None:
            for ev in stt.feed_frame(torch.from_numpy(pcm)):
                if ev.kind.value == "word":
                    state["words"].append(ev.piece)
                    print("  WORD:", ev.piece, " ->", "".join(state["words"][-12:]), flush=True)
                elif ev.kind.value == "end":
                    print("  [end-of-utterance]", flush=True)

    with sd.RawInputStream(samplerate=24000, blocksize=1920, dtype="float32",
                           channels=1, callback=on_audio):
        end = time.time() + PROBE_S
        while time.time() < end:
            time.sleep(0.2)

    words = "".join(state["words"]).replace("\u2581", " ")
    print(f"\nRESULT words transcribed: {words.strip()!r}", flush=True)

if __name__ == "__main__":
    main()
