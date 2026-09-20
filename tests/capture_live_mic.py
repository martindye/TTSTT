"""Live-mic forensic capture (queue-based, bridge-identical audio path).

60s window:
  - callback thread: copies each PCM block -> recording list AND worker queue
  - worker thread: feeds a live STT (exact same path as the voice bridge)
  - end: feeds the SAME recorded audio to a FRESH STT (control)

Compare:
  live-STT text  vs  fresh-STT text  vs  RMS profile
  -> level/SNR problem, room masking, or process-state bug.
"""
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

sys.path.insert(0, r"C:\Users\press\OneDrive\Projects\TTSTT")
from voice_stack.stt_engine import KyutaiSTT, SttEventKind  # noqa: E402

CAP_S = 60
OUT_WAV = r"C:\Users\press\OneDrive\Projects\TTSTT\tests\live_mic_capture.wav"
FRAME = 1920


def main():
    print(sd.query_devices(), flush=True)
    t0 = time.time()
    stt = KyutaiSTT(hf_repo="kyutai/stt-1b-en_fr", device="cuda")
    print(f"STT ready in {time.time() - t0:.1f}s - SPEAK NOW, window is {CAP_S}s",
          flush=True)

    q: "queue.Queue" = queue.Queue()
    chunks: "list[np.ndarray]" = []
    rms_state = {"v": 0.0, "mark": 0.0}

    def on_audio(indata, frames, time_info, status):
        pcm = np.frombuffer(indata, dtype=np.float32).copy()
        rms_state["v"] = 0.9 * rms_state["v"] + 0.1 * float(
            np.sqrt(np.mean(np.square(pcm))))
        now = time.time()
        if now - rms_state["mark"] >= 1.0:
            rms_state["mark"] = now
            print(f"t={now - t0:5.1f}s rms={rms_state['v']:.4f}", flush=True)
        chunks.append(pcm)   # recording
        q.put(pcm)           # STT worker

    def worker():
        live_words = []
        ends = 0
        while True:
            pcm = q.get()
            if pcm is None:  # sentinel: window over
                break
            for ev in stt.feed_frame(torch.from_numpy(pcm)):
                if ev.kind is SttEventKind.WORD:
                    live_words.append(ev.piece)
                    print("   LIVE WORD:", repr(ev.piece),
                          "->", repr("".join(live_words)[-60:]), flush=True)
                elif ev.kind is SttEventKind.END:
                    ends += 1
        print(f"LIVE STT words: {len(live_words)}, ends={ends}", flush=True)
        print("LIVE TEXT:",
              repr("".join(live_words).replace("\u2581", " ")), flush=True)

    wt = threading.Thread(target=worker, daemon=True)
    wt.start()

    with sd.RawInputStream(samplerate=24000, blocksize=FRAME, dtype="float32",
                           channels=1, callback=on_audio):
        end = time.time() + CAP_S
        while time.time() < end:
            time.sleep(0.1)

    q.put(None)  # sentinel: let the worker drain the rest
    wt.join(timeout=60)

    audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    sf.write(OUT_WAV, audio, 24000)
    print(f"\ncaptured {len(audio) / 24000:.1f}s -> {OUT_WAV}", flush=True)

    print("loading fresh STT for control transcription ...", flush=True)
    stt2 = KyutaiSTT(hf_repo="kyutai/stt-1b-en_fr", device="cuda")
    words = []
    for i in range(0, len(audio), FRAME):
        frame = audio[i:i + FRAME]
        if len(frame) < FRAME:
            frame = np.pad(frame, (0, FRAME - len(frame)))
        for ev in stt2.feed_frame(torch.from_numpy(frame)):
            if ev.kind is SttEventKind.WORD:
                words.append(ev.piece)
    text = "".join(words).replace("\u2581", " ")
    print(f"\nFRESH STT on the same recording ({len(words)} words):")
    print("   ", repr(text))


if __name__ == "__main__":
    main()
