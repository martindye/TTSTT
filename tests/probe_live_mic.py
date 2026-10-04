"""Direct microphone/STT check; no gateway injection or speaker playback."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import queue
import time
import numpy as np
import soundfile as sf
import torch
from voice_stack.stt_engine import KyutaiSTT, SttEventKind
from voice_stack.audio_io import MicCapture

stt = KyutaiSTT(device='cuda')
q = queue.Queue()
mic = MicCapture(q.put)
mic.start()
print('MIC READY', flush=True)
start = last = time.monotonic()
frames = []
try:
    while time.monotonic()-start < 120:
        pcm = q.get(timeout=5)
        frames.append(pcm)
        for ev in stt.feed_frame(torch.from_numpy(pcm)):
            if ev.kind is SttEventKind.WORD:
                print('WORD', repr(ev.piece), flush=True)
        if time.monotonic()-last > 10:
            print('MIC RMS', round(mic.input_rms,5), 'queue',q.qsize(),flush=True)
            last = time.monotonic()
finally:
    mic.close()
    sf.write('tests/mic_live_recovery_check.wav', np.concatenate(frames), 24000)
