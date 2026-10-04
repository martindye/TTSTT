"""Offline reproduction: main-thread load, worker STT, Pocket between turns."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import threading
import time
import numpy as np
import soundfile as sf
import torch
from voice_stack.stt_engine import KyutaiSTT, SttEventKind
from voice_stack.tts_engine import TTSEngine

stt = KyutaiSTT(device='cuda')
audio, sr = sf.read('voice_stack/ref_voices/british_p225.wav', dtype='float32')
assert sr == 24000
audio = np.concatenate([audio, np.zeros(3 * sr, dtype=np.float32)])

def feed(label, pcm):
    words = []
    kinds = {}
    start = time.perf_counter()
    for i in range(0, len(pcm), 1920):
        frame = np.pad(pcm[i:i+1920], (0, max(0, i+1920-len(pcm))))
        for ev in stt.feed_frame(torch.from_numpy(frame)):
            kinds[ev.kind.value] = kinds.get(ev.kind.value, 0) + 1
            if ev.kind is SttEventKind.WORD:
                words.append(ev.piece)
    print(label, round(time.perf_counter()-start, 2), kinds,
          repr(''.join(words).replace('\u2581', ' ')), flush=True)

feed('main before Pocket', audio)
tts = TTSEngine(voice='anna')
def worker():
    feed('worker after Pocket load', audio)
    for n in range(4):
        list(tts.stream_sentence('The test has finished. You can speak again.', threading.Event()))
        feed(f'worker after reply {n+1}', audio)
    feed('ninety seconds quiet', np.zeros(90*sr, dtype=np.float32))
    feed('worker after long quiet', audio)
    stt.reset()
    feed('worker after reset', audio)

t = threading.Thread(target=worker)
t.start()
t.join()
