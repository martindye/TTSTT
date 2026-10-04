"""Opt-in GPU integration: three turns, then a deliberately deaf decoder.

Uses local reference audio. Never opens the mic, plays sound, or injects chat.
Run from the repo root with the live bridge stopped to leave GPU headroom.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import threading
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
from voice_stack.coding_voice import CodingVoice, FRAME_S, STT_REPLAY_S
from voice_stack.stt_engine import KyutaiSTT

stt = KyutaiSTT(device='cuda')
cv = CodingVoice.__new__(CodingVoice)
cv.stt = stt
cv._stop = threading.Event()
cv._stt_reset_pending = threading.Event()
cv._stt_unheard = deque(maxlen=round(STT_REPLAY_S / FRAME_S))
cv._stt_wordless_frames = 0
cv._stt_raw_words = 0
cv._stt_replaying = False
cv.spoken = SimpleNamespace(busy_audio=False, tts_generating=False)
words = []
cv._on_word = lambda piece, mail=False: words.append(piece)
audio, sr = sf.read('voice_stack/ref_voices/british_p225.wav', dtype='float32')
assert sr == 24000
audio = np.pad(audio, (0, 4*sr))
results = []

def passage(label):
    words.clear()
    start = time.perf_counter()
    for i in range(0, len(audio), 1920):
        frame = audio[i:i+1920]
        frame = np.pad(frame, (0, 1920-len(frame)))
        cv._process_stt_frame(frame, False)
    text = ''.join(words).replace('\u2581', ' ').strip()
    result = dict(label=label, seconds=round(time.perf_counter()-start, 3),
                  text=text, allocated_mb=round(torch.cuda.memory_allocated()/1024**2))
    results.append(result)
    print(json.dumps(result), flush=True)
    assert 'red of a second bow' in text.lower(), text
    assert text.lower().count('red of a second bow') == 1, text

for turn in range(3):
    cv._stt_reset_pending.set()
    passage(f'after reply {turn+1}')

# Forced fault exercises the actual GPU stream reset and retained-audio retry.
# It is not evidence that corrupt samples caused the user's live failure.
# Bypass the bridge's sample sanitiser to poison the actual encoder caches.
stt.feed_frame(torch.full((1920,), float('nan')))
cv._stt_unheard.clear()
cv._stt_wordless_frames = 0
passage('forced deaf state, automatic retry')
assert results[-1]['allocated_mb'] <= results[0]['allocated_mb'] + 100
Path('tests/stt_recovery_result.json').write_text(json.dumps(results, indent=2))
stt.close()
print('PASS', flush=True)
