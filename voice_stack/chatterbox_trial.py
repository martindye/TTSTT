"""Trial bake for Chatterbox-Turbo as the GPU voice replacement.

Bakes TTSTT\\samples\\chatterbox-*.wav from the kit reference clips and
reports peak VRAM + real-time factor. Run with the chatterbox venv:
    C:\\Users\\press\\.dsh\\voice-chat\\venv-chatterbox\\Scripts\\python.exe
        voice_stack\\chatterbox_trial.py
"""
from __future__ import annotations

import os
import time

import torch
import torchaudio as ta

OUT_DIR = r"C:\Users\press\OneDrive\Projects\TTSTT\samples"
REF_DIR = r"C:\Users\press\OneDrive\Projects\DSH_TESTS\dsh-voice-chat\ref_voices"
TMP_DIR = os.path.join(os.environ["TEMP"], "chatterbox-refs")
os.makedirs(TMP_DIR, exist_ok=True)

# Same character as the earlier 17 s A/B passage.
TEXT = (
    "The kettle's just boiled, so have a cup of tea while this trial "
    "finishes. The weather's turned rather soft again this afternoon, "
    "exactly as the forecast promised. If the voice sounds right to you, "
    "we can make it the default for the replies, and the old GPU voice "
    "stays one word away if you miss it."
)

# Turbo's paralinguistic tags are its flavour channel.
TAGGED = (
    "So how did it go? [laugh] I knew you'd come running back for the "
    "results. [chuckle] Honestly, this one's faster than the old setup."
)


def prep_ref(name: str, src: str, max_seconds: float = 10.0) -> str:
    """Trim the kit reference clip to ~10 s and hand back a path (turbo
    wants a file, and ~10 s is its preferred prompt length)."""
    wav, sr = ta.load(src)
    if wav.shape[0] > 1:
        wav = wav[0:1]
    limit = int(max_seconds * sr)
    if wav.shape[-1] > limit:
        wav = wav[..., :limit]
    dst = os.path.join(TMP_DIR, f"ref-{name}.wav")
    ta.save(dst, wav, sr)
    return dst


def main() -> None:
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    t0 = time.time()
    model = ChatterboxTurboTTS.from_pretrained(device="cuda")
    print(f"loaded in {time.time() - t0:.1f}s, sr={model.sr}", flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    refs = {
        "pocket": prep_ref("pocket", f"{REF_DIR}\\pocket.wav"),
        "warm-brit": prep_ref("warm-brit", f"{REF_DIR}\\warm_brit.wav"),
    }

    for name, path in refs.items():
        torch.cuda.synchronize()
        t0 = time.time()
        wav = model.generate(TEXT, audio_prompt_path=path)
        elapsed = time.time() - t0
        dur = wav.shape[-1] / model.sr
        ta.save(os.path.join(OUT_DIR, f"chatterbox-{name}.wav"), wav, model.sr)
        print(f"{name}: {elapsed:.1f}s to make {dur:.1f}s of audio "
              f"(RTF {elapsed / dur:.2f})", flush=True)

    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 2**20
    free_mb = torch.cuda.mem_get_info()[0] / 2**20
    t0 = time.time()
    wav = model.generate(TAGGED, audio_prompt_path=refs["pocket"])
    ta.save(os.path.join(OUT_DIR, "chatterbox-pocket-tagged.wav"), wav, model.sr)
    print(f"tagged: {time.time() - t0:.1f}s")
    print(f"peak VRAM: {peak_mb:.0f} MiB allocated, {free_mb:.0f} MiB still free")
    print("done")


if __name__ == "__main__":
    main()
