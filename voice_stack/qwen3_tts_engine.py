"""Pro TTS engine: Qwen3-TTS 12 Hz 0.6B-Base, voice-cloned from a reference clip.

Drop-in alternative to :class:`voice_stack.kyutai_tts_engine.KyutaiTTS16B`
for the coding voice bridge. Same interface the bridge expects:

- ``sample_rate``  -- int, 24000
- ``stream_sentence(sentence, stop_event)``  -- generator of 1-D
  ``np.float32`` audio chunks in [-1, 1]

Backed by ``Qwen/Qwen3-TTS-12Hz-0.6B-Base`` (the ``qwen-tts`` package,
``Qwen3TTSModel``): a 0.6 B language model over a 12 Hz x 16-codebook
speech codec, bf16 on cuda:0, ~2.2 GB of VRAM. The model has no
pre-trained named voices (the Base line is clone-only), so the voice is a
clone of a reference clip built once at startup with
``create_voice_clone_prompt(ref_audio=..., x_vector_only_mode=True)``;
every sentence reuses that prompt.

Design notes
------------
- **Batch generation.** ``Qwen3TTSModel.generate_voice_clone`` has no
  frame callback: one call returns the *complete* audio for the sentence.
  So each sentence is fully generated on a worker thread before the first
  chunk can flow to the speaker. First-audio latency is the whole-sentence
  generation time (RTF ~4.6 with SDPA attention on this box; ~7 without),
  so the pause after each sentence is a few times the sentence's length;
  afterwards the audio streams out in 80 ms chunks (one codec frame
  each).
- **SDPA attention.** The model is loaded with
  ``attn_implementation="sdpa"`` (measured ~1.5x faster generation than
  the default eager attention on this machine; no flash-attn wheel for
  Windows).  Falls back to the default attention if SDPA is unavailable.
- **Silence trimming.** Each sentence's audio is trimmed of its leading
  silence and capped at ~150 ms of trailing silence so the model's own
  padding does not stack on top of the generation pause.
- Generation runs behind a lock on a worker thread, same as the Kyutai
  engine: if the caller stops consuming early (``stop_event`` set), the
  worker is left to finish (torn out mid-``generate`` the shared model
  state would be corrupted) and the audio is simply discarded; the next
  sentence waits for the lock.
- No moods: the engine exposes no ``set_mood``/``moods_enabled``, so the
  bridge's mood system is a silent no-op (tags stripped, fixed voice).
- The bridge runs offline (``HF_HUB_OFFLINE=1``), so the model is loaded
  from a **local snapshot directory**, never the hub id.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time

import numpy as np

log = logging.getLogger("voice_stack.qwen3_tts")

# Local HF snapshot of Qwen/Qwen3-TTS-12Hz-0.6B-Base (downloaded directly
# into the cache dir; the hub download path cannot create snapshot
# symlinks on this machine).
DEFAULT_MODEL_DIR = (
    r"C:\Users\press\.cache\huggingface\hub"
    r"\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots"
    r"\5d83992436eae1d760afd27aff78a71d676296fc"
)
# The current plain pro voice (VCTK 277), from the Kyutai voice repo cache:
# the clone target, so the Qwen3 engine replaces it 1:1.
DEFAULT_REF = (
    r"C:\Users\press\.cache\huggingface\hub"
    r"\models--kyutai--tts-voices\snapshots\323332d33f997de8394f24a193e1a76df720e01a"
    r"\vctk\p277_023.wav"
)
DEFAULT_LANGUAGE = "English"

# 80 ms of 24 kHz audio = one 12 Hz codec frame.
_FRAME_SAMPLES = 1920


def _trim_silence(audio: np.ndarray, sr: int = 24000,
                  threshold: float = 0.012, max_lead_ms: int = 100,
                  max_tail_ms: int = 150) -> np.ndarray:
    """Trim leading silence and cap trailing silence on generated audio.

    Qwen3-TTS pads each sentence with up to ~140 ms of leading and up to
    ~400 ms of trailing silence.  That trailing padding plays *after* the
    sentence, right into the gap before the next sentence's generation
    finishes, so it stacks on top of the generation pause.  Keep at most
    ``max_lead_ms`` of silence before the first energy and
    ``max_tail_ms`` after the last.
    """
    if audio.size < sr // 10:  # under 100 ms: nothing worth trimming
        return audio
    above = np.nonzero(np.abs(audio) > threshold)[0]
    if above.size == 0:
        return audio
    start = max(0, int(above[0]) - int(sr * max_lead_ms / 1000))
    end = min(audio.size - 1, int(above[-1]) + int(sr * max_tail_ms / 1000))
    return audio[start:end + 1]


class Qwen3TTS:
    """Qwen3-TTS 0.6B engine: a clone of ``ref_audio`` speaking English.

    ``model_dir`` and ``ref_audio`` default to the machine-local snapshot
    and the VCTK p277_023 clip (the current plain pro voice).
    """

    sample_rate = 24000  # confirmed from the model's returned sample rate

    def __init__(self, model_dir: str | None = None,
                 ref_audio: str | None = None,
                 language: str = DEFAULT_LANGUAGE) -> None:
        # The bridge runs fully offline; never let the hub client phone home.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        import torch  # noqa: F401  (imported here: heavy)
        from qwen_tts import Qwen3TTSModel

        self._model_dir = model_dir or DEFAULT_MODEL_DIR
        self._ref = ref_audio or DEFAULT_REF
        self._language = language

        t0 = time.monotonic()
        log.info("loading Qwen3-TTS 0.6B from %s (bf16, cuda:0, sdpa)...",
                 self._model_dir)
        # SDPA attention: measured ~1.5x faster generation than the default
        # eager ("manual PyTorch") attention on this box (RTF 4.6 vs 7.0).
        # Fall back to the default if this build lacks SDPA support.
        try:
            self._model = Qwen3TTSModel.from_pretrained(
                self._model_dir, device_map="cuda:0", dtype=torch.bfloat16,
                attn_implementation="sdpa")
        except Exception as e:
            log.warning("sdpa attention failed (%s: %s); "
                        "falling back to default attention",
                        type(e).__name__, e)
            self._model = Qwen3TTSModel.from_pretrained(
                self._model_dir, device_map="cuda:0", dtype=torch.bfloat16)
        self._torch = torch

        log.info("building voice clone prompt from %s (x-vector only)...",
                 self._ref)
        t0 = time.monotonic()
        self._prompt = self._model.create_voice_clone_prompt(
            ref_audio=self._ref, x_vector_only_mode=True)
        log.info("clone prompt ready in %.1f s", time.monotonic() - t0)

        if torch.cuda.is_available():
            log.info("Qwen3-TTS ready; VRAM allocated: %.2f GB",
                     torch.cuda.memory_allocated() / 2 ** 30)

        self._lock = threading.Lock()  # one generation at a time

    # ------------------------------------------------------------- speaking
    def _generate(self, sentence: str):
        return self._model.generate_voice_clone(
            text=sentence, language=self._language,
            voice_clone_prompt=self._prompt)

    def stream_sentence(self, sentence: str, stop_event=None):
        """Yield 1-D ``np.float32`` chunks in [-1, 1] for ``sentence``.

        Batch engine: the whole sentence is generated first (worker
        thread), then streamed out in 80 ms chunks. If ``stop_event`` is
        set before generation finishes the audio is discarded (the worker
        still runs to completion so the shared model state is never torn
        out from under it) and the generator yields nothing.
        """
        if not sentence or not sentence.strip():
            return

        with self._lock:
            if stop_event is not None and stop_event.is_set():
                return
            box: dict = {}

            def worker() -> None:
                try:
                    wavs, sr = self._generate(sentence)
                    box["wav"] = np.concatenate(
                        [np.asarray(w, dtype=np.float32) for w in wavs])
                    if box["wav"].ndim > 1:
                        box["wav"] = box["wav"].mean(axis=-1)
                    box["sr"] = int(sr)
                except Exception as e:  # surfaced to the caller
                    box["error"] = e
                finally:
                    _sentinel.put(None)

            _sentinel = queue.Queue()
            th = threading.Thread(target=worker, daemon=True,
                                  name="qwen3tts-gen")
            t0 = time.monotonic()
            th.start()
            _sentinel.get()  # sentinel only: generation must finish
            th.join()

            if "error" in box:
                raise box["error"]

            if stop_event is not None and stop_event.is_set():
                log.info("sentence cancelled: %.1f s of audio discarded",
                         len(box.get("wav", np.zeros(0))) / self.sample_rate)
                return

            sr = box["sr"]
            if sr != self.sample_rate:
                log.warning("model returned %d Hz, bridge expects %d",
                            sr, self.sample_rate)
            # Drop the model's padding silence so it does not stack on top
            # of the (already long) generation pause between sentences.
            wav = _trim_silence(box["wav"])
            log.info("generated %.2f s of audio in %.2f s "
                     "(RTF %.2f)", len(wav) / sr, time.monotonic() - t0,
                     (time.monotonic() - t0) / max(len(wav) / sr, 1e-9))

            for i in range(0, len(wav), _FRAME_SAMPLES):
                if stop_event is not None and stop_event.is_set():
                    return
                yield wav[i:i + _FRAME_SAMPLES].copy()


# --------------------------------------------------------------------- CLI
if __name__ == "__main__":  # pragma: no cover - manual preview
    import argparse
    import wave

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s: %(message)s")

    def _main() -> int:
        ap = argparse.ArgumentParser(
            description="Qwen3-TTS 0.6B preview: speak one sentence with the "
                        "cloned pro voice")
        ap.add_argument("--text",
                        default="Hello. This is the Qwen three point TTS "
                                "engine, speaking the pro voice.")
        ap.add_argument("--out", default=None, help="write a 16-bit wav here")
        ap.add_argument("--model", default=None,
                        help="model dir (default: local 0.6B snapshot)")
        ap.add_argument("--ref", default=None,
                        help="reference clip to clone (default: p277_023)")
        args = ap.parse_args()

        engine = Qwen3TTS(model_dir=args.model, ref_audio=args.ref)
        t0 = time.monotonic()
        chunks = list(engine.stream_sentence(args.text, None))
        elapsed = time.monotonic() - t0
        audio = (np.concatenate(chunks) if chunks
                 else np.zeros(0, dtype=np.float32))
        seconds = audio.size / engine.sample_rate
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        print(f"audio: {seconds:.2f} s generated in {elapsed:.1f} s "
              f"(RTF {elapsed / max(seconds, 1e-6):.2f}), "
              f"peak amplitude {peak:.3f}")
        if args.out:
            raw = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
            with wave.open(args.out, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(engine.sample_rate)
                w.writeframes(raw.tobytes())
            print(f"wrote {args.out}")
        return 0

    raise SystemExit(_main())
