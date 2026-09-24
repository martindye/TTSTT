"""Pro TTS engine: Kyutai TTS 1.6B (a.k.a. "1.8B"), int8-quantised on GPU.

Drop-in alternative to :class:`voice_stack.tts_engine.TTSEngine` (the small
"pocket" model) for the coding voice bridge. It exposes the same interface
the bridge expects:

- ``sample_rate``  -- int, 24000
- ``stream_sentence(sentence, stop_event)``  -- generator of 1-D
  ``np.float32`` audio chunks in [-1, 1]

Backed by the Kyutai TTS 1.6B delayed-streams model
(``kyutai/tts-1.6b-en_fr`` on Hugging Face; a 1.8B-parameter,
Moshi-derived streaming TTS: 12.5 audio frames/s, 32 codebooks, audio
delayed 1.28 s behind the text stream, CFG-distilled).

Design notes
------------
- The language model is int8-quantised via bitsandbytes by default: it is
  loaded in bf16 and then quantised in place with
  ``moshi.utils.quantize.replace_linear_with_qlinear`` (the published
  checkpoint is plain bf16, so get_moshi's own ``quantize`` kwarg -- which
  expects a checkpoint saved after quantisation -- cannot be used).
  ~2 GB of VRAM instead of ~3.7 GB; the bridge's llama-server keeps its
  own VRAM; the two share the GPU.
- Construction otherwise mirrors ``TTSModel.from_checkpoint_info``
  (moshi 0.2.13, ``moshi/models/tts.py``).
- ``generate()`` streams frames through ``on_frame``; the first
  ``delay_steps`` frames carry no audio yet. Every sentence is a fresh
  cold start: the decoder opens with a quiet 40-160 ms hum (measured
  <= ~0.02 RMS over 10 ms windows) before the first word, plus a small
  step at t=0. A fixed frame drop (the official recipe drops two)
  clips the first word whenever the model speaks early -- which is the
  normal case here, where every bridge sentence is its own cold start.
  So the lead-in is cleaned onset-aware instead: 10 ms windows below a
  measured hum ceiling are dropped, output starts at the first window
  above it, and a 20 ms fade-in covers the boundary (``_clean_lead``).
- Generation runs on a worker thread behind a lock: if the caller stops
  consuming early (``stop_event`` set, sentence cancelled), the worker is
  left to finish on its own and releases the lock, so the next sentence
  simply waits. Mimi's streaming state is never torn out from under it.
- First use of a new voice downloads a small per-voice embedding
  (~256 KB) into the Hugging Face cache.

The model is English (and French). Voices are VCTK speakers from the
``kyutai/tts-voices`` repo, referenced as ``vctk/p###_023.wav`` (or without
the ``.wav``; the model appends its own embedding suffix).
"""

from __future__ import annotations

import logging
import os
import queue
import threading

import numpy as np

log = logging.getLogger("voice_stack.kyutai_tts")

DEFAULT_REPO = "kyutai/tts-1.6b-en_fr"
# Voices are repo-relative paths to the source .wav in kyutai/tts-voices;
# the model appends its own <.sig@epoch.safetensors> embedding suffix.
DEFAULT_VOICE = "vctk/p277_023.wav"  # VCTK 277: UK, female

# The 23 "freeform" moods of the EARS p003 speaker (kyutai/tts-voices:
# ears/p003/emo_<mood>_freeform.wav plus a per-voice embedding each; the
# names keep the repo's original spelling: embarassment, extasy).
MOODS = (
    "adoration", "amazement", "amusement", "anger", "confusion",
    "contentment", "cuteness", "desire", "disappointment", "disgust",
    "distress", "embarassment", "extasy", "fear", "guilt", "interest",
    "neutral", "pain", "pride", "realization", "relief", "sadness",
    "serenity",
)
DEFAULT_MOOD_SPEAKER = "ears/p003"

# Cold-start lead-in cleanup (see _clean_lead). Measured on this box with
# drop_frames=0 over five sentences: the pre-word hum stays below
# 0.021 RMS in 10 ms windows (typically 40-160 ms long, plus a small
# step at t=0), while first-word energy reaches 0.03-0.35. The ceiling
# sits just above the measured hum and below real speech onsets.
LEAD_HUM_RMS = 0.022   # 10 ms window RMS ceiling of the pre-word hum
LEAD_FADE_S = 0.020    # fade-in applied at the first kept window
LEAD_MAX_S = 0.50      # never drop more lead-in than this (safety)
# When the gate opens on the first over-threshold window, also keep this much
# audio just before it. A plosive's attack burst (/d/, /k/) is a short low-RMS
# transient that can sit just under LEAD_HUM_RMS while the following vowel
# crosses it; without the look-back the consonant is cut and "Dropped" becomes
# "ropped". Sized to LEAD_FADE_S so the whole look-back sits inside the fade
# and any hum in it is inaudible.
LEAD_LOOKBACK_S = 0.020


# NOTE on bitsandbytes int8 on this box (RTX 5090 / sm_120, bnb 0.46.1):
# the quantise kernel stores row scales that are 127x the canonical
# dequantisation scale, and the int8 matmul kernel compensates for that
# internally, so the quantise+matmul *pair* is self-consistent and needs
# no workaround. Any code that dequantises QLinear weights directly
# would have to undo that 127x factor — so the cross-attention
# in_projs[0] (which must be a plain nn.Linear) is instead restored
# from the original pre-quantisation weights. Dissected 2026-09-23 with
# the scripts in DSH_TESTS (bnb_quant_check.py, bnb_disambiguate.py).


class KyutaiTTS16B:
    """Streaming TTS with the big Kyutai model (int8 on GPU by default)."""

    def __init__(self, voice: str = DEFAULT_VOICE, quantize: bool = True,
                 n_q: int = 32, temp: float = 0.6, cfg_coef: float = 2.0,
                 device: str = "cuda", repo: str = DEFAULT_REPO,
                 initial_padding: int = 2, max_padding: int = 8,
                 drop_frames: int = 0, moods: bool = False,
                 mood_speaker: str = DEFAULT_MOOD_SPEAKER):
        # Imported lazily so that merely importing this module (e.g. from
        # coding_voice.py's default pocket path) never pulls in torch.
        # This box cannot reach the Hugging Face hub; without this the
        # first cached lookup costs ~40 s of retry noise. Set before
        # huggingface_hub can be imported.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from moshi.models import loaders
        from moshi.models.tts import (
            DEFAULT_DSM_TTS_VOICE_REPO,
            StateMachine,
            TTSModel,
            TokenIds,
        )

        self._torch = torch
        dev = torch.device(device)
        self.quantized = bool(quantize)
        # Optional hard trim before the onset-aware gate (see
        # _clean_lead). Default 0: the gate removes the cold-start hum
        # on its own, and any fixed drop can clip first words (each
        # bridge sentence is its own cold start). Kept as an A/B knob
        # for the offline measurement runs.
        self._drop_frames = int(drop_frames)

        log.info("loading Kyutai TTS 1.6B from %s (quantized=%s, device=%s)",
                 repo, self.quantized, device)
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(repo)

        # Mirror of TTSModel.from_checkpoint_info (moshi 0.2.13) with one
        # addition: an optional int8 quantisation of the LM.
        #
        # The published checkpoint is plain bf16, so get_moshi's
        # ``quantize`` kwarg cannot be used: it expects a checkpoint that
        # was saved *after* quantisation (with the QLinear scale tensors in
        # the file). Instead we load the LM and quantise it in place on
        # the GPU. VRAM peaks at ~5.6 GB for the moment of conversion,
        # then settles at ~2 GB.
        #
        # Quantised loads must run in fp16, not bf16: the bnb int8 matmul
        # takes fp16 activations and returns fp16, and the transformer's
        # residual cast (x.to(update)) drags whole blocks to the attention
        # output's dtype. In bf16 the first cross-attention norm would hit
        # a dtype mismatch. Without quantisation there is no bnb in the
        # loop, so bf16 is fine.
        mimi = checkpoint_info.get_mimi(device=dev)
        tokenizer = checkpoint_info.get_text_tokenizer()
        lm_dtype = torch.float16 if self.quantized else torch.bfloat16
        lm = checkpoint_info.get_moshi(device=dev, dtype=lm_dtype)
        if self.quantized:
            from moshi.modules.transformer import StreamingMultiheadAttention
            from moshi.utils.quantize import replace_linear_with_qlinear
            # The cross-attention fast path (the voice-conditioning
            # attention) reads in_projs[0].weight directly as a plain
            # tensor (the fused q|k|v rows) and asserts a plain
            # nn.Linear, so it cannot run through a QLinear. Save the
            # original weights (to CPU, so no extra VRAM) before
            # quantisation and restore them below as plain fp16 Linears.
            # Restoring the original weights — rather than dequantising
            # bnb's int8 codes — sidesteps every platform-specific bnb
            # scale convention.
            kept_in_projs = []
            for m in lm.modules():
                if (isinstance(m, StreamingMultiheadAttention)
                        and getattr(m, "cross_attention", False)):
                    kept_in_projs.append(
                        (m, m.in_projs[0].weight.detach().cpu()))
            replace_linear_with_qlinear(lm)
            # Do NOT call lm.to() from here on: QLinear keeps an int8
            # weight plus a float32 scale (weight_scb) that must stay
            # float, or the bitsandbytes forward pass raises.
            for m, w0 in kept_in_projs:
                lin = torch.nn.Linear(w0.shape[1], w0.shape[0],
                                      bias=False, dtype=lm_dtype)
                with torch.no_grad():
                    lin.weight.copy_(w0.to(dev, dtype=lm_dtype))
                m.in_projs[0] = lin.to(dev)
                del w0

        model_id = checkpoint_info.raw_config["model_id"]
        voice_suffix = f".{model_id['sig']}@{model_id['epoch']}.safetensors"
        tts_config = checkpoint_info.tts_config
        self._delay_steps = int(tts_config["audio_delay"] * mimi.frame_rate)
        second_stream_ahead = int(tts_config.get("second_stream_ahead", 0))
        multistream = bool(tts_config.get("multistream", False))

        machine = StateMachine(
            token_ids=TokenIds(lm.text_card + 1),
            second_stream_ahead=second_stream_ahead,
            max_padding=max_padding,
            initial_padding=initial_padding,
        )
        self._model = TTSModel(
            lm=lm,
            mimi=mimi,
            tokenizer=tokenizer,
            voice_suffix=voice_suffix,
            voice_repo=DEFAULT_DSM_TTS_VOICE_REPO,
            machine=machine,
            delay_steps=self._delay_steps,
            multistream=multistream,
            n_q=n_q,
            temp=temp,
        )
        mimi.set_num_codebooks(n_q if not multistream else n_q // 2)
        self.sample_rate = int(mimi.sample_rate)

        # Mood mode: the voice is the speaker's neutral cut, and every one
        # of the speaker's moods is preloaded (background thread below) so
        # a later set_mood() is a dict lookup, not a re-encode. The `voice`
        # argument is ignored in this mode.
        if moods:
            voice_name = (f"{mood_speaker.rstrip('/')}"
                          "/emo_neutral_freeform.wav")
            shown = f"moods[{mood_speaker}]"
            if voice != DEFAULT_VOICE:
                log.info("moods on: --voice %s ignored "
                         "(the mood voice is the %s set)",
                         voice, mood_speaker)
        else:
            voice_name = self._normalise_voice(voice)
            shown = voice
        self._voice_path = self._model.get_voice_path(voice_name)
        log.info("voice: %s -> %s", shown, self._voice_path)
        self._conditions = self._model.make_condition_attributes(
            [self._voice_path], cfg_coef=cfg_coef)
        self._cfg_coef = cfg_coef
        self._moods: dict = {}  # mood -> condition attributes (preloaded)

        try:
            # torch.compile needs Triton (unavailable on Windows) ->
            # no_compile. CUDA graphs stay off here too: this warmup's
            # LMGen is discarded right after the call, so any graph
            # captured now would be wasted; the real per-sentence capture
            # happens in stream_sentence's worker.
            from moshi.utils.compile import no_compile, no_cuda_graph
            with no_compile(), no_cuda_graph():
                self._model.warmup([self._conditions], iters=2)
        except Exception:
            log.exception("Kyutai TTS warmup failed (continuing anyway)")

        if moods:
            # Preload in the background: 22 clips of ~60 s of reference
            # audio to encode, so keep the bridge usable (in neutral)
            # even if the first turns land before the last mood is in.
            # A mood that is not loaded yet simply keeps the current
            # voice (set_mood returns False).
            threading.Thread(
                target=self._preload_moods, args=(mood_speaker,),
                daemon=True, name="kyutai-moods").start()

        if torch.cuda.is_available():
            log.info("Kyutai TTS 1.6B ready; VRAM in use: %.2f GB",
                     torch.cuda.memory_allocated(dev) / 2 ** 30)

        self._lock = threading.Lock()  # one generation at a time

    # ------------------------------------------------------------------ util
    @staticmethod
    def _normalise_voice(voice: str) -> str:
        """Normalise to the repo-relative .wav path the model expects.

        Accepts ``vctk/p277_023``, ``vctk/p277_023.wav`` or the fully
        suffixed ``...wav.<sig>@<epoch>.safetensors``.
        """
        import re

        v = voice.strip()
        # Strip a full embedding suffix (the model appends its own).
        v = re.sub(r"\.[0-9a-f]{4,32}@[0-9]+\.safetensors$", "", v)
        if not v.lower().endswith(".wav"):
            v += ".wav"
        return v

    # --------------------------------------------------------------- moods
    def _preload_moods(self, mood_speaker: str) -> None:
        """Encode every freeform mood of the speaker once, up front.

        Each mood is a ~60 s reference clip. Encoded once at start-up the
        condition sets cost a few MB of VRAM in total, and from then on a
        mood switch is a dict lookup that touches nothing the model cares
        about mid-sentence (generate() reads ``self._conditions`` once, at
        the start of each sentence).
        """
        import time

        base = mood_speaker.rstrip("/")
        t0 = time.monotonic()
        for mood in MOODS:
            if mood == "neutral":
                continue  # the initial voice is already neutral
            rel = f"{base}/emo_{mood}_freeform.wav"
            try:
                path = self._model.get_voice_path(rel)
                self._moods[mood] = self._model.make_condition_attributes(
                    [path], cfg_coef=self._cfg_coef)
                log.info("mood ready: %s (%d loaded)", mood,
                         len(self._moods))
            except Exception:
                log.exception("mood preload failed: %s", rel)
        # Register neutral itself so a mood can always be *reset* back to
        # the starting voice (set_mood("neutral") works like any other).
        self._moods["neutral"] = self._conditions
        log.info("mood preload done: %d moods in %.1f s",
                 len(self._moods), time.monotonic() - t0)

    def set_mood(self, mood: str) -> bool:
        """Speak subsequent sentences in ``mood``.

        Call it between sentences (the bridge switches at turn
        boundaries): generation reads ``self._conditions`` at the start
        of each sentence, so the swap is free and never splits a
        sentence. Returns False when the mood is not loaded (the current
        voice is kept).
        """
        cond = self._moods.get(mood)
        if cond is None:
            return False
        self._conditions = cond
        return True

    def mood_loaded(self, mood: str) -> bool:
        return mood in self._moods

    # --------------------------------------------------------------- stream
    def _clean_lead(self, arr: np.ndarray, state: dict) -> np.ndarray:
        """Drop the cold-start hum before the first word, keep the word.

        Called with every decoded frame while ``state["lead"]``. Scans
        the frame in 10 ms windows: everything below ``LEAD_HUM_RMS``
        (the pre-word hum, the t=0 step) is dropped; the first window
        at or above the ceiling starts the output. When the gate opens,
        ``LEAD_LOOKBACK_S`` (20 ms, equal to the fade) just before the
        onset is kept as well: a plosive's attack burst (the /d/ of
        "Dropped", the /k/ of "Crackle") can sit just under the
        ceiling, and losing it turns "Dropped" into "ropped". Because
        the look-back equals the fade, that whole region is inside the
        20 ms fade-in, so any hum in it is inaudible and the cut
        itself cannot click.

        Safety valve: after ``LEAD_MAX_S`` of dropped lead-in (a
        pathologically soft sentence) everything passes through.
        Returns an empty array when the whole frame was dropped.
        """
        sr = self.sample_rate
        win = max(1, sr // 100)
        pos = 0
        while pos < len(arr):
            if state["lead_s"] >= LEAD_MAX_S:
                return self._fade_front(arr, state)
            chunk = arr[pos:pos + win]
            rms = float(np.linalg.norm(chunk) / np.sqrt(chunk.size)) \
                if chunk.size else 0.0
            if rms >= LEAD_HUM_RMS:
                break
            pos += win
            state["lead_s"] += win / sr
        if pos >= len(arr):
            if state["lead_s"] >= LEAD_MAX_S:
                return self._fade_front(arr, state)  # pass it through
            return np.zeros(0, dtype=np.float32)    # drop the frame
        # Look back before the trigger window: the plosive burst of the
        # first word (the /d/ of "Dropped") may sit just under the ceiling
        # in the 10-20 ms just before the first over-threshold window.
        start = max(pos - int(LEAD_LOOKBACK_S * sr), 0)
        return self._fade_front(arr[start:], state)

    def _fade_front(self, arr: np.ndarray, state: dict) -> np.ndarray:
        """20 ms linear fade-in, then disarm the lead gate for good."""
        fade = min(len(arr), int(LEAD_FADE_S * self.sample_rate))
        if fade:
            arr = arr.copy()
            arr[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        state["lead"] = False
        return arr

    def stream_sentence(self, sentence: str, stop_event):
        """Yield 1-D float32 chunks ([-1, 1]) of one spoken sentence.

        ``stop_event``: optional ``threading.Event``; when set, the
        generator returns as soon as it can. The worker that generates the
        rest of the sentence keeps running in the background (it is cheap
        relative to speaking and must not be torn out from under the
        model's streaming state); the next sentence waits for it.
        """
        sentence = (sentence or "").strip()
        if not sentence:
            return

        self._lock.acquire()
        q: "queue.Queue" = queue.Queue()
        done = object()
        state = {"frame": 0, "decoded": 0, "lead": True, "lead_s": 0.0}
        torch = self._torch

        def on_frame(frame):
            i = state["frame"]
            state["frame"] += 1
            if i < self._delay_steps:
                return  # audio is still inside the 1.28 s delay
            try:
                pcm = self._model.mimi.decode(frame[:, 1:, :])
                pcm = pcm.float().clamp(-1.0, 1.0)  # mimi may output bf16
            except Exception:
                log.exception("Kyutai TTS frame decode failed; "
                              "ending sentence")
                q.put(done)
                return
            state["decoded"] += 1
            if state["decoded"] <= self._drop_frames:
                return  # optional hard trim, before the onset-aware gate
            arr = (pcm[0, 0].detach().cpu().numpy()
                   .astype(np.float32, copy=False))
            if arr.size:
                if state["lead"]:
                    arr = self._clean_lead(arr, state)
                    if not arr.size:
                        return
                q.put(arr)

        def worker():
            try:
                # prepare_script returns one Entry PER WORD, so batch is a
                # list of entry-LISTS: exactly one item here (one sentence).
                batch = [self._model.prepare_script([sentence])]
                try:
                    # no_compile: torch.compile needs Triton (absent on
                    # Windows). CUDA graphs stay ON: capture is Triton-free
                    # and turns the eager ~700 ms/step loop (main LM plus
                    # the 32-step depformer) into cheap graph replays.
                    from moshi.utils.compile import no_compile
                    with no_compile(), \
                            self._model.mimi.streaming(len(batch)):
                        # generate() takes a list of entry-lists (one per
                        # condition set); we pass exactly one.
                        self._model.generate(batch, [self._conditions],
                                             on_frame=on_frame)
                except Exception:
                    # If CUDA-graph capture of the int8 kernels fails on
                    # this stack, fall back to a plain eager pass.
                    log.exception("graphed generation failed; "
                                  "retrying without CUDA graphs")
                    from moshi.utils.compile import no_compile, \
                        no_cuda_graph
                    with no_compile(), no_cuda_graph(), \
                            self._model.mimi.streaming(len(batch)):
                        self._model.generate(batch, [self._conditions],
                                             on_frame=on_frame)
            except Exception:
                log.exception("Kyutai TTS generation failed")
            finally:
                q.put(done)
                self._lock.release()

        threading.Thread(target=worker, daemon=True,
                         name="kyutai-tts-gen").start()
        while True:
            if stop_event is not None and stop_event.is_set():
                return  # worker finishes on its own and releases the lock
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is done:
                return
            yield item


def main(argv=None) -> int:
    """Offline smoke test: load the model, speak one line, write a WAV."""
    import argparse
    import time
    import wave

    ap = argparse.ArgumentParser(
        description="Kyutai TTS 1.6B smoke test (see module docstring)")
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--no-quantize", action="store_true",
                    help="load the LM unquantised (bf16, ~3.7 GB VRAM)")
    ap.add_argument("--n-q", type=int, default=32)
    ap.add_argument("--cfg-coef", type=float, default=2.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--moods", action="store_true",
                    help="mood mode: neutral cut of the EARS p003 "
                         "speaker, all 23 moods preloaded (voice arg "
                         "ignored)")
    ap.add_argument("--text",
                    default="This is the Pro voice. If you can hear this "
                            "clearly, the big model is working.")
    ap.add_argument("--out", default=None, help="write the audio to a WAV")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    engine = KyutaiTTS16B(voice=args.voice,
                          quantize=not args.no_quantize,
                          n_q=args.n_q, cfg_coef=args.cfg_coef,
                          device=args.device, repo=args.repo,
                          moods=args.moods)

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


if __name__ == "__main__":
    raise SystemExit(main())
