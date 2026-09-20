"""Command-line entry point for the local voice stack.

Live mode (default):
    python -m voice_stack

Offline self-test (no mic/speakers needed):
    python -m voice_stack --selftest path/to/speech.wav
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np
import torch

DEFAULT_LLM_URL = "http://127.0.0.1:8080/v1"
DEFAULT_LLM_MODEL = "qwen3.8-27b"


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="voice_stack", description=__doc__)
    p.add_argument("--selftest", metavar="WAV",
                   help="offline mode: transcribe WAV, answer with the LLM, "
                        "synthesize the answer to a wav file, and exit")
    p.add_argument("--llm-url", default=DEFAULT_LLM_URL)
    p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    p.add_argument("--llm-max-tokens", type=int, default=400)
    p.add_argument("--stt-repo", default="kyutai/stt-1b-en_fr")
    p.add_argument("--stt-device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--tts-language", default="english")
    p.add_argument("--tts-voice", default="alba")
    p.add_argument("--tts-quantize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--think", action="store_true",
                   help="allow the LLM to think (keeps internal reasoning; "
                        "slower). Off by default for low-latency voice.")
    p.add_argument("--mic-device", default=None, help="PortAudio input device index/name")
    p.add_argument("--spk-device", default=None, help="PortAudio output device index/name")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--echo-guard", type=int, default=3,
                   help="user words needed to barge in (barge-in mode)")
    p.add_argument("--barge-in", action="store_true",
                   help="allow interrupting the assistant mid-answer "
                        "(use headphones; with speakers the mic hears the "
                        "assistant's own voice and may false-trigger)")
    p.add_argument("--out", default="voice_answer.wav",
                   help="self-test output wav path")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("urllib3", "requests", "huggingface_hub", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def make_stt(repo: str, device: str):
    """Build the STT engine, falling back GPU -> CPU on OOM."""
    from .stt_engine import KyutaiSTT

    order = {"auto": ["cuda", "cpu"], "cuda": ["cuda"], "cpu": ["cpu"]}[device]
    last_err: Exception | None = None
    for dev in order:
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        try:
            log = logging.getLogger("voice")
            log.info("loading Kyutai STT on %s ...", dev)
            t0 = time.time()
            stt = KyutaiSTT(hf_repo=repo, device=dev)
            log.info("STT ready in %.1fs (frame=%d samples @ %d Hz, delay_tokens=%d)",
                     time.time() - t0, stt.frame_size, stt.sample_rate, stt.delay_tokens)
            return stt
        except Exception as e:  # OOM etc.
            last_err = e
            msg = str(e).lower()
            if dev == "cuda" and ("out of memory" in msg or "cuda" in msg):
                logging.getLogger("voice").warning(
                    "GPU load failed (%s); falling back to CPU", str(e)[:200])
                torch.cuda.empty_cache()
                continue
            raise
    raise RuntimeError(f"could not load STT on any device: {last_err}")


def make_llm(args):
    from .llm_client import LLMClient

    llm = LLMClient(
        base_url=args.llm_url,
        model=args.llm_model,
        max_tokens=args.llm_max_tokens,
        disable_thinking=not args.think,
    )
    if not llm.health():
        logging.getLogger("voice").warning(
            "LLM server at %s is not answering /v1/models - it may still be "
            "loading; the assistant will retry per turn.", args.llm_url)
    return llm


def make_tts(args):
    from .tts_engine import TTSEngine

    tts = TTSEngine(
        language=args.tts_language,
        voice=args.tts_voice,
        quantize=args.tts_quantize,
    )
    return tts


def run_selftest(args, stt, llm, tts):
    """Offline end-to-end: WAV in -> text -> LLM -> TTS wav out."""
    import soundfile as sf

    log = logging.getLogger("voice")
    log.info("selftest: reading %s", args.selftest)
    audio, sr = sf.read(args.selftest, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1) if audio.ndim == 2 else audio
    if sr != stt.sample_rate:
        import scipy.signal

        n = int(len(audio) * stt.sample_rate / sr)
        audio = scipy.signal.resample_poly(audio, stt.sample_rate, sr)[:n]
        log.info("resampled %d Hz -> %d Hz", sr, stt.sample_rate)

    t0 = time.time()
    text_parts: list[str] = []
    n_frames = len(audio) // stt.frame_size
    remaining = len(audio) - n_frames * stt.frame_size
    for i in range(n_frames):
        frame = audio[i * stt.frame_size:(i + 1) * stt.frame_size]
        for ev in stt.feed_frame(torch.from_numpy(frame)):
            if ev.kind.value == "word":
                text_parts.append(ev.piece)
    if remaining:
        pad = np.zeros(stt.frame_size - remaining, dtype=np.float32)
        for ev in stt.feed_frame(torch.from_numpy(np.concatenate([audio[-remaining:], pad]))):
            if ev.kind.value == "word":
                text_parts.append(ev.piece)
    # The model has a ~0.5 s built-in delay, so the trailing words only
    # materialize after ~1 s of following silence. Feed that explicitly for a
    # file input (the live mic provides it naturally).
    silent = torch.zeros(stt.frame_size, dtype=torch.float32)
    for _ in range(15):  # ~1.2 s of silence
        for ev in stt.feed_frame(silent):
            if ev.kind.value == "word":
                text_parts.append(ev.piece)
    transcript = "".join(text_parts).replace("▁", " ").strip()
    stt_time = time.time() - t0
    log.info("STT transcript (%.1fs): %r", stt_time, transcript)
    if not transcript:
        log.error("selftest: no transcript produced")
        return 1

    stt.reset()

    import threading

    cancel = threading.Event()
    messages = [
        {"role": "system", "content": llm.system_prompt},
        {"role": "user", "content": transcript},
    ]
    t0 = time.time()
    chunks: list[str] = []
    for delta in llm.stream_chat(messages, cancel):
        chunks.append(delta)
    answer = "".join(chunks).strip()
    llm_time = time.time() - t0
    log.info("LLM answer in %.1fs:\n%s", llm_time, answer)
    if not answer:
        log.error("selftest: empty LLM answer")
        return 1

    # Speak the answer (sentence by sentence) into one buffer.
    import re

    t0 = time.time()
    out: list[np.ndarray] = []
    buf = answer
    import math

    def feed(s: str):
        for chunk in tts.stream_sentence(s, cancel):
            out.append(chunk)

    # sentence split
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    for s in sentences:
        if s.strip():
            feed(s)
    tts_time = time.time() - t0
    audio_out = np.concatenate(out) if out else np.zeros(0)
    log.info("TTS: %.1fs of audio in %.1fs", len(audio_out) / tts.sample_rate, tts_time)

    import soundfile as sf2

    sf2.write(args.out, audio_out, tts.sample_rate)
    log.info("wrote %s (%.1f s @ %d Hz)", args.out, len(audio_out) / tts.sample_rate, tts.sample_rate)
    log.info("SELFTEST OK")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    log = logging.getLogger("voice")

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return 0

    # ---------------- selftest mode ----------------
    if args.selftest:
        stt = make_stt(args.stt_repo, args.stt_device)
        llm = make_llm(args)
        tts = make_tts(args)
        return run_selftest(args, stt, llm, tts)

    # ---------------- live mode ----------------
    import sounddevice as sd

    try:
        default_in = sd.default.device[0]
        default_out = sd.default.device[1]
    except Exception:
        default_in = default_out = None
    log.info("default input device: %s | output device: %s", default_in, default_out)
    try:
        print(sd.query_devices())
    except Exception:
        pass

    stt = make_stt(args.stt_repo, args.stt_device)
    llm = make_llm(args)
    tts = make_tts(args)

    from .assistant import VoiceAssistant

    assistant = VoiceAssistant(
        llm=llm,
        stt=stt,
        tts=tts,
        mic_device=args.mic_device,
        spk_device=args.spk_device,
        echo_guard_words=args.echo_guard,
        barge_in=args.barge_in,
    )
    try:
        assistant.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Ctrl-C: shutting down")
    finally:
        assistant.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
