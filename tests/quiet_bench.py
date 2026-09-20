"""Clean latency bench: wait for the mic to be quiet, then run two --once
turns and verify no user speech happened during them (single shared LLM slot:
any user conversation contaminates the numbers).

Writes tests/bench_result.json. Safe to run while the voice bridge is up.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "tests" / "live_dsh_log.txt"
OUT = ROOT / "tests" / "bench_result.json"
QUESTIONS = [
    "What is the capital of France?",
    "How are you doing today?",
]
QUIET_AFTER_S = 45        # newest mic activity must be older than this
MAX_ATTEMPTS = 4
RUN_TIMEOUT = 150

LINE_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2}) ")
STT_MARKERS = ("stt word", "USER:", "prompt sent")


def now_s():
    t = time.localtime()
    return t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec


def parse_hms(s):
    h, m, s2 = s
    return int(h) * 3600 + int(m) * 60 + int(s2)


def newest_activity_since(before_s):
    """Newest mic/turn activity in the log since `before_s` (seconds, same day).
    Returns seconds-since-epoch-ish value or None. Handles midnight wrap crudely
    by treating values < before_s-43200 as 'next day'."""
    newest = None
    if not LOG.exists():
        return None
    for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-400:]:
        m = LINE_RE.match(line)
        if not m:
            continue
        if not any(k in line for k in STT_MARKERS):
            continue
        v = parse_hms(m.groups())
        if before_s - 43200 < v <= before_s + 43200:
            if newest is None or v > newest:
                newest = v
    return newest


def wait_for_quiet(deadline_s):
    """Block until newest activity is older than QUIET_AFTER_S. False on timeout."""
    while time.time() < deadline_s:
        lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines() if LOG.exists() else []
        newest = None
        for line in lines[-400:]:
            m = LINE_RE.match(line)
            if not m or not any(k in line for k in STT_MARKERS):
                continue
            v = parse_hms(m.groups())
            if newest is None or v > newest:
                newest = v
        if newest is None:
            return True
        age = (now_s() - newest) % 86400
        if age > QUIET_AFTER_S:
            return True
        time.sleep(5)
    return False


def run_once(q):
    t0 = time.time()
    # --no-speak: full pipeline, but nothing reaches the speakers. Essential
    # while the live bridge's mic is open -- bench audio would be transcribed
    # as user speech (echo feedback loop) and clog the voice.
    # --no-speak must come AFTER the --once value: argparse would otherwise
    # read the flag as --once's argument.
    p = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "voice_stack.voice_dsh",
         "--once", q, "--no-speak"],
        cwd=ROOT, capture_output=True, text=True, timeout=RUN_TIMEOUT)
    out = p.stdout + p.stderr
    m = re.search(r"--- once --- ok=(\S+).*total=([\d.]+)s.*first_text=([\d.]+)s.*"
                  r"first_audio=([\d.]+)s.*first_reasoning=([\d.]+)s.*spoken=(\d+) chars "
                  r"\(thinking (\d+) chars", out)
    if not m:
        return {"ok": False, "raw_tail": out[-400:]}
    return {
        "ok": m.group(1) == "True",
        "total": float(m.group(2)),
        "first_text": float(m.group(3)),
        "first_audio": float(m.group(4)),
        "first_reasoning": float(m.group(5)),
        "spoken_chars": int(m.group(6)),
        "thinking_chars": int(m.group(7)),
    }


def main():
    result = {"attempts": []}
    runs = [None, None]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        deadline = time.time() + 600  # up to 10 min for a quiet window
        if not wait_for_quiet(deadline):
            result["abort"] = f"attempt {attempt}: no quiet window in 10 min"
            break
        q = QUESTIONS[attempt - 2 if attempt <= 2 else 0]  # alternate; see below
        # run both questions of this attempt
        attempt_res = {}
        contaminated = False
        for i, q in enumerate(QUESTIONS):
            before = now_s()
            try:
                r = run_once(q)
            except subprocess.TimeoutExpired:
                r = {"ok": False, "raw_tail": "TIMEOUT"}
            after = now_s()
            # contamination: any stt word / USER / prompt line between before/after
            contam = False
            if LOG.exists():
                for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-800:]:
                    m = LINE_RE.match(line)
                    if not m:
                        continue
                    if not any(k in line for k in ("stt word",)):
                        continue
                    v = parse_hms(m.groups())
                    if before - 1 <= v <= after + 1:
                        contam = True
            r["question"] = q
            r["contaminated"] = contam
            contaminated = contaminated or contam
            attempt_res[f"q{i+1}"] = r
        result["attempts"].append({
            "attempt": attempt,
            "contaminated": contaminated,
            "runs": attempt_res,
        })
        all_ok = all(r.get("ok") for r in attempt_res.values())
        if not contaminated and all_ok:
            result["clean"] = attempt_res
            break
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    sys.exit(main())
