"""Download Pocket TTS english model + alba voice + tokenizer (try full repo first,
fall back to the public without-voice-cloning repo)."""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from huggingface_hub import hf_hub_download

REV_MODEL_FULL = "39592ff23c9ef80098bb74895d104c26275fe2c9"
REV_MODEL_FBC = "d29db7978e464fb90cb3359ee0c69a273b9142cc"
REV_VOICE_FBC = "e81d79e8194ad4c7ce879c87a4258ef20cbf2487"

def get(url: str) -> str:
    # url like hf://owner/repo/path/to/file@rev
    body = url.removeprefix("hf://")
    if "@" in body:
        path, rev = body.rsplit("@", 1)
    else:
        path, rev = body, None
    parts = path.split("/", 2)
    repo = parts[0] + "/" + parts[1]
    file = parts[2] if len(parts) > 2 else ""
    p = hf_hub_download(repo, file, revision=rev)
    print(f"  OK {os.path.getsize(p)/1e6:8.1f} MB  {repo}::{file}@{rev}")
    return p

# 1) model weights: try the full (voice-cloning) repo first
try:
    get(f"hf://kyutai/pocket-tts/languages/english/model.safetensors@{REV_MODEL_FULL}")
    MODELS_REPO = "kyutai/pocket-tts"
except Exception as e:
    print("  full repo model failed:", type(e).__name__, str(e)[:200])
    MODELS_REPO = "kyutai/pocket-tts-without-voice-cloning"
    get(f"hf://kyutai/pocket-tts-without-voice-cloning/languages/english/model.safetensors@{REV_MODEL_FBC}")

# 2) tokenizer (from the without-voice-cloning repo, as in english.yaml)
get("hf://kyutai/pocket-tts-without-voice-cloning/languages/english/tokenizer.model@d29db7978e464fb90cb3359ee0c69a273b9142cc")

# 3) alba voice: from the full repo's languages/english/embeddings if available,
#    else the fallback repo
try:
    get(f"hf://kyutai/pocket-tts/languages/english/embeddings/alba.safetensors@{REV_MODEL_FULL}")
except Exception as e:
    print("  full repo voice failed:", type(e).__name__, str(e)[:120])
    get(f"hf://kyutai/pocket-tts-without-voice-cloning/languages/english/embeddings/alba.safetensors@{REV_VOICE_FBC}")

print("DONE repo:", MODELS_REPO)
