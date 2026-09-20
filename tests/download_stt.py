"""Download the Kyutai STT 1B en_fr model files to the HF cache."""
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from huggingface_hub import snapshot_download

path = snapshot_download(
    "kyutai/stt-1b-en_fr",
    allow_patterns=["config.json", "*.safetensors", "*.model", "*.json", "*.txt", "*.md"],
    max_workers=4,
)
print("STT_MODEL_DIR", path)
for f in sorted(os.listdir(path)):
    fp = os.path.join(path, f)
    if os.path.isfile(fp):
        print(f"  {os.path.getsize(fp)/1e6:10.1f} MB  {f}")
