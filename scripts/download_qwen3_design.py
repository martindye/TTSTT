"""Preview the pinned VoiceDesign download; require --download to fetch weights.

Uses a plain local directory, avoiding Windows cache symlink permissions.
"""
import argparse
import os
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
REVISION = "5ecdb67327fd37bb2e042aab12ff7391903235d3"
DEFAULT_DEST = Path.home() / "models" / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
PATTERNS = ["*.json", "*.safetensors", "merges.txt", "vocab.json"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    # Only this explicit setup process goes online; voice processes stay offline.
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import HfApi, snapshot_download
    from fnmatch import fnmatch

    info = HfApi().model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    files = [f for f in info.siblings
             if any(fnmatch(f.rfilename, pattern) for pattern in PATTERNS)]
    size = sum(f.size or 0 for f in files)
    print(f"Model: {MODEL_ID}\nRevision: {REVISION}\nDestination: {args.dest}")
    print(f"{len(files)} files, {size / 1e9:.2f} GB total (before cache reuse)")
    if not args.download:
        print("Preview only. Add --download to fetch the model.")
        return
    snapshot_download(MODEL_ID, revision=REVISION, local_dir=str(args.dest),
                      allow_patterns=PATTERNS)
    print("Download complete. Benchmark before enabling the live voice.")


if __name__ == "__main__":
    main()
