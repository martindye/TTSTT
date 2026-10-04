"""Fetch the GPU streaming VoiceDesign pair, pinned to a reviewed revision."""
import argparse
import os
from pathlib import Path

REPO = "Serveurperso/Qwen3-TTS-GGUF"
REVISION = "b7ee2e8c7459c3bea99da23e3d178125a7d1713c"
FILES = ["qwen-talker-1.7b-voicedesign-Q8_0.gguf", "qwen-tokenizer-12hz-Q8_0.gguf"]
DEST = Path.home() / "models" / "Qwen3-TTS-VoiceDesign-GGUF"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    print(f"{REPO} at {REVISION}\nDestination: {DEST}\nFiles: {FILES}", flush=True)
    if args.download:
        os.environ["HF_HUB_OFFLINE"] = "0"
        from huggingface_hub import snapshot_download
        snapshot_download(REPO, revision=REVISION, allow_patterns=FILES,
                          local_dir=str(DEST), max_workers=2)
        print("Download complete.", flush=True)


if __name__ == "__main__":
    main()
