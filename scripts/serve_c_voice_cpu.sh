#!/usr/bin/env bash
# CPU fallback until the WSL CUDA build is installed and verified.
set -euo pipefail
cd /home/press/ttstt/qwen3-tts
exec ./qwen_tts \
    -d /home/press/ttstt/models/qwen3-tts-0.6b \
    --load-voice /home/press/ttstt/voices/warm-brit.qvoice --icl-only \
    --int8 \
    --serve 8096
