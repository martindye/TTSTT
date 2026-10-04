#!/usr/bin/env bash
set -euo pipefail
cd /home/press/ttstt/qwen3-tts-cuda
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
export QWEN_CUDA_FUSED_TALKER=1
export QWEN_CUDA_DECODER=1
export QWEN_CUDA_CONVDEC=1
exec ./qwen_tts \
    -d /home/press/ttstt/models/qwen3-tts-0.6b \
    --load-voice /home/press/ttstt/voices/british-p225.qvoice --icl-only \
    --backend cuda --int8 \
    --serve "${1:-8096}"
