#!/usr/bin/env bash
set -euo pipefail
cd /home/press/ttstt
if [ ! -d qwen3-tts-cuda ]; then
    cp -a qwen3-tts qwen3-tts-cuda
fi
cd qwen3-tts-cuda
make cuda -j8 CUDA_HOME=/usr/local/cuda-12.8 NVCC_ARCH='-gencode arch=compute_120,code=sm_120'
sha256sum qwen_tts
