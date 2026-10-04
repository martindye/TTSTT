#!/usr/bin/env bash
set -euo pipefail
# Toolkit components only; WSL uses the existing Windows NVIDIA driver.
cd /tmp
curl -fL --retry 3 -o ttstt-cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i ttstt-cuda-keyring.deb
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cuda-nvcc-12-8 cuda-cudart-dev-12-8 libcublas-dev-12-8
/usr/local/cuda-12.8/bin/nvcc --version
