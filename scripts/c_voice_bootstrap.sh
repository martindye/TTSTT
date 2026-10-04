#!/usr/bin/env bash
# One-time bootstrap for the Qwen3-TTS C engine inside WSL2 (Ubuntu 24.04).
# Run as the normal user (press):  wsl -d Ubuntu-24.04 -u press bash /mnt/c/.../c_voice_bootstrap.sh
set -uo pipefail
T=/home/press/ttstt
mkdir -p "$T/models" "$T/voices"

# 1. Copy the source tree from the Windows side (9p) onto local ext4.
if [ ! -d "$T/qwen3-tts" ]; then
  cp -r /mnt/c/Users/press/OneDrive/Projects/TTSTT/tools/qwen3-tts "$T/qwen3-tts"
fi
cd "$T/qwen3-tts" || exit 1

# 2. WSL-side helper scripts.
cp /mnt/c/Users/press/OneDrive/Projects/TTSTT/scripts/serve_c_voice.sh "$T/serve_c_voice.sh"
sed -i "s/\r$//" "$T/serve_c_voice.sh"
chmod +x "$T/serve_c_voice.sh"

cat > "$T/create_warm_brit_voice.sh" <<'EOF'
#!/usr/bin/env bash
# One-time: clone the warm British reference clip into the warm-brit.qvoice graft.
set -e
cd /home/press/ttstt
./qwen3-tts/qwen_tts \
    -d /home/press/ttstt/models/qwen3-tts-0.6b-base \
    --ref-audio /mnt/c/Users/press/OneDrive/Projects/TTSTT/voice_stack/ref_voices/warm_brit.wav \
    --ref-text "$(cat /mnt/c/Users/press/OneDrive/Projects/TTSTT/voice_stack/ref_voices/warm_brit.txt)" \
    --save-voice /home/press/ttstt/voices/warm-brit.qvoice
EOF
chmod +x "$T/create_warm_brit_voice.sh"

# 3. Start the model downloads (network-bound) in the background.
if ! grep -q DOWNLOADS_DONE "$T/download.log" 2>/dev/null; then
  nohup bash -c 'cd /home/press/ttstt/qwen3-tts && bash download_model.sh --model small --dir /home/press/ttstt/models/qwen3-tts-0.6b && bash download_model.sh --model base-small --dir /home/press/ttstt/models/qwen3-tts-0.6b-base && echo DOWNLOADS_DONE' > "$T/download.log" 2>&1 &
  echo "model downloads started (pid $!)"
fi

# 4. Build (CPU-bound; overlaps the download).
make blas 2>&1 | tail -5

echo "=== caps ==="
./qwen_tts --caps 2>&1 | head -40
echo BOOTSTRAP_DONE
