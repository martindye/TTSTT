set -euo pipefail
cd /home/press/ttstt/qwen3-tts
./qwen_tts -d /home/press/ttstt/models/qwen3-tts-0.6b-base --ref-audio /mnt/c/Users/press/OneDrive/Projects/TTSTT/voice_stack/ref_voices/british_p225.wav --save-voice /home/press/ttstt/voices/british-p225.qvoice > /home/press/ttstt/create_british_p225.log 2>&1
