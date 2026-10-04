#!/usr/bin/env bash
set -euo pipefail
exec bash /home/press/ttstt/serve_c_voice.sh \
    >>/home/press/ttstt/c_voice.log 2>&1 </dev/null
