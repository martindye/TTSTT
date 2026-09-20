import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from huggingface_hub import hf_hub_download

p = hf_hub_download("kyutai/stt-1b-en_fr", "README.md")
text = open(p, encoding="utf-8").read()
print(text)
