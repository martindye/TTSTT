import json
from huggingface_hub import hf_hub_download

cfg_path = hf_hub_download("kyutai/stt-1b-en_fr", "config.json")
cfg = json.loads(open(cfg_path, "r", encoding="utf-8").read())
print(json.dumps(cfg, indent=2))
