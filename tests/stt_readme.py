from huggingface_hub import hf_hub_download

p = hf_hub_download("kyutai/stt-1b-en_fr", "README.md")
print(open(p, encoding="utf-8").read())
