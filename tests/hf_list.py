from huggingface_hub import HfApi

api = HfApi()
for repo in ["kyutai/stt-1b-en_fr"]:
    info = api.repo_info(repo)
    total = 0
    print(f"== {repo} ==")
    for f in info.siblings:
        size = getattr(f, "size", None)
        if size:
            total += size
        print(f"  {size if size else '?':>14}  {f.rfilename}")
    print(f"TOTAL: {total/1e9:.2f} GB")
