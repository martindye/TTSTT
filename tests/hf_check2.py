import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import warnings
warnings.filterwarnings("ignore")
from huggingface_hub import HfApi
from huggingface_hub.utils import GatedRepoError, EntryNotFoundError

api = HfApi()

for repo in ["kyutai/pocket-tts", "kyutai/pocket-tts-without-voice-cloning"]:
    print(f"\n=== {repo} ===")
    try:
        info = api.repo_info(repo, files_metadata=True)
        for f in info.siblings:
            size = getattr(f, "size", 0) or 0
            print(f"  {size/1e6:9.1f} MB  {f.rfilename}")
    except Exception as e:
        print("  ERROR:", type(e).__name__, e)
