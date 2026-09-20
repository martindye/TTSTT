"""Probe PyPI endpoints the way pip would, to find the hanging request."""
import re
import time
import urllib.request


def fetch(url, timeout=20, label=""):
    t0 = time.time()
    try:
        req = urllib.request.Request(url)
        # pip sends these headers for the simple index (PEP 658 + rich)
        req.add_header("Accept", "application/vnd.pypi.simple.v1+html")
        req.add_header("Accept-Encoding", "gzip")
        r = urllib.request.urlopen(req, timeout=timeout)
        n = len(r.read(100000))
        print(f"[{label}] OK  {time.time()-t0:6.1f}s  {n:8d}B  {url[:100]}")
        return r
    except Exception as e:
        print(f"[{label}] FAIL after {time.time()-t0:5.1f}s: {type(e).__name__}: {e}")
        return None


# 1) simple index for moshi
r = fetch("https://pypi.org/simple/moshi/", label="simple")
if r:
    body = r.read().decode("utf-8", "replace")
    # find the 0.2.13 wheel anchor and its metadata flag
    for m in re.finditer(
        r'<a href="([^"]+)" data-dist-info-metadata="([^"]*)"', body
    ):
        url, meta = m.group(1), m.group(2).strip()
        if "moshi-0.2.13" in url:
            print("wheel link:", url[:140])
            print("data-dist-info-metadata:", meta)
            if meta == "true":
                # metadata file: replace .whl with .whl.metadata
                meta_url = url.split("#")[0] + ".metadata"
                fetch(meta_url, label="pep658")
            else:
                print("no PEP658 metadata; pip will download full wheel:")
                fetch(url.split("#")[0], label="wheel")
                break

# 2) the actual wheel download (pip does this when no PEP658 or as fallback)
fetch(
    "https://files.pythonhosted.org/packages/placeholder",
    label="files-host-probe",
    timeout=5,
)
