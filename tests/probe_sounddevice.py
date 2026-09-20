"""Read the sounddevice simple page exactly like pip (PEP 503 + JSON accept),
with per-chunk timing, to find where it hangs."""
import time
import urllib.request

url = "https://pypi.org/simple/sounddevice/"
req = urllib.request.Request(url)
req.add_header("Accept", "application/vnd.pypi.simple.v1+json")
req.add_header("Accept-Encoding", "gzip")
req.add_header("User-Agent", "pip/25.3")

t0 = time.time()
r = urllib.request.urlopen(req, timeout=30)
print("status:", r.status, "headers CL:", r.headers.get("Content-Length"))
print("content-encoding:", r.headers.get("Content-Encoding"))
total = 0
chunk_no = 0
while True:
    tc = time.time()
    b = r.read(16384)
    dt = time.time() - tc
    total += len(b)
    chunk_no += 1
    print(f"chunk {chunk_no}: {len(b):6d} B in {dt:6.2f}s (total {total})")
    if not b:
        break
print("DONE, total bytes:", total, f"elapsed {time.time()-t0:.1f}s")
