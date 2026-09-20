import urllib.request

url = "https://pypi.org/simple/moshi/"
req = urllib.request.Request(url)
req.add_header("Accept", "application/vnd.pypi.simple.v1+html")
r = urllib.request.urlopen(req, timeout=20)
body = r.read().decode("utf-8", "replace")
print("status:", r.status)
print("len:", len(body))
print(body[:3000])
