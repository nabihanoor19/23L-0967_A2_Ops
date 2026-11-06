import re

import requests

BASE = "http://127.0.0.1:5000"
CASES = [("notes.txt", "wrong file type", 415),
         ("fake_image.jpg", "text pretending to be an image", 422),
         ("truncated.jpg", "damaged image", 422),
         ("too_big.jpg", "over the size limit", 413)]

for name, what, expected in CASES:
    try:
        with open(f"test_files/{name}", "rb") as f:
            r = requests.post(f"{BASE}/predict", files={"image": (name, f)}, timeout=60)
        title = re.search(r"<h1>(.*?)</h1>", r.text, re.S)
        print(f"{name:16} | {what:32} | status {r.status_code} (expected {expected}) | "
              f"page title: {title.group(1).strip() if title else 'NONE'}")
    except Exception as e:
        print(f"{name:16} | {what:32} | request failed: {repr(e)[:120]}")