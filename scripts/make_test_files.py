from pathlib import Path

import pandas as pd

out = Path("test_files")
out.mkdir(exist_ok=True)
val = pd.read_csv("data/splits/val.csv")
good = val[val["path"].str.lower().str.endswith((".jpg", ".jpeg"))].iloc[0]["path"]
data = open(good, "rb").read()

(out / "good_photo.jpg").write_bytes(data)
(out / "notes.txt").write_text("this is not an image")
(out / "fake_image.jpg").write_text("this is plain text pretending to be a JPG")
(out / "truncated.jpg").write_bytes(data[: len(data) // 3])
(out / "too_big.jpg").write_bytes(b"\xff\xd8" + b"0" * (10 * 1024 * 1024))
(out / "tëst photo; DROP TABLE x.jpg").write_bytes(data)
print("Created files in", out.resolve())