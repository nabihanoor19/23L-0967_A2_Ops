import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image

DATA_DIR = Path("data/raw")
OUT_CSV = Path("reports/audit.csv")
OUT_CSV.parent.mkdir(exist_ok=True)

def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()

rows = []
for path in sorted(DATA_DIR.rglob("*")):
    if not path.is_file():
        continue
    row = {
        "path": str(path),
        "class": path.relative_to(DATA_DIR).parts[0],
        "file_size": path.stat().st_size,
        "sha256": sha256_of(path),
        "width": None, "height": None, "mode": None, "format": None,
        "readable": True,
    }
    try:
        with Image.open(path) as img:
            img.verify()                      # checks file structure
        with Image.open(path) as img:         # verify() invalidates the object, so reopen
            img.load()                        # forces full pixel decode
            row.update(width=img.width, height=img.height,
                       mode=img.mode, format=img.format)
    except Exception as e:
        row["readable"] = False
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

ok = df[df["readable"]]

print("\n=== Files per class ===")
print(df.groupby("class").size())

print("\n=== Min / median / max (readable images) ===")
print(ok.groupby("class")[["width", "height", "file_size"]]
        .agg(["min", "median", "max"]).T)

print("\n=== Colour modes / formats ===")
print(ok.groupby(["class", "mode"]).size())
print(ok.groupby(["class", "format"]).size())

print("\n=== Corrupt / unreadable files ===")
print("Count:", (~df["readable"]).sum())
print(df.loc[~df["readable"], "path"].to_string(index=False))

print("\n=== Exact duplicates ===")
dups = df[df.duplicated("sha256", keep=False)]
print("Files involved in duplication:", len(dups))
print("Redundant copies (extra files):", df.duplicated("sha256").sum())
print("Duplicate groups:", dups["sha256"].nunique())
cross = dups.groupby("sha256")["class"].nunique()
print("Groups spanning BOTH classes:", (cross > 1).sum())