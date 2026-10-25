from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

SEED = 42
OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("reports/audit.csv")
df = df[df["readable"]].drop_duplicates("sha256")   # readable, de-duplicated
classes = sorted(df["class"].unique())

# ---------- Figure 1: sample grids (one per class) ----------
rng = np.random.default_rng(SEED)
for cls in classes:
    paths = df[df["class"] == cls]["path"].values
    picks = rng.choice(paths, size=16, replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, p in zip(axes.ravel(), picks):
        img = Image.open(p).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
    fig.suptitle(f"Random samples: {cls}", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT / f"samples_{cls}.png", dpi=120)
    plt.close()

# ---------- Figure 2: histograms per class ----------
def hist_by_class(col, title, xlabel, fname, log_x=False):
    plt.figure(figsize=(8, 5))
    data = [df[df["class"] == c][col] for c in classes]
    if log_x:
        bins = np.logspace(np.log10(df[col].min()), np.log10(df[col].max()), 40)
        plt.xscale("log")
    else:
        bins = 40
    for c, d in zip(classes, data):
        plt.hist(d, bins=bins, alpha=0.55, label=c)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of images")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / fname, dpi=120)
    plt.close()

hist_by_class("width",  "Image width per class",  "Width (pixels, log scale)",  "hist_width.png",  log_x=True)
hist_by_class("height", "Image height per class", "Height (pixels, log scale)", "hist_height.png", log_x=True)
hist_by_class("file_size", "File size per class", "File size (bytes, log scale)", "hist_filesize.png", log_x=True)

print("Saved figures to", OUT)