import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

thr = json.load(open("reports/test_results.json"))["transfer"]["test_at_chosen_threshold"]["threshold"]

pred = pd.read_csv("reports/test_predictions.csv")
audit = pd.read_csv("reports/audit.csv")
audit["path"] = audit["path"].str.replace("\\", "/", regex=False)
df = pred.merge(audit[["path", "width", "height", "file_size", "format"]], on="path", how="left")

df["p"] = df["p_transfer"]                           # probability the image is AI
df["pred"] = (df["p"] >= thr).astype(int)
df["error"] = df["pred"] != df["label"]
df["error_type"] = np.where(~df["error"], "correct",
                            np.where(df["label"] == 0, "FP", "FN"))
print(f"Threshold {thr} | test errors: {int(df['error'].sum())} of {len(df)}")
print(df["error_type"].value_counts().to_string(), "\n")

# ---- pick 5 most confident FP and 5 most confident FN ----
fp = df[df["error_type"] == "FP"].sort_values("p", ascending=False).head(5)
fn = df[df["error_type"] == "FN"].sort_values("p", ascending=True).head(5)
sel = pd.concat([fp, fn]).reset_index(drop=True)
sel.insert(0, "no", range(1, len(sel) + 1))

# ---- figure ----
fig, axes = plt.subplots(2, 5, figsize=(20, 9))
for ax, (_, r) in zip(axes.ravel(), sel.iterrows()):
    ax.imshow(Image.open(r["path"]).convert("RGB"))
    what = "Real photo called AI" if r["error_type"] == "FP" else "AI image called real"
    ax.set_title(f"#{r['no']}  {what}\nP(AI)={r['p']:.2f} | {int(r['width'])}x{int(r['height'])} "
                 f"| {r['format']} | {r['file_size'] / 1024:.0f} KB", fontsize=10)
    ax.axis("off")
plt.suptitle(f"Most confident test errors, transfer model (threshold {thr})", fontsize=14)
plt.tight_layout()
plt.savefig("reports/figures/misclassified.png", dpi=110)
plt.close()

sel[["no", "error_type", "p", "width", "height", "format", "file_size", "path"]] \
    .round(3).to_csv("reports/error_analysis.csv", index=False)
print(sel[["no", "error_type", "p", "width", "height", "format", "file_size"]].round(3).to_string(index=False))

# ---- do the errors cluster in any kind of image? ----
df["size_group"] = pd.qcut(df["file_size"], 3, labels=["small file", "medium file", "large file"])
for col in ["size_group", "format", "class"]:
    print(f"\nError rate by {col}:")
    print(df.groupby(col, observed=True)["error"].agg(error_rate="mean", images="count").round(2).to_string())