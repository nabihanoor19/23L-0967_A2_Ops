import time

import pandas as pd

from src.inference import load_model_once

p = load_model_once()
val = pd.read_csv("data/splits/val.csv")
audit = pd.read_csv("reports/audit.csv")
audit["path"] = audit["path"].str.replace("\\", "/", regex=False)
val = val.merge(audit[["path", "width", "height", "file_size"]], on="path").sort_values("file_size")

for name, row in [("smallest file", val.iloc[0]), ("largest file", val.iloc[-1])]:
    b = open(row["path"], "rb").read()
    t = time.perf_counter()
    for _ in range(5):
        x = p.preprocess(b)
    pre = (time.perf_counter() - t) / 5
    t = time.perf_counter()
    for _ in range(5):
        p._forward(x)
    fwd = (time.perf_counter() - t) / 5
    print(f"{name}: {int(row['width'])}x{int(row['height'])}, {row['file_size'] / 1024:.0f} KB "
          f"| preprocessing {pre * 1000:.0f} ms | model {fwd * 1000:.0f} ms")