import shutil

import pandas as pd

val = pd.read_csv("data/splits/val.csv")
audit = pd.read_csv("reports/audit.csv")
audit["path"] = audit["path"].str.replace("\\", "/", regex=False)
m = val.merge(audit[["path", "file_size"]], on="path")
m = m[m.path.str.lower().str.endswith((".jpg", ".jpeg")) & (m.file_size < 900_000)]
for label, name in [(1, "small_ai"), (0, "small_real")]:
    row = m[m.label == label].iloc[0]
    shutil.copy(row.path, f"test_files/{name}.jpg")
    print(name, "<-", row.path, f"({row.file_size / 1024:.0f} KB)")