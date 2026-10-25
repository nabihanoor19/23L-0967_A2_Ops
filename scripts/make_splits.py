from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
OUT = Path("data/splits")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("reports/audit.csv")
df = df[df["readable"]].drop_duplicates("sha256").copy()      # drop corrupt + duplicates
df["path"] = df["path"].str.replace("\\", "/", regex=False)    # same paths on Windows and Linux
df["label"] = (df["class"] == "AiArtData").astype(int)          # AI = 1, Real = 0

train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=SEED)

cols = ["path", "label", "class", "sha256"]
for name, part in [("train", train), ("val", val), ("test", test)]:
    part[cols].to_csv(OUT / f"{name}.csv", index=False)
    print(name, len(part), "| AI share:", round(part["label"].mean(), 3))

# leakage check: no image appears in two splits
assert not set(train.sha256) & set(val.sha256)
assert not set(train.sha256) & set(test.sha256)
assert not set(val.sha256) & set(test.sha256)
print("No overlap between splits. Total:", len(df))