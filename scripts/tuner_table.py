import glob, json, sys
import pandas as pd

project, metric = sys.argv[1], sys.argv[2]
rows = []
for f in glob.glob(f"tuner_runs/{project}/trial_*/trial.json"):
    t = json.load(open(f))
    rows.append({"trial": t["trial_id"], **t["hyperparameters"]["values"], metric: t["score"]})
df = pd.DataFrame(rows).sort_values(metric, ascending=(metric == "val_loss"))
df["learning_rate"] = df["learning_rate"].map(lambda x: f"{x:.2e}")
df[metric] = df[metric].round(4)
print(df.to_string(index=False))
df.to_csv(f"reports/tuner_{project}_table.csv", index=False)
