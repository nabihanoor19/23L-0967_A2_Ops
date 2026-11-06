import mlflow
from mlflow.tracking import MlflowClient

from src.data import make_dataset
from src.metrics import best_threshold
from src.tracking import dvc_data_version, git_commit, setup_mlflow

import pandas as pd
import tensorflow as tf

MODEL_NAME = "visionguard"
CANDIDATES = [   # (run name, saved model file, image size, alias)
    ("r1_base", "models/sweep_r1_base.keras", 224, "challenger"),
    ("r6_three_blocks", "models/sweep_r6_three_blocks.keras", 224, "champion"),
]

setup_mlflow("visionguard_registry")
client = MlflowClient()
y_val = pd.read_csv("data/splits/val.csv")["label"].values

for name, path, size, alias in CANDIDATES:        # list order = version order (1, then 2)
    model = tf.keras.models.load_model(path, compile=False)
    p_val = model.predict(make_dataset("data/splits/val.csv", img_size=size), verbose=0).ravel()
    thr = best_threshold(y_val, p_val)             # validation data only

    with mlflow.start_run(run_name=f"register_{name}") as run:
        mlflow.set_tags({"git_commit": git_commit(), "dvc_data_md5": dvc_data_version(),
                         "source_run": name, "uses_test_set": "false"})
        mlflow.log_params({"source_run": name, "img_size": size})
        mlflow.log_metric("val_threshold", thr)
        info = mlflow.keras.log_model(model, name="model", registered_model_name=MODEL_NAME)

    version = getattr(info, "registered_model_version", None)
    if version is None:                             # fall back to searching by run id
        mvs = [v for v in client.search_model_versions(f"name='{MODEL_NAME}'")
               if v.run_id == run.info.run_id]
        version = mvs[0].version
    client.set_model_version_tag(MODEL_NAME, version, "source_run", name)
    client.set_model_version_tag(MODEL_NAME, version, "val_threshold", f"{thr:.4f}")
    client.set_model_version_tag(MODEL_NAME, version, "img_size", str(size))
    client.set_registered_model_alias(MODEL_NAME, alias, version)
    print(f"Registered {name} as version {version}, alias '{alias}', val threshold {thr:.4f}")

print("\nVersions in the registry:")
for v in sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda v: int(v.version)):
    print(f"  version {v.version} | source_run={v.tags.get('source_run')} | aliases={v.aliases}")