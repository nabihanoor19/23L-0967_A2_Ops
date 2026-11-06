import json

import mlflow
from mlflow.tracking import MlflowClient

from src.tracking import setup_mlflow

setup_mlflow("visionguard_registry")
mv = MlflowClient().get_model_version_by_alias("visionguard", "champion")
model = mlflow.keras.load_model("models:/visionguard@champion")
model.save("models/champion.keras")
meta = {"version": mv.version, "val_threshold": float(mv.tags.get("val_threshold", 0.5)),
        "img_size": int(mv.tags.get("img_size", 224)), "source_run": mv.tags.get("source_run")}
json.dump(meta, open("models/champion_meta.json", "w"), indent=2)
print("Saved models/champion.keras and models/champion_meta.json:", meta)