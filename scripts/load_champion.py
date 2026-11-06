import sys

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from mlflow.tracking import MlflowClient

from src.tracking import setup_mlflow

MODEL_NAME = "visionguard"
setup_mlflow("visionguard_registry")

mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
threshold = float(mv.tags.get("val_threshold", 0.5))
model = mlflow.keras.load_model(f"models:/{MODEL_NAME}@champion")      # loaded by alias
size = model.input_shape[1]
print(f"Loaded {MODEL_NAME}@champion = version {mv.version} (source run: {mv.tags.get('source_run')})")

# a validation image by default, so the test set is not used for this demo
default = pd.read_csv("data/splits/val.csv").iloc[0]
path = sys.argv[1] if len(sys.argv) > 1 else default["path"]

raw = tf.io.read_file(path)
img = tf.image.decode_image(raw, channels=3, expand_animations=False)
img = tf.cast(tf.image.resize(img, (size, size)), tf.float32)           # same as training; scaling is inside the model
p_ai = float(model.predict(img[None, ...], verbose=0)[0][0])

print("Image:", path)
print(f"P(AI-generated) = {p_ai:.3f} | threshold = {threshold:.3f}")
print("Verdict:", "AI-generated" if p_ai >= threshold else "Real")
if len(sys.argv) == 1:
    print("True label:", "AI-generated" if default["label"] == 1 else "Real")