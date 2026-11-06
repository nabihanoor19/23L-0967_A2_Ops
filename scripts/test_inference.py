import time

import pandas as pd

from src.inference import InvalidImageError, load_model_once

t0 = time.perf_counter()
predictor = load_model_once()
load_seconds = time.perf_counter() - t0
print(f"Model loaded once in {load_seconds:.2f} s | version {predictor.model_version} "
      f"| image size {predictor.img_size} | threshold {predictor.threshold:.3f}\n")

val = pd.read_csv("data/splits/val.csv")
for label, name in [(1, "AI-generated"), (0, "Real")]:
    row = val[val["label"] == label].iloc[0]
    image_bytes = open(row["path"], "rb").read()
    out = predictor.predict(image_bytes)
    print(f"Known {name} image: {row['path']}")
    print(f"  -> verdict: {out['label']} | confidence: {out['confidence']:.3f} | "
          f"P(AI): {out['p_ai']:.3f} | correct: {out['label'] == name}\n")

t0 = time.perf_counter()
for _ in range(20):
    predictor.predict(image_bytes)
per_request = (time.perf_counter() - t0) / 20
print(f"Average time per prediction (model already loaded): {per_request * 1000:.0f} ms")
print(f"Loading the model took {load_seconds / per_request:.0f} times longer than one prediction.\n")

try:
    predictor.predict(b"this is not an image")
except InvalidImageError as e:
    print("Bad file rejected:", e)