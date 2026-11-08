import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import json
from pathlib import Path

import keras
import numpy as np
import onnxruntime as ort
import pandas as pd
import tensorflow as tf
from PIL import Image

OUT = Path("models/visionguard.onnx")
OUT.unlink(missing_ok=True)
threshold = float(json.loads(Path("models/champion_meta.json").read_text())["val_threshold"])

model = keras.models.load_model("models/champion.keras", compile=False)
size = int(model.input_shape[1])

# resizing INSIDE the exported model: any image size in, same preprocessing as training
inp = keras.Input(shape=(None, None, 3), dtype="float32", name="image")
x = keras.layers.Resizing(size, size, interpolation="bilinear", name="resize")(inp)
wrapper = keras.Model(inp, model(x, training=False), name="visionguard_onnx")
_ = wrapper(np.zeros((1, 300, 300, 3), dtype="float32"))   # call it once, as Keras requires
signature = [tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32, name="image")]
wrapper.export(str(OUT), format="onnx", input_signature=signature)
print(f"\nExported {OUT} ({OUT.stat().st_size / 1e6:.1f} MB), threshold {threshold}\n")

sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
val = pd.read_csv("data/splits/val.csv")
sample = pd.concat([val[val.label == 1].sample(6, random_state=0),
                    val[val.label == 0].sample(6, random_state=0)])

max_diff, agree = 0.0, 0
for path in sample["path"]:
    img = tf.image.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    small = tf.cast(tf.image.resize(img, (size, size)), tf.float32)[None]
    p_keras = float(model(small, training=False).numpy().ravel()[0])
    arr = np.asarray(Image.open(path).convert("RGB"), dtype="float32")[None]
    p_onnx = float(sess.run(None, {in_name: arr})[0].ravel()[0])
    max_diff = max(max_diff, abs(p_onnx - p_keras))
    agree += int((p_onnx >= threshold) == (p_keras >= threshold))
    print(f"{Path(path).name[:34]:34} keras={p_keras:.4f}  onnx={p_onnx:.4f}")

print(f"\nMax difference: {max_diff:.4f} | same verdict: {agree}/{len(sample)}")
print("PASS" if max_diff < 0.01 and agree == len(sample) else "CHECK THE DIFFERENCES")