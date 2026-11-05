import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()                                   # reads .env into environment variables
for key in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
    if not os.environ.get(key):
        raise SystemExit(f"{key} is missing. Fill in your .env file.")
print("Tracking URI:", os.environ["MLFLOW_TRACKING_URI"])      # the token is never printed

mlflow.set_experiment("toy_fashion_mnist")

(x, y), _ = tf.keras.datasets.fashion_mnist.load_data()
x, y = x[:10000] / 255.0, y[:10000]


class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch)


def run(lr):
    tf.keras.utils.set_random_seed(42)
    with mlflow.start_run(run_name=f"lr_{lr}"):
        mlflow.log_params({"learning_rate": lr, "units": 64, "epochs": 5, "batch_size": 64})
        mlflow.set_tag("purpose", "toy lab 3.2 on DagsHub")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        h = model.fit(x, y, validation_split=0.2, epochs=5, batch_size=64,
                      callbacks=[EpochLogger()], verbose=2).history

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(h["loss"], label="train loss")
        ax.plot(h["val_loss"], label="validation loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title(f"learning rate {lr}")
        ax.legend()
        mlflow.log_figure(fig, "loss_curve.png")
        plt.close(fig)

        try:
            mlflow.keras.log_model(model, "model")
        except Exception as e:                  # fallback if the server rejects the MLflow 3 call
            print("log_model failed, saving the model file as a plain artifact instead:", repr(e)[:200])
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, f"toy_model_lr_{lr}.keras")
                model.save(path)
                mlflow.log_artifact(path, "model_file")


for lr in (0.01, 0.0001):
    run(lr)
print("Done. Open the Experiments tab of your DagsHub project.")