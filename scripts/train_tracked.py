import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_curve
from tensorflow.keras import callbacks

from src.data import BATCH_SIZE, IMG_SIZE, make_dataset
from src.metrics import best_threshold, metrics_at
from src.models import build_tuned
from src.tracking import dvc_data_version, git_commit, git_dirty, setup_mlflow

mode = sys.argv[1] if len(sys.argv) > 1 else "manual"
assert mode in ("manual", "autolog"), "use: manual | autolog"

SEED = 42
HP = {"learning_rate": 0.000234, "dropout": 0.5, "unfrozen_blocks": 1}   # best tuner trial (rounded)
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", 20))
PATIENCE = 5

setup_mlflow(os.environ.get("EXPERIMENT", "visionguard"))
train_ds = make_dataset("data/splits/train.csv", training=True)
val_ds = make_dataset("data/splits/val.csv")
test_ds = make_dataset("data/splits/test.csv")


class MlflowEpochLogger(callbacks.Callback):
    """Manual logging: every number in `logs` goes to MLflow once per epoch."""
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)


tf.keras.utils.set_random_seed(SEED)
model = build_tuned(**HP)
cbs = [
    callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1),
    callbacks.ModelCheckpoint(f"models/tuned_{mode}.keras", monitor="val_loss", save_best_only=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
]

if mode == "autolog":
    mlflow.tensorflow.autolog(log_models=False, log_datasets=False)

with mlflow.start_run(run_name=f"{mode}_tuned_best"):
    # tags: autologging cannot know these, so both modes set them by hand
    mlflow.set_tags({"git_commit": git_commit(), "git_dirty": git_dirty(),
                     "dvc_data_md5": dvc_data_version(), "seed": str(SEED),
                     "logging_mode": mode})

    if mode == "manual":
        mlflow.log_params({**HP, "backbone": "MobileNetV2", "img_size": IMG_SIZE,
                           "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
                           "early_stopping_patience": PATIENCE, "optimizer": "Adam",
                           "augmentation": "flip+rotation0.05+zoom0.1",
                           "n_train": 681, "n_val": 146, "n_test": 147})
        cbs.append(MlflowEpochLogger())

    history = model.fit(train_ds, validation_data=val_ds, epochs=MAX_EPOCHS,
                        callbacks=cbs, verbose=2).history

    if mode == "manual":
        best_epoch = int(np.argmin(history["val_loss"])) + 1
        mlflow.log_metric("best_epoch", best_epoch)

        # final test metrics (threshold chosen on validation data only)
        y_val = pd.read_csv("data/splits/val.csv")["label"].values
        y_test = pd.read_csv("data/splits/test.csv")["label"].values
        p_val = model.predict(val_ds, verbose=0).ravel()
        p_test = model.predict(test_ds, verbose=0).ravel()
        m = metrics_at(y_test, p_test, best_threshold(y_val, p_val))
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in m.items()})
        print("Test metrics:", {k: round(v, 3) for k, v in m.items()})

        # artifacts (3.4)
        cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["pred Real", "pred AI"])
        ax.set_yticks([0, 1], ["true Real", "true AI"])
        ax.set_title("Test confusion matrix")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        fpr, tpr, _ = roc_curve(y_test, p_test)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(fpr, tpr, label=f"AUC = {m['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("Test ROC curve")
        ax.legend()
        mlflow.log_figure(fig, "roc_curve.png")
        plt.close(fig)

        mlflow.log_artifacts("data/splits", "splits")
        mlflow.log_artifact("requirements.txt")
        try:
            mlflow.keras.log_model(model, "model")
        except Exception as e:              # fallback if DagsHub rejects the MLflow 3 call
            print("log_model failed, saving the model file as a plain artifact:", repr(e)[:200])
            mlflow.log_artifact(f"models/tuned_{mode}.keras", "model_file")

print("Done:", mode)