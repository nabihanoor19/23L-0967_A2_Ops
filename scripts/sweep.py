import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks

from src.data import BATCH_SIZE, make_dataset
from src.models import build_tuned
from src.tracking import dvc_data_version, git_commit, git_dirty, setup_mlflow

SEED = 42
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", 20))
PATIENCE = 5
BASE = {"learning_rate": 0.000234, "dropout": 0.5, "unfrozen_blocks": 1,
        "img_size": 224, "augmentation": True}
RUNS = {
    "r1_base": {},
    "r2_lr_low": {"learning_rate": 0.00005},
    "r3_lr_high": {"learning_rate": 0.001},
    "r4_no_augmentation": {"augmentation": False},
    "r5_frozen_backbone": {"unfrozen_blocks": 0},
    "r6_three_blocks": {"unfrozen_blocks": 3},
    "r7_image_160": {"img_size": 160},
}

unknown = [n for n in sys.argv[1:] if n not in RUNS]
if unknown:
    sys.exit(f"Unknown run name(s): {unknown}. Choose from {list(RUNS)}")
wanted = sys.argv[1:] or list(RUNS)

setup_mlflow(os.environ.get("EXPERIMENT", "visionguard_comparison"))
CSV = Path("reports/sweep_runs_smoke.csv" if os.environ.get("EXPERIMENT") else "reports/sweep_runs.csv")
Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)
done = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame()


class EpochLogger(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)


for name in wanted:
    cfg = {**BASE, **RUNS[name]}
    changed = ", ".join(f"{k}={v}" for k, v in RUNS[name].items()) or "none (base)"
    print(f"\n=== {name}: change vs base = {changed} ===")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    train_ds = make_dataset("data/splits/train.csv", training=True,
                            img_size=cfg["img_size"], use_augmentation=cfg["augmentation"])
    val_ds = make_dataset("data/splits/val.csv", img_size=cfg["img_size"])
    model = build_tuned(dropout=cfg["dropout"], learning_rate=cfg["learning_rate"],
                        unfrozen_blocks=cfg["unfrozen_blocks"], img_size=cfg["img_size"])
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(f"models/sweep_{name}.keras", monitor="val_loss",
                                  save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                    min_lr=1e-7, verbose=1),
        EpochLogger(),
    ]

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"git_commit": git_commit(), "git_dirty": git_dirty(),
                         "dvc_data_md5": dvc_data_version(), "seed": str(SEED),
                         "changed_vs_base": changed, "uses_test_set": "false"})
        mlflow.log_params({**cfg, "backbone": "MobileNetV2", "batch_size": BATCH_SIZE,
                           "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE})
        h = model.fit(train_ds, validation_data=val_ds, epochs=MAX_EPOCHS,
                      callbacks=cbs, verbose=2).history
        i = int(np.argmin(h["val_loss"]))
        best = {"best_epoch": i + 1, "best_val_loss": h["val_loss"][i],
                "val_auc": h["val_auc"][i], "val_accuracy": h["val_accuracy"][i],
                "val_precision": h["val_precision"][i], "val_recall": h["val_recall"][i],
                "train_accuracy_at_best": h["accuracy"][i]}
        mlflow.log_metrics({k: float(v) for k, v in best.items()})
        row = {"run_name": name, "run_id": run.info.run_id, "changed_vs_base": changed,
               **{k: cfg[k] for k in BASE}, **best, "epochs_run": len(h["loss"])}

    keep = done[done["run_name"] != name] if len(done) else done
    done = pd.concat([keep, pd.DataFrame([row])], ignore_index=True)
    done.to_csv(CSV, index=False)             # saved after every run, so progress is never lost
    print(f"[{name}] best epoch {best['best_epoch']} | val_loss {best['best_val_loss']:.4f} "
          f"| val_auc {best['val_auc']:.4f} | val_acc {best['val_accuracy']:.4f}")

print("\nAll requested runs finished.")