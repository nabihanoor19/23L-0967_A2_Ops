import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks

SEED = 42
Path("models").mkdir(exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)


def set_seed(seed=SEED):
    tf.keras.utils.set_random_seed(seed)      # seeds Python, NumPy and TensorFlow


def plot_curves(h, best_epoch, name):
    epochs = range(1, len(h["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in [(axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")]:
        ax.plot(epochs, h[key], label="train")
        ax.plot(epochs, h["val_" + key], label="validation")
        ax.axvline(best_epoch, color="red", linestyle="--", label=f"best epoch ({best_epoch})")
        ax.set_title(f"{name}: {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"reports/figures/curves_{name}.png", dpi=120)
    plt.close()


def run_training(model, name, train_ds, val_ds, epochs, patience=5):
    epochs = min(epochs, int(os.environ.get("EPOCH_CAP", epochs)))   # EPOCH_CAP = quick test
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(f"models/{name}.keras", monitor="val_loss",
                                  save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                    min_lr=1e-7, verbose=1),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        callbacks=cbs, verbose=2)
    h = history.history
    best_epoch = h["val_loss"].index(min(h["val_loss"])) + 1
    Path(f"reports/history_{name}.json").write_text(
        json.dumps({"history": h, "best_epoch": best_epoch}, indent=2, default=float))
    plot_curves(h, best_epoch, name)
    print(f"\n[{name}] best epoch = {best_epoch} | val_loss = {min(h['val_loss']):.4f} "
          f"| val_accuracy at best = {h['val_accuracy'][best_epoch - 1]:.4f}")
    return h, best_epoch