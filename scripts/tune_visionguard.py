import os
from pathlib import Path

import keras_tuner as kt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers

from src.data import make_dataset
from src.models import _compile, build_transfer

SEED = 42
TRIALS = int(os.environ.get("TUNER_TRIALS", 10))
EPOCHS = int(os.environ.get("TUNER_EPOCHS", 8))
UNFREEZE_FROM = {0: None, 1: "block_16_expand", 2: "block_15_expand", 3: "block_14_expand"}
Path("reports").mkdir(exist_ok=True)

train_ds = make_dataset("data/splits/train.csv", training=True)
val_ds = make_dataset("data/splits/val.csv")          # the test set is never loaded here


def build_model(hp):
    tf.keras.utils.set_random_seed(SEED)
    lr = hp.Float("learning_rate", 1e-5, 1e-3, sampling="log")
    dropout = hp.Float("dropout", 0.1, 0.5, step=0.1)
    blocks = hp.Choice("unfrozen_blocks", [0, 1, 2, 3])

    model = build_transfer(dropout=dropout, learning_rate=lr)    # backbone frozen
    first = UNFREEZE_FROM[blocks]
    if first is not None:
        base = model.get_layer("backbone")
        base.trainable = True
        reached = False
        for layer in base.layers:
            if layer.name == first:
                reached = True
            layer.trainable = reached and not isinstance(layer, layers.BatchNormalization)
        _compile(model, lr)                                       # recompile after changing trainable
    return model


tuner = kt.RandomSearch(build_model,
                        objective=kt.Objective("val_loss", direction="min"),
                        max_trials=TRIALS, seed=SEED,
                        directory="tuner_runs", project_name="visionguard",
                        overwrite=True)
tuner.search(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2,
             callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                                restore_best_weights=True)])

rows = [{"trial": t.trial_id, **t.hyperparameters.values, "val_loss": t.score}
        for t in tuner.oracle.trials.values()]
table = pd.DataFrame(rows).sort_values("val_loss").round(4)
print("\n=== VisionGuard tuner results (best first) ===")
print(table.to_string(index=False))
table.to_csv("reports/tuner_visionguard_results.csv", index=False)
print("\nBest settings:", tuner.get_best_hyperparameters(1)[0].values)

print("\nAverage val_loss by number of unfrozen blocks:")
print(table.groupby("unfrozen_blocks")["val_loss"].agg(["mean", "min", "count"]).round(4))
print("\nSpearman correlation with val_loss (positive = larger value gives worse loss):")
for col in ["learning_rate", "dropout"]:
    print(f"  {col}: {table[col].corr(table['val_loss'], method='spearman'):.3f}")
print(f"\nSpread between best and worst trial: {table['val_loss'].max() - table['val_loss'].min():.4f}")