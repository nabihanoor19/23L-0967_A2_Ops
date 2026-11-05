from pathlib import Path

import keras_tuner as kt
import pandas as pd
import tensorflow as tf

SEED = 42
Path("reports").mkdir(exist_ok=True)

(x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
x_train, y_train = x_train[:20000] / 255.0, y_train[:20000]      # subset for speed


def build_model(hp):
    tf.keras.utils.set_random_seed(SEED)
    units = hp.Int("units", 32, 256, step=32)
    lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


tuner = kt.RandomSearch(build_model, objective="val_accuracy", max_trials=8, seed=SEED,
                        directory="tuner_runs", project_name="toy_fashion_mnist",
                        overwrite=True)
tuner.search(x_train, y_train, epochs=5, validation_split=0.2, verbose=2)

rows = [{**t.hyperparameters.values, "val_accuracy": t.score}
        for t in tuner.oracle.trials.values()]
table = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False).round(4)
print("\n=== Toy results (best first) ===")
print(table.to_string(index=False))
table.to_csv("reports/tuner_toy_results.csv", index=False)
print("\nBest settings:", tuner.get_best_hyperparameters(1)[0].values)