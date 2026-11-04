import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from src.data import make_dataset

OUT = Path("reports/test_results.json")
if OUT.exists() and "--force" not in sys.argv:
    sys.exit("The test set was already evaluated (reports/test_results.json exists). "
             "Evaluating again would be a second look at the test data.")

MODELS = {"baseline": "models/baseline.keras", "transfer": "models/transfer_stage2.keras"}
COLORS = {"baseline": "tab:orange", "transfer": "tab:blue"}

val_df = pd.read_csv("data/splits/val.csv")
test_df = pd.read_csv("data/splits/test.csv")
val_ds = make_dataset("data/splits/val.csv")      # no shuffle: order matches the CSV
test_ds = make_dataset("data/splits/test.csv")
y_val, y_test = val_df["label"].values, test_df["label"].values


def best_threshold(y, p):
    """Threshold that maximises F1 on VALIDATION data."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[np.argmax(f1[:-1])])


def metrics_at(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": round(t, 4),
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, p),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


results, probs, rows = {}, {}, []
for name, path in MODELS.items():
    model = tf.keras.models.load_model(path, compile=False)
    p_val = model.predict(val_ds, verbose=0).ravel()
    p_test = model.predict(test_ds, verbose=0).ravel()
    t = best_threshold(y_val, p_val)                       # chosen on validation only
    results[name] = {"val_at_chosen_threshold": metrics_at(y_val, p_val, t),
                     "test_at_chosen_threshold": metrics_at(y_test, p_test, t),
                     "test_at_0.5": metrics_at(y_test, p_test, 0.5)}
    probs[name] = p_test
    for setting, key in [("chosen threshold", "test_at_chosen_threshold"),
                         ("threshold 0.5", "test_at_0.5")]:
        rows.append({"model": name, "setting": setting, **results[name][key]})

table = pd.DataFrame(rows).round(3)
pd.set_option("display.width", 200)
print(table.to_string(index=False))
table.to_csv("reports/test_metrics.csv", index=False)
OUT.write_text(json.dumps(results, indent=2, default=float))

pred_df = test_df[["path", "label", "class"]].copy()
for name in MODELS:
    pred_df[f"p_{name}"] = probs[name]
pred_df.to_csv("reports/test_predictions.csv", index=False)

# Confusion matrices (chosen threshold)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, name in zip(axes, MODELS):
    m = results[name]["test_at_chosen_threshold"]
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred Real", "pred AI"])
    ax.set_yticks([0, 1], ["true Real", "true AI"])
    ax.set_title(f"{name} (threshold {m['threshold']:.2f})")
plt.tight_layout()
plt.savefig("reports/figures/confusion_matrices.png", dpi=120)
plt.close()

# ROC curves
plt.figure(figsize=(6, 5.5))
for name in MODELS:
    fpr, tpr, _ = roc_curve(y_test, probs[name])
    auc = results[name]["test_at_chosen_threshold"]["roc_auc"]
    plt.plot(fpr, tpr, color=COLORS[name], label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="chance")
plt.xlabel("False positive rate (real photos called AI)")
plt.ylabel("True positive rate (AI images caught)")
plt.title("ROC curves on the test set")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/roc_curves.png", dpi=120)
plt.close()
print("\nSaved: reports/test_metrics.csv, test_results.json, test_predictions.csv, "
      "figures/confusion_matrices.png, figures/roc_curves.png")