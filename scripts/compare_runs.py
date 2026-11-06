import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("reports/sweep_runs.csv")
cols = ["run_name", "changed_vs_base", "epochs_run", "best_epoch", "best_val_loss",
        "val_auc", "val_accuracy", "val_precision", "val_recall", "train_accuracy_at_best"]
table = df[cols].sort_values("best_val_loss").round(4)
print(table.to_string(index=False))
table.to_csv("reports/comparison_table.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col, title in [(axes[0], "best_val_loss", "Validation loss (lower is better)"),
                       (axes[1], "val_auc", "Validation AUC (higher is better)")]:
    d = df.sort_values(col, ascending=(col != "best_val_loss"))      # best run ends up on top
    colors = ["tab:red" if n == "r1_base" else "tab:blue" for n in d["run_name"]]
    ax.barh(d["run_name"], d[col], color=colors)
    for y, v in enumerate(d[col]):
        ax.text(v, y, f" {v:.3f}", va="center")
    ax.set_xlim(d[col].min() * 0.9, d[col].max() * 1.08)
    ax.set_title(title)
    ax.set_xlabel("(axis does not start at zero)")
plt.suptitle("Comparison of 7 runs, each with one change from the base (red)")
plt.tight_layout()
plt.savefig("reports/figures/run_comparison.png", dpi=120)