from mlflow.tracking import MlflowClient

from src.tracking import setup_mlflow

setup_mlflow("visionguard")
c = MlflowClient()
exp = c.get_experiment_by_name("visionguard")
for mode in ("manual", "autolog"):
    runs = c.search_runs([exp.experiment_id], filter_string=f"tags.logging_mode = '{mode}'",
                         order_by=["attribute.start_time DESC"], max_results=1)
    if not runs:
        print(f"\n=== {mode}: no run found ===")
        continue
    r = runs[0]
    print(f"\n=== {mode} ===")
    print("PARAMS :", dict(sorted(r.data.params.items())))
    print("METRICS:", sorted(r.data.metrics.keys()))
    print("TAGS   :", {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")})
    print("Per-epoch 'loss' values logged:", len(c.get_metric_history(r.info.run_id, "loss")))