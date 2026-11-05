from mlflow.tracking import MlflowClient

from src.tracking import setup_mlflow

setup_mlflow("visionguard")
client = MlflowClient()
exp = client.get_experiment_by_name("visionguard")
run = client.search_runs([exp.experiment_id], filter_string="tags.logging_mode = 'manual'",
                         order_by=["attribute.start_time DESC"], max_results=1)[0]
print("Run:", run.info.run_name, "|", run.info.run_id)
print("Git commit tag:", run.data.tags.get("git_commit"))


def walk(path=""):
    for item in client.list_artifacts(run.info.run_id, path):
        if item.is_dir:
            walk(item.path)
        else:
            print(f"  {item.path}  ({item.file_size} bytes)")


print("Artifacts:")
walk()