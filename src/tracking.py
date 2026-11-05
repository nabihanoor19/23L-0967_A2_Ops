import os
import subprocess

import mlflow
import yaml
from dotenv import load_dotenv


def setup_mlflow(experiment):
    load_dotenv()
    for key in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
        if not os.environ.get(key):
            raise SystemExit(f"{key} is missing. Fill in your .env file.")
    mlflow.set_experiment(experiment)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def git_dirty():
    try:
        return str(bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())).lower()
    except Exception:
        return "unknown"


def dvc_data_version():
    """MD5 hash of the tracked dataset folder, read from the DVC pointer file."""
    with open("data/raw.dvc") as f:
        return yaml.safe_load(f)["outs"][0]["md5"]