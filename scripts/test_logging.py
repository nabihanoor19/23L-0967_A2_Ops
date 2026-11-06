from dotenv import load_dotenv
load_dotenv()

import requests
from sqlalchemy import text

from src.db import engine

BASE = "http://127.0.0.1:5000"


def count():
    with engine.connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar()


def send(path):
    with open(path, "rb") as f:
        return requests.post(f"{BASE}/predict", files={"image": (path.split("/")[-1], f)},
                             timeout=120).status_code


before = count()
print(f"Rows before: {before}")

status = send("test_files/good_photo.jpg")
after_good = count()
print(f"Valid image  -> HTTP {status} | rows now {after_good} (added {after_good - before})")

status = send("test_files/notes.txt")
after_bad = count()
print(f"Bad file     -> HTTP {status} | rows now {after_bad} (added {after_bad - after_good})")

status = send("test_files/fake_image.jpg")
print(f"Fake image   -> HTTP {status} | rows now {count()} (added {count() - after_bad})")