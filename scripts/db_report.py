from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import inspect, text

from src.db import DATABASE_URL, engine

print("Database engine:", engine.url.render_as_string(hide_password=True))
print("\n=== Table structure ===")
for col in inspect(engine).get_columns("predictions"):
    print(f"  {col['name']:15} {str(col['type'])}")


def show(title, sql):
    print(f"\n=== {title} ===")
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        print("  " + " | ".join(result.keys()))
        for row in result:
            print("  " + " | ".join(str(v)[:19] if v is not None else "-" for v in row))


show("Newest 8 decisions",
     "SELECT id, created_at, filename, label, ROUND(confidence, 3) AS confidence, "
     "model_version FROM predictions ORDER BY id DESC LIMIT 8")
show("Decisions per label",
     "SELECT label, COUNT(*) AS n, ROUND(AVG(confidence), 3) AS avg_confidence "
     "FROM predictions GROUP BY label")
show("Decisions per model version",
     "SELECT model_version, COUNT(*) AS n, MIN(created_at) AS first_use, "
     "MAX(created_at) AS last_use FROM predictions GROUP BY model_version")
show("Least confident decisions",
     "SELECT id, filename, label, ROUND(confidence, 3) AS confidence, "
     "ROUND(p_ai, 3) AS p_ai FROM predictions ORDER BY confidence ASC LIMIT 5")