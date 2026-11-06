import os
from datetime import datetime, timezone

from sqlalchemy import (Column, DateTime, Float, Integer, MetaData, String, Table,
                        create_engine, insert, select, text)

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///visionguard.db")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
metadata = MetaData()

predictions = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False),   # stored in UTC
    Column("filename", String(120), nullable=False),                 # sanitised name only
    Column("file_size_kb", Float),
    Column("label", String(20), nullable=False),
    Column("confidence", Float, nullable=False),
    Column("p_ai", Float, nullable=False),
    Column("threshold", Float, nullable=False),
    Column("model_version", String(40), nullable=False),
    Column("latency_ms", Float),
)


def init_db():
    metadata.create_all(engine)


def save_prediction(**row):
    with engine.begin() as conn:
        conn.execute(insert(predictions).values(created_at=datetime.now(timezone.utc), **row))


def recent_predictions(limit=100):
    with engine.connect() as conn:
        stmt = select(predictions).order_by(predictions.c.id.desc()).limit(limit)
        return conn.execute(stmt).mappings().all()


def db_ok():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False