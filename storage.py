import json
import sqlite3
from pathlib import Path
from datetime import datetime

from config import Config

DB_PATH = Path(Config.SQLITE_PATH)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            dataset_key TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            models TEXT NOT NULL,
            sample_size INTEGER NOT NULL,
            mode TEXT NOT NULL, -- 'single' or 'compare'
            summary TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS layer_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            layer_index INTEGER NOT NULL,
            accuracy REAL,
            f1 REAL,
            ece REAL,
            extra JSON,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )
        """
    )

    conn.commit()
    conn.close()


def save_run(dataset_key, dataset_name, models, sample_size, mode, summary, metrics):
    """
    metrics: dict[model_name] -> list[dict(layer_index, accuracy, f1, ece, extra)]
    """
    conn = get_conn()
    cur = conn.cursor()

    created_at = datetime.utcnow().isoformat()
    cur.execute(
        """
        INSERT INTO runs (created_at, dataset_key, dataset_name, models,
                          sample_size, mode, summary)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            created_at,
            dataset_key,
            dataset_name,
            json.dumps(models),
            sample_size,
            mode,
            summary,
        ),
    )
    run_id = cur.lastrowid

    for model_name, layer_list in metrics.items():
        for layer in layer_list:
            cur.execute(
                """
                INSERT INTO layer_metrics
                    (run_id, model_name, layer_index, accuracy, f1, ece, extra)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    model_name,
                    layer["layer_index"],
                    layer["accuracy"],
                    layer["f1"],
                    layer["ece"],
                    json.dumps(layer.get("extra", {})),
                ),
            )

    conn.commit()
    conn.close()
    return run_id


def list_runs(limit=50):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM runs ORDER BY datetime(created_at) DESC LIMIT ?", (limit,)
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def get_run(run_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
    row = cur.fetchone()
    conn.close()
    return row


def get_run_metrics(run_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT model_name, layer_index, accuracy, f1, ece, extra
        FROM layer_metrics
        WHERE run_id = ?
        ORDER BY model_name, layer_index
        """,
        (run_id,),
    )
    rows = cur.fetchall()
    conn.close()

    metrics = {}
    for r in rows:
        metrics.setdefault(r["model_name"], []).append(
            {
                "layer_index": r["layer_index"],
                "accuracy": r["accuracy"],
                "f1": r["f1"],
                "ece": r["ece"],
                "extra": json.loads(r["extra"] or "{}"),
            }
        )
    return metrics
