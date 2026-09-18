import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "audit.db"


def initialize_database(db_path: Path = DEFAULT_DB_PATH) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS extraction_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                redaction_summary TEXT NOT NULL,
                model_used TEXT NOT NULL,
                retries_needed INTEGER NOT NULL,
                validation_success INTEGER NOT NULL
            )"""
        )


def log_extraction(
    model_used: str,
    redaction_mapping: list[dict],
    retries_needed: int,
    validation_success: bool,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    initialize_database(db_path)
    summary = {"entity_types": sorted({item["entity_type"] for item in redaction_mapping}), "count": len(redaction_mapping)}
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO extraction_audit (timestamp, redaction_summary, model_used, retries_needed, validation_success) VALUES (?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), json.dumps(summary), model_used, retries_needed, int(validation_success)),
        )