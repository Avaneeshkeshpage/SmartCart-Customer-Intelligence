"""SQLite persistence for SmartCart's local/demo deployment.

For a multi-user production deployment, replace SQLite with PostgreSQL and
configure DATABASE_URL through the deployment's secret manager.
"""
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "smartcart.db"
CSV_PATH = BASE_DIR / "model" / "segmented_customers.csv"


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    """Create tables and import the bundled segmented CSV once if DB is empty."""
    with _connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            data_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS profile_assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            assignment_type TEXT NOT NULL,
            cluster_id INTEGER,
            segment TEXT NOT NULL,
            profile_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        if count == 0 and CSV_PATH.exists():
            df = pd.read_csv(CSV_PATH, index_col="ID")
            now = datetime.now(timezone.utc).isoformat()
            rows = []
            for customer_id, row in df.iterrows():
                values = json.loads(pd.DataFrame([row]).to_json(orient="records", date_format="iso"))[0]
                rows.append((str(customer_id), json.dumps(values, allow_nan=False), now))
            conn.executemany(
                "INSERT OR IGNORE INTO customers(customer_id, data_json, updated_at) VALUES (?, ?, ?)", rows
            )


def load_customers() -> pd.DataFrame:
    initialize_database()
    with _connect() as conn:
        rows = conn.execute("SELECT customer_id, data_json FROM customers ORDER BY customer_id").fetchall()
    if not rows:
        return pd.DataFrame()
    records = []
    ids = []
    for row in rows:
        ids.append(str(row["customer_id"]))
        records.append(json.loads(row["data_json"]))
    df = pd.DataFrame(records, index=pd.Index(ids, name="Customer ID"))
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    return df


def save_profile_assignment(profile: dict, segment: str, cluster_id=None, customer_id=None,
                            assignment_type="new_profile"):
    initialize_database()
    now = datetime.now(timezone.utc).isoformat()
    clean_profile = {}
    for key, value in profile.items():
        if hasattr(value, "item"):
            value = value.item()
        if pd.isna(value) if not isinstance(value, (dict, list, str, bool)) else False:
            value = None
        clean_profile[key] = value
    with _connect() as conn:
        conn.execute(
            """INSERT INTO profile_assignments
               (customer_id, assignment_type, cluster_id, segment, profile_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (str(customer_id) if customer_id is not None else None, assignment_type,
             int(cluster_id) if cluster_id is not None else None, str(segment),
             json.dumps(clean_profile, allow_nan=False), now),
        )


def list_profile_assignments(limit=100):
    initialize_database()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, customer_id, assignment_type, cluster_id, segment, profile_json, created_at "
            "FROM profile_assignments ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["profile"] = json.loads(item.pop("profile_json"))
        result.append(item)
    return result
