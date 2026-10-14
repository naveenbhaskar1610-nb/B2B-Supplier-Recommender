"""Tiny SQLite layer so market analytics are plain, readable SQL (see sql/analytics.sql)."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

SQL_FILE = Path(__file__).resolve().parent.parent / "sql" / "analytics.sql"


def load_named_queries(path: Path = SQL_FILE) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^-- name:\s*(\w+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def run_analytics(suppliers: pd.DataFrame, offers: pd.DataFrame) -> dict[str, pd.DataFrame]:
    queries = load_named_queries()
    with sqlite3.connect(":memory:") as con:
        suppliers.to_sql("suppliers", con, index=False)
        offers.to_sql("offers", con, index=False)
        return {name: pd.read_sql_query(sql, con) for name, sql in queries.items()}
