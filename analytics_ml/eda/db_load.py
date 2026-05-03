"""Shared DB helpers for EDA scripts."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[2]


def load_dotenv_from_project() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        pass


def database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "mysql+pymysql://appuser:apppassword@localhost:3307/job_market_db",
    )


def fetch_jobs_flat(db_url: str) -> pd.DataFrame:
    engine = create_engine(db_url)
    q = """
    SELECT
        j.job_id,
        j.title,
        j.location,
        j.source,
        j.description,
        j.created_at AS job_created_at,
        c.name AS company_name,
        agg.min_salary,
        agg.max_salary,
        agg.salary_currency
    FROM jobs j
    LEFT JOIN companies c ON j.company_id = c.company_id
    LEFT JOIN (
        SELECT job_id,
               MIN(min_salary) AS min_salary,
               MAX(max_salary) AS max_salary,
               MIN(currency) AS salary_currency
        FROM salaries
        GROUP BY job_id
    ) agg ON agg.job_id = j.job_id
    """
    return pd.read_sql(text(q), engine)


def table_row_counts(conn) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for table in ("jobs", "companies", "salaries", "skills", "job_skills"):
        try:
            r = conn.execute(text(f"SELECT COUNT(*) AS n FROM {table}"))
            row = r.fetchone()
            out[table] = int(row[0]) if row is not None else 0
        except Exception:
            out[table] = -1
    return out


def json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, pd.Series):
        return json_safe(obj.to_dict())
    return obj
