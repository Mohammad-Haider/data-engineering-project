#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for job_market_db.

Reads ``DATABASE_URL`` (or ``.env``), pulls jobs + companies + salaries into a flat
DataFrame, prints a short summary, writes ``eda_report.json``, and saves a few PNG plots.

Usage from project root:
  PYTHONPATH=. python analytics_ml/eda/run_eda.py
  PYTHONPATH=. python analytics_ml/eda/run_eda.py --no-plots
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from analytics_ml.eda.db_load import (
    ROOT,
    database_url,
    fetch_jobs_flat,
    json_safe,
    load_dotenv_from_project,
    table_row_counts,
)
from analytics_ml.eda.source_filters import drop_placeholder_sources


def build_report(df: pd.DataFrame, table_counts: Dict[str, int]) -> Dict[str, Any]:
    n = len(df)
    desc_len = df["description"].fillna("").astype(str).str.len()
    title_len = df["title"].fillna("").astype(str).str.len()

    has_salary = df["min_salary"].notna() & df["max_salary"].notna()
    mid = pd.Series(dtype=float)
    if has_salary.any():
        mid = (df.loc[has_salary, "min_salary"].astype(float) + df.loc[has_salary, "max_salary"].astype(float)) / 2.0

    report: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_counts_tables": table_counts,
        "flat_jobs_rows": n,
        "columns": list(df.columns),
        "missingness": {
            "company_name_pct": round(100.0 * df["company_name"].isna().mean(), 2) if n else 0.0,
            "description_empty_or_null_pct": round(
                100.0 * (df["description"].fillna("").astype(str).str.len() == 0).mean(), 2
            )
            if n
            else 0.0,
            "salary_both_present_pct": round(100.0 * has_salary.mean(), 2) if n else 0.0,
        },
        "title_length_chars": json_safe(title_len.describe().to_dict()) if n else {},
        "description_length_chars": json_safe(desc_len.describe().to_dict()) if n else {},
        "sources_top": json_safe(df["source"].fillna("(null)").value_counts().head(25)),
        "locations_top": json_safe(df["location"].fillna("(null)").value_counts().head(25)),
        "companies_top": json_safe(df["company_name"].fillna("(null)").value_counts().head(25)),
    }

    if len(mid) > 0:
        report["salary_midpoint_pkr"] = json_safe(mid.describe().to_dict())
        report["salary_midpoint_histogram_bins"] = json_safe(
            np.histogram(mid.dropna().values, bins=30, density=False)[0].tolist()
        )
    else:
        report["salary_midpoint_pkr"] = {}

    return report


def save_plots(df: pd.DataFrame, out_dir: Path) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    saved: List[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    vc = df["source"].fillna("(null)").value_counts().head(20)
    if len(vc):
        fig, ax = plt.subplots(figsize=(10, 6))
        vc.sort_values().plot.barh(ax=ax, color="#2c5282")
        ax.set_title("Job count by source (top 20)")
        ax.set_xlabel("Count")
        fig.tight_layout()
        p = out_dir / "eda_jobs_by_source.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        saved.append(str(p))

    loc = df["location"].fillna("(null)").value_counts().head(20)
    if len(loc):
        fig, ax = plt.subplots(figsize=(10, 6))
        loc.sort_values().plot.barh(ax=ax, color="#276749")
        ax.set_title("Job count by location (top 20)")
        ax.set_xlabel("Count")
        fig.tight_layout()
        p = out_dir / "eda_jobs_by_location.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        saved.append(str(p))

    has_salary = df["min_salary"].notna() & df["max_salary"].notna()
    if has_salary.sum() > 5:
        mid = (df.loc[has_salary, "min_salary"].astype(float) + df.loc[has_salary, "max_salary"].astype(float)) / 2.0
        mid = mid.replace([np.inf, -np.inf], np.nan).dropna()
        mid = mid[mid <= np.percentile(mid, 99)]
        if len(mid) > 5:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(mid, bins=40, color="#805ad5", edgecolor="white")
            ax.set_title("Distribution of midpoint salary (min+max)/2 — trimmed at 99th pct")
            ax.set_xlabel("PKR (approx)")
            ax.set_ylabel("Frequency")
            fig.tight_layout()
            p = out_dir / "eda_salary_midpoint_hist.png"
            fig.savefig(p, dpi=120)
            plt.close(fig)
            saved.append(str(p))

    return saved


def main() -> int:
    load_dotenv_from_project()
    parser = argparse.ArgumentParser(description="Run EDA on job_market_db.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analytics_ml" / "eda" / "output",
        help="Directory for eda_report.json and plots",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib PNG exports")
    args = parser.parse_args()

    db_url = database_url()
    print(f"Using database URL host: {db_url.split('@')[-1] if '@' in db_url else db_url[:40]}...")

    try:
        df = fetch_jobs_flat(db_url)
    except Exception as e:
        print(f"ERROR: could not read database: {e}", file=sys.stderr)
        return 1

    df, n_drop_ph = drop_placeholder_sources(df)
    if n_drop_ph:
        print(f"Dropped {n_drop_ph} row(s) with placeholder job sources (e.g. seed, Rozee.pk fallback).")

    engine = create_engine(db_url)
    with engine.connect() as conn:
        counts = table_row_counts(conn)

    report = build_report(df, counts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "eda_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {report_path}")

    if not args.no_plots:
        for p in save_plots(df, args.output_dir):
            print(f"Wrote {p}")

    print("\n--- Quick summary ---")
    print(f"jobs table rows: {counts.get('jobs', '?')}")
    print(f"flat extract rows: {len(df)}")
    print(f"rows with salary: {(df['min_salary'].notna() & df['max_salary'].notna()).sum()}")
    print(f"unique sources: {df['source'].nunique()}")
    print(f"unique locations: {df['location'].nunique()}")
    print(f"unique companies: {df['company_name'].nunique()}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
