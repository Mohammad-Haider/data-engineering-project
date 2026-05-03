#!/usr/bin/env python3
"""
One-shot helper: download the four Kaggle bundles (see datasets_manifest.json),
then load them into MySQL using the same logic as the Prefect pipeline.

Prerequisites (non-technical checklist):
  1. MySQL is running and reachable (e.g. ``docker compose up -d db`` from project root).
  2. Kaggle credentials (pick one):
        - Project root file ``.env`` with ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` (see ``.env.example``), or
        - ``~/.kaggle/kaggle.json`` from Kaggle: Settings -> API -> Create New Token (ZIP).
  3. From the project root, install deps: ``pip install -r requirements.txt``
  4. Set ``DATABASE_URL`` if you do not use the default local URL (see main_pipeline.py).

Usage (from project root):
  python scripts/bootstrap_kaggle_to_database.py

Options:
  --skip-download     Only load CSVs already under data/static_jobs/kaggle/
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_SCRIPT = ROOT / "scripts" / "download_kaggle_datasets.py"
KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"


def _load_project_env_file() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        pass


def _ensure_path() -> None:
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)


def _validate_kaggle_credentials() -> bool:
    user = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    if user and key:
        print("[OK] Kaggle credentials from .env / environment (username and key present — not shown).")
        return True

    if not KAGGLE_JSON.is_file():
        print(
            "\n[MISSING] Kaggle credentials.\n"
            "  Option A: In the project folder, copy .env.example to .env and set:\n"
            "            KAGGLE_USERNAME=...   and   KAGGLE_KEY=...\n"
            "  Option B: On kaggle.com -> Settings -> API -> Create New Token;\n"
            "            put kaggle.json in ~/.kaggle/\n"
        )
        return False
    try:
        data = json.loads(KAGGLE_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"\n[ERROR] ~/.kaggle/kaggle.json is not valid JSON: {e}\n")
        return False
    if "username" not in data or "key" not in data:
        print("\n[ERROR] kaggle.json must contain 'username' and 'key' fields.\n")
        return False
    print("[OK] Found ~/.kaggle/kaggle.json (username present, key present — not shown).")
    return True


def _run_download() -> bool:
    print("\n==> Downloading Kaggle datasets (this can take several minutes)...")
    r = subprocess.run([sys.executable, str(DOWNLOAD_SCRIPT)], cwd=str(ROOT))
    if r.returncode != 0:
        print("\n[ERROR] Download step failed. Fix Kaggle auth or network, then retry.\n")
        return False
    print("[OK] Download step finished.")
    return True


def _run_db_ingest() -> bool:
    _ensure_path()
    import pandas as pd

    from data_ingestion.static_datasets.csv_job_loader import load_job_dicts_from_csv_paths
    from data_transformation.cleaners import clean_jobs_dataframe
    from orchestration.flows.main_pipeline import load_to_mysql

    print("\n==> Loading CSV rows into MySQL...")
    rows = load_job_dicts_from_csv_paths()
    if not rows:
        print(
            "\n[ERROR] No rows loaded. Either run without --skip-download first, or place CSVs under\n"
            "        data/static_jobs/kaggle/<bundle_name>/ and ensure columns include job title + company.\n"
        )
        return False

    df = clean_jobs_dataframe(pd.DataFrame(rows))
    if df.empty:
        print("\n[ERROR] After cleaning, the dataframe is empty (check CSV columns).\n")
        return False

    print(f"    Rows to insert: {len(df)}")
    load_to_mysql.fn(df)
    print("[OK] MySQL load finished.")
    return True


def main() -> int:
    _load_project_env_file()
    parser = argparse.ArgumentParser(description="Download Kaggle job CSVs and load into MySQL.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Kaggle download; only ingest existing CSVs under data/static_jobs/kaggle/",
    )
    args = parser.parse_args()

    print("=== Kaggle -> MySQL bootstrap ===\n")
    print("Security: never paste API keys into chat, email, or git. Use only ~/.kaggle/kaggle.json.\n")

    if not _validate_kaggle_credentials():
        return 1

    if not args.skip_download:
        if not _run_download():
            return 1
    else:
        print("\n[INFO] Skipping download (--skip-download).")

    if not _run_db_ingest():
        return 1

    print("\n=== All done. You can open the Streamlit dashboard to explore the data. ===\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
