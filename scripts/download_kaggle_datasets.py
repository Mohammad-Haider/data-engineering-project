#!/usr/bin/env python3
"""
Download the four configured Pakistan job datasets from Kaggle into
data/static_jobs/kaggle/<name>/ (unzipped).

Setup:
  pip install kaggle python-dotenv
  Either create project root ".env" with KAGGLE_USERNAME and KAGGLE_KEY (see .env.example),
  or place API credentials in ~/.kaggle/kaggle.json from
  https://www.kaggle.com/settings -> API -> Create New Token.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/static_jobs/kaggle/datasets_manifest.json"
KAGGLE_ROOT = ROOT / "data/static_jobs/kaggle"


def _load_project_env_file() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        pass


def _kaggle_cmd_prefix() -> list[str]:
    exe = shutil.which("kaggle")
    if exe:
        return [exe]
    return [sys.executable, "-m", "kaggle"]


def _credentials_present() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    kaggle_dir = Path.home() / ".kaggle"
    return (kaggle_dir / "kaggle.json").is_file()


def main() -> int:
    _load_project_env_file()
    parser = argparse.ArgumentParser(description="Download Kaggle Pakistan job CSV bundles.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to kaggle datasets download.",
    )
    args = parser.parse_args()

    if not MANIFEST.is_file():
        print(f"Manifest not found: {MANIFEST}", file=sys.stderr)
        return 1

    if not _credentials_present():
        print(
            "Kaggle API credentials not found.\n"
            "  - Add ~/.kaggle/kaggle.json (from Kaggle Account -> API -> Create New API Token), or\n"
            "  - Export KAGGLE_USERNAME and KAGGLE_KEY.\n"
            "See https://github.com/Kaggle/kaggle-api/",
            file=sys.stderr,
        )
        return 1

    spec = json.loads(MANIFEST.read_text(encoding="utf-8"))
    datasets = spec.get("datasets") or []
    if len(datasets) != 4:
        print(f"Warning: manifest lists {len(datasets)} datasets (expected 4).", file=sys.stderr)

    KAGGLE_ROOT.mkdir(parents=True, exist_ok=True)
    prefix = _kaggle_cmd_prefix()

    failures = 0
    for ds in datasets:
        slug = ds["slug"]
        name = ds["name"]
        target = KAGGLE_ROOT / name
        target.mkdir(parents=True, exist_ok=True)
        print(f"\n==> {slug} -> {target}")

        cmd = prefix + [
            "datasets",
            "download",
            "-d",
            slug,
            "-p",
            str(target),
            "--unzip",
        ]
        if args.force:
            cmd.append("--force")

        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            failures += 1
            print(f"ERROR: kaggle exited {r.returncode} for {slug}", file=sys.stderr)
            continue

        for z in target.glob("*.zip"):
            try:
                z.unlink()
            except OSError:
                pass

        csvs = list(target.rglob("*.csv"))
        print(f"    CSV files found: {len(csvs)}")
        for c in sorted(csvs)[:12]:
            print(f"      - {c.relative_to(ROOT)}")
        if len(csvs) > 12:
            print(f"      ... and {len(csvs) - 12} more")

    if failures:
        print(f"\nCompleted with {failures} failure(s).", file=sys.stderr)
        return 1
    print("\nAll configured datasets downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
