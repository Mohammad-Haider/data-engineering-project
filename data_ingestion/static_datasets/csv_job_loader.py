"""
Load job postings from local CSV exports (e.g. Kaggle Pakistan job datasets)
into the same dict shape as API extractors (title, company, location, ...).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    """Project root (parent of ``data_ingestion``)."""
    return Path(__file__).resolve().parents[2]


def _norm_key(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def _first_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return the first DataFrame column whose normalized name matches a candidate."""
    col_by_key = {_norm_key(c): c for c in df.columns}
    for cand in candidates:
        k = _norm_key(cand)
        if k in col_by_key:
            return col_by_key[k]
    return None


def _cell_str(row: pd.Series, col: Optional[str]) -> str:
    if not col:
        return ""
    v = row.get(col)
    if pd.isna(v):
        return ""
    return str(v).strip()


def _cell_scalar(row: pd.Series, col: Optional[str]) -> Any:
    if not col:
        return None
    v = row.get(col)
    if pd.isna(v):
        return None
    return v


def _to_numeric_salary(val: Any) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _build_salary_raw(row: pd.Series, salary_col: Optional[str], smin: Optional[str], smax: Optional[str]) -> Optional[str]:
    if salary_col:
        v = row.get(salary_col)
        if pd.isna(v):
            return None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return str(int(v)) if float(v) == int(v) else str(v)
        s = str(v).strip()
        return s if s else None
    lo = _to_numeric_salary(_cell_scalar(row, smin))
    hi = _to_numeric_salary(_cell_scalar(row, smax))
    if lo is not None and hi is not None:
        return f"{int(lo)} - {int(hi)}" if lo >= 1000 else f"{lo} - {hi}"
    if lo is not None:
        return str(int(lo)) if lo >= 1000 else str(lo)
    return None


def _parse_csv_paths(raw: str) -> List[str]:
    if not raw or not str(raw).strip():
        return []
    parts = re.split(r"[\n;]+", str(raw).strip())
    return [p.strip() for p in parts if p.strip()]


def _infer_kaggle_bundle_source_label(path: Path) -> Optional[str]:
    """Use folder name under ``.../kaggle/<bundle>/`` as stable source, e.g. ``Kaggle/rozee_pk``."""
    p = path.resolve()
    parts = p.parts
    for i, seg in enumerate(parts):
        if seg == "kaggle" and i + 1 < len(parts):
            return f"Kaggle/{parts[i + 1]}"
    return None


def discover_kaggle_bundle_csv_paths() -> List[str]:
    """
    Discover ``*.csv`` under ``data/static_jobs/kaggle/<bundle_name>/`` (one level of bundle dirs).

    Disabled when env ``STATIC_JOBS_KAGGLE_AUTO`` is ``0``, ``false``, or ``no``.
    """
    flag = os.environ.get("STATIC_JOBS_KAGGLE_AUTO", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return []
    root = _project_root() / "data" / "static_jobs" / "kaggle"
    if not root.is_dir():
        return []
    paths: List[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        for csv_path in sorted(child.rglob("*.csv")):
            if "__MACOSX" in csv_path.parts or ".ipynb_checkpoints" in csv_path.parts:
                continue
            paths.append(str(csv_path.resolve()))
    return paths


def dataframe_to_job_dicts(
    df: pd.DataFrame,
    *,
    source_label: str,
    file_stem: str,
) -> List[Dict[str, Any]]:
    """
    Map heterogeneous CSV columns to pipeline keys.
    """
    if df.empty:
        return []

    title_c = _first_column(
        df,
        (
            "title",
            "jobtitle",
            "job_title",
            "job title",
            "position",
            "designation",
            "role",
            "name",
            "posting_title",
            "job_opening_title",
        ),
    )
    company_c = _first_column(
        df,
        (
            "company",
            "companyname",
            "company_name",
            "employer",
            "employer_name",
            "organization",
            "organisation",
            "hiring_company",
            "org",
            "website_domain",
            "ticker",
        ),
    )
    loc_c = _first_column(
        df,
        (
            "location",
            "city",
            "job_location",
            "job location",
            "place",
            "region",
            "job_city",
            "area",
            "geo",
        ),
    )
    desc_c = _first_column(
        df,
        (
            "description",
            "job_description",
            "job description",
            "desc",
            "details",
            "summary",
            "skills",
        ),
    )
    salary_one = _first_column(
        df,
        (
            "salary",
            "salary_raw",
            "salaryrange",
            "compensation",
            "pay",
            "expected_salary",
            "salary_range",
            "compensation_range",
            "annual_salary",
        ),
    )
    smin_c = _first_column(
        df,
        (
            "min_salary",
            "salary_min",
            "salary_from",
            "comp_min",
            "minsalary",
            "from_salary",
        ),
    )
    smax_c = _first_column(
        df,
        (
            "max_salary",
            "salary_max",
            "salary_to",
            "comp_max",
            "maxsalary",
            "to_salary",
        ),
    )
    id_c = _first_column(
        df,
        (
            "job_id",
            "id",
            "listing_id",
            "uniq_id",
            "unique_id",
            "url",
            "link",
            "job_link",
            "job_url",
        ),
    )

    if not title_c:
        logger.warning(
            "CSV %s: missing title-like column. Found: %s",
            file_stem,
            list(df.columns),
        )
        return []

    fallback_company = os.environ.get(
        "STATIC_JOBS_FALLBACK_COMPANY",
        "Employer not listed (CSV)",
    )
    if not company_c:
        logger.info(
            "CSV %s: no company column; using fallback employer name for all rows.",
            file_stem,
        )

    if not loc_c:
        logger.info("CSV %s: no location column; defaulting location to 'Pakistan'.", file_stem)

    out: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        title = _cell_str(row, title_c)
        company = _cell_str(row, company_c) if company_c else fallback_company
        location = _cell_str(row, loc_c) if loc_c else "Pakistan"
        if not title:
            continue
        if not company:
            company = fallback_company

        raw_id = _cell_str(row, id_c) if id_c else ""
        static_key = f"{file_stem}:{raw_id}" if raw_id else f"{file_stem}:row_{idx}"

        salary_raw = _build_salary_raw(row, salary_one, smin_c, smax_c)

        rec: Dict[str, Any] = {
            "title": title,
            "company": company,
            "location": location or "Pakistan",
            "source": source_label[:250],
            "description": _cell_str(row, desc_c),
            "static_row_id": static_key[:512],
        }
        if salary_raw:
            rec["salary_raw"] = salary_raw
        out.append(rec)

    return out


def load_csv_file(path: Path, *, source_label: Optional[str] = None) -> List[Dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        logger.warning("Static jobs CSV not found: %s", path)
        return []

    stem = path.stem.replace(" ", "_")[:80]
    label = source_label or _infer_kaggle_bundle_source_label(path) or f"Kaggle/{stem}"

    enc_primary = os.environ.get("STATIC_JOBS_CSV_ENCODING", "utf-8")
    seen: set[str] = set()
    encodings: List[str] = []
    for e in (enc_primary, "utf-8-sig", "latin-1", "cp1252"):
        if e not in seen:
            seen.add(e)
            encodings.append(e)
    df = None
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines="skip", low_memory=False)
            if enc != enc_primary:
                logger.info("Read %s using encoding %s", path.name, enc)
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        logger.error("Failed reading CSV %s: %s", path, last_err)
        return []

    logger.info("Loaded CSV %s: %s rows, columns=%s", path.name, len(df), list(df.columns)[:20])
    return dataframe_to_job_dicts(df, source_label=label, file_stem=stem)


def load_job_dicts_from_csv_paths(paths: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """
    Load and merge all configured CSV paths.

    Resolution order:

    1. Explicit ``paths`` argument.
    2. Env ``STATIC_JOBS_CSV_PATHS`` (newline or ``;`` separated).
    3. Auto-discovery: all ``*.csv`` under ``data/static_jobs/kaggle/*/`` after running
       ``scripts/download_kaggle_datasets.py`` (unless ``STATIC_JOBS_KAGGLE_AUTO=0``).
    """
    if paths is not None:
        path_list = [str(p).strip() for p in paths if str(p).strip()]
    else:
        raw = os.environ.get("STATIC_JOBS_CSV_PATHS", "")
        path_list = _parse_csv_paths(raw)
        if not path_list:
            path_list = discover_kaggle_bundle_csv_paths()
            if path_list:
                logger.info(
                    "Using %s CSV(s) from data/static_jobs/kaggle/ (auto-discovery).",
                    len(path_list),
                )

    if not path_list:
        return []

    all_rows: List[Dict[str, Any]] = []
    for p in path_list:
        all_rows.extend(load_csv_file(Path(p)))
    logger.info("Static CSV ingest total rows: %s (from %s files)", len(all_rows), len(path_list))
    return all_rows
