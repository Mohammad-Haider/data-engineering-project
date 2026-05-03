import hashlib
import os
import re

import pandas as pd


def generate_fingerprint(title, company, location, external_id=None):
    """Generates a unique hash for a job posting to handle deduplication."""
    if external_id is not None and pd.notna(external_id) and str(external_id).strip():
        raw = f"jsearch:{str(external_id).strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if not all([title, company, location]):
        return None
    raw_string = f"{str(title).lower().strip()}|{str(company).lower().strip()}|{str(location).lower().strip()}"
    return hashlib.sha256(raw_string.encode("utf-8")).hexdigest()


def fingerprint_from_static_row_id(static_row_id):
    """Stable dedupe key for CSV / Kaggle rows (separate namespace from API ids)."""
    if static_row_id is None or (isinstance(static_row_id, float) and pd.isna(static_row_id)):
        return None
    s = str(static_row_id).strip()
    if not s:
        return None
    raw = f"static:{s}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_city(city_name):
    """Normalizes common Pakistani city names."""
    if not city_name:
        return "Unknown"

    city_name = str(city_name).lower()

    mapping = {
        "lhr": "Lahore",
        "isb": "Islamabad",
        "khi": "Karachi",
        "pindi": "Rawalpindi",
        "rwp": "Rawalpindi",
        "fsd": "Faisalabad",
    }

    for key, value in mapping.items():
        if key in city_name:
            return value

    return city_name.title()


def compact_location(value, max_len: int = 255):
    """Collapse whitespace/newlines (common in scraped CSVs) and cap length for MySQL VARCHAR."""
    s = " ".join(str(value or "").split()).strip()
    if not s:
        return "Unknown"
    return s[:max_len]


def extract_salary(salary_str):
    """Extracts min and max salary from a string like 'PKR 50K - 80K', or a bare number."""
    if pd.isna(salary_str):
        return None, None
    if isinstance(salary_str, (int, float)) and not isinstance(salary_str, bool):
        v = float(salary_str)
        return v, v
    if not isinstance(salary_str, str):
        return None, None

    numbers = re.findall(r"\d+", salary_str)

    if not numbers:
        return None, None

    if len(numbers) == 1:
        val = float(numbers[0]) * 1000 if "k" in salary_str.lower() else float(numbers[0])
        return val, val

    if len(numbers) >= 2:
        min_val = float(numbers[0]) * 1000 if "k" in salary_str.lower() else float(numbers[0])
        max_val = float(numbers[1]) * 1000 if "k" in salary_str.lower() else float(numbers[1])
        return min_val, max_val


def _sanitize_salary_bounds(min_v, max_v):
    """Cap to MySQL DECIMAL(10,2) range and drop absurd ranges (e.g. misparsed phone/year digits)."""
    cap = 99_999_999.99
    if pd.isna(min_v) or pd.isna(max_v):
        return None, None
    try:
        a, b = float(min_v), float(max_v)
    except (TypeError, ValueError):
        return None, None
    a, b = max(0.0, min(a, cap)), max(0.0, min(b, cap))
    if a > b:
        a, b = b, a
    if a > 0 and b > 20 * a:
        b = a
    return a, b


def _fallback_company_name() -> str:
    return os.environ.get(
        "PIPELINE_FALLBACK_COMPANY",
        os.environ.get("STATIC_JOBS_FALLBACK_COMPANY", "Employer not listed"),
    )


def _null_location_default() -> str:
    return os.environ.get("PIPELINE_NULL_LOCATION_DEFAULT", "Unknown")


def _coalesce_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip strings, fill NaN text fields, drop rows with no usable title."""
    if "title" not in df.columns:
        df["title"] = ""
    df["title"] = df["title"].fillna("").astype(str).str.strip()

    if "company" not in df.columns:
        df["company"] = ""
    df["company"] = df["company"].fillna("").astype(str).str.strip()
    df.loc[df["company"].str.len() == 0, "company"] = _fallback_company_name()

    if "location" not in df.columns:
        df["location"] = ""
    df["location"] = df["location"].fillna("").astype(str).str.strip()
    df.loc[df["location"].str.len() == 0, "location"] = _null_location_default()

    if "description" in df.columns:
        df["description"] = df["description"].fillna("").astype(str)
    else:
        df["description"] = ""

    df = df[df["title"].str.len() > 0]
    return df


def _dedupe_by_fingerprint_prefer_rich_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per fingerprint; when duplicates exist, keep the row with the richest payload
    (salary text present, longer description).
    """
    if "salary_raw" in df.columns:
        has_salary = df["salary_raw"].fillna("").astype(str).str.strip().ne("")
    else:
        has_salary = pd.Series(False, index=df.index)
    if "description" in df.columns:
        desc_len = df["description"].fillna("").astype(str).str.len()
    else:
        desc_len = pd.Series(0, index=df.index)
    df = df.copy()
    df["_dedupe_score"] = has_salary.astype(int) * 10_000 + desc_len
    df = df.sort_values("_dedupe_score", ascending=False).drop_duplicates(subset=["fingerprint"], keep="first")
    return df.drop(columns=["_dedupe_score"], errors="ignore")


def _repair_partial_salaries(df: pd.DataFrame) -> pd.DataFrame:
    """If only min or only max is set after parsing, mirror to the other bound."""
    if "min_salary" not in df.columns or "max_salary" not in df.columns:
        return df
    only_min = df["min_salary"].notna() & df["max_salary"].isna()
    only_max = df["min_salary"].isna() & df["max_salary"].notna()
    df.loc[only_min, "max_salary"] = df.loc[only_min, "min_salary"]
    df.loc[only_max, "min_salary"] = df.loc[only_max, "max_salary"]
    return df


def _impute_missing_salaries_with_batch_median(df: pd.DataFrame) -> pd.DataFrame:
    """
    For rows still missing both min and max salary, optionally fill with the batch **median**
    of mid-salaries from rows that do have both bounds (guarded by minimum count).

    Set ``IMPUTE_SALARY_WITH_BATCH_MEDIAN=0`` to disable (default: enabled).
    """
    if os.environ.get("IMPUTE_SALARY_WITH_BATCH_MEDIAN", "1").strip().lower() in ("0", "false", "no", "off"):
        return df
    if "min_salary" not in df.columns or "max_salary" not in df.columns:
        return df

    min_k = int(os.environ.get("IMPUTE_SALARY_MIN_KNOWN_ROWS", "15"))
    both = df["min_salary"].notna() & df["max_salary"].notna()
    if int(both.sum()) < min_k:
        return df

    mid = (df.loc[both, "min_salary"] + df.loc[both, "max_salary"]) / 2.0
    median_mid = float(mid.median())
    if pd.isna(median_mid) or median_mid <= 0:
        return df

    missing_both = df["min_salary"].isna() & df["max_salary"].isna()
    df.loc[missing_both, "min_salary"] = median_mid
    df.loc[missing_both, "max_salary"] = median_mid
    df[["min_salary", "max_salary"]] = df.apply(
        lambda row: pd.Series(_sanitize_salary_bounds(row["min_salary"], row["max_salary"])),
        axis=1,
    )
    return df


def clean_jobs_dataframe(df):
    """
    Normalize nulls, fingerprint, deduplicate (keeping best row per id), normalize location,
    parse and repair salaries, optional median imputation for missing salaries.
    """
    if df.empty:
        return df

    df = _coalesce_text_columns(df)
    if df.empty:
        return df

    def _fingerprint_row(row):
        fp_static = fingerprint_from_static_row_id(row.get("static_row_id"))
        if fp_static:
            return fp_static
        ext = row.get("jsearch_job_id") or row.get("adzuna_job_id")
        return generate_fingerprint(row.get("title"), row.get("company"), row.get("location"), ext)

    df["fingerprint"] = df.apply(_fingerprint_row, axis=1)
    df = df[df["fingerprint"].notna()]
    if df.empty:
        return df

    df = _dedupe_by_fingerprint_prefer_rich_row(df)

    df["location_clean"] = df["location"].apply(normalize_city).apply(compact_location)

    if "salary_raw" in df.columns:
        df[["min_salary", "max_salary"]] = df.apply(
            lambda row: pd.Series(extract_salary(row["salary_raw"])), axis=1
        )
        df[["min_salary", "max_salary"]] = df.apply(
            lambda row: pd.Series(_sanitize_salary_bounds(row["min_salary"], row["max_salary"])),
            axis=1,
        )
        df = _repair_partial_salaries(df)
        df[["min_salary", "max_salary"]] = df.apply(
            lambda row: pd.Series(_sanitize_salary_bounds(row["min_salary"], row["max_salary"])),
            axis=1,
        )
        df = _impute_missing_salaries_with_batch_median(df)

    return df
