#!/usr/bin/env python3
"""
Advanced EDA: correlations, time trends, salary by source, IQR outliers, title token
frequencies, source×location heatmap, and richer JSON metrics.

By default, analysis is restricted to **IT-related jobs** whose text signals **Pakistan
(geography)** OR **remote / WFH / hybrid-remote** (see ``analytics_ml/eda/filters.py``).

Usage from project root:
  PYTHONPATH=. python analytics_ml/eda/run_advanced_eda.py
  PYTHONPATH=. python analytics_ml/eda/run_advanced_eda.py --no-plots
  PYTHONPATH=. python analytics_ml/eda/run_advanced_eda.py --full-corpus   # skip IT/geo filter
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from analytics_ml.eda.filters import (
    filter_description,
    filter_it_pakistan_remote,
    it_mask,
    pakistan_geo_mask,
    remote_mask,
)
from analytics_ml.eda.source_filters import drop_placeholder_sources

# Minimal English stopwords + noisy job-listing tokens
_STOP = frozenset(
    """
    the a an and or for with from to of in on at as by is are was were be been being
    this that these those our your their we you it its they them will can must should
    all any not no yes has have had having do does did doing done about into over out
    job jobs work works working worker team role open opening position candidate candidates
    experience experienced years year skills skill required requirement requirements looking
    seek seeking apply application applications based using used use well other such both
    each more most less least one two new senior junior lead manager engineer developer
    data software full stack front end back end quality assurance qa devops cloud mobile
    business analyst project product designer designer designers intern internship graduate
    level mid high low strong good excellent knowledge understanding ability able need
    needs needed including include included various multiple across join joining us company
    industry industries organization organizations ltd pvt private limited inc llc corp
    """.split()
)


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["title_len"] = d["title"].fillna("").astype(str).str.len()
    d["desc_len"] = d["description"].fillna("").astype(str).str.len()
    hs = d["min_salary"].notna() & d["max_salary"].notna()
    d["salary_mid"] = np.nan
    d.loc[hs, "salary_mid"] = (
        d.loc[hs, "min_salary"].astype(float) + d.loc[hs, "max_salary"].astype(float)
    ) / 2.0
    d["salary_span"] = np.nan
    d.loc[hs, "salary_span"] = d.loc[hs, "max_salary"].astype(float) - d.loc[hs, "min_salary"].astype(float)
    d["salary_span_ratio"] = np.nan
    m = d["salary_mid"] > 0
    d.loc[hs & m, "salary_span_ratio"] = d.loc[hs & m, "salary_span"] / d.loc[hs & m, "salary_mid"]
    ts = pd.to_datetime(d["job_created_at"], errors="coerce")
    d["job_month"] = ts.dt.to_period("M").astype(str)
    d["has_salary"] = hs
    return d


def _iqr_outlier_bounds(series: pd.Series) -> Tuple[float, float]:
    s = series.dropna()
    if len(s) < 10:
        return float("-inf"), float("inf")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return float(lo), float(hi)


def _entropy(counts: pd.Series) -> float:
    p = counts.values.astype(float)
    p = p[p > 0]
    p = p / p.sum()
    return float(-(p * np.log(p + 1e-15)).sum())


def _title_tokens(titles: pd.Series, top_n: int = 50) -> Dict[str, int]:
    tok = re.compile(r"[A-Za-z]{4,}")
    c: Counter[str] = Counter()
    for t in titles.fillna("").astype(str):
        for w in tok.findall(t.lower()):
            if w in _STOP:
                continue
            c[w] += 1
    return dict(c.most_common(top_n))


def build_advanced_report(
    d: pd.DataFrame,
    table_counts: Dict[str, int],
    segment: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    n = len(d)
    hs = d["has_salary"]
    mid = d.loc[hs, "salary_mid"]

    report: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_counts_tables": table_counts,
        "segment_filter": segment or {},
        "flat_jobs_rows": n,
        "salary_by_source": {},
        "iqr_outliers_salary_mid_overall": {},
        "salary_span_ratio_on_mid": {},
        "duplicate_titles": {},
        "time_series_jobs_by_month": {},
        "source_distribution_entropy": None,
        "numeric_correlations": {},
        "title_token_counts_top": {},
        "location_concentration_top10_share_pct": None,
    }

    if n == 0:
        return report

    # Salary by source (only rows with salary)
    if hs.any():
        g = (
            d.loc[hs]
            .groupby(d.loc[hs, "source"].fillna("(null)"))["salary_mid"]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .sort_values("count", ascending=False)
        )
        report["salary_by_source"] = json_safe(g.head(25).round(2))

        lo, hi = _iqr_outlier_bounds(mid)
        out = (mid < lo) | (mid > hi)
        report["iqr_outliers_salary_mid_overall"] = {
            "lower_fence": lo,
            "upper_fence": hi,
            "outlier_count": int(out.sum()),
            "outlier_pct_of_salary_rows": round(100.0 * out.mean(), 2) if len(mid) else 0.0,
        }

        ratio = d.loc[hs, "salary_span_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio):
            report["salary_span_ratio_on_mid"] = json_safe(ratio.describe().round(4).to_dict())

    # Duplicate titles (exact string)
    tc = d["title"].fillna("").astype(str).str.strip()
    vc = tc.value_counts()
    dup = vc[vc > 1]
    report["duplicate_titles"] = {
        "unique_titles": int(vc.shape[0]),
        "titles_appearing_more_than_once": int(dup.shape[0]),
        "rows_in_duplicate_title_groups": int(dup.sum()),
        "top_duplicate_titles": json_safe(dup.head(20)),
    }

    # Time series
    mcounts = d["job_month"].replace("NaT", np.nan).dropna().value_counts().sort_index()
    report["time_series_jobs_by_month"] = json_safe(mcounts.tail(36))

    ent = _entropy(d["source"].fillna("(null)").value_counts())
    report["source_distribution_entropy"] = round(ent, 4)

    loc_vc = d["location"].fillna("(null)").value_counts()
    if loc_vc.sum() > 0:
        report["location_concentration_top10_share_pct"] = round(
            100.0 * loc_vc.head(10).sum() / loc_vc.sum(), 2
        )

    # Correlations on numeric slice
    num = d[["title_len", "desc_len", "salary_mid"]].copy()
    corr = num.corr(method="pearson")
    report["numeric_correlations"] = json_safe(corr.round(4).to_dict())

    report["title_token_counts_top"] = _title_tokens(d["title"], top_n=50)

    # Within-segment signal mix (IT ∧ filter already applied)
    if n > 0:
        pk = pakistan_geo_mask(d)
        rm = remote_mask(d)
        report["segment_geo_remote_breakdown"] = {
            "rows_with_pakistan_geo_signal": int(pk.sum()),
            "rows_with_remote_signal": int(rm.sum()),
            "rows_with_both_signals": int((pk & rm).sum()),
            "rows_pk_only": int((pk & ~rm).sum()),
            "rows_remote_only": int((~pk & rm).sum()),
        }

    return report


def save_advanced_plots(d: pd.DataFrame, out_dir: Path, *, subtitle: str = "") -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    # Correlation heatmap
    num = d[["title_len", "desc_len", "salary_mid"]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(num.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num.columns)))
    ax.set_yticks(range(len(num.index)))
    ax.set_xticklabels(num.columns, rotation=35, ha="right")
    ax.set_yticklabels(num.index)
    for i in range(num.shape[0]):
        for j in range(num.shape[1]):
            ax.text(j, i, f"{num.values[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Pearson correlation (title_len, desc_len, salary_mid)" + subtitle)
    fig.tight_layout()
    p = out_dir / "advanced_corr_heatmap.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(str(p))

    # Jobs by month
    mcounts = d["job_month"].replace("NaT", np.nan).dropna().value_counts().sort_index()
    if len(mcounts) > 1:
        fig, ax = plt.subplots(figsize=(12, 4))
        mcounts.plot(ax=ax, kind="line", marker="o", color="#3182ce")
        ax.set_title("Job postings by month (created_at)" + subtitle)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        p2 = out_dir / "advanced_jobs_by_month.png"
        fig.savefig(p2, dpi=120)
        plt.close(fig)
        saved.append(str(p2))

    # Boxplot salary_mid by top sources
    hs = d["has_salary"]
    if hs.sum() > 20:
        top_src = d.loc[hs, "source"].value_counts().head(6).index.tolist()
        data = [d.loc[(d["source"] == s) & hs, "salary_mid"].dropna().values for s in top_src]
        lens = [len(x) for x in data]
        if max(lens) > 5:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.boxplot(data, showfliers=False)
            ax.set_xticklabels([s[:28] for s in top_src], rotation=25)
            ax.set_ylabel("Salary midpoint (PKR)")
            ax.set_title("Salary midpoint distribution by source (top 6 by count)" + subtitle)
            fig.tight_layout()
            p3 = out_dir / "advanced_salary_boxplot_by_source.png"
            fig.savefig(p3, dpi=120)
            plt.close(fig)
            saved.append(str(p3))

    # Source × location heatmap (top categories)
    top_s = d["source"].fillna("(null)").value_counts().head(8).index
    top_l = d["location"].fillna("(null)").value_counts().head(8).index
    sub = d[d["source"].isin(top_s) & d["location"].isin(top_l)]
    if len(sub) > 30:
        ct = pd.crosstab(sub["source"], sub["location"])
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(ct.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(ct.columns)))
        ax.set_yticks(range(len(ct.index)))
        ax.set_xticklabels([c[:22] for c in ct.columns], rotation=45, ha="right")
        ax.set_yticklabels([r[:35] for r in ct.index])
        ax.set_title("Heatmap: job counts (top 8 sources × top 8 locations)" + subtitle)
        fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        p4 = out_dir / "advanced_source_location_heatmap.png"
        fig.savefig(p4, dpi=120)
        plt.close(fig)
        saved.append(str(p4))

    # Title tokens barh
    tok = _title_tokens(d["title"], top_n=30)
    if tok:
        fig, ax = plt.subplots(figsize=(8, 8))
        words = list(tok.keys())[::-1]
        vals = [tok[w] for w in words]
        ax.barh(words, vals, color="#553c9a")
        ax.set_title("Most frequent title words (length ≥4, stopwords removed)" + subtitle)
        ax.set_xlabel("Count")
        fig.tight_layout()
        p5 = out_dir / "advanced_title_tokens.png"
        fig.savefig(p5, dpi=120)
        plt.close(fig)
        saved.append(str(p5))

    return saved


def main() -> int:
    load_dotenv_from_project()
    parser = argparse.ArgumentParser(description="Advanced EDA on job_market_db.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analytics_ml" / "eda" / "output",
        help="Output directory for JSON and PNGs",
    )
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="Do not apply IT + Pakistan/remote filter (entire jobs table).",
    )
    parser.add_argument("--no-plots", action="store_true")
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

    n_full = len(df)
    segment: Dict[str, Any]
    plot_sub = ""
    if args.full_corpus:
        segment = {
            "filter_applied": False,
            "description": "Full corpus (no IT / Pakistan / remote text filter).",
            "rows_before_filter": n_full,
            "rows_after_filter": n_full,
        }
        df_f = df
        print(f"Rows (full corpus): {n_full}")
    else:
        df_f, n_before, n_after = filter_it_pakistan_remote(df)
        segment = {
            "filter_applied": True,
            "description": filter_description(),
            "rows_before_filter": n_before,
            "rows_after_filter": n_after,
            "it_rows_in_full_corpus": int(it_mask(df).sum()),
        }
        plot_sub = "\n(IT · Pakistan geo OR remote/WFH)"
        print(f"Segment filter: IT + (Pakistan geo OR remote/WFH)")
        print(f"  Rows before filter: {n_before}  after: {n_after}  ({100.0 * n_after / max(n_before, 1):.1f}% kept)")
        print(f"  IT-matching rows in full corpus (for reference): {segment['it_rows_in_full_corpus']}")

    d = _enrich(df_f)

    engine = create_engine(db_url)
    with engine.connect() as conn:
        counts = table_row_counts(conn)

    report = build_advanced_report(d, counts, segment=segment)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "advanced_eda_report.json"
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_json}")

    if not args.no_plots:
        for p in save_advanced_plots(d, args.output_dir, subtitle=plot_sub):
            print(f"Wrote {p}")

    print("\n--- Advanced summary ---")
    print(f"Analyzed rows (after filter): {report.get('flat_jobs_rows')}")
    print(f"Source entropy: {report.get('source_distribution_entropy')}")
    print(f"Top-10 location share %: {report.get('location_concentration_top10_share_pct')}")
    br = report.get("segment_geo_remote_breakdown") or {}
    if br:
        print(
            f"Geo/remote mix: PK-signal={br.get('rows_with_pakistan_geo_signal')} "
            f"remote-signal={br.get('rows_with_remote_signal')} "
            f"both={br.get('rows_with_both_signals')}"
        )
    dup = report.get("duplicate_titles") or {}
    print(f"Duplicate title groups: {dup.get('titles_appearing_more_than_once', 'n/a')}")
    iqr = report.get("iqr_outliers_salary_mid_overall") or {}
    print(f"IQR salary-mid outliers: {iqr.get('outlier_count', 'n/a')} ({iqr.get('outlier_pct_of_salary_rows', 'n/a')}%)")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
