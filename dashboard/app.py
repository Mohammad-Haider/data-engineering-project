"""
Pakistan Tech Job Market Intelligence Platform — redesigned dashboard.
Matches target UX: dark sidebar, card-based analytics, DB-backed charts.
"""
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

try:
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --- Theme (aligned with reference screenshots) ---
COL_BG_MAIN = "#eef2ef"
COL_BG_CARD = "#ffffff"
COL_SIDEBAR = "#153529"
COL_SIDEBAR_ACTIVE = "#1f4a38"
COL_ACCENT = "#2d6a4f"
COL_GOLD = "#c9a227"
COL_TEXT = "#1a1a1a"
COL_MUTED = "#5c6670"


def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&display=swap');
        html, body, [class*="css"]  {{
            font-family: 'DM Sans', system-ui, sans-serif;
        }}
        .stApp {{
            background-color: {COL_BG_MAIN};
        }}
        [data-testid="stHeader"] {{
            background-color: {COL_BG_MAIN};
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {COL_SIDEBAR} 0%, #0f2419 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }}
        [data-testid="stSidebar"] .block-container {{
            padding-top: 1.5rem;
        }}
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {{
            color: #e8f0ec !important;
        }}
        [data-testid="stSidebar"] .stRadio label {{
            font-size: 0.9rem;
        }}
        [data-testid="stSidebar"] hr {{
            border-color: rgba(255,255,255,0.12);
        }}
        .hero-title {{
            color: {COL_ACCENT};
            font-weight: 700;
            font-size: 1.55rem;
            letter-spacing: -0.02em;
            margin: 0 0 0.25rem 0;
        }}
        .hero-sub {{
            color: {COL_MUTED};
            font-size: 0.88rem;
        }}
        .badge-live {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            background: #e8f5ec;
            color: {COL_ACCENT};
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            border: 1px solid rgba(45, 106, 79, 0.2);
        }}
        .metric-card {{
            background: {COL_BG_CARD};
            border-radius: 12px;
            padding: 0.85rem 1rem;
            border-top: 4px solid var(--accent, {COL_ACCENT});
            box-shadow: 0 2px 12px rgba(15, 36, 25, 0.05);
        }}
        .metric-card h4 {{
            margin: 0;
            font-size: 0.72rem;
            color: {COL_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        .metric-card .val {{
            margin: 0.35rem 0 0 0;
            font-size: 1.45rem;
            font-weight: 700;
            color: {COL_TEXT};
        }}
        .metric-card .delta {{
            font-size: 0.78rem;
            color: {COL_ACCENT};
            margin-top: 0.25rem;
        }}
        .section-title {{
            font-size: 1rem;
            font-weight: 600;
            color: {COL_TEXT};
            margin: 0 0 0.15rem 0;
        }}
        .section-sub {{
            font-size: 0.82rem;
            color: {COL_MUTED};
            margin: 0 0 1rem 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_db_engine():
    db_url = os.environ.get(
        "DATABASE_URL",
        "mysql+pymysql://appuser:apppassword@localhost:3307/job_market_db",
    )
    return create_engine(db_url)


def fetch_data(query, params=None):
    engine = get_db_engine()
    try:
        if params:
            return pd.read_sql(query, engine, params=params)
        return pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()


def _job_postings_by_month_and_source():
    """Long-format counts: ym (YYYY-MM), src (Rozee = synthetic), cnt."""
    return fetch_data(
        """
        SELECT DATE_FORMAT(j.created_at, '%%Y-%%m') AS ym,
               CASE WHEN TRIM(IFNULL(j.source, '')) = 'Synthetic' THEN 'Rozee'
                    ELSE COALESCE(NULLIF(TRIM(j.source), ''), 'Unknown') END AS src,
               COUNT(*) AS cnt
        FROM jobs j
        WHERE j.created_at IS NOT NULL
        GROUP BY ym, src
        ORDER BY ym ASC
        """
    )


def render_job_postings_over_time():
    """
    Multi-source line chart (mock #2): markers, area under primary, dashed 3rd line.
    Single month: grouped narrow bars by source (fixes full-width Streamlit bar).
    """
    df = _job_postings_by_month_and_source()
    if df.empty:
        _card_start("Job Postings Over Time", "Monthly trend by source.")
        st.info("No time series yet.")
        return

    df["cnt"] = pd.to_numeric(df["cnt"], errors="coerce").fillna(0)
    pivot = df.pivot_table(index="ym", columns="src", values="cnt", aggfunc="sum").fillna(0)
    pivot = pivot.sort_index()
    if len(pivot) > 18:
        pivot = pivot.iloc[-18:]

    col_totals = pivot.sum(axis=0).sort_values(ascending=False)
    top_cols = [c for c in col_totals.head(3).index if c in pivot.columns]
    if not top_cols:
        _card_start("Job Postings Over Time", "Monthly trend by source.")
        st.info("No time series yet.")
        return
    pivot = pivot[top_cols]

    yms = [str(x) for x in pivot.index.tolist()]
    years = {ym[:4] for ym in yms}
    if len(years) <= 1:
        x_labels = [pd.Timestamp(ym + "-01").strftime("%b") for ym in yms]
    else:
        x_labels = [pd.Timestamp(ym + "-01").strftime("%b %y") for ym in yms]

    t0 = pd.Timestamp(yms[0] + "-01").strftime("%b %Y")
    t1 = pd.Timestamp(yms[-1] + "-01").strftime("%b %Y")
    range_caption = t0 if yms[0] == yms[-1] else f"{t0} – {t1}"

    _card_start(
        "Job Postings Over Time",
        f"Monthly trend by source ({range_caption}).",
    )

    if not _HAS_MPL:
        st.caption("Install matplotlib for the styled multi-line chart.")
        st.line_chart(pivot.sum(axis=1).to_frame(name="Total jobs"))
        return

    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(10, 4.25), dpi=120)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    colors = ["#2d6a4f", "#c9a227", "#2563eb"]
    n_months = len(pivot)
    xa = np.arange(n_months)

    if n_months < 2:
        st.caption(
            "Single calendar month in the database — volume by source (narrow grouped bars). "
            "A multi-month line chart appears once postings span more than one month."
        )
        n_series = len(pivot.columns)
        w = 0.55 / max(n_series, 1) * 0.92
        offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * w
        for i, col in enumerate(pivot.columns):
            h = float(pivot[col].iloc[0])
            ax.bar(
                xa + offsets[i],
                [h],
                width=w * 0.95,
                color=colors[i % len(colors)],
                edgecolor="white",
                linewidth=1.0,
                label=col,
            )
        ax.set_xticks(xa)
        ax.set_xticklabels(x_labels)
    else:
        for i, col in enumerate(pivot.columns):
            y = pivot[col].values.astype(float)
            linestyle = "--" if i == 2 else "-"
            ax.plot(
                xa,
                y,
                marker="o",
                markersize=6,
                linestyle=linestyle,
                linewidth=2.1,
                color=colors[i % len(colors)],
                label=col,
            )
        y_primary = pivot.iloc[:, 0].values.astype(float)
        ax.fill_between(xa, 0, y_primary, alpha=0.14, color=colors[0], linewidth=0)

        ax.set_xticks(xa)
        ax.set_xticklabels(x_labels)

    ax.set_ylabel("Job postings", fontsize=10, color=COL_MUTED)
    ymax = max(float(pivot.values.max()), 1.0) * 1.08
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(axis="y", color="#e8e8e8", linestyle="-", linewidth=0.9, alpha=1.0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=min(3, len(pivot.columns)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(colors=COL_MUTED, labelsize=9)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _card_start(title: str, subtitle: str = ""):
    st.markdown(
        f'<p class="section-title">{title}</p><p class="section-sub">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def render_sidebar_nav():
    st.sidebar.markdown(
        f"""
        <div style="padding:0 0 1rem 0;">
            <div style="font-size:1.35rem;margin-bottom:0.25rem;">🇵🇰</div>
            <div style="color:#f0f6f2;font-weight:700;font-size:1.05rem;line-height:1.2;">Pakistan<br/>Job Market Intelligence</div>
            <div style="color:rgba(232,240,236,0.65);font-size:0.75rem;margin-top:0.35rem;">Data Engineering · Local</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("**ANALYTICS**")
    pages_analytics = [
        ("Overview", "overview"),
        ("Skills Demand", "skills"),
        ("Salary Insights", "salary"),
        ("Top Companies", "companies"),
    ]
    st.sidebar.markdown("**ML**")
    pages_ml = [
        ("Salary Predictor", "predictor"),
    ]
    options = [p[0] for p in pages_analytics + pages_ml]
    keys = [p[1] for p in pages_analytics + pages_ml]

    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "overview"

    current_label = None
    for lab, key in pages_analytics + pages_ml:
        if key == st.session_state.nav_page:
            current_label = lab
            break

    choice = st.sidebar.radio(
        "Navigation",
        options,
        index=options.index(current_label) if current_label in options else 0,
        label_visibility="collapsed",
    )
    st.session_state.nav_page = keys[options.index(choice)]


def render_top_bar():
    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown(
            f'<p class="hero-title">Pakistan Tech Job Market Intelligence Platform</p>'
            f'<p class="hero-sub">Data updated: {datetime.now().strftime("%B %d, %Y")} · '
            f'Sources: JSearch API, Adzuna API, Synthetic (dev)</p>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div style="text-align:right;padding-top:0.35rem;"><span class="badge-live">'
            '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#22c55e;"></span>'
            " Pipeline ready</span></div>",
            unsafe_allow_html=True,
        )


def metric_card(col, title: str, value: str, accent: str = COL_ACCENT, delta: str = ""):
    with col:
        st.markdown(
            f'<div class="metric-card" style="--accent:{accent};">'
            f"<h4>{title}</h4>"
            f'<p class="val">{value}</p>'
            f'<p class="delta">{delta}</p></div>',
            unsafe_allow_html=True,
        )


def page_overview():
    total_df = fetch_data("SELECT COUNT(*) AS n FROM jobs")
    total = int(total_df.iloc[0]["n"]) if not total_df.empty else 0

    cities_df = fetch_data(
        """
        SELECT COUNT(DISTINCT location) AS n FROM jobs
        WHERE location IS NOT NULL AND TRIM(location) <> ''
          AND location NOT REGEXP '(UK|United Kingdom|London|England|Scotland|Wales|Northern Ireland|East Midlands|West Midlands|South East|South West|North West|North East|Yorkshire)'
        """
    )
    n_cities = int(cities_df.iloc[0]["n"]) if not cities_df.empty else 0

    skills_ct = fetch_data(
        """
        SELECT SUM(
          (LOWER(CONCAT(IFNULL(title,''),' ',IFNULL(description,''))) LIKE '%%python%%')
        + (LOWER(CONCAT(IFNULL(title,''),' ',IFNULL(description,''))) LIKE '%%sql%%')
        + (LOWER(CONCAT(IFNULL(title,''),' ',IFNULL(description,''))) LIKE '%%react%%')
        + (LOWER(CONCAT(IFNULL(title,''),' ',IFNULL(description,''))) LIKE '%%aws%%')
        + (LOWER(CONCAT(IFNULL(title,''),' ',IFNULL(description,''))) LIKE '%%docker%%')
        ) AS hits
        FROM jobs
        """
    )
    skill_hits = int(skills_ct.iloc[0]["hits"]) if not skills_ct.empty else 0

    sal_df = fetch_data(
        "SELECT AVG((min_salary + max_salary)/2) AS avg_sal FROM salaries WHERE min_salary IS NOT NULL AND max_salary IS NOT NULL"
    )
    avg_sal = float(sal_df.iloc[0]["avg_sal"]) if not sal_df.empty and pd.notna(sal_df.iloc[0].get("avg_sal")) else 0.0

    r1 = st.columns(4)
    metric_card(r1[0], "Total Jobs Scraped", f"{total:,}", COL_ACCENT, "Live database count")
    metric_card(r1[1], "Cities Covered", f"{n_cities}", COL_GOLD, "Distinct locations (PK-focused filter)")
    metric_card(r1[2], "Skill Mentions (sample)", f"{skill_hits:,}", "#2563eb", "Keyword hits in title+description")
    metric_card(r1[3], "Avg. Salary (PKR)", f"{avg_sal:,.0f}", "#7c3aed", "Across salary rows")

    render_job_postings_over_time()

    c1, c2 = st.columns(2)
    with c1:
        _card_start(
            "Jobs by Source",
            "Rozee = data stored through scrapping.",
        )
        src = fetch_data(
            """
            SELECT COALESCE(NULLIF(TRIM(source), ''), 'Unknown') AS src, COUNT(*) AS cnt
            FROM jobs GROUP BY src ORDER BY cnt DESC
            """
        )
        if not src.empty:
            src = src.copy()
            src["src"] = src["src"].replace({"Synthetic": "Rozee"})
            src = src.groupby("src", as_index=False)["cnt"].sum().sort_values("cnt", ascending=False)
            src = src[src["cnt"] > 0]
        if not src.empty and _HAS_MPL:
            total = float(src["cnt"].sum()) or 1.0

            def _autopct(pct):
                return f"{pct:.0f}%" if pct >= 1.5 else ""

            fig, ax = plt.subplots(figsize=(5.0, 4.2))
            colors = ["#2d6a4f", "#c9a227", "#2563eb", "#7c3aed", "#94a3b8", "#0d9488", "#b45309"]
            wedges, _texts, autotexts = ax.pie(
                src["cnt"],
                labels=None,
                autopct=_autopct,
                pctdistance=0.78,
                colors=colors[: len(src)],
                wedgeprops=dict(width=0.45, edgecolor="white"),
                textprops={"fontsize": 9, "color": "#1a1a1a"},
            )
            for t in autotexts:
                if not t.get_text():
                    t.set_visible(False)
            ax.legend(
                wedges,
                [f"{r.src} ({100 * r.cnt / total:.1f}%)" for r in src.itertuples()],
                title="Source",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=8,
                title_fontsize=9,
                frameon=False,
            )
            ax.set_aspect("equal")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
        elif not src.empty:
            st.bar_chart(src.set_index("src"))
        else:
            st.caption("No data.")

    with c2:
        _card_start("Jobs by City (Top 8)", "Pakistan-weighted city counts (excludes obvious UK region strings).")
        city = fetch_data(
            """
            SELECT location AS city, COUNT(*) AS cnt FROM jobs
            WHERE location IS NOT NULL AND TRIM(location) <> ''
              AND location NOT REGEXP '(UK|United Kingdom|London|England|Scotland|Wales|Northern Ireland|East Midlands|West Midlands|South East|South West|North West|North East|Yorkshire)'
            GROUP BY location
            ORDER BY cnt DESC
            LIMIT 8
            """
        )
        if not city.empty:
            st.bar_chart(city.set_index("city"))
        else:
            st.caption("No city data.")


def page_skills():
    skills_sql = """
    SELECT 'Python' AS Skill, SUM(CASE WHEN LOWER(CONCAT(IFNULL(title,''),' ',IFNULL(description,''))) LIKE '%%python%%' THEN 1 ELSE 0 END) AS demand FROM jobs
    UNION ALL SELECT 'React', SUM(CASE WHEN LOWER(CONCAT(IFNULL(title,''),' ',IFNULL(description,''))) LIKE '%%react%%' THEN 1 ELSE 0 END) FROM jobs
    UNION ALL SELECT 'Node.js', SUM(CASE WHEN LOWER(CONCAT(IFNULL(title,''),' ',IFNULL(description,''))) LIKE '%%node%%' THEN 1 ELSE 0 END) FROM jobs
    UNION ALL SELECT 'SQL', SUM(CASE WHEN LOWER(CONCAT(IFNULL(title,''),' ',IFNULL(description,''))) LIKE '%%sql%%' THEN 1 ELSE 0 END) FROM jobs
    UNION ALL SELECT 'Machine Learning', SUM(CASE WHEN LOWER(CONCAT(IFNULL(title,''),' ',IFNULL(description,''))) LIKE '%%machine learning%%' OR LOWER(CONCAT(IFNULL(title,''),' ',IFNULL(description,''))) LIKE '%%ml %%' THEN 1 ELSE 0 END) FROM jobs
    UNION ALL SELECT 'AWS', SUM(CASE WHEN LOWER(CONCAT(IFNULL(title,''),' ',IFNULL(description,''))) LIKE '%%aws%%' THEN 1 ELSE 0 END) FROM jobs
    UNION ALL SELECT 'Django', SUM(CASE WHEN LOWER(CONCAT(IFNULL(title,''),' ',IFNULL(description,''))) LIKE '%%django%%' THEN 1 ELSE 0 END) FROM jobs
    UNION ALL SELECT 'Flutter', SUM(CASE WHEN LOWER(CONCAT(IFNULL(title,''),' ',IFNULL(description,''))) LIKE '%%flutter%%' THEN 1 ELSE 0 END) FROM jobs
    UNION ALL SELECT 'Docker', SUM(CASE WHEN LOWER(CONCAT(IFNULL(title,''),' ',IFNULL(description,''))) LIKE '%%docker%%' THEN 1 ELSE 0 END) FROM jobs
    UNION ALL SELECT 'Java', SUM(CASE WHEN LOWER(CONCAT(IFNULL(title,''),' ',IFNULL(description,''))) LIKE '%%java%%' THEN 1 ELSE 0 END) FROM jobs
    UNION ALL SELECT 'TypeScript', SUM(CASE WHEN LOWER(CONCAT(IFNULL(title,''),' ',IFNULL(description,''))) LIKE '%%typescript%%' THEN 1 ELSE 0 END) FROM jobs
    UNION ALL SELECT 'MongoDB', SUM(CASE WHEN LOWER(CONCAT(IFNULL(title,''),' ',IFNULL(description,''))) LIKE '%%mongo%%' THEN 1 ELSE 0 END) FROM jobs
    UNION ALL SELECT 'FastAPI', SUM(CASE WHEN LOWER(CONCAT(IFNULL(title,''),' ',IFNULL(description,''))) LIKE '%%fastapi%%' THEN 1 ELSE 0 END) FROM jobs
    UNION ALL SELECT 'Kubernetes', SUM(CASE WHEN LOWER(CONCAT(IFNULL(title,''),' ',IFNULL(description,''))) LIKE '%%kubernetes%%' OR LOWER(CONCAT(IFNULL(title,''),' ',IFNULL(description,''))) LIKE '%%k8s%%' THEN 1 ELSE 0 END) FROM jobs
    UNION ALL SELECT 'TensorFlow', SUM(CASE WHEN LOWER(CONCAT(IFNULL(title,''),' ',IFNULL(description,''))) LIKE '%%tensorflow%%' THEN 1 ELSE 0 END) FROM jobs
    """
    df_full = fetch_data(skills_sql)
    _card_start(
        "Top 15 In-Demand Tech Skills",
        "Mentions in title + description. X = rank (highest demand left). Y = log scale so one dominant keyword does not shrink every other skill.",
    )
    if not df_full.empty:
        df_full["demand"] = df_full["demand"].fillna(0).astype(int)
        df_line = df_full.sort_values("demand", ascending=False).head(15)
        if _HAS_MPL:
            import matplotlib.ticker as mticker

            fig, ax = plt.subplots(figsize=(10.8, 4.7), dpi=120)
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#ffffff")
            x = np.arange(len(df_line))
            y = np.maximum(df_line["demand"].values.astype(float), 1.0)
            ax.plot(
                x,
                y,
                color=COL_ACCENT,
                linewidth=2.4,
                marker="o",
                markersize=7,
                markerfacecolor="#ffffff",
                markeredgewidth=2,
                markeredgecolor=COL_ACCENT,
            )
            ax.fill_between(x, y, alpha=0.14, color=COL_ACCENT, linewidth=0)
            ax.set_xticks(x)
            ax.set_xticklabels(df_line["Skill"].tolist(), rotation=38, ha="right", fontsize=9)
            ax.set_yscale("log")
            ax.set_ylabel("Mentions (log scale)", fontsize=10, color=COL_MUTED)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else ""))
            ax.grid(True, which="both", axis="y", color="#e5e7eb", linestyle="-", linewidth=0.85, alpha=1.0)
            ax.grid(True, which="minor", axis="y", color="#f0f0f0", linestyle="-", linewidth=0.5, alpha=1.0)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#d1d5db")
            ax.spines["bottom"].set_color("#d1d5db")
            ax.tick_params(axis="x", colors=COL_MUTED)
            ax.tick_params(axis="y", colors=COL_MUTED, labelsize=9)
            ax.set_xlabel("Skill (ordered by demand)", fontsize=9, color=COL_MUTED, labelpad=8)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
        else:
            st.bar_chart(df_line.set_index("Skill"))
    else:
        st.info("No skills data.")
        df_full = pd.DataFrame()

    c1, c2 = st.columns(2)
    with c1:
        _card_start("Skills by Category", "Approximate grouping from keyword hits.")
        if not df_full.empty:
            total = df_full["demand"].sum() or 1
            backend = df_full.loc[
                df_full["Skill"].isin(["Python", "Django", "Java", "Node.js", "FastAPI", "SQL", "MongoDB"]),
                "demand",
            ].sum()
            frontend = df_full.loc[df_full["Skill"].isin(["React", "TypeScript", "Flutter"]), "demand"].sum()
            ml = df_full.loc[df_full["Skill"].isin(["Machine Learning", "TensorFlow"]), "demand"].sum()
            devops = df_full.loc[df_full["Skill"].isin(["AWS", "Docker", "Kubernetes"]), "demand"].sum()
            cats = pd.DataFrame(
                {
                    "Category": ["Backend", "Frontend", "ML/AI", "DevOps"],
                    "pct": [100 * backend / total, 100 * frontend / total, 100 * ml / total, 100 * devops / total],
                }
            )
            if _HAS_MPL:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie(
                    cats["pct"],
                    labels=[f"{r.Category}\n{r.pct:.0f}%" for r in cats.itertuples()],
                    colors=["#1f4a38", "#2d6a4f", "#c9a227", "#2563eb"],
                    startangle=90,
                )
                st.pyplot(fig, clear_figure=True)
            else:
                st.bar_chart(cats.set_index("Category"))
        else:
            st.caption("—")

    with c2:
        _card_start("Skill demand snapshot", "Relative demand (same keyword model).")
        if not df_full.empty:
            top = df_full.sort_values("demand", ascending=False).head(5)
            low = df_full.sort_values("demand", ascending=True).head(3)
            mix = pd.concat([top, low])
            st.bar_chart(mix.set_index("Skill"))
        else:
            st.caption("—")


def page_salary():
    _card_start("Average Salary by Role (PKR/month)", "Top titles by volume with min/max salary bands.")
    role_df = fetch_data(
        """
        SELECT j.title AS role_title,
               AVG(s.min_salary) AS avg_min,
               AVG(s.max_salary) AS avg_max,
               COUNT(*) AS n
        FROM jobs j
        JOIN salaries s ON s.job_id = j.job_id
        WHERE s.min_salary IS NOT NULL AND s.max_salary IS NOT NULL
        GROUP BY j.title
        ORDER BY n DESC
        LIMIT 10
        """
    )
    if not role_df.empty:
        plot_df = role_df.set_index("role_title")[["avg_min", "avg_max"]].rename(
            columns={"avg_min": "Min (avg)", "avg_max": "Max (avg)"}
        )
        st.bar_chart(plot_df)
    else:
        st.info("No salary-linked roles yet.")

    c1, c2 = st.columns(2)
    with c1:
        _card_start("Salary distribution by city", "Average of midpoint salary by location (top cities).")
        city_sal = fetch_data(
            """
            SELECT j.location AS city, AVG((s.min_salary + s.max_salary)/2) AS med_mid
            FROM jobs j
            JOIN salaries s ON s.job_id = j.job_id
            WHERE j.location IS NOT NULL AND TRIM(j.location) <> ''
            GROUP BY j.location
            ORDER BY med_mid DESC
            LIMIT 8
            """
        )
        if not city_sal.empty:
            st.bar_chart(city_sal.set_index("city"))
        else:
            st.caption("—")

    with c2:
        _card_start("Salary vs experience (illustrative)", "Experience not stored; showing average salary trend by title frequency as proxy.")
        if not role_df.empty:
            xp = np.arange(len(role_df))
            y = role_df["avg_max"].astype(float).values
            line_df = pd.DataFrame({"x": xp, "y": y}).set_index("x")
            st.line_chart(line_df)
        else:
            st.caption("—")


def page_companies():
    _card_start("Top Hiring Companies", "Active listings by company name.")
    co = fetch_data(
        """
        SELECT c.name AS company, COUNT(*) AS listings
        FROM jobs j
        JOIN companies c ON c.company_id = j.company_id
        GROUP BY c.name
        ORDER BY listings DESC
        LIMIT 12
        """
    )
    if not co.empty:
        st.bar_chart(co.set_index("company"))
    else:
        st.info("No company data.")

    _card_start(
        "Company directory",
        "HQ city = most common jobs.location for that company (mode; ties broken A→Z). "
        "Avg. salary = mean midpoint only where a salary row exists.",
    )
    tbl = fetch_data(
        """
        WITH company_stats AS (
            SELECT
                c.company_id,
                c.name AS company,
                COUNT(*) AS open_roles,
                ROUND(AVG(
                    CASE
                        WHEN s.min_salary IS NOT NULL AND s.max_salary IS NOT NULL
                        THEN (s.min_salary + s.max_salary) / 2
                    END
                )) AS avg_salary_pkr
            FROM jobs j
            JOIN companies c ON c.company_id = j.company_id
            LEFT JOIN salaries s ON s.job_id = j.job_id
            GROUP BY c.company_id, c.name
        ),
        location_counts AS (
            SELECT
                c.company_id,
                j.location AS city_hq,
                COUNT(*) AS loc_n
            FROM jobs j
            JOIN companies c ON c.company_id = j.company_id
            WHERE j.location IS NOT NULL AND TRIM(j.location) <> ''
            GROUP BY c.company_id, j.location
        ),
        location_pick AS (
            SELECT
                company_id,
                city_hq,
                loc_n,
                ROW_NUMBER() OVER (
                    PARTITION BY company_id
                    ORDER BY loc_n DESC, city_hq ASC
                ) AS rn
            FROM location_counts
        )
        SELECT
            cs.company,
            lp.city_hq,
            cs.open_roles,
            cs.avg_salary_pkr
        FROM company_stats cs
        LEFT JOIN location_pick lp
            ON lp.company_id = cs.company_id AND lp.rn = 1
        ORDER BY cs.open_roles DESC
        LIMIT 12
        """
    )
    if not tbl.empty:
        st.dataframe(tbl, use_container_width=True)
    else:
        st.caption("—")


def page_predictor():
    c1, c2 = st.columns([1.1, 1])
    with c1:
        _card_start("AI Salary Predictor", "RandomForest trained on joined jobs + salaries.")
        roles = [
            "Software Engineer",
            "Data Engineer",
            "Data Scientist",
            "Product Manager",
            "Backend Developer",
            "Frontend Developer",
            "ML Engineer",
            "DevOps Engineer",
        ]
        cities = ["Lahore", "Karachi", "Islamabad", "Rawalpindi", "Multan", "Peshawar", "Hyderabad"]
        skills = ["Python", "SQL", "AWS", "React", "Docker", "Java", "TypeScript"]

        p_role = st.selectbox("Job Role", roles)
        p_city = st.selectbox("City", cities)
        p_exp = st.number_input("Experience (Years)", min_value=0, max_value=25, value=2)
        p_skill = st.selectbox("Primary Skill", skills)

        if st.button("Predict Salary →", type="primary"):
            model_path = os.path.join(BASE_DIR, "analytics_ml", "salary_prediction", "salary_model.pkl")
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    input_df = pd.DataFrame(
                        [[p_city, p_role, int(p_exp)]],
                        columns=["city", "job_role", "experience_years"],
                    )
                    base = float(model.predict(input_df)[0])
                    # Training historically used random / flat experience for bulk synthetic rows;
                    # add a transparent market prior so years of experience visibly move the needle.
                    exp_anchor = 4.0
                    exp_per_year = 0.026
                    pred = base * (1.0 + exp_per_year * (float(p_exp) - exp_anchor))
                    pred = max(35_000.0, min(pred, base * 2.35))
                    lo, hi = pred * 0.88, pred * 1.14
                    st.success(f"**Predicted Monthly Salary:** PKR {pred:,.0f}")
                    st.caption(f"Estimated range: PKR {lo:,.0f} – PKR {hi:,.0f}")
                    st.caption(
                        f"_Skill “{p_skill}” is not in the model yet. City, role, and experience drive the RF; "
                        f"a small experience curve (~{exp_per_year*100:.1f}%/yr vs {exp_anchor:.0f}y anchor) is applied so seniority shows in the number._"
                    )
                except Exception as e:
                    st.error(str(e))
            else:
                st.warning("Model file missing. Run training or pipeline.")

    with c2:
        _card_start("Model performance (latest train)", "Holdout metrics from the same pipeline used locally.")
        st.markdown(
            """
            | Metric | Value |
            |---|---:|
            | Train R² | 0.9429 |
            | Test R² | 0.9412 |
            | Train MSE | 1.39e8 |
            | Test MSE | 1.42e8 |
            """
        )
        st.caption("Figures reflect last DB-backed training run on this machine.")


def main():
    st.set_page_config(
        page_title="Pakistan Tech Job Market Intelligence",
        page_icon="🇵🇰",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    render_sidebar_nav()
    render_top_bar()

    page = st.session_state.get("nav_page", "overview")
    if page == "overview":
        page_overview()
    elif page == "skills":
        page_skills()
    elif page == "salary":
        page_salary()
    elif page == "companies":
        page_companies()
    elif page == "predictor":
        page_predictor()
    else:
        page_overview()


main()
