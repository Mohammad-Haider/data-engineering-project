"""Remove non-production ``jobs.source`` placeholders from EDA extracts."""

from __future__ import annotations

from typing import Tuple

import pandas as pd

# Dev / seed / legacy labels that should not appear in analytics charts
_PLACEHOLDER_SOURCES_LOWER = frozenset(
    {
        "seed",
        "rozee.pk (fallback)",
    }
)


def drop_placeholder_sources(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Drop rows whose ``source`` is a known placeholder (case-insensitive).

    Returns ``(filtered_df, n_dropped)``.
    """
    if df.empty or "source" not in df.columns:
        return df, 0
    s = df["source"].fillna("").astype(str).str.strip().str.lower()
    mask = ~s.isin(_PLACEHOLDER_SOURCES_LOWER)
    n_dropped = int((~mask).sum())
    return df.loc[mask].copy(), n_dropped
