"""
Segment filters for EDA: IT-related roles and (Pakistan geography OR remote / WFH).

Matching is regex-based on title, description, and location fields (case-insensitive).
"""

from __future__ import annotations

import re
from typing import Tuple

import pandas as pd

# Broad IT / tech role signals (title or description)
_IT_PATTERN = re.compile(
    r"(?is)\b("
    r"software|developer|development|programming|programmer|"
    r"engineer|engineering|devops|sre\b|sysadmin|system\s*admin|network\s*admin|"
    r"data\s*scientist|data\s*engineer|ml\s|machine\s*learning|ai\b|deep\s*learning|"
    r"cloud|aws|azure|gcp|kubernetes|k8s|docker|terraform|ansible|"
    r"full\s*stack|fullstack|backend|back-end|front\s*end|front-end|frontend|"
    r"web\s*developer|mobile\s*developer|ios|android|flutter|react|angular|vue|node\.?js|"
    r"python|java\b|c\+\+|\.net|php|ruby|golang|go\s*lang|rust\b|scala\b|kotlin|swift|"
    r"cyber\s*security|infosec|penetration|security\s*analyst|"
    r"database|dba\b|sql\b|nosql|mongodb|postgres|mysql|oracle|"
    r"qa\b|quality\s*assurance|test\s*automation|selenium|cypress|"
    r"business\s*intelligence|\bbi\b|power\s*bi|tableau|looker|"
    r"product\s*manager|technical\s*product|scrum\s*master|agile\s*coach|"
    r"it\s|ict\b|information\s*technology|technology\s*stack|tech\s*stack|"
    r"blockchain|crypto\s*dev|embedded|firmware|hardware\s*engineer|"
    r"ui\s*/?\s*ux|ux\s*designer|ui\s*designer|graphic\s*designer\s*\(?tech|"
    r"salesforce|servicenow|sap\b|oracle\s*apps|"
    r"support\s*engineer|technical\s*support|helpdesk|help\s*desk|l[123]\s*support|"
    r"architect\b|solution\s*architect|enterprise\s*architect|"
    r"site\s*reliability|platform\s*engineer|release\s*engineer|"
    r"analytics\s*engineer|research\s*scientist\s*\(?ml"
    r")\b"
)

# Pakistan geography (location or description)
_PAKISTAN_GEO_PATTERN = re.compile(
    r"(?is)\b("
    r"pakistan|\bp\.?k\.?\b|pakistani|"
    r"karachi|lahore|islamabad|rawalpindi|pindi\b|"
    r"peshawar|faisalabad|multan|hyderabad\s*\(?pak|sialkot|gujranwala|"
    r"quetta|abbottabad|bahawalpur|sukkur|mirpur|ajk\b|gwadar|"
    r"islamabad\s*capital|capital\s*territory"
    r")\b"
)

# Remote / hybrid-remote / work-from-anywhere (location or description)
_REMOTE_PATTERN = re.compile(
    r"(?is)\b("
    r"remote\b|fully\s*remote|100%\s*remote|remote-first|remote\s*first|"
    r"work\s*from\s*home|wfh\b|work-from-home|"
    r"virtual\s*office|virtual\s*position|distributed\s*team|anywhere\s*in|"
    r"location\s*flexible|flexible\s*location|open\s*to\s*remote|"
    r"hybrid\s*[-–/]?\s*remote|remote\s*[-–/]?\s*hybrid|hybrid.{0,40}?remote|remote.{0,40}?hybrid|"
    r"globally\s*remote|worldwide\s*remote|international\s*remote|"
    r"remote\s*\(|remote\s*\-|remote\s*:|remote\s*/|"
    r"based\s*anywhere|work\s*anywhere|from\s*anywhere"
    r")\b"
)


def it_pakistan_remote_mask(df: pd.DataFrame) -> pd.Series:
    """
    True where row is IT-like AND (Pakistan geography OR remote/WFH signals).

    Parameters
    ----------
    df : DataFrame
        Must include columns ``title``, ``description``, ``location`` (nullable).
    """
    return it_mask(df) & (pakistan_geo_mask(df) | remote_mask(df))


def filter_it_pakistan_remote(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """Return filtered copy, (n_before, n_after)."""
    n_before = len(df)
    m = it_pakistan_remote_mask(df)
    out = df.loc[m].copy()
    return out, n_before, len(out)


def filter_description() -> str:
    return (
        "IT-related: title or description matches technology / engineering role patterns. "
        "Geography: location or description mentions Pakistan / major PK cities OR "
        "remote / WFH / hybrid-remote / distributed-team style wording."
    )


def pakistan_geo_mask(df: pd.DataFrame) -> pd.Series:
    title = df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
    desc = df.get("description", pd.Series("", index=df.index)).fillna("").astype(str)
    loc = df.get("location", pd.Series("", index=df.index)).fillna("").astype(str)
    return (
        loc.map(lambda s: bool(_PAKISTAN_GEO_PATTERN.search(s)))
        | desc.map(lambda s: bool(_PAKISTAN_GEO_PATTERN.search(s)))
        | title.map(lambda s: bool(_PAKISTAN_GEO_PATTERN.search(s)))
    )


def remote_mask(df: pd.DataFrame) -> pd.Series:
    title = df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
    desc = df.get("description", pd.Series("", index=df.index)).fillna("").astype(str)
    loc = df.get("location", pd.Series("", index=df.index)).fillna("").astype(str)
    return (
        loc.map(lambda s: bool(_REMOTE_PATTERN.search(s)))
        | desc.map(lambda s: bool(_REMOTE_PATTERN.search(s)))
        | title.map(lambda s: bool(_REMOTE_PATTERN.search(s)))
    )


def it_mask(df: pd.DataFrame) -> pd.Series:
    title = df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
    desc = df.get("description", pd.Series("", index=df.index)).fillna("").astype(str)
    return title.map(lambda s: bool(_IT_PATTERN.search(s))) | desc.map(
        lambda s: bool(_IT_PATTERN.search(s))
    )
