"""Clean and subset the merged NHAMCS ED dataframe for downstream analysis."""

from __future__ import annotations

import pandas as pd

NONNEG_FILTER_COLS = ["TOTPROC", "TOTCHRON", "TOTDIAG", "PAINSCALE"]

SELECTED_FEATURES = [
    "LOV",
    "WAITTIME",
    "IMMEDR",
    "AGE",
    "PAINSCALE",
    "TOTCHRON",
    "TOTDIAG",
]


def drop_negative_sentinels(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Remove rows where any of *cols* is negative (NHAMCS uses negatives as missing sentinels)."""
    cols = cols or NONNEG_FILTER_COLS
    mask = pd.Series(True, index=df.index)
    for col in cols:
        if col in df.columns:
            mask &= df[col] >= 0
    return df.loc[mask].reset_index(drop=True)


def filter_nonnegative(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where TOTPROC, TOTCHRON, TOTDIAG, and PAINSCALE are all >= 0."""
    df = df[
        (df["TOTPROC"] >= 0)
        & (df["TOTCHRON"] >= 0)
        & (df["TOTDIAG"] >= 0)
        & (df["PAINSCALE"] >= 0)
    ].reset_index(drop=True)
    return df


def select_features(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Keep only the requested feature columns."""
    features = features or SELECTED_FEATURES
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in dataframe: {missing}")
    return df[features].copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: filter negative sentinels then select features."""
    print(f"Before cleaning: {df.shape}")
    df = filter_nonnegative(df)
    df = select_features(df)
    print(f"After cleaning:  {df.shape}")
    return df
