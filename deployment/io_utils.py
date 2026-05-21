"""CSV I/O helpers for deployment preprocessing."""

from __future__ import annotations

import pandas as pd


def parse_timestamp_series(
    series: pd.Series,
    name: str = "TIMESTAMP",
    *,
    strict: bool = True,
) -> pd.Series:
    """Parse TIMESTAMP strings (day-first / mixed formats)."""
    raw = series.astype(str).str.strip()
    parsed = pd.to_datetime(raw, dayfirst=True, format="mixed", errors="coerce")

    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(
            raw.loc[mask],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )

    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(
            raw.loc[mask],
            format="%Y-%m-%d %H:%M",
            errors="coerce",
        )

    n_bad = int(parsed.isna().sum())
    if strict and n_bad:
        raise ValueError(f"{name}: failed to parse {n_bad} timestamp(s).")
    return parsed


def read_vibration_export_csv(path: str) -> pd.DataFrame:
    """Load a multi-sensor vibration export CSV."""
    df = pd.read_csv(path, low_memory=False)
    required = {"TIMESTAMP", "SENSOR_DESC"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    if "STN_CODE" in df.columns:
        df["STN_CODE"] = df["STN_CODE"].astype(str)
    df["SENSOR_DESC"] = df["SENSOR_DESC"].astype(str).str.strip()
    return df
