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


def effective_timestamp_column(df: pd.DataFrame) -> str:
    """
    Pick the column that best resolves reading time.

    Exports like ``Jun.csv`` store date-only values in ``TIMESTAMP`` while
    ``TIMESTAMP_DISPLAY`` carries the true sub-day timestamp.
    """
    if "TIMESTAMP_DISPLAY" not in df.columns:
        return "TIMESTAMP"
    n = len(df)
    if n == 0:
        return "TIMESTAMP"
    n_unique_ts = df["TIMESTAMP"].astype(str).nunique()
    if n_unique_ts < max(n // 2, 2):
        return "TIMESTAMP_DISPLAY"
    return "TIMESTAMP"


def parse_dataframe_timestamps(df: pd.DataFrame, *, strict: bool = False) -> pd.Series:
    """Parse the best available timestamp column for a vibration export frame."""
    col = effective_timestamp_column(df)
    return parse_timestamp_series(df[col], name=col, strict=strict)


def prepare_vibration_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize columns for inference / preprocessing."""
    required = {"SENSOR_DESC"}
    if "TIMESTAMP" not in df.columns and "TIMESTAMP_DISPLAY" not in df.columns:
        required.add("TIMESTAMP")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")
    out = df.copy()
    if "STN_CODE" in out.columns:
        out["STN_CODE"] = out["STN_CODE"].astype(str)
    out["SENSOR_DESC"] = out["SENSOR_DESC"].astype(str).str.strip()
    ts_col = effective_timestamp_column(out)
    out["_EFFECTIVE_TIMESTAMP"] = parse_dataframe_timestamps(out, strict=False)
    out["TIMESTAMP"] = out["_EFFECTIVE_TIMESTAMP"]
    return out


def read_vibration_export_csv(path: str) -> pd.DataFrame:
    """Load a multi-sensor vibration export CSV."""
    return prepare_vibration_dataframe(pd.read_csv(path, low_memory=False))
