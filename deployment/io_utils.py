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
    parsed = pd.to_datetime(raw, dayfirst=False, format="mixed", errors="coerce")

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


def prepare_vibration_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize columns for inference / preprocessing."""
    out = df.copy()
    if "SENSOR_DESC" not in out.columns and "SENSOR_NAME" in out.columns:
        out["SENSOR_DESC"] = out["SENSOR_NAME"]
    if "Acceleration RMS" not in out.columns and "DATA12" in out.columns:
        out["Acceleration RMS"] = pd.to_numeric(out["DATA12"], errors="coerce")

    required = {"TIMESTAMP", "SENSOR_DESC"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")
    if "STN_CODE" in out.columns:
        out["STN_CODE"] = out["STN_CODE"].astype("string").str.strip()
    if "SENSOR_CODE" in out.columns:
        out["SENSOR_CODE"] = out["SENSOR_CODE"].astype("string").str.strip()
    out["SENSOR_DESC"] = out["SENSOR_DESC"].astype(str).str.strip()
    return out


def read_vibration_export_csv(path: str) -> pd.DataFrame:
    """Load a multi-sensor vibration export CSV."""
    # Preserve hex-like IDs (e.g. 91B8) as text. Registry-aware matching also
    # handles files where Excel already converted 91E2 to 9100.
    return prepare_vibration_dataframe(
        pd.read_csv(
            path,
            low_memory=False,
            dtype={"SENSOR_CODE": "string", "STN_CODE": "string"},
        )
    )
