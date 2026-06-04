"""CSV I/O helpers."""

from __future__ import annotations

import pandas as pd


def parse_timestamp_series(
    series: pd.Series,
    name: str = "TIMESTAMP",
    *,
    strict: bool = True,
) -> pd.Series:
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


def prepare_vibration_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = {"TIMESTAMP", "SENSOR_DESC"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")
    out = df.copy()
    if "STN_CODE" in out.columns:
        out["STN_CODE"] = out["STN_CODE"].astype(str)
    out["SENSOR_DESC"] = out["SENSOR_DESC"].astype(str).str.strip()
    return out


def read_vibration_export_csv(path: str) -> pd.DataFrame:
    return prepare_vibration_dataframe(pd.read_csv(path, low_memory=False))
