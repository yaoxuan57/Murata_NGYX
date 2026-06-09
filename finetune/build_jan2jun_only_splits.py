"""Build train/val/test from Jan–Jun 30 min CSV only (70/15/15 chrono)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "finetune" / "data_AHU_2_9_Blower_DE_A"
OUT_DIR = DATA_DIR / "splits"
SOURCE = DATA_DIR / "AHU_2_9_Blower_DE_A_jan2_jun_30_min.csv"
TIME_COL, VALUE_COL, SENSOR_COL = "TIMESTAMP", "Acceleration RMS", "SENSOR_DESC"
SENSOR = "AHU 2-9 Blower DE A"
TR, VA, TE = 0.70, 0.15, 0.15


def parse_ts(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.strip()
    p = pd.to_datetime(raw, dayfirst=True, format="mixed", errors="coerce")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        m = p.isna()
        if not m.any():
            break
        p.loc[m] = pd.to_datetime(raw.loc[m], format=fmt, errors="coerce")
    return p


def main() -> None:
    df = pd.read_csv(SOURCE, low_memory=False)
    if VALUE_COL not in df.columns and "DATA12" in df.columns:
        df[VALUE_COL] = pd.to_numeric(df["DATA12"], errors="coerce")
    if SENSOR_COL not in df.columns and "SENSOR_NAME" in df.columns:
        df[SENSOR_COL] = df["SENSOR_NAME"].astype(str).str.strip()
    df[TIME_COL] = parse_ts(df[TIME_COL])
    df = df[df[SENSOR_COL].astype(str).str.strip() == SENSOR]
    df = df.sort_values(TIME_COL).drop_duplicates(TIME_COL, keep="last").reset_index(drop=True)

    n = len(df)
    n_tr, n_va = int(np.floor(n * TR)), int(np.floor(n * VA))
    train, val, test = df.iloc[:n_tr], df.iloc[n_tr : n_tr + n_va], df.iloc[n_tr + n_va :]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    val.to_csv(OUT_DIR / "val.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)

    manifest = {
        "mode": "jan2jun_only_70_15_15",
        "source": str(SOURCE),
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "time": {
            "train": [str(train[TIME_COL].min()), str(train[TIME_COL].max())],
            "val": [str(val[TIME_COL].min()), str(val[TIME_COL].max())],
            "test": [str(test[TIME_COL].min()), str(test[TIME_COL].max())],
        },
    }
    (OUT_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
