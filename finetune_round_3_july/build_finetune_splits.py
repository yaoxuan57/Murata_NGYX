"""Build finetune train/val/test: full historical + chronological June split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
FINETUNE_DIR = Path(__file__).resolve().parent
JUN_DIR = FINETUNE_DIR / "jun_by_sensor"
HIST_DIR = REPO_ROOT / "data_30mins_frequency"

TIME_COL = "TIMESTAMP"
VALUE_COL = "Acceleration RMS"
SENSOR_COL = "SENSOR_DESC"
SENSOR_NAME_COL = "SENSOR_NAME"


def stem_to_sensor_name(stem: str) -> str:
    text = stem.replace("2_9", "2-9").replace("4_4", "4-4").replace("_", " ")
    return " ".join(text.split())


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    parsed = pd.to_datetime(raw, dayfirst=True, format="mixed", errors="coerce")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        mask = parsed.isna()
        if not mask.any():
            break
        parsed.loc[mask] = pd.to_datetime(raw.loc[mask], format=fmt, errors="coerce")
    if int(parsed.isna().sum()):
        raise ValueError(f"Failed to parse {int(parsed.isna().sum())} timestamps.")
    return parsed


def load_and_prepare(path: Path, sensor_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if TIME_COL not in df.columns:
        raise ValueError(f"{path.name}: missing {TIME_COL}")
    if VALUE_COL not in df.columns and "DATA12" in df.columns:
        df = df.copy()
        df[VALUE_COL] = pd.to_numeric(df["DATA12"], errors="coerce")
    if VALUE_COL not in df.columns:
        raise ValueError(f"{path.name}: missing {VALUE_COL}")

    df = df.copy()
    df[TIME_COL] = parse_timestamp_series(df[TIME_COL])
    name_col = SENSOR_COL if SENSOR_COL in df.columns else SENSOR_NAME_COL
    if name_col in df.columns:
        df[name_col] = df[name_col].astype(str).str.strip()
        df = df[df[name_col] == sensor_name]
    df = df.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)
    return df.drop_duplicates(subset=[TIME_COL], keep="last")


def chrono_row_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rsum = float(train_ratio + val_ratio + test_ratio)
    if abs(rsum - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1, got {rsum}")
    n = len(df)
    n_tr = int(np.floor(n * train_ratio))
    n_va = int(np.floor(n * val_ratio))
    n_te = n - n_tr - n_va
    if min(n_tr, n_va, n_te) < 1:
        raise ValueError(f"Empty split: n={n}, train={n_tr}, val={n_va}, test={n_te}")
    return (
        df.iloc[:n_tr].copy(),
        df.iloc[n_tr : n_tr + n_va].copy(),
        df.iloc[n_tr + n_va :].copy(),
    )


def summarize_frame(name: str, df: pd.DataFrame) -> dict:
    ts = df[TIME_COL]
    rms = pd.to_numeric(df[VALUE_COL], errors="coerce")
    return {
        "name": name,
        "rows": int(len(df)),
        "t_min": str(ts.min()),
        "t_max": str(ts.max()),
        "rms_mean": float(rms.mean()),
        "rms_median": float(rms.median()),
    }


def resolve_historical_csv(stem: str, historical_csv: Path | None = None) -> Path:
    if historical_csv is not None:
        return historical_csv
    candidates = [
        HIST_DIR / f"{stem}_30_min.csv",
        HIST_DIR / f"{stem}_merged.csv",
        HIST_DIR / f"{stem}.csv",
    ]
    # On-disk AHU 4-4 files often use a hyphen (AHU_4-4_...). Prefer the full
    # historical export (same as finetune_round_1) before the short _30_min cut.
    hyphen_stem = stem.replace("4_4", "4-4").replace("2_9", "2-9")
    if hyphen_stem != stem:
        candidates.extend(
            [
                HIST_DIR / f"{hyphen_stem}.csv",
                HIST_DIR / f"{hyphen_stem}_merged.csv",
                HIST_DIR / f"{hyphen_stem}_30_min.csv",
            ]
        )
    for path in candidates:
        if path.is_file():
            return path
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"No historical CSV for stem {stem!r}. Tried: {tried}")


def build_splits(
    *,
    stem: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    sensor_name: str | None = None,
    historical_csv: Path | None = None,
    jun_csv: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    sensor_name = sensor_name or stem_to_sensor_name(stem)
    historical_csv = resolve_historical_csv(stem, historical_csv)
    jun_csv = jun_csv or (JUN_DIR / f"{stem}.csv")
    out_dir = out_dir or (FINETUNE_DIR / f"data_{stem}" / "splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_hist = load_and_prepare(historical_csv, sensor_name)
    df_jun = load_and_prepare(jun_csv, sensor_name)

    df_jun_train, df_jun_val, df_jun_test = chrono_row_split(
        df_jun, train_ratio, val_ratio, test_ratio
    )

    hist_cols = list(df_hist.columns)
    df_train = pd.concat(
        [df_hist, df_jun_train.reindex(columns=hist_cols)],
        ignore_index=True,
    ).sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)
    df_val = df_jun_val.reindex(columns=hist_cols).sort_values(TIME_COL).reset_index(drop=True)
    df_test = df_jun_test.reindex(columns=hist_cols).sort_values(TIME_COL).reset_index(drop=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    ratio_tag = f"{int(train_ratio * 100)}_{int(val_ratio * 100)}_{int(test_ratio * 100)}"
    manifest = {
        "mode": f"historical_full_plus_jun_chrono_{ratio_tag}",
        "sensor": sensor_name,
        "historical_csv": str(historical_csv.resolve()),
        "new_period_csv": str(jun_csv.resolve()),
        "use_full_historical_in_train": True,
        "new_period_split": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "outputs": {
            "train": str(train_path.resolve()),
            "val": str(val_path.resolve()),
            "test": str(test_path.resolve()),
        },
        "summary": {
            "train": summarize_frame("train", df_train),
            "val": summarize_frame("val", df_val),
            "test": summarize_frame("test", df_test),
            "hist": summarize_frame("hist", df_hist),
            "jun": summarize_frame("jun", df_jun),
        },
    }
    manifest_path = out_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote splits for {stem} ({ratio_tag}):")
    for split in ("train", "val", "test"):
        s = manifest["summary"][split]
        print(f"  {split}: {s['rows']:,} rows  {s['t_min']} -> {s['t_max']}")
    print(f"  manifest: {manifest_path}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", required=True, help="e.g. AHU_2_9_Blower_NDE_A")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--sensor-name", default=None)
    parser.add_argument("--historical-csv", type=Path, default=None)
    parser.add_argument("--jun-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    build_splits(
        stem=args.stem,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        sensor_name=args.sensor_name,
        historical_csv=args.historical_csv,
        jun_csv=args.jun_csv,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
