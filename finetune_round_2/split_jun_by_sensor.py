"""Split finetune/Jan2Jun.csv (or Jun.csv) into per-sensor CSVs under jun_by_sensor/."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

FINETUNE_DIR = Path(__file__).resolve().parent
TIME_COL = "TIMESTAMP"


def sensor_desc_to_stem(desc: str) -> str:
    slug = str(desc).strip().replace(" ", "_")
    slug = re.sub(r"[^\w\-]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug.replace("2-9", "2_9").replace("4-4", "4_4")


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


def downsample_pick_last_in_bin(df: pd.DataFrame) -> pd.DataFrame:
    work = df.sort_values(TIME_COL, kind="mergesort").copy()
    bins = work[TIME_COL].dt.floor("30min")
    pick_idx = work.groupby(bins, sort=True)[TIME_COL].idxmax()
    return df.loc[pick_idx].sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)


def split_by_sensor(
    src: Path,
    out_dir: Path,
    *,
    june_only: bool,
    downsample_30min: bool,
) -> list[dict]:
    raw = pd.read_csv(src, low_memory=False)
    if "SENSOR_DESC" in raw.columns:
        sensor_col = "SENSOR_DESC"
    elif "SENSOR_NAME" in raw.columns:
        raw = raw.copy()
        raw["SENSOR_DESC"] = raw["SENSOR_NAME"].astype(str).str.strip()
        sensor_col = "SENSOR_DESC"
    else:
        raise ValueError(f"{src}: need SENSOR_DESC or SENSOR_NAME")

    if "DATA12" in raw.columns and "Acceleration RMS" not in raw.columns:
        raw["Acceleration RMS"] = pd.to_numeric(raw["DATA12"], errors="coerce")

    raw[TIME_COL] = parse_timestamp_series(raw[TIME_COL])
    raw = raw.sort_values(TIME_COL, kind="mergesort")
    if june_only:
        raw = raw[raw[TIME_COL].dt.month == 6].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    for desc, grp in raw.groupby(sensor_col, sort=True):
        stem = sensor_desc_to_stem(desc)
        part = grp.drop_duplicates(subset=[TIME_COL], keep="last").reset_index(drop=True)
        n_before = len(part)
        if downsample_30min and n_before > 0:
            part = downsample_pick_last_in_bin(part)
        out_path = out_dir / f"{stem}.csv"
        part.to_csv(out_path, index=False)
        med = (
            part[TIME_COL].diff().dt.total_seconds().dropna().median()
            if len(part) > 1
            else None
        )
        summary.append(
            {
                "sensor": str(desc),
                "stem": stem,
                "rows_before_downsample": int(n_before),
                "rows": int(len(part)),
                "t_min": str(part[TIME_COL].min()) if len(part) else None,
                "t_max": str(part[TIME_COL].max()) if len(part) else None,
                "median_step_seconds": float(med) if med is not None and pd.notna(med) else None,
                "path": str(out_path.resolve()),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=FINETUNE_DIR / "Jun.csv",
        help="Multi-sensor export (default: finetune_round_2/Jun.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FINETUNE_DIR / "jun_by_sensor",
        help="Output folder (default: finetune/jun_by_sensor).",
    )
    parser.add_argument(
        "--june-only",
        action="store_true",
        help="Keep only June rows (default: full Jan–Jun from input file).",
    )
    parser.add_argument(
        "--downsample-30min",
        action="store_true",
        default=True,
        help="Pick last row per 30-min bin (default: on).",
    )
    parser.add_argument(
        "--no-downsample-30min",
        dest="downsample_30min",
        action="store_false",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    june_only = bool(args.june_only)
    summary = split_by_sensor(
        args.input,
        args.out_dir,
        june_only=june_only,
        downsample_30min=args.downsample_30min,
    )

    manifest = {
        "source": str(args.input.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "june_only": june_only,
        "downsample_30min": args.downsample_30min,
        "sensors": summary,
    }
    manifest_path = args.out_dir / "split_summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Split {args.input.name} -> {args.out_dir}/ ({len(summary)} sensors)")
    print(f"  june_only={june_only}  downsample_30min={args.downsample_30min}")
    for row in summary:
        med = row["median_step_seconds"]
        med_s = f"{med:.0f}s" if med is not None else "n/a"
        print(
            f"  {row['rows']:5,} rows ({row['rows_before_downsample']:,} raw)  "
            f"{row['stem']}.csv  median step {med_s}  {row['t_min']} -> {row['t_max']}"
        )
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
