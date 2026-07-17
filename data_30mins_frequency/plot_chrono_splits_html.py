#!/usr/bin/env python3
"""Plot Acceleration RMS timelines with chronological train/val/test splits (HTML).

Finds sensor CSVs in this folder, applies a chronological train/val/test split,
and writes interactive Plotly HTML per file.

Also writes a separate smoothed HTML: causal MA applied **per split** (no leakage
across train/val/test boundaries). Default window = 48.

Usage:
  python data_30mins_frequency/plot_chrono_splits_html.py --pattern merged
  python data_30mins_frequency/plot_chrono_splits_html.py --pattern merged --smooth-window 48
  python data_30mins_frequency/plot_chrono_splits_html.py --smoothed-only
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError as exc:
    raise SystemExit("plotly is required: pip install plotly") from exc

DATA_DIR = Path(__file__).resolve().parent
TIME_COL = "TIMESTAMP"
VALUE_COL = "Acceleration RMS"
OUT_DIR = DATA_DIR / "chrono_split_plots"

SPLIT_COLORS = {
    "train": "#2563eb",
    "val": "#16a34a",
    "test": "#ea580c",
}


def discover_sensor_csvs(data_dir: Path, pattern: str = "merged") -> List[Path]:
    """Sensor CSVs matching ``pattern``: merged | plain | 30_min."""
    skip_names = {
        "Jan2Jun.csv",
        "sensor_id_name_mapping.csv",
        "vibration_May.csv",
        "_downsample_summary.csv",
        "_train_rms_full_bins.csv",
        "_window_rms_summary.csv",
    }
    out: List[Path] = []
    for path in sorted(data_dir.glob("*.csv")):
        name = path.name
        if name in skip_names:
            continue
        if not re.match(r"^AHU[_\-]", name, flags=re.IGNORECASE):
            continue
        is_merged = name.endswith("_merged.csv")
        is_30 = name.endswith("_30_min.csv")
        if pattern == "merged":
            if is_merged:
                out.append(path)
        elif pattern == "30_min":
            if is_30:
                out.append(path)
        elif pattern == "plain":
            if not is_merged and not is_30:
                out.append(path)
        else:
            raise ValueError(f"Unknown pattern: {pattern!r} (use merged|plain|30_min)")
    return out


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    parsed = pd.to_datetime(raw, dayfirst=True, format="mixed", errors="coerce")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        mask = parsed.isna()
        if not mask.any():
            break
        parsed.loc[mask] = pd.to_datetime(raw.loc[mask], format=fmt, errors="coerce")
    return parsed


def load_sensor_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if TIME_COL not in df.columns:
        raise ValueError(f"{path.name}: missing {TIME_COL}")
    if VALUE_COL not in df.columns:
        raise ValueError(f"{path.name}: missing {VALUE_COL}")
    df = df.copy()
    df[TIME_COL] = parse_timestamp_series(df[TIME_COL])
    df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, VALUE_COL])
    df = df.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)
    return df.drop_duplicates(subset=[TIME_COL], keep="last").reset_index(drop=True)


def chrono_row_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, pd.DataFrame]:
    rsum = float(train_ratio + val_ratio + test_ratio)
    if abs(rsum - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1, got {rsum}")
    n = len(df)
    n_tr = int(np.floor(n * train_ratio))
    n_va = int(np.floor(n * val_ratio))
    n_te = n - n_tr - n_va
    if min(n_tr, n_va, n_te) < 1:
        raise ValueError(f"Split too small: n={n}, train={n_tr}, val={n_va}, test={n_te}")
    return {
        "train": df.iloc[:n_tr].copy(),
        "val": df.iloc[n_tr : n_tr + n_va].copy(),
        "test": df.iloc[n_tr + n_va :].copy(),
    }


def smooth_target_series_1d(vec: np.ndarray, window: int) -> np.ndarray:
    """Causal (trailing) MA; window W, min_periods=1 at the start of *this* series only."""
    if window <= 1:
        return np.asarray(vec, dtype=np.float32)
    w = int(window)
    y = np.asarray(vec, dtype=np.float64)
    n = len(y)
    if n == 0:
        return np.asarray(y, dtype=np.float32)
    csum = np.concatenate(([0.0], np.cumsum(y)))
    idx = np.arange(n, dtype=np.int64)
    start = np.maximum(0, idx - w + 1)
    counts = (idx - start + 1).astype(np.float64)
    return ((csum[idx + 1] - csum[start]) / counts).astype(np.float32)


def apply_causal_smooth_per_split(
    splits: Dict[str, pd.DataFrame],
    window: int,
) -> Dict[str, pd.DataFrame]:
    """Smooth each split independently — val/test never use train points."""
    out: Dict[str, pd.DataFrame] = {}
    for name, part in splits.items():
        smoothed = part.copy()
        smoothed[VALUE_COL] = smooth_target_series_1d(
            smoothed[VALUE_COL].to_numpy(dtype=np.float32), window
        )
        out[name] = smoothed
    return out


def split_stats(name: str, df: pd.DataFrame) -> Dict[str, Any]:
    y = df[VALUE_COL].astype(float)
    ts = df[TIME_COL]
    return {
        "split": name,
        "rows": int(len(df)),
        "t_min": str(ts.min()),
        "t_max": str(ts.max()),
        "mean": float(y.mean()),
        "median": float(y.median()),
        "std": float(y.std(ddof=0)),
        "min": float(y.min()),
        "max": float(y.max()),
        "p05": float(y.quantile(0.05)),
        "p95": float(y.quantile(0.95)),
    }


def format_stats_block(stats: Dict[str, Any], ratio: float) -> str:
    return (
        f"<b>{stats['split'].upper()}</b> ({ratio:.0%}, n={stats['rows']:,})<br>"
        f"{stats['t_min']} → {stats['t_max']}<br>"
        f"mean={stats['mean']:.4f}  median={stats['median']:.4f}  std={stats['std']:.4f}<br>"
        f"min={stats['min']:.4f}  max={stats['max']:.4f}  "
        f"p05={stats['p05']:.4f}  p95={stats['p95']:.4f}"
    )


def stats_table_html(stats: Dict[str, Dict[str, Any]], ratios: Tuple[float, float, float]) -> str:
    """HTML table below the chart — page-scrollable, never clipped by Plotly."""
    ratio_map = dict(zip(("train", "val", "test"), ratios))
    rows = []
    for name in ("train", "val", "test"):
        s = stats[name]
        color = SPLIT_COLORS[name]
        rows.append(
            "<tr>"
            f"<td style='border-left:6px solid {color};font-weight:700'>{name}</td>"
            f"<td>{ratio_map[name]:.0%}</td>"
            f"<td>{s['rows']:,}</td>"
            f"<td>{s['t_min']}</td>"
            f"<td>{s['t_max']}</td>"
            f"<td>{s['mean']:.4f}</td>"
            f"<td>{s['median']:.4f}</td>"
            f"<td>{s['std']:.4f}</td>"
            f"<td>{s['min']:.4f}</td>"
            f"<td>{s['max']:.4f}</td>"
            f"<td>{s['p05']:.4f}</td>"
            f"<td>{s['p95']:.4f}</td>"
            "</tr>"
        )
    return f"""
<section class="stats-panel">
  <h2>Split statistics (Acceleration RMS)</h2>
  <table>
    <thead>
      <tr>
        <th>split</th><th>ratio</th><th>rows</th>
        <th>t_min</th><th>t_max</th>
        <th>mean</th><th>median</th><th>std</th>
        <th>min</th><th>max</th><th>p05</th><th>p95</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
</section>
"""


def write_split_html(
    *,
    csv_path: Path,
    splits: Dict[str, pd.DataFrame],
    stats: Dict[str, Dict[str, Any]],
    ratios: Tuple[float, float, float],
    out_path: Path,
    title_suffix: str = "",
    y_axis_label: Optional[str] = None,
    meta_extra: Optional[Dict[str, Any]] = None,
) -> None:
    sensor_label = csv_path.stem.replace("_", " ").replace("2 9", "2-9").replace("4 4", "4-4")
    fig = go.Figure()
    y_label = y_axis_label or VALUE_COL

    for name in ("train", "val", "test"):
        part = splits[name]
        fig.add_trace(
            go.Scatter(
                x=part[TIME_COL],
                y=part[VALUE_COL],
                mode="lines",
                name=f"{name} ({stats[name]['rows']:,})",
                line=dict(color=SPLIT_COLORS[name], width=1.2),
                hovertemplate=(
                    f"{name}<br>%{{x|%Y-%m-%d %H:%M}}<br>"
                    f"RMS=%{{y:.4f}}<extra></extra>"
                ),
            )
        )

    t_val_start = splits["val"][TIME_COL].iloc[0]
    t_test_start = splits["test"][TIME_COL].iloc[0]
    for t_bound in (t_val_start, t_test_start):
        fig.add_vline(
            x=t_bound,
            line_dash="dash",
            line_color="rgba(80,80,80,0.85)",
            line_width=1.2,
        )

    subtitle = (
        f"Chronological split {ratios[0]:.0%} / {ratios[1]:.0%} / {ratios[2]:.0%} "
        f"(train → val → test in time)"
    )
    if title_suffix:
        subtitle = f"{subtitle} | {title_suffix}"

    fig.update_layout(
        title=(f"{sensor_label} — {y_label}<br><sup>{subtitle}</sup>"),
        xaxis_title="TIMESTAMP",
        yaxis_title=y_label,
        template="plotly_white",
        hovermode="x unified",
        height=560,
        margin=dict(l=60, r=30, t=90, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(rangeslider=dict(visible=False))

    plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{sensor_label} — {y_label}</title>
  <style>
    body {{
      margin: 0;
      padding: 16px 20px 40px;
      font-family: Segoe UI, system-ui, sans-serif;
      background: #fafafa;
      color: #111;
    }}
    .wrap {{ max-width: 1400px; margin: 0 auto; }}
    .plot-card, .stats-panel {{
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 12px 16px 20px;
      margin-bottom: 20px;
    }}
    .stats-panel h2 {{
      margin: 0 0 12px;
      font-size: 16px;
      font-weight: 650;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-family: ui-monospace, Consolas, monospace;
      font-size: 12px;
    }}
    th, td {{
      border-bottom: 1px solid #eee;
      padding: 8px 10px;
      text-align: left;
      white-space: nowrap;
    }}
    th {{ background: #f3f4f6; font-weight: 650; }}
    tbody tr:hover {{ background: #f9fafb; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="plot-card">
      {plot_html}
    </div>
    {stats_table_html(stats, ratios)}
  </div>
</body>
</html>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")

    meta: Dict[str, Any] = {
        "source_csv": str(csv_path.resolve()),
        "output_html": str(out_path.resolve()),
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "stats": stats,
    }
    if meta_extra:
        meta.update(meta_extra)
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def process_csv(
    csv_path: Path,
    *,
    out_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    smooth_window: int = 48,
    write_raw: bool = True,
    write_smoothed: bool = True,
) -> List[Path]:
    df = load_sensor_csv(csv_path)
    splits = chrono_row_split(df, train_ratio, val_ratio, test_ratio)
    pct = "_".join(str(int(r * 100)) for r in (train_ratio, val_ratio, test_ratio))
    ratios = (train_ratio, val_ratio, test_ratio)
    written: List[Path] = []

    if write_raw:
        stats = {name: split_stats(name, part) for name, part in splits.items()}
        out_path = out_dir / f"{csv_path.stem}_chrono_{pct}_rms.html"
        write_split_html(
            csv_path=csv_path,
            splits=splits,
            stats=stats,
            ratios=ratios,
            out_path=out_path,
            title_suffix="raw (no smoothing)",
            y_axis_label=VALUE_COL,
            meta_extra={"smoothing": None},
        )
        print(
            f"Wrote {out_path.name}  "
            f"(train={stats['train']['rows']:,}, val={stats['val']['rows']:,}, test={stats['test']['rows']:,})"
        )
        written.append(out_path)

    if write_smoothed and smooth_window > 1:
        # Causal MA separately inside each split — first val point never uses train.
        smooth_splits = apply_causal_smooth_per_split(splits, smooth_window)
        smooth_stats = {name: split_stats(name, part) for name, part in smooth_splits.items()}
        out_path = out_dir / f"{csv_path.stem}_chrono_{pct}_rms_smooth{smooth_window}.html"
        write_split_html(
            csv_path=csv_path,
            splits=smooth_splits,
            stats=smooth_stats,
            ratios=ratios,
            out_path=out_path,
            title_suffix=(
                f"causal MA window={smooth_window} "
                f"(applied independently per split — no cross-split leakage)"
            ),
            y_axis_label=f"{VALUE_COL} (smoothed)",
            meta_extra={
                "smoothing": {
                    "method": "causal_trailing_ma",
                    "window": int(smooth_window),
                    "scope": "per_split_independent",
                    "note": "val/test MA never includes train (or each other) rows",
                }
            },
        )
        print(
            f"Wrote {out_path.name}  "
            f"(smooth={smooth_window}, per-split; "
            f"train mean={smooth_stats['train']['mean']:.4f}, "
            f"val mean={smooth_stats['val']['mean']:.4f}, "
            f"test mean={smooth_stats['test']['mean']:.4f})"
        )
        written.append(out_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--csv", type=str, default=None, help="Single file name or path")
    parser.add_argument(
        "--pattern",
        choices=("merged", "plain", "30_min"),
        default="merged",
        help="Which CSVs to plot (default: merged).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=48,
        help="Causal MA window for separate smoothed HTMLs (default 48; 0/1 disables).",
    )
    parser.add_argument(
        "--smoothed-only",
        action="store_true",
        help="Only write smoothed HTMLs (skip raw).",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Only write raw HTMLs (skip smoothed).",
    )
    args = parser.parse_args()

    if args.csv:
        p = Path(args.csv)
        paths = [p if p.is_file() else args.data_dir / args.csv]
    else:
        paths = discover_sensor_csvs(args.data_dir, pattern=args.pattern)

    if not paths:
        raise SystemExit(f"No matching sensor CSVs in {args.data_dir} (pattern={args.pattern})")

    write_raw = not args.smoothed_only
    write_smoothed = not args.raw_only
    print(f"Found {len(paths)} sensor CSV(s) (pattern={args.pattern})")
    for path in paths:
        if not path.is_file():
            print(f"[skip] missing {path}")
            continue
        try:
            process_csv(
                path,
                out_dir=args.out_dir,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                smooth_window=args.smooth_window,
                write_raw=write_raw,
                write_smoothed=write_smoothed,
            )
        except Exception as exc:
            print(f"[error] {path.name}: {exc}")

    print(f"\nHTML plots: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
