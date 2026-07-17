#!/usr/bin/env python3
"""Plot train/val/test from merged CSVs plus deployment test holdout (HTML).

For each ``*_merged.csv`` in this folder:
  - train / val / test: chronological split of the merged historical CSV
  - deployment: ``June_data_deployment/data_<STEM>/splits/test.csv``

Writes raw + per-split smoothed HTMLs to ``chrono_split_plots_with_deployment/``.

Usage:
  python data_30mins_frequency/plot_chrono_splits_with_deployment_html.py
  python data_30mins_frequency/plot_chrono_splits_with_deployment_html.py --smooth-window 48
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go

from plot_chrono_splits_html import (
    DATA_DIR,
    TIME_COL,
    VALUE_COL,
    apply_causal_smooth_per_split,
    chrono_row_split,
    discover_sensor_csvs,
    load_sensor_csv,
    parse_timestamp_series,
    split_stats,
    smooth_target_series_1d,
)

DEPLOYMENT_ROOT = DATA_DIR / "June_data_deployment"
OUT_DIR = DATA_DIR / "chrono_split_plots_with_deployment"

SPLIT_COLORS = {
    "train": "#2563eb",
    "val": "#16a34a",
    "test": "#ea580c",
    "deployment": "#9333ea",
}

SPLIT_ORDER = ("train", "val", "test", "deployment")


def stem_from_merged_csv(csv_path: Path) -> str:
    stem = csv_path.stem
    if stem.endswith("_merged"):
        stem = stem[: -len("_merged")]
    # Deployment folders use underscore (AHU_4_4_...), merged files may use hyphen.
    return stem.replace("4-4", "4_4").replace("2-9", "2_9")


def deployment_test_csv(stem: str, deployment_root: Path) -> Path:
    candidates = [
        deployment_root / f"data_{stem}" / "splits" / "test.csv",
        deployment_root / f"data_{stem.replace('4_4', '4-4')}" / "splits" / "test.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def load_deployment_test(path: Path) -> Any:
    import pandas as pd

    df = pd.read_csv(path, low_memory=False)
    if VALUE_COL not in df.columns and "DATA12" in df.columns:
        df[VALUE_COL] = pd.to_numeric(df["DATA12"], errors="coerce")
    if VALUE_COL not in df.columns:
        raise ValueError(f"{path.name}: missing {VALUE_COL}")
    df = df.copy()
    df[TIME_COL] = parse_timestamp_series(df[TIME_COL])
    df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, VALUE_COL])
    return (
        df.sort_values(TIME_COL, kind="mergesort")
        .drop_duplicates(subset=[TIME_COL], keep="last")
        .reset_index(drop=True)
    )


def stats_table_html(
    stats: Dict[str, Dict[str, Any]],
    ratios: Tuple[float, float, float],
) -> str:
    ratio_map = dict(zip(("train", "val", "test"), ratios))
    rows = []
    for name in SPLIT_ORDER:
        s = stats[name]
        color = SPLIT_COLORS[name]
        ratio_text = "—" if name == "deployment" else f"{ratio_map[name]:.0%}"
        rows.append(
            "<tr>"
            f"<td style='border-left:6px solid {color};font-weight:700'>{name}</td>"
            f"<td>{ratio_text}</td>"
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
    splits: Dict[str, Any],
    stats: Dict[str, Dict[str, Any]],
    ratios: Tuple[float, float, float],
    out_path: Path,
    deployment_csv: Path,
    title_suffix: str = "",
    y_axis_label: Optional[str] = None,
    meta_extra: Optional[Dict[str, Any]] = None,
) -> None:
    sensor_label = csv_path.stem.replace("_", " ").replace("2 9", "2-9").replace("4 4", "4-4")
    fig = go.Figure()
    y_label = y_axis_label or VALUE_COL

    for name in SPLIT_ORDER:
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

    for t_bound in (
        splits["val"][TIME_COL].iloc[0],
        splits["test"][TIME_COL].iloc[0],
        splits["deployment"][TIME_COL].iloc[0],
    ):
        fig.add_vline(
            x=t_bound,
            line_dash="dash",
            line_color="rgba(80,80,80,0.85)",
            line_width=1.2,
        )

    subtitle = (
        f"Chronological split {ratios[0]:.0%} / {ratios[1]:.0%} / {ratios[2]:.0%} "
        f"+ deployment test holdout"
    )
    if title_suffix:
        subtitle = f"{subtitle} | {title_suffix}"

    fig.update_layout(
        title=(f"{sensor_label} — {y_label}<br><sup>{subtitle}</sup>"),
        xaxis_title="TIMESTAMP",
        yaxis_title=y_label,
        template="plotly_white",
        hovermode="x unified",
        height=580,
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
        "deployment_csv": str(deployment_csv.resolve()),
        "output_html": str(out_path.resolve()),
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "stats": stats,
    }
    if meta_extra:
        meta.update(meta_extra)
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def build_splits_with_deployment(
    csv_path: Path,
    deployment_root: Path,
) -> Tuple[Dict[str, Any], Path, Tuple[float, float, float]]:
    df = load_sensor_csv(csv_path)
    stem = stem_from_merged_csv(csv_path)
    deploy_path = deployment_test_csv(stem, deployment_root)
    if not deploy_path.is_file():
        raise FileNotFoundError(f"Missing deployment test CSV: {deploy_path}")

    train_ratio, val_ratio, test_ratio = 0.60, 0.20, 0.20
    splits = chrono_row_split(df, train_ratio, val_ratio, test_ratio)
    splits["deployment"] = load_deployment_test(deploy_path)
    return splits, deploy_path, (train_ratio, val_ratio, test_ratio)


def process_csv(
    csv_path: Path,
    *,
    out_dir: Path,
    deployment_root: Path,
    smooth_window: int = 48,
    write_raw: bool = True,
    write_smoothed: bool = True,
) -> List[Path]:
    splits, deploy_path, ratios = build_splits_with_deployment(csv_path, deployment_root)
    pct = "_".join(str(int(r * 100)) for r in ratios)
    written: List[Path] = []

    if write_raw:
        stats = {name: split_stats(name, part) for name, part in splits.items()}
        out_path = out_dir / f"{csv_path.stem}_chrono_{pct}_deploy_rms.html"
        write_split_html(
            csv_path=csv_path,
            splits=splits,
            stats=stats,
            ratios=ratios,
            out_path=out_path,
            deployment_csv=deploy_path,
            title_suffix="raw (no smoothing)",
            y_axis_label=VALUE_COL,
            meta_extra={"smoothing": None},
        )
        print(
            f"Wrote {out_path.name}  "
            f"(train={stats['train']['rows']:,}, val={stats['val']['rows']:,}, "
            f"test={stats['test']['rows']:,}, deployment={stats['deployment']['rows']:,})"
        )
        written.append(out_path)

    if write_smoothed and smooth_window > 1:
        smooth_splits = apply_causal_smooth_per_split(splits, smooth_window)
        smooth_stats = {name: split_stats(name, part) for name, part in smooth_splits.items()}
        out_path = out_dir / f"{csv_path.stem}_chrono_{pct}_deploy_rms_smooth{smooth_window}.html"
        write_split_html(
            csv_path=csv_path,
            splits=smooth_splits,
            stats=smooth_stats,
            ratios=ratios,
            out_path=out_path,
            deployment_csv=deploy_path,
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
                    "note": "deployment MA computed only on deployment rows",
                }
            },
        )
        print(
            f"Wrote {out_path.name}  "
            f"(smooth={smooth_window}; deploy mean={smooth_stats['deployment']['mean']:.4f})"
        )
        written.append(out_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--deployment-dir", type=Path, default=DEPLOYMENT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--csv", type=str, default=None, help="Single file name or path")
    parser.add_argument(
        "--pattern",
        choices=("merged", "plain", "30_min"),
        default="merged",
    )
    parser.add_argument("--smooth-window", type=int, default=48)
    parser.add_argument("--smoothed-only", action="store_true")
    parser.add_argument("--raw-only", action="store_true")
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
    print(f"Deployment root: {args.deployment_dir}")
    for path in paths:
        if not path.is_file():
            print(f"[skip] missing {path}")
            continue
        try:
            process_csv(
                path,
                out_dir=args.out_dir,
                deployment_root=args.deployment_dir,
                smooth_window=args.smooth_window,
                write_raw=write_raw,
                write_smoothed=write_smoothed,
            )
        except Exception as exc:
            print(f"[error] {path.name}: {exc}")

    print(f"\nHTML plots: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
