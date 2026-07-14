#!/usr/bin/env python3
"""One HTML per sensor: x=timestamp, y=consecutive interval (minutes)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "jan2jun_by_sensor"

Y_CAP_THRESHOLD = 120.0
Y_CAP_DISPLAY = 130.0
Y_TICKS = list(range(10, 121, 10)) + [130]
Y_TICK_LABELS = [str(t) for t in range(10, 121, 10)] + [">120"]

FILES = [
    (
        "AHU_2_9_Blower_DE_A_intervals.html",
        "AHU 2-9 Blower DE A",
        DATA_DIR / "AHU_2_9_Blower_DE_A.csv",
    ),
    (
        "AHU_2_9_Blower_DE_Vibration_X_intervals.html",
        "AHU 2-9 Blower DE Vibration X",
        DATA_DIR / "AHU_2_9_Blower_DE_Vibration_X.csv",
    ),
]


def parse_ts(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    ts = pd.to_datetime(raw, dayfirst=True, format="mixed", errors="coerce")
    bad = ts.isna()
    if bad.any():
        ts.loc[bad] = pd.to_datetime(raw.loc[bad], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    return ts


def load_intervals(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["TIMESTAMP"] = parse_ts(df["TIMESTAMP"])
    df = df.dropna(subset=["TIMESTAMP"]).sort_values("TIMESTAMP").reset_index(drop=True)
    df = df.drop_duplicates(subset=["TIMESTAMP"], keep="last").reset_index(drop=True)

    interval_min = df["TIMESTAMP"].diff().dt.total_seconds().iloc[1:] / 60.0
    return pd.DataFrame(
        {
            "timestamp": df["TIMESTAMP"].iloc[1:].values,
            "interval_min": interval_min.values,
        }
    )


def stats_text(intervals: pd.DataFrame) -> str:
    s = intervals["interval_min"]
    pct30 = float(s.between(29.5, 30.5).mean() * 100)
    n_capped = int((s > Y_CAP_THRESHOLD).sum())
    capped_note = f" | >{Y_CAP_THRESHOLD:.0f} min (shown at 130): {n_capped}" if n_capped else ""
    return (
        f"n={len(s):,} intervals | min={s.min():.2f} | median={s.median():.2f} | "
        f"mean={s.mean():.2f} | max={s.max():.2f} min | ~30 min: {pct30:.1f}%"
        f"{capped_note}"
    )


def display_interval(interval_min: np.ndarray) -> np.ndarray:
    """Plot values >120 at y=130 so large gaps do not stretch the scale."""
    return np.where(interval_min > Y_CAP_THRESHOLD, Y_CAP_DISPLAY, interval_min)


def write_html(out_path: Path, title: str, intervals: pd.DataFrame) -> None:
    import plotly.graph_objects as go

    raw_y = intervals["interval_min"].to_numpy(dtype=np.float64)
    plot_y = display_interval(raw_y)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=intervals["timestamp"],
            y=plot_y,
            customdata=raw_y,
            mode="lines",
            name="Interval",
            line=dict(color="#2563eb", width=1),
            hovertemplate=(
                "%{x}<br>interval=%{customdata:.2f} min"
                "<extra></extra>"
            ),
        )
    )
    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="#94a3b8",
        annotation_text="30 min",
        annotation_position="top right",
    )
    fig.add_hline(
        y=Y_CAP_DISPLAY,
        line_dash="dot",
        line_color="#cbd5e1",
        annotation_text=f">{Y_CAP_THRESHOLD:.0f} min capped here",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title=title,
        xaxis_title="Timestamp",
        yaxis_title="Interval (minutes)",
        template="plotly_white",
        height=520,
        width=1100,
        margin=dict(l=60, r=30, t=70, b=60),
    )
    fig.update_yaxes(
        range=[0, Y_CAP_DISPLAY],
        tickmode="array",
        tickvals=Y_TICKS,
        ticktext=Y_TICK_LABELS,
    )

    summary = stats_text(intervals)
    note = (
        f"<br><i>Y-axis: ticks 10–120 min; intervals &gt; {Y_CAP_THRESHOLD:.0f} min "
        f"are drawn at {Y_CAP_DISPLAY:.0f} (hover shows true value).</i>"
    )
    html = fig.to_html(full_html=True, include_plotlyjs="cdn", config={"scrollZoom": True})
    html = html.replace(
        "</body>",
        f"<p style='font-family:Segoe UI,Arial,sans-serif;padding:0 24px 16px;color:#475569'>"
        f"<b>Summary:</b> {summary}{note}</p></body>",
    )
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    for out_name, title, csv_path in FILES:
        if not csv_path.is_file():
            raise SystemExit(f"Missing: {csv_path}")
        intervals = load_intervals(csv_path)
        out_path = DATA_DIR / out_name
        write_html(out_path, f"{title} — timestamp intervals", intervals)
        print(f"{title}: {stats_text(intervals)}")
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
