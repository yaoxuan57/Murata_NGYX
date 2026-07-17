"""Interactive Plotly HTML timelines for finetune train/val/test CSVs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SPLIT_DIR = Path(__file__).resolve().parent / "data_AHU_2_9_Blower_DE_A" / "splits"
TIME_COL = "TIMESTAMP"
VALUE_COL = "Acceleration RMS"
SENSOR = "AHU 2-9 Blower DE A"


def load_split(name: str) -> pd.DataFrame:
    path = SPLIT_DIR / f"{name}.csv"
    df = pd.read_csv(path, low_memory=False)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], dayfirst=True, format="mixed", errors="coerce")
    if "SENSOR_DESC" in df.columns:
        df = df[df["SENSOR_DESC"].astype(str).str.strip() == SENSOR]
    df = df.dropna(subset=[TIME_COL, VALUE_COL]).sort_values(TIME_COL)
    df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce")
    return df


def write_single_html(name: str, df: pd.DataFrame, out_path: Path) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[TIME_COL],
            y=df[VALUE_COL],
            mode="lines",
            name=VALUE_COL,
            line=dict(width=1.2, color="#2563eb"),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>RMS=%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{SENSOR} — {name} split (n={len(df):,})",
        xaxis_title="Time",
        yaxis_title=VALUE_COL,
        template="plotly_white",
        hovermode="x unified",
        height=520,
        margin=dict(l=60, r=30, t=60, b=50),
    )
    fig.update_xaxes(rangeslider=dict(visible=True))
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)


def write_overview_html(splits: dict[str, pd.DataFrame], out_path: Path) -> None:
    n_tr = len(splits["train"])
    n_va = len(splits["val"])
    n_te = len(splits["test"])
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.06,
        subplot_titles=(
            f"train (n={n_tr:,})",
            f"val (n={n_va:,})",
            f"test (n={n_te:,})",
        ),
    )
    colors = {"train": "#2563eb", "val": "#16a34a", "test": "#ea580c"}
    for i, name in enumerate(("train", "val", "test"), start=1):
        df = splits[name]
        fig.add_trace(
            go.Scatter(
                x=df[TIME_COL],
                y=df[VALUE_COL],
                mode="lines",
                name=name,
                line=dict(width=1, color=colors[name]),
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>RMS=%{y:.3f}<extra></extra>",
            ),
            row=i,
            col=1,
        )
    fig.update_layout(
        title=f"{SENSOR} — train / val / test splits (zoom & pan per panel)",
        template="plotly_white",
        height=900,
        hovermode="x unified",
    )
    for i in range(1, 4):
        fig.update_yaxes(title_text=VALUE_COL, row=i, col=1)
        fig.update_xaxes(rangeslider=dict(visible=(i == 3)), row=i, col=1)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)


def main() -> None:
    splits = {name: load_split(name) for name in ("train", "val", "test")}
    for name, df in splits.items():
        out = SPLIT_DIR / f"{name}_rms_timeline.html"
        write_single_html(name, df, out)
        print(f"Wrote {out}")
    overview = SPLIT_DIR / "train_val_test_rms_overview.html"
    write_overview_html(splits, overview)
    print(f"Wrote {overview}")


if __name__ == "__main__":
    main()
