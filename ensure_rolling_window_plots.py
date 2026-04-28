import argparse
import os
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate missing rolling-window plot PNGs from saved CSV files."
    )
    parser.add_argument("--runs-root", type=str, required=True)
    return parser.parse_args()


def regenerate_run_plots(run_dir: str) -> Tuple[int, int]:
    windows_dir = os.path.join(run_dir, "rolling_window_forecasts")
    csv_dir = os.path.join(windows_dir, "csv")
    plots_dir = os.path.join(windows_dir, "plots")

    if not os.path.isdir(csv_dir):
        return 0, 0

    os.makedirs(plots_dir, exist_ok=True)
    csv_files = sorted(
        name for name in os.listdir(csv_dir) if name.startswith("window_") and name.endswith(".csv")
    )
    if not csv_files:
        return 0, 0

    generated = 0
    for csv_name in csv_files:
        png_name = os.path.splitext(csv_name)[0] + ".png"
        png_path = os.path.join(plots_dir, png_name)
        if os.path.isfile(png_path):
            continue

        csv_path = os.path.join(csv_dir, csv_name)
        window_df = pd.read_csv(csv_path)
        if window_df.empty:
            continue

        plt.figure(figsize=(8, 3))
        plt.plot(window_df["step_ahead"], window_df["actual"], label="Actual")
        plt.plot(window_df["step_ahead"], window_df["predicted"], label="Predicted")
        window_idx = int(window_df["window_index"].iloc[0])
        pred_len = len(window_df)
        plt.title(f"Window {window_idx} Forecast ({pred_len}-step)")
        plt.xlabel("Step Ahead")
        plt.ylabel("Acceleration RMS")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(png_path, dpi=140)
        plt.close()
        generated += 1

    return len(csv_files), generated


def main():
    args = parse_args()
    if not os.path.isdir(args.runs_root):
        raise FileNotFoundError(f"Runs root not found: {args.runs_root}")

    total_runs = 0
    total_windows = 0
    total_generated = 0

    for run_name in sorted(os.listdir(args.runs_root)):
        run_dir = os.path.join(args.runs_root, run_name)
        if not os.path.isdir(run_dir):
            continue

        total_csv, generated = regenerate_run_plots(run_dir)
        if total_csv == 0:
            continue

        total_runs += 1
        total_windows += total_csv
        total_generated += generated
        print(f"{run_name}: regenerated {generated}/{total_csv} plot files")

    print(
        f"Done. Processed {total_runs} runs, {total_windows} window CSVs, generated {total_generated} missing PNGs."
    )


if __name__ == "__main__":
    main()
