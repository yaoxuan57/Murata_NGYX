import argparse
import json
import os
from typing import Dict, List

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Select best sweep run from multiple run directories.")
    parser.add_argument("--runs-root", type=str, required=True)
    parser.add_argument("--out-json", type=str, default="best_run_selection.json")
    parser.add_argument(
        "--objective",
        type=str,
        default="auto",
        help="Ranking metric. Use 'auto' to prefer best_val_window_rmse, then best_val_loss.",
    )
    return parser.parse_args()


def infer_objective(summary_df: pd.DataFrame) -> str:
    if "best_val_window_rmse" in summary_df.columns:
        return "best_val_window_rmse"
    if "best_val_loss" in summary_df.columns:
        return "best_val_loss"
    raise RuntimeError(
        "Could not infer ranking metric from experiment_summary.csv. "
        "Expected best_val_window_rmse or best_val_loss."
    )


def to_float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_run_rows(runs_root: str, objective_arg: str) -> pd.DataFrame:
    rows: List[Dict] = []

    if not os.path.isdir(runs_root):
        raise FileNotFoundError(f"Runs root not found: {runs_root}")

    for run_name in sorted(os.listdir(runs_root)):
        run_dir = os.path.join(runs_root, run_name)
        if not os.path.isdir(run_dir):
            continue

        summary_path = os.path.join(run_dir, "experiment_summary.csv")
        cfg_path = os.path.join(run_dir, "best_config.json")
        if not os.path.isfile(summary_path) or not os.path.isfile(cfg_path):
            continue

        summary_df = pd.read_csv(summary_path)
        if summary_df.empty:
            continue

        objective = infer_objective(summary_df) if objective_arg == "auto" else objective_arg
        if objective not in summary_df.columns:
            raise RuntimeError(
                f"Objective '{objective}' not found in {summary_path}. "
                f"Columns: {list(summary_df.columns)}"
            )

        best_row = summary_df.sort_values(objective).iloc[0].to_dict()
        with open(cfg_path, "r", encoding="utf-8") as fp:
            cfg = json.load(fp)

        model_config = cfg.get("model_config", {})
        row = {
            "run_name": run_name,
            "run_dir": run_dir,
            "objective_name": objective,
            "objective_value": float(best_row[objective]),
            "input_len": int(best_row["input_len"]) if "input_len" in best_row else None,
            "pred_len": int(best_row["pred_len"]) if "pred_len" in best_row else None,
            "test_mse": to_float_or_none(best_row.get("test_mse")),
            "test_rmse": to_float_or_none(best_row.get("test_rmse")),
            "test_mae": to_float_or_none(best_row.get("test_mae")),
            "test_mape": to_float_or_none(best_row.get("test_mape")),
            "test_r2": to_float_or_none(best_row.get("test_r2")),
            "lr": to_float_or_none(cfg.get("lr")),
            "weight_decay": to_float_or_none(cfg.get("weight_decay")),
            "save_window_plots": cfg.get("save_window_plots"),
            "model_config": model_config,
            "best_config": cfg,
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("No completed runs found. Missing experiment_summary.csv or best_config.json.")

    return pd.DataFrame(rows).sort_values("objective_value").reset_index(drop=True)


def main():
    args = parse_args()
    df = collect_run_rows(args.runs_root, args.objective)

    ranking_path = os.path.join(args.runs_root, "sweep_run_ranking.csv")
    df_for_csv = df.drop(columns=["model_config", "best_config"])
    df_for_csv.to_csv(ranking_path, index=False)

    best = df.iloc[0].to_dict()
    out_path = args.out_json
    if not os.path.isabs(out_path):
        out_path = os.path.join(args.runs_root, out_path)

    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(best, fp, indent=2)

    print("Saved ranking:", ranking_path)
    print("Saved best run selection:", out_path)
    print("Best run:", best["run_name"])
    print(f"{best['objective_name']}: {best['objective_value']}")


if __name__ == "__main__":
    main()
