import argparse
import json
import os
from typing import List

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select best DLinear run by test_composite_loss (min), else val composite, else val window RMSE."
    )
    parser.add_argument("--runs-root", type=str, required=True)
    parser.add_argument("--out-json", type=str, default="best_run_selection.json")
    return parser.parse_args()


def collect_run_rows(runs_root: str) -> pd.DataFrame:
    rows: List[dict] = []

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

        if "test_composite_loss" in summary_df.columns:
            sort_key = "test_composite_loss"
        elif "best_val_loss" in summary_df.columns:
            sort_key = "best_val_loss"
        else:
            sort_key = "best_val_window_rmse"
        best_row = summary_df.sort_values(sort_key).iloc[0].to_dict()
        with open(cfg_path, "r", encoding="utf-8") as fp:
            cfg = json.load(fp)

        rows.append(
            {
                "run_name": run_name,
                "run_dir": run_dir,
                "best_val_loss": float(best_row["best_val_loss"])
                if pd.notna(best_row.get("best_val_loss"))
                else None,
                "best_val_window_rmse": float(best_row["best_val_window_rmse"]),
                "test_composite_loss": float(best_row["test_composite_loss"])
                if pd.notna(best_row.get("test_composite_loss"))
                else None,
                "test_composite_loss_pct": float(best_row["test_composite_loss_pct"])
                if pd.notna(best_row.get("test_composite_loss_pct"))
                else None,
                "test_mse": float(best_row["test_mse"]),
                "test_rmse": float(best_row["test_rmse"]),
                "test_mae": float(best_row["test_mae"]),
                "test_mape": float(best_row["test_mape"]),
                "test_r2": float(best_row["test_r2"]),
                "input_len": int(best_row["input_len"]),
                "pred_len": int(best_row["pred_len"]),
                "kernel_size": cfg.get("model_config", {}).get("kernel_size"),
                "lr": cfg.get("lr"),
                "weight_decay": cfg.get("weight_decay"),
                "loss_point_weight": cfg.get("loss_point_weight"),
                "loss_diff_weight": cfg.get("loss_diff_weight"),
                "loss_curvature_weight": cfg.get("loss_curvature_weight"),
                "loss_variance_weight": cfg.get("loss_variance_weight"),
                "use_residual_head": cfg.get("model_config", {}).get("use_residual_head"),
                "residual_hidden": cfg.get("model_config", {}).get("residual_hidden"),
                "residual_dropout": cfg.get("model_config", {}).get("residual_dropout"),
                "residual_weight": cfg.get("model_config", {}).get("residual_weight"),
            }
        )

    if not rows:
        raise RuntimeError("No completed runs found. Missing experiment_summary.csv or best_config.json.")

    df_r = pd.DataFrame(rows)
    if "test_composite_loss" in df_r.columns and df_r["test_composite_loss"].notna().all():
        rank_col = "test_composite_loss"
    elif "best_val_loss" in df_r.columns and df_r["best_val_loss"].notna().all():
        rank_col = "best_val_loss"
    else:
        rank_col = "best_val_window_rmse"
    return df_r.sort_values(rank_col).reset_index(drop=True)


def main():
    args = parse_args()
    df = collect_run_rows(args.runs_root)

    ranking_path = os.path.join(args.runs_root, "dlinear_run_ranking.csv")
    df.to_csv(ranking_path, index=False)

    best = df.iloc[0].to_dict()
    out_path = args.out_json
    if not os.path.isabs(out_path):
        out_path = os.path.join(args.runs_root, out_path)

    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(best, fp, indent=2)

    print("Saved ranking:", ranking_path)
    print("Saved best run selection:", out_path)
    print("Best run:", best["run_name"])
    if best.get("test_composite_loss") is not None:
        print("test_composite_loss (min over candidates):", best["test_composite_loss"])
    if best.get("best_val_loss") is not None:
        print("best_val_loss (val composite at best epoch):", best["best_val_loss"])
    print("best_val_window_rmse:", best["best_val_window_rmse"])
    if best.get("test_composite_loss_pct") is not None:
        print("test_composite_loss_pct (loss×100):", best["test_composite_loss_pct"])


if __name__ == "__main__":
    main()
