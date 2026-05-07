import argparse
import json
import os
import shlex
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Re-run selected best sweep config with plots enabled.")
    parser.add_argument("--selection-json", type=str, required=True)
    parser.add_argument("--train-script", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--scheduler-patience", type=int, default=6)
    parser.add_argument("--pred-lens", type=int, default=288)
    parser.add_argument("--loss-point-weight", type=float, default=0.7)
    parser.add_argument("--loss-diff-weight", type=float, default=0.9)
    parser.add_argument("--loss-curvature-weight", type=float, default=0.5)
    parser.add_argument("--loss-variance-weight", type=float, default=0.2)
    parser.add_argument("--loss-laplacian-weight", type=float, default=0.3)
    parser.add_argument(
        "--pred-smoothing-window",
        type=int,
        default=1,
        help="Passed to train script; 1 disables post-smoothing (default).",
    )
    parser.add_argument("--train-csv", type=str, default=None, help="With --val-csv, explicit 3-way split.")
    parser.add_argument("--val-csv", type=str, default=None)
    parser.add_argument("--train-val-csv", type=str, default=None)
    parser.add_argument("--test-csv", type=str, default=None)
    parser.add_argument(
        "--single-csv",
        type=str,
        default=None,
        help="Override or set path for runs that used --single-csv.",
    )
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument(
        "--rolling-window-artifact-limit",
        type=int,
        default=None,
        help="If set, passed through to the train script (per-window CSV/PNG cap). "
        "When omitted, uses best_config.rolling_window_artifact_limit when present.",
    )
    return parser.parse_args()


def append_optional(cmd, flag, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def model_config_to_flags(model_config):
    args = []
    skip = {"model_type", "input_len", "pred_len"}
    for key, value in model_config.items():
        if key in skip:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            args.append(flag if value else f"--no-{key.replace('_', '-')}")
        else:
            args.extend([flag, str(value)])
    return args


def main():
    args = parse_args()
    if (args.train_csv is None) ^ (args.val_csv is None):
        raise SystemExit("Provide both --train-csv and --val-csv or neither.")

    with open(args.selection_json, "r", encoding="utf-8") as fp:
        best = json.load(fp)

    run_name = best["run_name"]
    input_len = int(best["input_len"])
    model_config = best.get("model_config", {})
    best_config = best.get("best_config", {})
    supports_save_window_plots = bool(best_config.get("supports_save_window_plots", True))
    supports_loss_weights = "loss_point_weight" in best_config
    lr = best.get("lr")
    weight_decay = best.get("weight_decay")

    out_dir = os.path.join(args.output_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "python",
        "-u",
        args.train_script,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--scheduler-patience",
        str(args.scheduler_patience),
        "--pred-lens",
        str(args.pred_lens),
        "--input-lens",
        str(input_len),
    ]
    explicit_cfg = best_config.get("data_split_mode") == "explicit_train_val_test"
    single_cfg = best_config.get("data_split_mode") == "single_csv_window_ratios"
    cli_explicit = args.train_csv is not None and args.val_csv is not None
    single_path = args.single_csv or (best_config.get("single_csv") if single_cfg else None)

    if cli_explicit:
        cmd.extend(["--train-csv", args.train_csv, "--val-csv", args.val_csv])
        test_p = args.test_csv or best_config.get("test_csv")
        if test_p:
            cmd.extend(["--test-csv", str(test_p)])
    elif single_path:
        cmd.extend(["--single-csv", str(single_path)])
        tr = (
            args.train_ratio
            if args.train_ratio is not None
            else best_config.get("train_ratio_window")
        )
        va = (
            args.val_ratio
            if args.val_ratio is not None
            else best_config.get("val_ratio_window")
        )
        te = (
            args.test_ratio
            if args.test_ratio is not None
            else best_config.get("test_ratio_window")
        )
        if tr is not None:
            cmd.extend(["--train-ratio", str(tr)])
        if va is not None:
            cmd.extend(["--val-ratio", str(va)])
        if te is not None:
            cmd.extend(["--test-ratio", str(te)])
        mw = best_config.get("min_windows_per_split")
        if mw is not None:
            cmd.extend(["--min-windows-per-split", str(int(mw))])
    elif explicit_cfg and best_config.get("train_csv") and best_config.get("val_csv"):
        cmd.extend(
            [
                "--train-csv",
                str(best_config["train_csv"]),
                "--val-csv",
                str(best_config["val_csv"]),
            ]
        )
        if best_config.get("test_csv"):
            cmd.extend(["--test-csv", str(best_config["test_csv"])])
    else:
        if args.train_val_csv is not None:
            cmd.extend(["--train-val-csv", args.train_val_csv])
        elif best_config.get("train_val_csv"):
            cmd.extend(["--train-val-csv", str(best_config["train_val_csv"])])
        if args.test_csv is not None:
            cmd.extend(["--test-csv", args.test_csv])
        elif best_config.get("test_csv"):
            cmd.extend(["--test-csv", str(best_config["test_csv"])])
    if supports_loss_weights:
        loss_lap = best_config.get("loss_laplacian_weight", args.loss_laplacian_weight)
        pred_smooth = args.pred_smoothing_window
        cmd.extend(
            [
                "--loss-point-weight",
                str(args.loss_point_weight),
                "--loss-diff-weight",
                str(args.loss_diff_weight),
                "--loss-curvature-weight",
                str(args.loss_curvature_weight),
                "--loss-variance-weight",
                str(args.loss_variance_weight),
                "--loss-laplacian-weight",
                str(loss_lap),
                "--pred-smoothing-window",
                str(pred_smooth),
            ]
        )
    if supports_save_window_plots:
        cmd.append("--save-window-plots")

    rolling_limit = args.rolling_window_artifact_limit
    if rolling_limit is None:
        rolling_limit = best_config.get("rolling_window_artifact_limit")
    if rolling_limit is not None:
        cmd.extend(["--rolling-window-artifact-limit", str(rolling_limit)])

    append_optional(cmd, "--lr", lr)
    append_optional(cmd, "--weight-decay", weight_decay)
    value_column = best_config.get("value_column")
    if value_column is not None:
        cmd.extend(["--value-column", str(value_column)])
    feature_columns = best_config.get("feature_columns")
    if isinstance(feature_columns, list) and len(feature_columns) > 0:
        cmd.append("--feature-columns")
        cmd.extend([str(c) for c in feature_columns])
    if bool(best_config.get("use_all_numeric_features", False)):
        cmd.append("--use-all-numeric-features")
    cmd.extend(model_config_to_flags(model_config))
    cmd.extend(["--output-dir", out_dir])

    print("Re-running best config with window plots:")
    print(" ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
