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
    parser.add_argument("--train-val-csv", type=str, default=None)
    parser.add_argument("--test-csv", type=str, default=None)
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
    if args.train_val_csv is not None:
        cmd.extend(["--train-val-csv", args.train_val_csv])
    if args.test_csv is not None:
        cmd.extend(["--test-csv", args.test_csv])
    if supports_loss_weights:
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
            ]
        )
    if supports_save_window_plots:
        cmd.append("--save-window-plots")

    append_optional(cmd, "--lr", lr)
    append_optional(cmd, "--weight-decay", weight_decay)
    cmd.extend(model_config_to_flags(model_config))
    cmd.extend(["--output-dir", out_dir])

    print("Re-running best config with window plots:")
    print(" ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
