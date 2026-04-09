import argparse
import json
import os
import shlex
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Re-run best DLinear config with window plots enabled")
    parser.add_argument("--selection-json", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--scheduler-patience", type=int, default=6)
    parser.add_argument("--pred-lens", type=int, default=36)
    parser.add_argument("--loss-point-weight", type=float, default=0.7)
    parser.add_argument("--loss-diff-weight", type=float, default=0.9)
    parser.add_argument("--loss-curvature-weight", type=float, default=0.5)
    parser.add_argument("--loss-variance-weight", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.selection_json, "r", encoding="utf-8") as fp:
        best = json.load(fp)

    run_name = best["run_name"]
    input_len = int(best["input_len"])
    kernel_size = int(best["kernel_size"])
    lr = float(best["lr"])
    weight_decay = float(best["weight_decay"])

    out_dir = os.path.join(args.output_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "python",
        "-u",
        "train_dlinear_sweep.py",
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
        "--loss-point-weight",
        str(args.loss_point_weight),
        "--loss-diff-weight",
        str(args.loss_diff_weight),
        "--loss-curvature-weight",
        str(args.loss_curvature_weight),
        "--loss-variance-weight",
        str(args.loss_variance_weight),
        "--save-window-plots",
        "--input-lens",
        str(input_len),
        "--kernel-size",
        str(kernel_size),
        "--lr",
        str(lr),
        "--weight-decay",
        str(weight_decay),
        "--output-dir",
        out_dir,
    ]

    print("Re-running best config with window plots:")
    print(" ".join(shlex.quote(x) for x in cmd))

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
