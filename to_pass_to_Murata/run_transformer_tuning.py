#!/usr/bin/env python3
"""Portable replacement for run_transformer_tuning.sbatch (no Slurm required).

Runs the full transformer tuning pipeline:
  1. Optional chronological CSV split
  2. Train candidate config(s) with window plots + quantile bands
  3. Rank runs and pick the best
  4. Re-run the best config into best_model/
  5. Rank best_model/ outputs

Example (default: split data/AHU_2_9_Blower_DE_A.csv then train):
  python run_transformer_tuning.py

Example (pre-split files):
  python run_transformer_tuning.py \\
    --split-source-csv "" \\
    --train-csv data/splits/my_run/train.csv \\
    --val-csv data/splits/my_run/val.csv \\
    --test-csv data/splits/my_run/test.csv

Example (quick smoke test on CPU):
  python run_transformer_tuning.py --epochs 5 --split-source-csv data/AHU_2_9_Blower_DE_A.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_transformer_sweep.py"
SPLIT_SCRIPT = SCRIPT_DIR / "split_csv_chronological_train_val_test.py"
SELECT_SCRIPT = SCRIPT_DIR / "select_best_sweep_run.py"
RERUN_SCRIPT = SCRIPT_DIR / "rerun_best_sweep_with_plots.py"


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd or SCRIPT_DIR))


def _parse_input_features(text: str | None) -> list[str]:
    if not text or not str(text).strip():
        return []
    return [part.strip() for part in str(text).split(";") if part.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run transformer tuning pipeline (split → train → select → rerun → select)."
    )

    # --- Data ---
    data = p.add_argument_group("data")
    data.add_argument(
        "--split-source-csv",
        type=str,
        default=None,
        help="Chronologically split this CSV into train/val/test before training. "
        "Ignored when --train-csv/--val-csv/--test-csv are set.",
    )
    data.add_argument("--split-out-dir", type=str, default=None)
    data.add_argument("--chrono-train-ratio", type=float, default=0.6)
    data.add_argument("--chrono-val-ratio", type=float, default=0.2)
    data.add_argument("--chrono-test-ratio", type=float, default=0.2)
    data.add_argument("--train-csv", type=str, default=None)
    data.add_argument("--val-csv", type=str, default=None)
    data.add_argument("--test-csv", type=str, default=None)
    data.add_argument("--single-csv", type=str, default=None)
    data.add_argument("--train-ratio", type=float, default=0.80)
    data.add_argument("--val-ratio", type=float, default=0.10)
    data.add_argument("--test-ratio", type=float, default=None)

    # --- Target / features ---
    feat = p.add_argument_group("target and features")
    feat.add_argument("--value-column", type=str, default="Acceleration RMS")
    feat.add_argument(
        "--input-features",
        type=str,
        default="Acceleration RMS",
        help="Semicolon-separated feature column names (e.g. 'Acceleration RMS;Kurtosis').",
    )
    feat.add_argument(
        "--use-all-numeric-features",
        action="store_true",
        help="Use every numeric column except TIMESTAMP as input features.",
    )
    feat.add_argument(
        "--target-smoothing-window",
        type=int,
        default=48,
        help="Causal trailing MA on value column per CSV before windowing (1 = off).",
    )

    # --- Window / timestamp rules ---
    win = p.add_argument_group("window rules")
    win.add_argument(
        "--require-uniform-timestep",
        dest="require_uniform_timestep",
        action="store_true",
        default=True,
    )
    win.add_argument("--no-require-uniform-timestep", dest="require_uniform_timestep", action="store_false")
    win.add_argument("--uniform-step-seconds", type=float, default=1800.0)
    win.add_argument("--uniform-step-tolerance-seconds", type=float, default=60.0)
    win.add_argument(
        "--max-consecutive-timestamp-gap-seconds",
        type=float,
        default=1860.0,
        help="Max allowed Δt between consecutive rows inside a window. Use 0 to disable.",
    )

    # --- Training ---
    train = p.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--early-stopping-patience", type=int, default=50)
    train.add_argument("--min-delta", type=float, default=1e-6)
    train.add_argument("--scheduler-patience", type=int, default=6)
    train.add_argument("--input-lens", type=int, nargs="+", default=[48])
    train.add_argument("--pred-lens", type=int, nargs="+", default=[48])
    train.add_argument("--lr", type=float, default=5e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--pred-smoothing-window", type=int, default=1)
    train.add_argument("--train-window-stride", type=int, default=1)
    train.add_argument("--val-window-stride", type=int, default=0)
    train.add_argument("--test-window-stride", type=int, default=0)

    # --- Loss weights ---
    loss = p.add_argument_group("loss weights")
    loss.add_argument("--loss-point-weight", type=float, default=0.7)
    loss.add_argument("--loss-diff-weight", type=float, default=0.9)
    loss.add_argument("--loss-curvature-weight", type=float, default=0.5)
    loss.add_argument("--loss-variance-weight", type=float, default=0.2)
    loss.add_argument("--loss-tail-weight", type=float, default=1.0)

    # --- Quantiles / plots ---
    plots = p.add_argument_group("quantiles and plots")
    plots.add_argument(
        "--enable-forecast-quantiles",
        dest="enable_forecast_quantiles",
        action="store_true",
        default=True,
    )
    plots.add_argument("--no-forecast-quantiles", dest="enable_forecast_quantiles", action="store_false")
    plots.add_argument(
        "--save-stitched-test-html",
        dest="save_stitched_test_html",
        action="store_true",
        default=True,
    )
    plots.add_argument("--no-stitched-test-html", dest="save_stitched_test_html", action="store_false")
    plots.add_argument("--rolling-window-artifact-limit", type=int, default=None)

    # --- Output ---
    out = p.add_argument_group("output")
    out.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root output directory. Default: outputs_transformer_tuning_<data_tag>[_<run-id>].",
    )
    out.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional suffix on output dir (replaces Slurm job id) to avoid collisions.",
    )
    out.add_argument(
        "--sweep-run-tag",
        type=str,
        default=None,
        help="Subfolder name under runs/. Default: cfg_48_k11_lr5e4_wd<weight-decay>.",
    )
    out.add_argument(
        "--skip-rerun",
        action="store_true",
        help="Stop after sweep + ranking (skip best-model rerun step).",
    )

    meta = p.add_argument_group("model metadata")
    meta.add_argument(
        "--model-version",
        type=str,
        default="v1",
        help="Version tag for <checkpoint>_meta.json modelName (e.g. v1).",
    )
    meta.add_argument(
        "--sensor-mapping-csv",
        type=str,
        default=None,
        help="SENSOR_CODE/SENSOR_NAME mapping CSV (default: data/sensor_id_name_mapping.csv).",
    )
    meta.add_argument(
        "--no-model-meta-json",
        dest="write_model_meta_json",
        action="store_false",
        help="Skip writing model metadata JSON beside each .pth checkpoint.",
    )
    p.set_defaults(write_model_meta_json=True)

    return p.parse_args()


class DataConfig:
    def __init__(
        self,
        *,
        mode: str,
        data_flags: list[str],
        data_tag: str,
        ref_parent: Path,
        train_csv: str | None = None,
        val_csv: str | None = None,
        test_csv: str | None = None,
        single_csv: str | None = None,
        train_ratio: float | None = None,
        val_ratio: float | None = None,
        test_ratio: float | None = None,
    ):
        self.mode = mode
        self.data_flags = data_flags
        self.data_tag = data_tag
        self.ref_parent = ref_parent
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.single_csv = single_csv
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def rerun_data_flags(self) -> list[str]:
        if self.mode == "single":
            flags = [
                "--single-csv",
                self.single_csv,
                "--train-ratio",
                str(self.train_ratio),
                "--val-ratio",
                str(self.val_ratio),
            ]
            if self.test_ratio is not None:
                flags.extend(["--test-ratio", str(self.test_ratio)])
            return flags
        return [
            "--train-csv",
            self.train_csv,
            "--val-csv",
            self.val_csv,
            "--test-csv",
            self.test_csv,
        ]


def _resolve_data_paths(args: argparse.Namespace) -> DataConfig:
    train_csv = args.train_csv
    val_csv = args.val_csv
    test_csv = args.test_csv
    single_csv = args.single_csv
    split_out_dir = args.split_out_dir

    if train_csv and val_csv and test_csv:
        pass  # explicit three-file split — use as-is
    elif single_csv:
        pass  # handled below
    elif split_out_dir:
        split_path = Path(split_out_dir)
        if not split_path.is_absolute():
            split_path = SCRIPT_DIR / split_path
        for name in ("train.csv", "val.csv", "test.csv"):
            if not (split_path / name).is_file():
                raise FileNotFoundError(f"Expected {split_path / name} when using --split-out-dir")
        train_csv = str(split_path / "train.csv")
        val_csv = str(split_path / "val.csv")
        test_csv = str(split_path / "test.csv")
        single_csv = None
    else:
        split_source = (args.split_source_csv or "").strip()
        if not split_source:
            raise SystemExit(
                "Set one of: --split-source-csv, --split-out-dir, "
                "--train-csv/--val-csv/--test-csv, or --single-csv."
            )
        src = Path(split_source)
        if not src.is_absolute():
            src = SCRIPT_DIR / src
        if not src.is_file():
            raise FileNotFoundError(f"Split source CSV not found: {src}")
        if not split_out_dir:
            stem = src.stem
            split_out_dir = str(
                SCRIPT_DIR
                / "data"
                / "splits"
                / f"{stem}_r{args.chrono_train_ratio}_{args.chrono_val_ratio}_{args.chrono_test_ratio}"
            )
        split_path = Path(split_out_dir)
        split_path.mkdir(parents=True, exist_ok=True)
        _run(
            [
                sys.executable,
                "-u",
                str(SPLIT_SCRIPT),
                "--input",
                str(src),
                "--out-dir",
                str(split_path),
                "--train-ratio",
                str(args.chrono_train_ratio),
                "--val-ratio",
                str(args.chrono_val_ratio),
                "--test-ratio",
                str(args.chrono_test_ratio),
            ]
        )
        train_csv = str(split_path / "train.csv")
        val_csv = str(split_path / "val.csv")
        test_csv = str(split_path / "test.csv")
        single_csv = None

    if train_csv and val_csv and test_csv:
        ref_parent = Path(train_csv).resolve().parent
        return DataConfig(
            mode="explicit",
            data_flags=["--train-csv", train_csv, "--val-csv", val_csv, "--test-csv", test_csv],
            data_tag=f"chrono_{ref_parent.name}",
            ref_parent=ref_parent,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
        )
    if single_csv:
        sc = Path(single_csv)
        if not sc.is_absolute():
            sc = SCRIPT_DIR / sc
        if not sc.is_file():
            raise FileNotFoundError(f"Single CSV not found: {sc}")
        data_flags = [
            "--single-csv",
            str(sc),
            "--train-ratio",
            str(args.train_ratio),
            "--val-ratio",
            str(args.val_ratio),
        ]
        if args.test_ratio is not None:
            data_flags.extend(["--test-ratio", str(args.test_ratio)])
        return DataConfig(
            mode="single",
            data_flags=data_flags,
            data_tag=f"{sc.stem}_single_r{args.train_ratio}_v{args.val_ratio}",
            ref_parent=sc.parent,
            single_csv=str(sc),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )

    raise SystemExit(
        "Set one of: --split-source-csv, --split-out-dir, "
        "--train-csv/--val-csv/--test-csv, or --single-csv."
    )


def _meta_flags(args: argparse.Namespace) -> list[str]:
    if not getattr(args, "write_model_meta_json", True):
        return ["--no-model-meta-json"]
    flags = ["--model-version", str(args.model_version)]
    if args.sensor_mapping_csv:
        flags.extend(["--sensor-mapping-csv", str(args.sensor_mapping_csv)])
    return flags


def _feature_flags(args: argparse.Namespace) -> list[str]:
    if args.use_all_numeric_features:
        return ["--use-all-numeric-features"]
    cols = _parse_input_features(args.input_features)
    if cols:
        return ["--feature-columns", *cols]
    return []


def _uniform_flags(args: argparse.Namespace) -> list[str]:
    if args.require_uniform_timestep:
        return [
            "--require-uniform-timestep",
            "--uniform-step-seconds",
            str(args.uniform_step_seconds),
            "--uniform-step-tolerance-seconds",
            str(args.uniform_step_tolerance_seconds),
        ]
    return ["--no-require-uniform-timestep"]


def _quantile_flags(args: argparse.Namespace) -> list[str]:
    if args.enable_forecast_quantiles:
        return ["--forecast-quantiles", "0.05", "0.5", "0.95"]
    return []


def _stitch_flags(args: argparse.Namespace) -> list[str]:
    return ["--save-stitched-test-html"] if args.save_stitched_test_html else []


def _rolling_flags(args: argparse.Namespace) -> list[str]:
    if args.rolling_window_artifact_limit is not None and args.rolling_window_artifact_limit > 0:
        return ["--rolling-window-artifact-limit", str(args.rolling_window_artifact_limit)]
    return []


def _target_smooth_flags(args: argparse.Namespace) -> list[str]:
    if args.target_smoothing_window > 1:
        return ["--target-smoothing-window", str(args.target_smoothing_window)]
    return []


def _common_train_flags(args: argparse.Namespace) -> list[str]:
    flags = [
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--min-delta",
        str(args.min_delta),
        "--scheduler-patience",
        str(args.scheduler_patience),
        "--pred-smoothing-window",
        str(args.pred_smoothing_window),
        "--train-window-stride",
        str(args.train_window_stride),
        "--val-window-stride",
        str(args.val_window_stride),
        "--test-window-stride",
        str(args.test_window_stride),
        "--max-consecutive-timestamp-gap-seconds",
        str(args.max_consecutive_timestamp_gap_seconds),
        *_uniform_flags(args),
    ]
    flags.extend(["--pred-lens", *[str(x) for x in args.pred_lens]])
    return flags


def _common_loss_flags(args: argparse.Namespace) -> list[str]:
    return [
        "--loss-point-weight",
        str(args.loss_point_weight),
        "--loss-diff-weight",
        str(args.loss_diff_weight),
        "--loss-curvature-weight",
        str(args.loss_curvature_weight),
        "--loss-variance-weight",
        str(args.loss_variance_weight),
        "--loss-tail-weight",
        str(args.loss_tail_weight),
    ]


def _output_root(args: argparse.Namespace, ref_parent: Path, data_tag: str) -> Path:
    if args.output_root:
        root = Path(args.output_root)
        return root if root.is_absolute() else SCRIPT_DIR / root

    folder_name = ref_parent.name
    suffix = "" if folder_name == "." else f"_{folder_name}"
    job_suffix = f"_run{args.run_id}" if args.run_id else ""
    return SCRIPT_DIR / f"outputs_transformer_tuning{suffix}_{data_tag}{job_suffix}"


def main() -> None:
    args = parse_args()
    os.chdir(SCRIPT_DIR)
    print(f"Using project dir: {SCRIPT_DIR}")

    for script in (TRAIN_SCRIPT, SPLIT_SCRIPT, SELECT_SCRIPT, RERUN_SCRIPT):
        if not script.is_file():
            raise FileNotFoundError(f"Missing required script: {script}")

    data = _resolve_data_paths(args)
    feature_flags = _feature_flags(args)
    quant_flags = _quantile_flags(args)
    stitch_flags = _stitch_flags(args)
    rolling_flags = _rolling_flags(args)
    target_smooth_flags = _target_smooth_flags(args)
    meta_flags = _meta_flags(args)
    common_train = _common_train_flags(args)
    common_loss = _common_loss_flags(args)

    root = _output_root(args, data.ref_parent, data.data_tag)
    runs = root / "runs"
    best = root / "best_model"
    runs.mkdir(parents=True, exist_ok=True)
    best.mkdir(parents=True, exist_ok=True)

    sweep_run_tag = args.sweep_run_tag or f"cfg_48_k11_lr5e4_wd{args.weight_decay}"
    sweep_out = runs / sweep_run_tag

    print(f"Output root: {root}")
    print(f"Quantile forecast: {'on (0.05 0.5 0.95)' if args.enable_forecast_quantiles else 'off'}")
    print(f"Stitched test HTML: {'on' if args.save_stitched_test_html else 'off'}")
    print(f"AdamW weight decay: {args.weight_decay}")

    # 1) Train candidate config(s)
    train_cmd = [
        sys.executable,
        "-u",
        str(TRAIN_SCRIPT),
        *common_train,
        *common_loss,
        *quant_flags,
        *target_smooth_flags,
        "--value-column",
        args.value_column,
        *feature_flags,
        *data.data_flags,
        "--save-window-plots",
        *stitch_flags,
        *rolling_flags,
        *meta_flags,
        "--input-lens",
        *[str(x) for x in args.input_lens],
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--output-dir",
        str(sweep_out),
    ]
    _run(train_cmd)

    # 2) Rank sweep runs
    _run(
        [
            sys.executable,
            "-u",
            str(SELECT_SCRIPT),
            "--runs-root",
            str(runs),
            "--out-json",
            "best_run_selection.json",
        ]
    )

    if args.skip_rerun:
        print("\nSkipping best-model rerun (--skip-rerun).")
        print(f"Sweep output: {sweep_out}")
        if args.save_stitched_test_html:
            print(f"Stitched HTML: {sweep_out / 'rolling_window_forecasts' / 'stitched_test_forecast.html'}")
        return

    # 3) Re-run best config with plots
    rerun_cmd = [
        sys.executable,
        "-u",
        str(RERUN_SCRIPT),
        "--selection-json",
        str(runs / "best_run_selection.json"),
        "--train-script",
        str(TRAIN_SCRIPT),
        "--output-root",
        str(best),
        *data.rerun_data_flags(),
        "--value-column",
        args.value_column,
        *feature_flags,
        *stitch_flags,
        *rolling_flags,
        *quant_flags,
        *common_train,
        *common_loss,
        *meta_flags,
        "--target-smoothing-window",
        str(args.target_smoothing_window),
    ]
    _run(rerun_cmd)

    # 4) Rank best_model runs
    _run(
        [
            sys.executable,
            "-u",
            str(SELECT_SCRIPT),
            "--runs-root",
            str(best),
            "--out-json",
            "best_model_selection.json",
        ]
    )

    print("\nTransformer tuning completed.")
    print(f"Candidates ranking: {runs / 'sweep_run_ranking.csv'}")
    print(f"Best selection:     {runs / 'best_run_selection.json'}")
    print(f"Sweep plots:        {sweep_out / 'rolling_window_forecasts' / 'plots'}")
    if args.save_stitched_test_html:
        print(f"Stitched HTML:      {sweep_out / 'rolling_window_forecasts' / 'stitched_test_forecast.html'}")
    print(f"Best model root:    {best}")


if __name__ == "__main__":
    main()
