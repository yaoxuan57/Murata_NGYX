"""Build purged random window splits for Jan–Jun finetune (no overlap leakage).

Modes
-----
full_random (default)
  Entire Jan–Jun CSV. Randomly assign 7-day chunks to train / val / test (70/15/15).
  May–Jun high-RMS periods can land in train. Test windows are random chunks too
  (not a pure temporal tail). Chunks are purged: no window crosses train/val/test.

temporal_holdout
  Legacy: last 15%% rows frozen as holdout; random train/val chunks on dev only.

Outputs (full_random)
  full.csv, window_manifest.json  (train + test both index into full.csv)

Training:
  --train-csv .../full.csv --test-csv .../full.csv --window-manifest .../window_manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

DATA_DIR = REPO / "finetune" / "data_AHU_2_9_Blower_DE_A"
DEFAULT_SOURCE = DATA_DIR / "AHU_2_9_Blower_DE_A_jan2_jun_30_min.csv"
DEFAULT_OUT = DATA_DIR / "splits_purged"
TIME_COL, VALUE_COL, SENSOR_COL = "TIMESTAMP", "Acceleration RMS", "SENSOR_DESC"
SENSOR = "AHU 2-9 Blower DE A"

INPUT_LEN = 48
PRED_LEN = 48
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
HOLDOUT_FRAC = 0.15
CHUNK_ROWS = 336
SEED = 42


def row_indices_covered_by_windows(starts: np.ndarray, span: int, n_rows: int) -> np.ndarray:
    starts = np.asarray(starts, dtype=np.int64)
    if starts.size == 0:
        return np.zeros(0, dtype=np.int64)
    if starts.size >= 2 and np.all(np.diff(starts) == 1):
        lo = int(starts[0])
        hi = int(starts[-1]) + int(span)
        return np.arange(lo, min(hi, n_rows), dtype=np.int64)
    mask = np.zeros(int(n_rows), dtype=bool)
    for s in starts:
        ss = int(s)
        if ss < 0:
            continue
        mask[ss : min(ss + int(span), n_rows)] = True
    return np.flatnonzero(mask)


def parse_ts(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.strip()
    p = pd.to_datetime(raw, dayfirst=True, format="mixed", errors="coerce")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        m = p.isna()
        if not m.any():
            break
        p.loc[m] = pd.to_datetime(raw.loc[m], format=fmt, errors="coerce")
    return p


def load_sensor_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if VALUE_COL not in df.columns and "DATA12" in df.columns:
        df[VALUE_COL] = pd.to_numeric(df["DATA12"], errors="coerce")
    if SENSOR_COL not in df.columns and "SENSOR_NAME" in df.columns:
        df[SENSOR_COL] = df["SENSOR_NAME"].astype(str).str.strip()
    df[TIME_COL] = parse_ts(df[TIME_COL])
    df = df[df[SENSOR_COL].astype(str).str.strip() == SENSOR]
    return df.sort_values(TIME_COL).drop_duplicates(TIME_COL, keep="last").reset_index(drop=True)


def split_chunk_counts(
    total: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    *,
    min_each: int = 1,
) -> tuple[int, int, int]:
    if total < 3 * min_each:
        raise ValueError(f"Need at least {3 * min_each} chunks (got {total}).")
    w = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(w < 0) or float(w.sum()) <= 0:
        raise ValueError("train/val/test ratios must be positive.")
    w /= w.sum()
    left = total - 3 * min_each
    raw = left * w
    extra = np.floor(raw).astype(int)
    rem = left - int(extra.sum())
    frac_order = np.argsort(-(raw - extra))
    for k in range(rem):
        extra[int(frac_order[k % 3])] += 1
    parts = min_each + extra
    return int(parts[0]), int(parts[1]), int(parts[2])


def chunk_window_starts(
    n_rows: int,
    *,
    input_len: int,
    pred_len: int,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    span = input_len + pred_len
    if n_rows < span:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    n_chunks = n_rows // chunk_rows
    if n_chunks < 3:
        raise ValueError(
            f"Need at least 3 full chunks of {chunk_rows} rows; got {n_chunks} from {n_rows} rows."
        )

    starts: list[int] = []
    chunk_ids: list[int] = []
    for cid in range(n_chunks):
        lo = cid * chunk_rows
        hi = lo + chunk_rows
        last_start = hi - span
        if last_start < lo:
            continue
        for s in range(lo, last_start + 1):
            starts.append(s)
            chunk_ids.append(cid)

    return np.asarray(starts, dtype=np.int64), np.asarray(chunk_ids, dtype=np.int64)


def purged_chunk_split_three_way(
    starts: np.ndarray,
    chunk_ids: np.ndarray,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int], list[int]]:
    unique_chunks = np.unique(chunk_ids)
    n_tr, n_va, n_te = split_chunk_counts(
        len(unique_chunks), train_ratio, val_ratio, test_ratio, min_each=1
    )
    rng = np.random.default_rng(seed)
    perm = unique_chunks.copy()
    rng.shuffle(perm)

    train_chunks = set(int(c) for c in perm[:n_tr])
    val_chunks = set(int(c) for c in perm[n_tr : n_tr + n_va])
    test_chunks = set(int(c) for c in perm[n_tr + n_va : n_tr + n_va + n_te])

    train_mask = np.array([int(c) in train_chunks for c in chunk_ids], dtype=bool)
    val_mask = np.array([int(c) in val_chunks for c in chunk_ids], dtype=bool)
    test_mask = np.array([int(c) in test_chunks for c in chunk_ids], dtype=bool)
    return (
        starts[train_mask],
        starts[val_mask],
        starts[test_mask],
        sorted(train_chunks),
        sorted(val_chunks),
        sorted(test_chunks),
    )


def purged_chunk_split_two_way(
    starts: np.ndarray,
    chunk_ids: np.ndarray,
    *,
    train_chunk_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    unique_chunks = np.unique(chunk_ids)
    rng = np.random.default_rng(seed)
    perm = unique_chunks.copy()
    rng.shuffle(perm)

    n_train_chunks = max(1, int(np.floor(len(perm) * train_chunk_frac)))
    if n_train_chunks >= len(perm):
        n_train_chunks = len(perm) - 1
    train_chunks = set(int(c) for c in perm[:n_train_chunks])
    val_chunks = set(int(c) for c in perm[n_train_chunks:])

    train_mask = np.array([int(c) in train_chunks for c in chunk_ids], dtype=bool)
    val_mask = np.array([int(c) in val_chunks for c in chunk_ids], dtype=bool)
    return starts[train_mask], starts[val_mask], sorted(train_chunks), sorted(val_chunks)


def rms_summary(frame: pd.DataFrame) -> dict:
    s = pd.to_numeric(frame[VALUE_COL], errors="coerce")
    return {
        "rows": int(len(frame)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "median": float(s.median()),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def window_start_rms_stats(series: np.ndarray, starts: np.ndarray, input_len: int) -> dict:
    if len(starts) == 0:
        return {"windows": 0, "last_val_mean": None, "last_val_ge_5": 0, "last_val_ge_5_pct": 0.0}
    last_vals = series[starts + input_len - 1]
    ge5 = int((last_vals >= 5.0).sum())
    return {
        "windows": int(len(starts)),
        "last_val_mean": float(last_vals.mean()),
        "last_val_ge_5": ge5,
        "last_val_ge_5_pct": float(100.0 * ge5 / len(starts)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-mode",
        choices=("full_random", "temporal_holdout"),
        default="full_random",
        help="full_random: 70/15/15 chunk split on entire Jan–Jun (default). "
        "temporal_holdout: legacy dev + chrono tail holdout.",
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--input-len", type=int, default=INPUT_LEN)
    parser.add_argument("--pred-len", type=int, default=PRED_LEN)
    parser.add_argument("--holdout-frac", type=float, default=HOLDOUT_FRAC)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC)
    parser.add_argument("--test-frac", type=float, default=TEST_FRAC)
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    span = args.input_len + args.pred_len
    df = load_sensor_frame(args.source)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    series = pd.to_numeric(df[VALUE_COL], errors="coerce").to_numpy()

    if args.split_mode == "full_random":
        starts, chunk_ids = chunk_window_starts(
            len(df),
            input_len=args.input_len,
            pred_len=args.pred_len,
            chunk_rows=args.chunk_rows,
        )
        train_starts, val_starts, test_starts, train_chunks, val_chunks, test_chunks = (
            purged_chunk_split_three_way(
                starts,
                chunk_ids,
                train_ratio=args.train_frac,
                val_ratio=args.val_frac,
                test_ratio=args.test_frac,
                seed=args.seed,
            )
        )

        full_path = args.out_dir / "full.csv"
        df.to_csv(full_path, index=False)
        row_tr = row_indices_covered_by_windows(train_starts, span, len(df))
        train_mean = float(series[row_tr].mean())
        train_std = float(series[row_tr].std()) + 1e-8

        manifest = {
            "mode": "purged_chunk_full_random_70_15_15",
            "window_manifest_version": 2,
            "same_series_test": True,
            "source_csv": str(args.source.resolve()),
            "series_csv": str(full_path.resolve()),
            "input_len": int(args.input_len),
            "pred_len": int(args.pred_len),
            "span": int(span),
            "chunk_rows": int(args.chunk_rows),
            "train_frac": float(args.train_frac),
            "val_frac": float(args.val_frac),
            "test_frac": float(args.test_frac),
            "seed": int(args.seed),
            "train_chunks": train_chunks,
            "val_chunks": val_chunks,
            "test_chunks": test_chunks,
            "train_window_starts": [int(x) for x in train_starts],
            "val_window_starts": [int(x) for x in val_starts],
            "test_window_starts": [int(x) for x in test_starts],
            "counts": {
                "series_rows": len(df),
                "train_windows": int(len(train_starts)),
                "val_windows": int(len(val_starts)),
                "test_windows": int(len(test_starts)),
            },
            "time": {
                "full": [str(df[TIME_COL].iloc[0]), str(df[TIME_COL].iloc[-1])],
            },
            "rms": {
                "full": rms_summary(df),
                "train_rows_covered": rms_summary(df.iloc[row_tr]),
                "train_mean_for_norm": train_mean,
                "train_std_for_norm": train_std,
                "window_last_val": {
                    "train": window_start_rms_stats(series, train_starts, args.input_len),
                    "val": window_start_rms_stats(series, val_starts, args.input_len),
                    "test": window_start_rms_stats(series, test_starts, args.input_len),
                },
            },
        }
    else:
        train_chunk_frac = args.train_frac / (args.train_frac + args.val_frac)
        n = len(df)
        n_holdout = int(np.floor(n * args.holdout_frac))
        n_dev = n - n_holdout
        if n_dev < args.chunk_rows * 2 or n_holdout < span:
            raise ValueError(f"Not enough rows for split: n={n}, dev={n_dev}, holdout={n_holdout}")

        dev = df.iloc[:n_dev].copy()
        holdout = df.iloc[n_dev:].copy()
        dev_series = pd.to_numeric(dev[VALUE_COL], errors="coerce").to_numpy()

        starts, chunk_ids = chunk_window_starts(
            len(dev),
            input_len=args.input_len,
            pred_len=args.pred_len,
            chunk_rows=args.chunk_rows,
        )
        train_starts, val_starts, train_chunks, val_chunks = purged_chunk_split_two_way(
            starts,
            chunk_ids,
            train_chunk_frac=train_chunk_frac,
            seed=args.seed,
        )

        dev_path = args.out_dir / "dev.csv"
        holdout_path = args.out_dir / "holdout.csv"
        dev.to_csv(dev_path, index=False)
        holdout.to_csv(holdout_path, index=False)

        row_tr = row_indices_covered_by_windows(train_starts, span, len(dev))
        train_mean = float(dev_series[row_tr].mean())
        train_std = float(dev_series[row_tr].std()) + 1e-8

        manifest = {
            "mode": "purged_chunk_random_dev_temporal_holdout",
            "window_manifest_version": 1,
            "same_series_test": False,
            "source_csv": str(args.source.resolve()),
            "dev_csv": str(dev_path.resolve()),
            "holdout_csv": str(holdout_path.resolve()),
            "input_len": int(args.input_len),
            "pred_len": int(args.pred_len),
            "span": int(span),
            "chunk_rows": int(args.chunk_rows),
            "holdout_frac": float(args.holdout_frac),
            "train_frac": float(args.train_frac),
            "val_frac": float(args.val_frac),
            "train_chunk_frac_of_dev": float(train_chunk_frac),
            "seed": int(args.seed),
            "holdout_start": str(holdout[TIME_COL].iloc[0]),
            "train_chunks": train_chunks,
            "val_chunks": val_chunks,
            "train_window_starts": [int(x) for x in train_starts],
            "val_window_starts": [int(x) for x in val_starts],
            "counts": {
                "dev_rows": len(dev),
                "holdout_rows": len(holdout),
                "train_windows": int(len(train_starts)),
                "val_windows": int(len(val_starts)),
                "holdout_windows_dense": int(max(0, len(holdout) - span + 1)),
            },
            "time": {
                "dev": [str(dev[TIME_COL].iloc[0]), str(dev[TIME_COL].iloc[-1])],
                "holdout": [str(holdout[TIME_COL].iloc[0]), str(holdout[TIME_COL].iloc[-1])],
            },
            "rms": {
                "dev": rms_summary(dev),
                "holdout": rms_summary(holdout),
                "train_rows_covered": rms_summary(dev.iloc[row_tr]),
                "train_mean_for_norm": train_mean,
                "train_std_for_norm": train_std,
                "window_last_val": {
                    "train": window_start_rms_stats(dev_series, train_starts, args.input_len),
                    "val": window_start_rms_stats(dev_series, val_starts, args.input_len),
                },
            },
        }

    manifest_path = args.out_dir / "window_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
