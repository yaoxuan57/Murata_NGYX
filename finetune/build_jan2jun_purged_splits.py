"""Build purged random dev splits + temporal holdout for Jan–Jun finetune.

Protocol (no leakage):
  1) Cut holdout by time first (last HOLDOUT_FRAC rows).
  2) On dev only: assign contiguous time chunks randomly to train vs val.
  3) Within each chunk: all sliding windows whose full span stays in the chunk.
  4) Write dev.csv, holdout.csv, window_manifest.json for --window-manifest training.

Submit training with:
  --train-csv .../purged/dev.csv
  --test-csv .../purged/holdout.csv
  --window-manifest .../purged/window_manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


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

DATA_DIR = REPO / "finetune" / "data_AHU_2_9_Blower_DE_A"
DEFAULT_SOURCE = DATA_DIR / "AHU_2_9_Blower_DE_A_jan2_jun_30_min.csv"
DEFAULT_OUT = DATA_DIR / "splits_purged"
TIME_COL, VALUE_COL, SENSOR_COL = "TIMESTAMP", "Acceleration RMS", "SENSOR_DESC"
SENSOR = "AHU 2-9 Blower DE A"

INPUT_LEN = 48
PRED_LEN = 48
HOLDOUT_FRAC = 0.15
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
CHUNK_ROWS = 336  # 7 days at 30-min cadence
SEED = 42


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


def chunk_window_starts(
    n_rows: int,
    *,
    input_len: int,
    pred_len: int,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (all_valid_starts, chunk_id_per_start) for windows fully inside one chunk."""
    span = input_len + pred_len
    if n_rows < span:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    n_chunks = n_rows // chunk_rows
    if n_chunks < 2:
        raise ValueError(
            f"Need at least 2 full chunks of {chunk_rows} rows for purged split; "
            f"dev has {n_rows} rows ({n_chunks} chunks)."
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


def purged_chunk_split(
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--input-len", type=int, default=INPUT_LEN)
    parser.add_argument("--pred-len", type=int, default=PRED_LEN)
    parser.add_argument("--holdout-frac", type=float, default=HOLDOUT_FRAC)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC)
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    span = args.input_len + args.pred_len
    dev_frac = 1.0 - args.holdout_frac
    train_chunk_frac = args.train_frac / (args.train_frac + args.val_frac)

    df = load_sensor_frame(args.source)
    n = len(df)
    n_holdout = int(np.floor(n * args.holdout_frac))
    n_dev = n - n_holdout
    if n_dev < args.chunk_rows * 2 or n_holdout < span:
        raise ValueError(f"Not enough rows for split: n={n}, dev={n_dev}, holdout={n_holdout}")

    dev = df.iloc[:n_dev].copy()
    holdout = df.iloc[n_dev:].copy()

    starts, chunk_ids = chunk_window_starts(
        len(dev),
        input_len=args.input_len,
        pred_len=args.pred_len,
        chunk_rows=args.chunk_rows,
    )
    train_starts, val_starts, train_chunks, val_chunks = purged_chunk_split(
        starts,
        chunk_ids,
        train_chunk_frac=train_chunk_frac,
        seed=args.seed,
    )

    row_tr = row_indices_covered_by_windows(train_starts, span, len(dev))
    train_mean = float(np.mean(dev[VALUE_COL].to_numpy()[row_tr]))
    train_std = float(np.std(dev[VALUE_COL].to_numpy()[row_tr])) + 1e-8

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dev_path = args.out_dir / "dev.csv"
    holdout_path = args.out_dir / "holdout.csv"
    dev.to_csv(dev_path, index=False)
    holdout.to_csv(holdout_path, index=False)

    manifest = {
        "mode": "purged_chunk_random_dev_temporal_holdout",
        "window_manifest_version": 1,
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
        },
    }
    manifest_path = args.out_dir / "window_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
