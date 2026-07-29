#!/usr/bin/env python3
"""Build sensor_id_name_mapping.csv from unique sensors in a vibration export CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model_meta import build_sensor_mapping_dataframe, default_sensor_mapping_path, write_sensor_mapping_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a vibration CSV and write sensor_id_name_mapping.csv "
            "(one row per unique SENSOR_NAME / SENSOR_DESC + SENSOR_CODE)."
        ),
    )
    parser.add_argument(
        "source_csv",
        type=Path,
        help="Input vibration CSV (single- or multi-sensor export).",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help=f"Output mapping CSV (default: {default_sensor_mapping_path()}).",
    )
    args = parser.parse_args()

    out = write_sensor_mapping_csv(args.source_csv, args.out)
    mapping_df = build_sensor_mapping_dataframe(args.source_csv)
    print(f"Wrote {len(mapping_df)} sensor(s) -> {out}")
    print(mapping_df.to_string(index=False))


if __name__ == "__main__":
    main()
