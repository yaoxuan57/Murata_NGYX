"""Fixed target for this deployment bundle."""

from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import DEPLOY_ROOT

# Canonical SENSOR_DESC in the vibration export CSV.
SENSOR_DESC = "AHU 2-9 Blower DE A"

# Checkpoint file stem (spaces → underscores, 2-9 → 2_9).
CHECKPOINT_STEM = "AHU_2_9_Blower_DE_A"

VALUE_COLUMN = "Acceleration RMS"

DEFAULT_INPUT_CSV = DEPLOY_ROOT / "data" / "vibration_May.csv"
DEFAULT_CHECKPOINT = DEPLOY_ROOT / "models" / f"{CHECKPOINT_STEM}.pth"
DEFAULT_OUTPUT_JSON = DEPLOY_ROOT / "output" / "predictions.json"
DEFAULT_PLOT_PATH = DEPLOY_ROOT / "output" / "plots" / f"{CHECKPOINT_STEM}_forecast.png"
