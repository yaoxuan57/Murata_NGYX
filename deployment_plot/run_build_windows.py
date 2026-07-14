#!/usr/bin/env python3
"""Deprecated: use ``deployment/run_inference.py`` (single JSON output)."""

from __future__ import annotations

import sys


def main() -> None:
    print(
        "run_build_windows.py is deprecated.\n"
        "Use:  python deployment/run_inference.py --sensor \"...\" --checkpoint path/to/model.pth\n"
        "That script filters the multi-sensor CSV, preprocesses in memory, and writes one predictions JSON.",
        file=sys.stderr,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
