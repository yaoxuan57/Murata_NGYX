"""Ensure deployment/ is on sys.path for flat local imports."""

from __future__ import annotations

import sys
from pathlib import Path

_DEPLOY_ROOT = Path(__file__).resolve().parent
if str(_DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_ROOT))

DEPLOY_ROOT = _DEPLOY_ROOT
