"""JSON serialization helpers shared by dataset and experiment reports."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


def json_default(value: Any) -> Any:
    """Convert common scientific/Python objects into JSON-compatible values.

    Raises TypeError for unsupported objects rather than silently stringifying them.
    """
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
