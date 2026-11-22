from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    """Sets seeds for reproducibility across Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_directories(paths: list[Path]) -> None:
    """Creates directories if they are missing."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _to_json_ready(value: Any) -> Any:
    """Converts common Python, NumPy, pandas, and dataclass values to JSON."""

    if is_dataclass(value):
        return _to_json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return value


def save_json(data: Any, path: Path) -> None:
    """Saves JSON with consistent formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_to_json_ready(data), file, indent=2, ensure_ascii=False)


def save_dataframe(dataframe: pd.DataFrame, path: Path) -> None:
    """Saves a DataFrame as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
