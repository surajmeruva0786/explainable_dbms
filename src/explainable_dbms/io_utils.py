"""
Helper utilities for reading/writing pipeline artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Persist dictionary data as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True, default=_json_serializer)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON content from disk if the file exists."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Write a dataframe to CSV for quick inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _json_serializer(obj: Any) -> Any:
    """Fallback serializer for NumPy/pandas types."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

