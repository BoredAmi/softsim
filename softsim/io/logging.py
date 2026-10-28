"""Tabular output helpers (Parquet preferred, CSV fallback)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_table(df: pd.DataFrame, path: str | Path, fmt: str = "parquet") -> Path:
    """Write ``df`` to ``path`` (suffix set from ``fmt``); falls back to CSV without pyarrow."""
    path = Path(path).with_suffix(f".{fmt}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except ImportError:
            path = path.with_suffix(".csv")
    df.to_csv(path, index=False)
    return path


def load_table(path: str | Path) -> pd.DataFrame:
    """Read a table written by :func:`save_table`."""
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
