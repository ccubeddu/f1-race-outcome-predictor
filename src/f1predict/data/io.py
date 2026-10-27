from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

import pandas as pd


def atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_text_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_parquet_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".parquet") as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_partitioned_table(root: Path, table: str) -> pd.DataFrame:
    files = sorted((root / table).glob("season=*/data.parquet"))
    return read_parquet_files(files)


def read_parquet_files(files: list[Path]) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()
    frames = [frame for path in files if not (frame := pd.read_parquet(path)).empty]
    if not frames:
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA entries",
            category=FutureWarning,
        )
        return pd.concat(frames, ignore_index=True)
