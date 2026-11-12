"""Lightweight CSV reader compatible with the pandas calls in this repository.

Avoids binary-ABI failures when pandas is absent or compiled against an older NumPy,
while keeping attribute access, multi-column slicing, boolean indexing, and basic
summary statistics identical for all agents, plotting scripts, and tests.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def _try_float(value: str):
    if value == "" or value.lower() in {"nan", "na", "null", "none"}:
        return np.nan
    try:
        num = float(value)
        if num.is_integer() and "." not in value and "e" not in value.lower():
            return int(value)
        return num
    except ValueError:
        return value


class _ILocIndexer:
    def __init__(self, array: np.ndarray):
        self._array = array

    def __getitem__(self, idx):
        val = self._array[idx]
        return val.item() if hasattr(val, "item") else val


class _NullFrame:
    def __init__(self, has_nulls: bool):
        self._has_nulls = has_nulls

    def any(self):
        return self

    def __bool__(self) -> bool:
        return self._has_nulls


class CsvColumn:
    """Minimal 1D column wrapper supporting Series methods used in this repo."""

    def __init__(self, values: np.ndarray, name: str = ""):
        self._data = np.asarray(values)
        self.name = name
        self.iloc = _ILocIndexer(self._data)

    @property
    def values(self) -> np.ndarray:
        return self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def unique(self) -> np.ndarray:
        seen = []
        seen_set = set()
        for item in self._data.tolist():
            if item not in seen_set:
                seen_set.add(item)
                seen.append(item)
        return np.asarray(seen, dtype=self._data.dtype)

    def mean(self) -> float:
        return float(np.mean(self._data))

    def std(self, ddof: int = 1) -> float:
        return float(np.std(self._data, ddof=ddof))

    def min(self) -> float:
        return float(np.min(self._data))

    def max(self) -> float:
        return float(np.max(self._data))

    def tolist(self) -> list:
        return self._data.tolist()

    @property
    def is_unique(self) -> bool:
        items = self._data.tolist()
        return len(set(items)) == len(items)

    def __eq__(self, other):  # type: ignore[override]
        return self._data == other

    def __ne__(self, other):  # type: ignore[override]
        return self._data != other

    def __gt__(self, other):
        return self._data > other

    def __ge__(self, other):
        return self._data >= other

    def __lt__(self, other):
        return self._data < other

    def __le__(self, other):
        return self._data <= other


class CsvTable:
    """Minimal 2D table wrapper supporting DataFrame indexing used in this repo."""

    def __init__(
        self,
        columns: list[str],
        data_dict: dict[str, np.ndarray],
        row_index: np.ndarray | None = None,
    ):
        self.columns = list(columns)
        self._cols = {col: np.asarray(data_dict[col]) for col in self.columns}
        self._nrows = len(next(iter(self._cols.values()))) if self.columns else 0
        self._row_index = (
            np.asarray(row_index, dtype=np.int64)
            if row_index is not None
            else np.arange(self._nrows, dtype=np.int64)
        )

    @property
    def index(self) -> np.ndarray:
        return self._row_index

    def __len__(self) -> int:
        return self._nrows

    @property
    def values(self) -> np.ndarray:
        if not self.columns:
            return np.empty((0, 0), dtype=np.float32)
        return np.column_stack([self._cols[c] for c in self.columns])

    def __getitem__(self, key):
        if isinstance(key, str):
            return CsvColumn(self._cols[key], name=key)
        if isinstance(key, (list, tuple)):
            return CsvTable(list(key), {k: self._cols[k] for k in key}, self._row_index)
        mask = np.asarray(key)
        return CsvTable(
            self.columns,
            {k: v[mask] for k, v in self._cols.items()},
            self._row_index[mask],
        )

    def __getattr__(self, name: str):
        cols = object.__getattribute__(self, "_cols")
        if name in cols:
            return CsvColumn(cols[name], name=name)
        raise AttributeError(f"'CsvTable' object has no attribute '{name}'")

    def sample(self, n: int, random_state: int | None = None) -> "CsvTable":
        rng = np.random.default_rng(random_state)
        indices = rng.choice(self._nrows, size=min(n, self._nrows), replace=False)
        return CsvTable(
            self.columns,
            {k: v[indices] for k, v in self._cols.items()},
            self._row_index[indices],
        )

    def reset_index(self, drop: bool = True) -> "CsvTable":
        return CsvTable(self.columns, self._cols)

    def isnull(self) -> _NullFrame:
        has_nulls = False
        for arr in self._cols.values():
            if np.issubdtype(arr.dtype, np.floating) and np.isnan(arr).any():
                has_nulls = True
                break
        return _NullFrame(has_nulls)


def read_csv(path: str | Path) -> CsvTable:
    """Read CSV file into a lightweight CsvTable without requiring C extensions."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols: list[list] = [[] for _ in header]
        for row in reader:
            if not row:
                continue
            for idx, raw in enumerate(row):
                cols[idx].append(_try_float(raw))

    data_dict: dict[str, np.ndarray] = {}
    clean_cols: list[str] = []
    for i, name in enumerate(header):
        col_name = name if name != "" else f"Unnamed: {i}"
        clean_cols.append(col_name)
        col_vals = cols[i]
        if all(isinstance(v, (int, np.integer)) for v in col_vals):
            arr = np.asarray(col_vals, dtype=np.int64)
        elif all(isinstance(v, (int, float, np.floating, np.integer)) for v in col_vals):
            arr = np.asarray(col_vals, dtype=np.float64)
        else:
            arr = np.asarray(col_vals, dtype=object)
        data_dict[col_name] = arr
    return CsvTable(clean_cols, data_dict)
