"""Strict numeric CSV reading shared by the PVT, history and aquifer-influx inputs."""
from __future__ import annotations

import csv
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .exceptions import InputValidationError


@dataclass(frozen=True)
class NumericTable:
    path: Path
    columns: tuple[str, ...]
    _data: dict[str, np.ndarray]
    line_numbers: tuple[int, ...]

    @property
    def n_rows(self) -> int:
        return len(self.line_numbers)

    def column(self, name: str) -> np.ndarray:
        """Return a copy of a column; blank cells are NaN."""
        return self._data[name].copy()

    def has_blank(self, name: str) -> bool:
        return bool(np.any(np.isnan(self._data[name])))


#: Bounded LRU cache keyed by (path, mtime, size, kind); a changed file is simply a new key.
_CACHE: "OrderedDict[tuple[str, int, int, str], NumericTable]" = OrderedDict()
_CACHE_MAX = 32


def read_numeric_csv(path: str | Path, *, kind: str) -> NumericTable:
    """Read a CSV whose cells are numbers or blanks.

    Errors identify the file, line and column.  Fully blank rows are skipped.  Results are
    cached by path, modification time and size because history matching rebuilds the model
    many times from the same files.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise InputValidationError(f"{kind} file not found: {path}")
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size, kind)
    cached = _CACHE.get(key)
    if cached is not None:
        _CACHE.move_to_end(key)
        return cached

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise InputValidationError(f"{kind} file is empty: {path.name}") from None
        columns = [h.strip() for h in header]
        if any(c == "" for c in columns):
            raise InputValidationError(f"{kind} file {path.name} has an empty column name in its header")
        duplicated = sorted({c for c in columns if columns.count(c) > 1})
        if duplicated:
            raise InputValidationError(f"{kind} file {path.name} repeats column(s) {duplicated}")

        values: dict[str, list[float]] = {c: [] for c in columns}
        lines: list[int] = []
        for row in reader:
            line = reader.line_num
            cells = [cell.strip() for cell in row]
            if not any(cells):
                continue
            if len(cells) > len(columns) and any(cells[len(columns):]):
                raise InputValidationError(
                    f"{kind} file {path.name}, line {line}: row has {len(cells)} values but the header has {len(columns)} columns"
                )
            cells = (cells + [""] * len(columns))[: len(columns)]
            for col, cell in zip(columns, cells):
                if cell == "":
                    values[col].append(math.nan)
                    continue
                try:
                    number = float(cell)
                except ValueError:
                    raise InputValidationError(
                        f"{kind} file {path.name}, line {line}, column {col}: cannot read {cell!r} as a number"
                    ) from None
                if not math.isfinite(number):
                    raise InputValidationError(
                        f"{kind} file {path.name}, line {line}, column {col}: value {cell!r} is not finite"
                    )
                values[col].append(number)
            lines.append(line)

    if not lines:
        raise InputValidationError(f"{kind} file {path.name} has a header but no data rows")
    table = NumericTable(
        path=path,
        columns=tuple(columns),
        _data={c: np.asarray(v, dtype=float) for c, v in values.items()},
        line_numbers=tuple(lines),
    )
    _CACHE[key] = table
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return table


def clear_cache() -> None:
    _CACHE.clear()
