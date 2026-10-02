"""History input, per-step results and CSV output."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .exceptions import InputValidationError
from .tabular import read_numeric_csv


@dataclass(frozen=True)
class HistoryRecord:
    """One row of cumulative production/injection history (SPE symbols, field units).

    ``Np`` [STB], ``Gp`` [scf], ``Wp`` [STB], ``Winj`` [STB], ``Ginj`` [scf] are cumulative from
    t = 0.  ``pressure_psia`` is the observed average reservoir pressure (``None`` = not measured).
    """

    time_days: float
    np_stb: float = 0.0
    gp_scf: float = 0.0
    wp_stb: float = 0.0
    winj_stb: float = 0.0
    ginj_scf: float = 0.0
    pressure_psia: float | None = None
    pressure_sigma_psia: float | None = None
    pressure_weight: float = 1.0


@dataclass
class SimulationRow:
    time_days: float
    pressure_psia: float
    pressure_observed_psia: float | None
    water_influx_rb: float
    aquifer_pressure_psia: float | None
    underground_withdrawal_rb: float
    hydrocarbon_expansion_rb: float
    oil_expansion_rb: float
    gas_expansion_rb: float
    gas_cap_expansion_rb: float
    rock_water_expansion_rb: float
    water_injection_rb: float
    gas_injection_rb: float
    material_balance_residual_rb: float
    closure_relative: float
    extras: dict[str, float] = field(default_factory=dict)

    @property
    def total_injection_rb(self) -> float:
        return self.water_injection_rb + self.gas_injection_rb

    @property
    def total_support_rb(self) -> float:
        return (
            self.hydrocarbon_expansion_rb
            + self.rock_water_expansion_rb
            + self.water_influx_rb
            + self.total_injection_rb
        )


@dataclass
class SimulationResult:
    rows: list[SimulationRow]
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def pressures(self) -> list[float]:
        return [r.pressure_psia for r in self.rows]

    def observed_pairs(self) -> list[tuple[float, float, float]]:
        out: list[tuple[float, float, float]] = []
        for r in self.rows:
            if r.pressure_observed_psia is not None:
                out.append((r.time_days, r.pressure_observed_psia, r.pressure_psia))
        return out

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "time_days",
            "pressure_psia",
            "pressure_observed_psia",
            "water_influx_rb",
            "aquifer_pressure_psia",
            "underground_withdrawal_rb",
            "hydrocarbon_expansion_rb",
            "oil_expansion_rb",
            "gas_expansion_rb",
            "gas_cap_expansion_rb",
            "rock_water_expansion_rb",
            "water_injection_rb",
            "gas_injection_rb",
            "material_balance_residual_rb",
            "closure_relative",
        ]
        extra_keys = sorted({k for r in self.rows for k in r.extras})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields + extra_keys)
            writer.writeheader()
            for r in self.rows:
                row = {name: getattr(r, name) for name in fields}
                row.update(r.extras)
                writer.writerow(row)


HISTORY_GAS_COLUMNS = {
    "gp": [("Gp_scf", 1.0), ("Gp_Mscf", 1.0e3), ("Gp_MMscf", 1.0e6)],
    "ginj": [("Ginj_scf", 1.0), ("Ginj_Mscf", 1.0e3), ("Ginj_MMscf", 1.0e6)],
}
HISTORY_SIMPLE_COLUMNS = (
    "time_days",
    "Np_stb",
    "Wp_stb",
    "Winj_stb",
    "Rp_scf_per_stb",
    "pressure_psia",
    "pressure_sigma_psia",
    "pressure_weight",
)
HISTORY_INFORMATIONAL_PREFIXES = ("note", "comment", "date", "well")


def _gas_column(table, path: Path, kind: str) -> tuple[np.ndarray | None, str | None]:
    present = [(c, s) for c, s in HISTORY_GAS_COLUMNS[kind] if c in table.columns]
    if len(present) > 1:
        raise InputValidationError(
            f"History CSV {path.name} gives {kind} in more than one unit ({[c for c, _ in present]}); keep one column"
        )
    if not present:
        return None, None
    col, scale = present[0]
    return np.nan_to_num(table.column(col), nan=0.0) * scale, col


def read_history_csv(path: str | Path, *, warnings: list[str] | None = None) -> list[HistoryRecord]:
    """Read cumulative production/injection history.

    Gas may be supplied in scf, Mscf or MMscf (``Gp_scf``/``Gp_Mscf``/``Gp_MMscf`` and the
    matching ``Ginj_*`` columns), or as a cumulative producing gas-oil ratio
    ``Rp_scf_per_stb`` from which ``Gp = Np * Rp``.  Blank cumulative cells are read as zero;
    a blank observed pressure means "not measured".  The internal gas unit is scf.
    """
    path = Path(path)
    table = read_numeric_csv(path, kind="History")
    cols = set(table.columns)
    if "time_days" not in cols:
        raise InputValidationError(f"History CSV {path.name} requires a time_days column")
    if table.has_blank("time_days"):
        line = table.line_numbers[int(np.flatnonzero(np.isnan(table.column("time_days")))[0])]
        raise InputValidationError(f"History CSV {path.name}, line {line}: time_days is blank")

    def col(name: str) -> np.ndarray:
        if name in cols:
            return np.nan_to_num(table.column(name), nan=0.0)
        return np.zeros(table.n_rows)

    np_stb = col("Np_stb")
    gp, gp_col = _gas_column(table, path, "gp")
    ginj, _ = _gas_column(table, path, "ginj")
    if "Rp_scf_per_stb" in cols:
        rp_raw = table.column("Rp_scf_per_stb")
        missing = (np_stb > 0) & np.isnan(rp_raw)
        if np.any(missing):
            line = table.line_numbers[int(np.flatnonzero(missing)[0])]
            raise InputValidationError(f"History CSV {path.name}, line {line}: Rp_scf_per_stb is blank while Np_stb > 0")
        gp_from_rp = np_stb * np.nan_to_num(rp_raw, nan=0.0)
        if gp is not None:
            scale = np.maximum(np.abs(gp), 1.0)
            bad = np.abs(gp - gp_from_rp) > 1e-6 * scale
            if np.any(bad):
                line = table.line_numbers[int(np.flatnonzero(bad)[0])]
                raise InputValidationError(
                    f"History CSV {path.name}, line {line}: {gp_col} is inconsistent with Np_stb * Rp_scf_per_stb; give only one"
                )
        gp = gp_from_rp
    if gp is None:
        gp = np.zeros(table.n_rows)
    if ginj is None:
        ginj = np.zeros(table.n_rows)

    pressure = table.column("pressure_psia") if "pressure_psia" in cols else np.full(table.n_rows, np.nan)
    sigma = table.column("pressure_sigma_psia") if "pressure_sigma_psia" in cols else np.full(table.n_rows, np.nan)
    weight = table.column("pressure_weight") if "pressure_weight" in cols else np.full(table.n_rows, np.nan)

    known = set(HISTORY_SIMPLE_COLUMNS) | {c for v in HISTORY_GAS_COLUMNS.values() for c, _ in v}
    unknown = [c for c in table.columns if c not in known and not c.lower().startswith(HISTORY_INFORMATIONAL_PREFIXES)]
    if unknown and warnings is not None:
        warnings.append(f"History CSV {path.name}: ignored unrecognised column(s) {unknown}")

    # Build every column once (the previous per-row column copies made this O(n^2)).
    time_days = table.column("time_days")
    wp, winj = col("Wp_stb"), col("Winj_stb")
    records = [
        HistoryRecord(
            time_days=float(t), np_stb=float(n), gp_scf=float(g), wp_stb=float(w),
            winj_stb=float(wi), ginj_scf=float(gi),
            pressure_psia=None if math.isnan(p) else float(p),
            pressure_sigma_psia=None if math.isnan(sd) else float(sd),
            pressure_weight=1.0 if math.isnan(wt) else float(wt),
        )
        for t, n, g, w, wi, gi, p, sd, wt in zip(time_days, np_stb, gp, wp, winj, ginj, pressure, sigma, weight)
    ]

    try:
        validate_history(records)
    except InputValidationError as exc:
        raise InputValidationError(f"History CSV {path.name}: {exc}") from exc
    return records


CUMULATIVE_FIELDS = (
    ("np_stb", "Np_stb"),
    ("gp_scf", "Gp"),
    ("wp_stb", "Wp_stb"),
    ("winj_stb", "Winj_stb"),
    ("ginj_scf", "Ginj"),
)


def validate_history(records: Iterable[HistoryRecord]) -> None:
    records = list(records)
    if not records:
        raise InputValidationError("At least one history record is required")
    previous = None
    for i, row in enumerate(records):
        if not math.isfinite(row.time_days) or row.time_days < 0:
            raise InputValidationError(f"Time must be finite and non-negative (history row {i + 1})")
        for attr, label in CUMULATIVE_FIELDS:
            value = getattr(row, attr)
            if not math.isfinite(value) or value < 0:
                raise InputValidationError(f"{label} must be finite and non-negative at time {row.time_days:g} days")
        if row.pressure_psia is not None and not (math.isfinite(row.pressure_psia) and row.pressure_psia > 0):
            raise InputValidationError(f"Observed pressure must be positive at time {row.time_days:g} days")
        if row.pressure_sigma_psia is not None and not (math.isfinite(row.pressure_sigma_psia) and row.pressure_sigma_psia > 0):
            raise InputValidationError(f"pressure_sigma_psia must be positive at time {row.time_days:g} days")
        if not (math.isfinite(row.pressure_weight) and row.pressure_weight >= 0):
            raise InputValidationError(f"pressure_weight must be non-negative at time {row.time_days:g} days")
        if previous is not None:
            if row.time_days <= previous.time_days:
                raise InputValidationError(
                    f"History time must be strictly increasing (time {row.time_days:g} days follows {previous.time_days:g})"
                )
            for attr, label in CUMULATIVE_FIELDS:
                now, before = getattr(row, attr), getattr(previous, attr)
                if now + 1e-9 * max(abs(before), 1.0) < before:
                    raise InputValidationError(
                        f"Cumulative {label} decreases at time {row.time_days:g} days "
                        f"({before:g} -> {now:g}; blank cells are read as zero)"
                    )
        previous = row
