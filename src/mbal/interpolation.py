from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exceptions import PVTError

VALID_MODES = {"linear_pressure", "reciprocal_pressure"}
VALID_EXTRAPOLATION = {"linear", "error", "constant"}


def check_mode(mode: str) -> None:
    if mode not in VALID_MODES:
        raise PVTError(f"Unknown interpolation mode {mode!r}; valid modes are {sorted(VALID_MODES)}")


def check_extrapolation(extrapolation: str) -> None:
    if extrapolation not in VALID_EXTRAPOLATION:
        raise PVTError(
            f"Unknown extrapolation mode {extrapolation!r}; valid modes are {sorted(VALID_EXTRAPOLATION)}"
        )


@dataclass(frozen=True)
class PreparedTable:
    """A one-dimensional table transformed once for repeated interpolation.

    ``x`` is the interpolation abscissa (pressure or 1/pressure) in ascending order and
    ``y`` the corresponding values.  ``p_min``/``p_max`` are the pressure limits of the
    underlying data so that error messages can be reported in pressure units.
    """

    x: np.ndarray
    y: np.ndarray
    mode: str
    p_min: float
    p_max: float

    @classmethod
    def build(cls, pressure: np.ndarray, values: np.ndarray, mode: str) -> "PreparedTable":
        check_mode(mode)
        p = np.asarray(pressure, dtype=float)
        y = np.asarray(values, dtype=float)
        if p.ndim != 1 or y.ndim != 1 or len(p) != len(y):
            raise PVTError("Interpolation table must contain one-dimensional arrays of equal length")
        if len(p) < 2:
            raise PVTError("Interpolation requires at least two pressure/value pairs")
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(y))):
            raise PVTError("Interpolation table contains non-finite values")
        if np.any(np.diff(p) <= 0):
            raise PVTError("Interpolation pressures must be strictly increasing")
        if mode == "reciprocal_pressure":
            if p[0] <= 0:
                raise PVTError("Reciprocal-pressure interpolation requires positive table pressures")
            x = (1.0 / p)[::-1]
            y = y[::-1]
        else:
            x = p
        return cls(np.ascontiguousarray(x), np.ascontiguousarray(y), mode, float(p[0]), float(p[-1]))

    def __call__(self, pressure: float, extrapolation: str = "error", name: str = "property") -> float:
        if not np.isfinite(pressure):
            raise PVTError(f"Cannot evaluate {name} at non-finite pressure {pressure!r}")
        if self.mode == "reciprocal_pressure":
            if pressure <= 0.0:
                raise PVTError("Pressure must be positive for reciprocal-pressure interpolation")
            xq = 1.0 / pressure
        else:
            xq = pressure
        x, y = self.x, self.y
        if x[0] <= xq <= x[-1]:
            return float(np.interp(xq, x, y))
        if extrapolation == "error":
            raise PVTError(
                f"Pressure {pressure:g} psia lies outside the {name} table "
                f"[{self.p_min:g}, {self.p_max:g}] psia (pvt.interpolation.extrapolation is 'error')"
            )
        if extrapolation == "constant":
            return float(y[0] if xq < x[0] else y[-1])
        if extrapolation != "linear":
            check_extrapolation(extrapolation)
        if xq < x[0]:
            x1, x2, y1, y2 = x[0], x[1], y[0], y[1]
        else:
            x1, x2, y1, y2 = x[-2], x[-1], y[-2], y[-1]
        return float(y1 + (y2 - y1) * (xq - x1) / (x2 - x1))


def interpolate_1d(
    pressure: float,
    pressure_table: np.ndarray,
    values: np.ndarray,
    *,
    mode: str = "linear_pressure",
    extrapolation: str = "linear",
) -> float:
    """Interpolate a single value (kept for backward compatibility and tests)."""
    check_extrapolation(extrapolation)
    if pressure <= 0.0 and mode == "reciprocal_pressure":
        raise PVTError("Pressure must be positive for reciprocal-pressure interpolation")
    table = PreparedTable.build(pressure_table, values, mode)
    return table(pressure, extrapolation)
