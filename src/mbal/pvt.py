"""Tabulated PVT properties, unit handling, interpolation at the saturation pressure and sanity checks."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .exceptions import InputValidationError, PVTError
from .interpolation import PreparedTable, check_extrapolation, check_mode
from .tabular import read_numeric_csv

#: Bt mode that evaluates Bo + (Rsi - Rs) Bg from separately interpolated properties instead of
#: interpolating a tabulated Bt (the convention of most spreadsheet/R material-balance tools).
BT_FROM_COMPONENTS = "components"


@dataclass(frozen=True)
class PVTInterpolation:
    bo: str = "linear_pressure"
    rs: str = "linear_pressure"
    # Bg = 0.00504 z T / p is linear in 1/p when z is linear in p, so reciprocal-pressure
    # interpolation is exact for constant z and removes the convexity bias of linear interpolation.
    bg: str = "reciprocal_pressure"
    rv: str = "linear_pressure"
    bw: str = "linear_pressure"
    bt_below_saturation: str = "linear_pressure"
    bt_above_saturation: str = "linear_pressure"
    extrapolation: str = "error"
    # Interpolate saturated (p < Psat) and undersaturated (p >= Psat) states from their own
    # rows so that a table segment straddling the saturation pressure is never used.
    split_at_saturation: bool = True

    def __post_init__(self):
        for mode in (self.bo, self.rs, self.bg, self.rv, self.bw):
            check_mode(mode)
        for mode in (self.bt_below_saturation, self.bt_above_saturation):
            if mode != BT_FROM_COMPONENTS:
                check_mode(mode)
        check_extrapolation(self.extrapolation)


# Accepted CSV columns: internal property -> [(column name, multiplier to internal unit)].
PVT_COLUMN_UNITS: dict[str, list[tuple[str, float]]] = {
    "bo": [("Bo_rb_per_stb", 1.0)],
    "rs": [("Rs_scf_per_stb", 1.0)],
    "bg": [("Bg_rb_per_scf", 1.0), ("Bg_rb_per_Mscf", 1.0e-3)],
    "rv": [("Rv_stb_per_scf", 1.0), ("Rv_stb_per_Mscf", 1.0e-3), ("Rv_stb_per_MMscf", 1.0e-6)],
    "bw": [("Bw_rb_per_stb", 1.0)],
    "z": [("z", 1.0)],
}
# Informational columns that are accepted but not used by the material balance.
PVT_INFORMATIONAL_PREFIXES = ("mu", "visc", "density", "rho", "note", "comment", "liquid_volume")

PROPERTY_LABELS = {"bo": "Bo", "rs": "Rs", "bg": "Bg", "rv": "Rv", "bw": "Bw", "z": "z"}


class PVTTable:
    """Pressure-dependent PVT data in a consistent internal unit system.

    Internal units:
      * pressure: psia
      * Bo, Bw: rb/STB
      * Bg: rb/scf
      * Rs: scf/STB
      * Rv: STB/scf

    A property may be undefined (NaN) over a contiguous part of the pressure range, for
    example Bo/Rs above the dew point of a gas condensate.  When a saturation pressure is
    assigned and ``split_at_saturation`` is enabled, values below the saturation pressure are
    interpolated only from rows at or below it, and values at or above it only from rows at or
    above it; the short gap between the last row of a branch and the saturation pressure is
    bridged by linear extrapolation of that branch.
    """

    def __init__(
        self,
        pressure_psia: np.ndarray,
        *,
        bo_rb_per_stb: np.ndarray | None = None,
        rs_scf_per_stb: np.ndarray | None = None,
        bg_rb_per_scf: np.ndarray | None = None,
        rv_stb_per_scf: np.ndarray | None = None,
        bw_rb_per_stb: np.ndarray | None = None,
        z: np.ndarray | None = None,
        interpolation: PVTInterpolation | None = None,
        saturation_pressure_psia: float | None = None,
        source: str | None = None,
    ):
        raw_p = np.asarray(pressure_psia, dtype=float)
        if raw_p.ndim != 1 or len(raw_p) < 2:
            raise PVTError("PVT pressure values must contain at least two points")
        if not np.all(np.isfinite(raw_p)):
            raise PVTError("PVT pressures must be finite numbers")
        if np.any(raw_p <= 0):
            raise PVTError("PVT pressures must be positive (psia)")
        order = np.argsort(raw_p, kind="stable")
        self.pressure = raw_p[order]
        duplicates = self.pressure[1:][np.diff(self.pressure) <= 0]
        if duplicates.size:
            raise PVTError(f"PVT pressures must be unique; duplicated pressure(s): {sorted(set(duplicates.tolist()))}")
        self.source = source

        def sort_optional(a, name):
            if a is None:
                return None
            arr = np.asarray(a, dtype=float)
            if arr.shape != raw_p.shape:
                raise PVTError("All PVT columns must have the same number of rows")
            arr = arr[order]
            if np.any(np.isinf(arr)):
                raise PVTError(f"PVT column {PROPERTY_LABELS[name]} contains infinite values")
            finite = np.isfinite(arr)
            if finite.sum() < 2:
                raise PVTError(f"PVT column {PROPERTY_LABELS[name]} needs at least two defined values")
            idx = np.flatnonzero(finite)
            if idx[-1] - idx[0] + 1 != idx.size:
                gaps = self.pressure[idx[0]: idx[-1] + 1][~finite[idx[0]: idx[-1] + 1]]
                raise PVTError(
                    f"PVT column {PROPERTY_LABELS[name]} has blank values inside its defined pressure range "
                    f"(at {gaps.tolist()} psia); blanks are only allowed at the top or bottom of the table"
                )
            return arr

        self.bo = sort_optional(bo_rb_per_stb, "bo")
        self.rs = sort_optional(rs_scf_per_stb, "rs")
        self.bg = sort_optional(bg_rb_per_scf, "bg")
        self.rv = sort_optional(rv_stb_per_scf, "rv")
        self.bw = sort_optional(bw_rb_per_stb, "bw")
        self.z = sort_optional(z, "z")
        self.interpolation = interpolation or PVTInterpolation()
        self._saturation_pressure: float | None = None
        self._tables: dict[tuple, PreparedTable] = {}
        self._bt_tables: dict[tuple, PreparedTable] = {}
        self.saturation_pressure_psia = saturation_pressure_psia

    # ------------------------------------------------------------------ configuration
    @property
    def saturation_pressure_psia(self) -> float | None:
        return self._saturation_pressure

    @saturation_pressure_psia.setter
    def saturation_pressure_psia(self, value: float | None) -> None:
        if value is not None:
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise PVTError("Saturation pressure must be a finite, non-negative number")
            if value == 0.0:
                value = None
        self._saturation_pressure = value
        self._tables.clear()
        self._bt_tables.clear()

    def with_interpolation(self, interpolation: PVTInterpolation) -> "PVTTable":
        clone = object.__new__(PVTTable)
        clone.__dict__.update(self.__dict__)
        clone.interpolation = interpolation
        clone._tables = {}
        clone._bt_tables = {}
        return clone

    @property
    def p_min(self) -> float:
        return float(self.pressure[0])

    @property
    def p_max(self) -> float:
        return float(self.pressure[-1])

    def has(self, name: str) -> bool:
        return getattr(self, name) is not None

    # ------------------------------------------------------------------ interpolation core
    def _split_active(self, sat: float | None, p_rows: np.ndarray) -> bool:
        return (
            sat is not None
            and self.interpolation.split_at_saturation
            and p_rows[0] < sat < p_rows[-1]
        )

    def _branch_rows(self, p_rows: np.ndarray, y_rows: np.ndarray, side: str | None, sat: float | None):
        if side is None or not self._split_active(sat, p_rows):
            return p_rows, y_rows
        mask = p_rows <= sat if side == "below" else p_rows >= sat
        if int(mask.sum()) >= 2:
            return p_rows[mask], y_rows[mask]
        return p_rows, y_rows

    def _side(self, p: float, sat: float | None) -> str | None:
        if sat is None or not self.interpolation.split_at_saturation:
            return None
        return "below" if p < sat else "above"

    def _extrapolation_for(self, p: float) -> str:
        # Bridging the gap between a branch and the saturation pressure, or between a partially
        # defined column and the table limits, is allowed: the pressure is inside the table.
        if self.p_min <= p <= self.p_max:
            return "linear"
        return self.interpolation.extrapolation

    def _value(self, p: float, name: str, mode: str) -> float:
        values = getattr(self, name)
        if values is None:
            raise PVTError(f"PVT table does not contain {PROPERTY_LABELS[name]}")
        sat = self._saturation_pressure
        side = self._side(p, sat)
        key = (name, side, mode)
        table = self._tables.get(key)
        if table is None:
            finite = np.isfinite(values)
            p_rows, y_rows = self._branch_rows(self.pressure[finite], values[finite], side, sat)
            table = PreparedTable.build(p_rows, y_rows, mode)
            self._tables[key] = table
        return table(p, self._extrapolation_for(p), PROPERTY_LABELS[name])

    def Bo(self, p: float) -> float:
        return self._value(p, "bo", self.interpolation.bo)

    def Rs(self, p: float) -> float:
        return self._value(p, "rs", self.interpolation.rs)

    def Bg(self, p: float) -> float:
        return self._value(p, "bg", self.interpolation.bg)

    def Rv(self, p: float) -> float:
        return self._value(p, "rv", self.interpolation.rv)

    def Bw(self, p: float, default: float = 1.0) -> float:
        if self.bw is None:
            return default
        return self._value(p, "bw", self.interpolation.bw)

    def Z(self, p: float) -> float:
        return self._value(p, "z", "linear_pressure")

    # ------------------------------------------------------------------ derived properties
    def Bt(self, p: float, *, saturation_pressure_psia: float, rsi_scf_per_stb: float) -> float:
        """Conventional two-phase oil FVF, Bo + (Rsi - Rs) Bg below the bubble point."""
        if self.bo is None or self.bg is None:
            raise PVTError("Bt requires Bo and Bg columns")
        sat = float(saturation_pressure_psia)
        rsi = float(rsi_scf_per_stb)
        below_sat = p < sat
        if below_sat and self.rs is None:
            raise PVTError("Bt below the bubble point requires an Rs column (Bt = Bo + (Rsi - Rs) Bg)")
        mode = self.interpolation.bt_below_saturation if below_sat else self.interpolation.bt_above_saturation
        split = self.interpolation.split_at_saturation
        side = ("below" if below_sat else "above") if split else None
        if mode == BT_FROM_COMPONENTS:
            if not below_sat:
                return self.Bo(p)
            rs = self.Rs(p)
            return self.Bo(p) + (rsi - rs) * self.Bg(p)
        key = (sat, rsi, side, mode)
        table = self._bt_tables.get(key)
        if table is None:
            rs_rows = self.rs if self.rs is not None else np.full_like(self.pressure, rsi)
            two_phase_bt = self.bo + (rsi - rs_rows) * self.bg
            if split:
                if side == "below":
                    rows_bt = two_phase_bt
                    use_rows = self.pressure <= sat
                else:
                    rows_bt = self.bo.copy()
                    use_rows = self.pressure >= sat
                valid = np.isfinite(rows_bt)
                candidate = valid & use_rows
                if int(candidate.sum()) < 2:
                    # Fall back to the conventional single table when one branch has too few rows.
                    rows_bt = np.where(self.pressure < sat, two_phase_bt, self.bo)
                    candidate = np.isfinite(rows_bt)
            else:
                rows_bt = np.where(self.pressure < sat, two_phase_bt, self.bo)
                candidate = np.isfinite(rows_bt)
            table = PreparedTable.build(self.pressure[candidate], rows_bt[candidate], mode)
            self._bt_tables[key] = table
        return table(p, self._extrapolation_for(p), "Bt")

    def _modified_black_oil_state(self, p: float) -> tuple[float, float, float, float, float]:
        bo, bg, rs, rv = self.Bo(p), self.Bg(p), self.Rs(p), self.Rv(p)
        den = 1.0 - rs * rv
        if den <= 1e-12:
            raise PVTError(
                f"Invalid modified-black-oil state at {p:g} psia: 1 - Rs*Rv = {den:.3e} <= 0 "
                f"(Rs={rs:g} scf/STB, Rv={rv:.4g} STB/scf)"
            )
        return bo, bg, rs, rv, den

    def generalized_Bto(self, p: float, *, rsi_scf_per_stb: float) -> float:
        bo, bg, rs, rv, den = self._modified_black_oil_state(p)
        return (bo * (1.0 - rsi_scf_per_stb * rv) + bg * (rsi_scf_per_stb - rs)) / den

    def generalized_Btg(self, p: float, *, rvi_stb_per_scf: float) -> float:
        bo, bg, rs, rv, den = self._modified_black_oil_state(p)
        return (bg * (1.0 - rvi_stb_per_scf * rs) + bo * (rvi_stb_per_scf - rv)) / den

    def generalized_withdrawal_coefficients(self, p: float) -> tuple[float, float]:
        """Return reservoir-volume coefficients multiplying Np and Gp.

        This is the generalized two-component withdrawal expression:
        oil coefficient [rb/STB], gas coefficient [rb/scf].
        """
        bo, bg, rs, rv, den = self._modified_black_oil_state(p)
        return (bo - rs * bg) / den, (bg - rv * bo) / den

    def saturated_value(self, name: str, saturation_pressure_psia: float) -> float:
        """Property on the *saturated* (two-phase) branch evaluated at the saturation pressure.

        With ``split_at_saturation`` the two-phase rows are extrapolated up to Psat, so this is the
        limit p -> Psat from below, i.e. the equilibrium-phase property at saturation.
        """
        p = float(saturation_pressure_psia) * (1.0 - 1e-12)
        if self.p_min * (1.0 - 1e-9) <= float(saturation_pressure_psia) <= self.p_max * (1.0 + 1e-9):
            p = min(max(p, self.p_min), self.p_max)
        return {"bo": self.Bo, "rs": self.Rs, "bg": self.Bg, "rv": self.Rv}[name](p)


# ---------------------------------------------------------------------- CSV input
def read_pvt_csv(
    path: str | Path,
    interpolation: PVTInterpolation | None = None,
    *,
    saturation_pressure_psia: float | None = None,
    warnings: list[str] | None = None,
    temperature_f: float | None = None,
) -> PVTTable:
    """Read a PVT CSV into a :class:`PVTTable` (internal units, see the class docstring).

    If the table gives ``z`` but no Bg column, Bg is derived as ``0.00504 z T / p`` [rb/scf], which
    requires the reservoir temperature ``temperature_f`` [degF].  If both are given, Bg is checked
    against z and a warning is issued above 1 % disagreement.
    """
    path = Path(path)
    table = read_numeric_csv(path, kind="PVT")
    if "pressure_psia" not in table.columns:
        raise InputValidationError(f"PVT CSV {path.name} requires a pressure_psia column")
    if table.has_blank("pressure_psia"):
        raise InputValidationError(f"PVT CSV {path.name}: every row needs a pressure_psia value")

    used = {"pressure_psia"}
    arrays: dict[str, np.ndarray | None] = {}
    for prop, options in PVT_COLUMN_UNITS.items():
        present = [(col, scale) for col, scale in options if col in table.columns]
        if len(present) > 1:
            names = ", ".join(c for c, _ in present)
            raise InputValidationError(
                f"PVT CSV {path.name} gives {PROPERTY_LABELS[prop]} in more than one unit ({names}); keep one column"
            )
        if present:
            col, scale = present[0]
            used.add(col)
            arrays[prop] = table.column(col) * scale
        else:
            arrays[prop] = None

    unknown = [
        c for c in table.columns
        if c not in used and not c.lower().startswith(PVT_INFORMATIONAL_PREFIXES)
    ]
    if unknown and warnings is not None:
        warnings.append(f"PVT CSV {path.name}: ignored unrecognised column(s) {unknown}")

    pressure = table.column("pressure_psia")
    if arrays["z"] is not None and temperature_f is not None:
        z = arrays["z"]
        if np.any(np.isfinite(z) & (z <= 0)):
            raise InputValidationError(f"PVT CSV {path.name}: z must be positive")
        bg_z = bg_from_z_field_units(pressure, z, temperature_f)
        if arrays["bg"] is None:
            arrays["bg"] = bg_z
        elif warnings is not None:
            both = np.isfinite(arrays["bg"]) & np.isfinite(bg_z)
            if np.any(both):
                worst = float(np.max(np.abs(arrays["bg"][both] / bg_z[both] - 1.0)))
                if worst > 0.01:
                    warnings.append(
                        f"PVT CSV {path.name}: Bg differs from 0.00504 z T / p by up to {100 * worst:.1f}%; "
                        "the Bg column is used"
                    )
    elif arrays["z"] is not None and arrays["bg"] is None:
        raise InputValidationError(
            f"PVT CSV {path.name} gives z but no Bg; set reservoir.T_degF so Bg = 0.00504 z T / p can be derived"
        )

    try:
        return PVTTable(
            pressure,
            bo_rb_per_stb=arrays["bo"],
            rs_scf_per_stb=arrays["rs"],
            bg_rb_per_scf=arrays["bg"],
            rv_stb_per_scf=arrays["rv"],
            bw_rb_per_stb=arrays["bw"],
            z=arrays["z"],
            interpolation=interpolation,
            saturation_pressure_psia=saturation_pressure_psia,
            source=str(path),
        )
    except PVTError as exc:
        raise InputValidationError(f"PVT CSV {path.name}: {exc}") from exc


# ---------------------------------------------------------------------- sanity checks
FLUID_SYSTEMS = {"dry_gas", "black_oil", "volatile_oil", "gas_condensate", "two_phase"}

REQUIRED_PROPERTIES = {
    "dry_gas": ("bg",),
    "black_oil": ("bo", "rs", "bg"),
    "volatile_oil": ("bo", "rs", "bg", "rv"),
    "gas_condensate": ("bg",),
    "two_phase": ("bo", "rs", "bg", "rv"),
}


def _defined(values: np.ndarray | None, mask: np.ndarray) -> np.ndarray:
    if values is None:
        return np.zeros_like(mask, dtype=bool)
    return mask & np.isfinite(values)


def check_pvt_table(
    pvt: PVTTable,
    *,
    fluid_system: str,
    initial_pressure_psia: float,
    saturation_pressure_psia: float | None,
    needs_two_phase_properties: bool = True,
) -> list[str]:
    """Check a PVT table for errors and suspicious data.

    Physically impossible data raise :class:`InputValidationError`.  Suspicious but usable
    data are returned as warning strings.  Checks are restricted to the pressure region in
    which a property is physically meaningful for the selected fluid system, so fictitious
    placeholder values (for example Bg above the bubble point of an oil) are tolerated.
    """
    if fluid_system not in FLUID_SYSTEMS:
        raise ValueError(f"Unknown fluid system {fluid_system!r}")
    warnings: list[str] = []
    label = Path(pvt.source).name if pvt.source else "PVT table"
    p = pvt.pressure
    sat = saturation_pressure_psia if saturation_pressure_psia else None

    required = list(REQUIRED_PROPERTIES[fluid_system])
    if fluid_system == "gas_condensate" and needs_two_phase_properties and sat is not None:
        required += ["bo", "rs", "rv"]
    missing = [PROPERTY_LABELS[r] for r in required if not pvt.has(r)]
    if missing:
        raise InputValidationError(f"{label}: missing PVT column(s) required for {fluid_system}: {missing}")

    # Validate only properties that participate in the selected fluid formulation.
    # This deliberately tolerates harmless placeholder columns (for example Bo=0 in a
    # dry-gas table) instead of rejecting an otherwise valid case.
    positive_properties = {"bg", "bw", "z"}
    if fluid_system != "dry_gas":
        positive_properties.add("bo")
    for name in positive_properties:
        values = getattr(pvt, name)
        if values is not None and np.any(values[np.isfinite(values)] <= 0):
            bad = p[np.isfinite(values) & (values <= 0)]
            raise InputValidationError(f"{label}: {PROPERTY_LABELS[name]} must be positive (check rows at {bad.tolist()} psia)")
    for name in ("rs", "rv"):
        values = getattr(pvt, name)
        if values is not None and np.any(values[np.isfinite(values)] < 0):
            bad = p[np.isfinite(values) & (values < 0)]
            raise InputValidationError(f"{label}: {PROPERTY_LABELS[name]} cannot be negative (rows at {bad.tolist()} psia)")

    # Region where a free gas phase coexists with oil.
    if fluid_system == "two_phase":
        two_phase = p <= initial_pressure_psia
    elif sat is not None and fluid_system in {"black_oil", "volatile_oil", "gas_condensate"}:
        two_phase = p < sat
    else:
        two_phase = np.zeros_like(p, dtype=bool)

    if pvt.has("rs") and pvt.has("rv"):
        both = _defined(pvt.rs, two_phase) & _defined(pvt.rv, two_phase)
        den = 1.0 - pvt.rs[both] * pvt.rv[both]
        if np.any(den <= 0):
            bad = p[both][den <= 0]
            raise InputValidationError(
                f"{label}: 1 - Rs*Rv <= 0 in the two-phase region at {bad.tolist()} psia; "
                "the modified-black-oil table is inconsistent"
            )

    gas_region = np.ones_like(p, dtype=bool) if fluid_system in {"dry_gas", "gas_condensate"} else two_phase
    oil_region = (p >= 0) if fluid_system in {"black_oil", "volatile_oil", "two_phase"} else two_phase

    def monotonic_warning(name: str, mask: np.ndarray, increasing_with_pressure: bool, text: str):
        values = getattr(pvt, name)
        sel = _defined(values, mask)
        if sel.sum() < 2:
            return
        d = np.diff(values[sel])
        bad = d < 0 if increasing_with_pressure else d > 0
        if np.any(bad):
            where = p[sel][1:][bad]
            warnings.append(f"{label}: {PROPERTY_LABELS[name]} {text} (check rows near {where.tolist()} psia)")

    monotonic_warning("bg", gas_region, False, "should decrease as pressure increases")
    if fluid_system in {"black_oil", "volatile_oil", "two_phase"} and sat is not None:
        below = oil_region & (p <= sat)
        monotonic_warning("rs", below, True, "should increase with pressure below the saturation pressure")
        monotonic_warning("bo", below, True, "should increase with pressure below the saturation pressure")
        monotonic_warning("bo", oil_region & (p >= sat), False, "should decrease with pressure above the bubble point")
        if pvt.has("rs"):
            above = _defined(pvt.rs, p >= sat)
            if above.sum() >= 2:
                rs_above = pvt.rs[above]
                spread = (rs_above.max() - rs_above.min()) / max(abs(rs_above.max()), 1e-12)
                if spread > 0.005:
                    warnings.append(f"{label}: Rs varies by {100 * spread:.2f}% above the bubble point; it should equal Rsi there")

    if not (pvt.p_min <= initial_pressure_psia <= pvt.p_max):
        message = (
            f"{label}: initial pressure {initial_pressure_psia:g} psia lies outside the PVT table "
            f"[{pvt.p_min:g}, {pvt.p_max:g}] psia"
        )
        if pvt.interpolation.extrapolation == "error":
            raise InputValidationError(message + " and extrapolation is 'error'")
        warnings.append(message + "; initial properties are extrapolated")
    if sat is not None and not (pvt.p_min <= sat <= pvt.p_max):
        warnings.append(
            f"{label}: saturation pressure {sat:g} psia lies outside the PVT table [{pvt.p_min:g}, {pvt.p_max:g}] psia"
        )
    return warnings


def bg_from_z_field_units(p_psia, z, temperature_f: float):
    """Gas FVF [rb/scf] = 0.02827 z T / p / 5.615 = 0.005035 z T[degR] / p[psia].

    Standard conditions 14.7 psia and 60 degF.  Accepts scalars or NumPy arrays (NaN propagates).
    """
    p = np.asarray(p_psia, dtype=float)
    zz = np.asarray(z, dtype=float)
    if np.any(p <= 0) or np.any(np.isfinite(zz) & (zz <= 0)):
        raise PVTError("Pressure and z must be positive")
    if not math.isfinite(temperature_f) or temperature_f <= -459.67:
        raise PVTError("Temperature must be above absolute zero")
    bg = (0.02827 / 5.615) * zz * (temperature_f + 459.67) / p
    return float(bg) if bg.ndim == 0 else bg
