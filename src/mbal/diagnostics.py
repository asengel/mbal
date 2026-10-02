"""Classical material-balance diagnostics evaluated at *measured* pressures.

These functions are independent of the forward pressure solve and are used both for plotting and
as independent cross-checks of history-matched inventories:

* :func:`material_balance_points` - F, Et and We at every measured pressure
* :func:`havlena_odeh_gas_cap`    - F/(Eo+Efw) vs (Eg+Efw)/(Eo+Efw) regression -> N and m
* :func:`havlena_odeh_oil`        - (F - We) vs Et regression through the origin -> N
* :func:`pz_regression`           - (p/z)/(p/z)_i vs Gp regression -> G
* :func:`drive_indices`           - drive-index time series from a simulation result
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .data import SimulationResult
from .exceptions import InputValidationError
from .material_balance import BlackOilMaterialBalance
from .simulator import Simulator


@dataclass(frozen=True)
class MaterialBalancePoints:
    """Material-balance quantities at the measured pressures (rows with production only).

    ``F_net`` [rb] is underground withdrawal minus injection, ``Et`` [rb per unit inventory] is the
    total expansion (hydrocarbon + rock/connate water) per STB of N or scf of G, and ``We`` [rb] is
    the aquifer influx computed along the measured pressure history.
    """

    time_days: np.ndarray
    pressure_psia: np.ndarray
    production: np.ndarray          # Np [STB] for oil inventories, Gp [scf] for gas inventories
    F_net: np.ndarray
    Et: np.ndarray
    We: np.ndarray
    inventory: float
    symbol: str


def _measured_pressure_path(sim: Simulator) -> tuple[np.ndarray, np.ndarray]:
    t = [0.0]
    p = [sim.reservoir.initial_pressure_psia]
    for h in sim.history:
        if h.pressure_psia is not None and h.time_days > 0:
            t.append(h.time_days)
            p.append(h.pressure_psia)
    return np.asarray(t), np.asarray(p)


def influx_along_measured_pressure(sim: Simulator) -> np.ndarray:
    """Cumulative influx [rb] at every history row when the aquifer is driven by the *measured*
    pressure history (linearly interpolated in time between measurements)."""
    t_obs, p_obs = _measured_pressure_path(sim)
    aquifer = copy.deepcopy(sim.aquifer)
    pi = sim.reservoir.initial_pressure_psia
    aquifer.reset(pi, 0.0)
    out = np.zeros(len(sim.history))
    prev_p, prev_t = pi, 0.0
    for row, report in sim._steps():
        p = float(np.interp(row.time_days, t_obs, p_obs))
        if row.time_days > prev_t:
            trial = aquifer.preview(p, prev_p, row.time_days, prev_t)
            aquifer.commit(trial, p, row.time_days)
            we = trial.cumulative_influx_rb
            prev_p, prev_t = p, row.time_days
        else:
            we = 0.0
        if report is not None:
            out[report] = we
    return out


def material_balance_points(sim: Simulator) -> MaterialBalancePoints:
    """Evaluate F, Et and We at measured pressures for rows with non-zero production."""
    inventory, symbol = sim.reservoir.inventory()
    we_all = influx_along_measured_pressure(sim)
    cols: dict[str, list[float]] = {k: [] for k in ("t", "p", "prod", "F", "Et", "We")}
    for i, h in enumerate(sim.history):
        if h.pressure_psia is None:
            continue
        prod = h.np_stb if symbol == "N" else h.gp_scf
        if prod <= 0:
            continue
        terms = sim.reservoir.balance(float(h.pressure_psia), h, 0.0)
        et = (terms.hydrocarbon_expansion_rb + terms.rock_water_expansion_rb) / inventory
        if abs(et) <= 1e-15:
            continue
        cols["t"].append(h.time_days)
        cols["p"].append(float(h.pressure_psia))
        cols["prod"].append(prod)
        cols["F"].append(terms.underground_withdrawal_rb - terms.water_injection_rb - terms.gas_injection_rb)
        cols["Et"].append(et)
        cols["We"].append(float(we_all[i]))
    a = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
    return MaterialBalancePoints(a["t"], a["p"], a["prod"], a["F"], a["Et"], a["We"], float(inventory), symbol)


@dataclass(frozen=True)
class LineFit:
    x: np.ndarray
    y: np.ndarray
    slope: float
    intercept: float
    r_squared: float


def _fit(x: np.ndarray, y: np.ndarray, through_origin: bool = False) -> LineFit:
    if x.size < 2:
        raise InputValidationError("A straight-line diagnostic needs at least two measured points")
    if through_origin:
        slope = float(x @ y / (x @ x))
        intercept = 0.0
    else:
        slope, intercept = (float(v) for v in np.polyfit(x, y, 1))
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return LineFit(x, y, slope, intercept, r2)


def havlena_odeh_gas_cap(sim: Simulator, points: MaterialBalancePoints | None = None) -> tuple[LineFit, float, float]:
    """Gas-cap Havlena-Odeh straight line, independent of the fitted m.

    ``F - We = N (Eo + Efw) + m N (Eg + Efw)``  ->  ``y = N + mN x`` with
    ``y = (F - We)/(Eo + Efw)`` and ``x = (Eg + Efw)/(Eo + Efw)`` (Efw per STB of oil-zone).

    Returns the fit, N [STB] and m.
    """
    res = sim.reservoir
    if not isinstance(res, BlackOilMaterialBalance):
        raise InputValidationError("havlena_odeh_gas_cap applies to black-oil reservoirs")
    pts = points or material_balance_points(sim)
    x, y = [], []
    for p, f, we in zip(pts.pressure_psia, pts.F_net, pts.We):
        e = res.unit_expansions(float(p))
        denom = e["Eo"] + e["Efw"]
        if abs(denom) < 1e-15:
            continue
        x.append((e["Eg"] + e["Efw"]) / denom)
        y.append((f - we) / denom)
    fit = _fit(np.asarray(x), np.asarray(y))
    n = fit.intercept
    return fit, n, fit.slope / n if n != 0 else float("nan")


def havlena_odeh_oil(points: MaterialBalancePoints) -> tuple[LineFit, float]:
    """``F - We = N Et`` through the origin; returns the fit and the inventory (N or G)."""
    fit = _fit(points.Et, points.F_net - points.We, through_origin=True)
    return fit, fit.slope


def pz_regression(sim: Simulator) -> tuple[LineFit, float]:
    """Normalised p/z plot: ``(p/z)/(p/z)_i = Bgi/Bg`` vs Gp, fitted with a straight line.

    For a volumetric gas reservoir without rock/water expansion the line is ``1 - Gp/G``; the
    extrapolated Gp at zero gives G [scf].  Below the dew point the two-phase equivalent
    ``Bgi/Btg`` is used.
    """
    ratio = getattr(sim.reservoir, "gas_fvf_ratio", None)
    if ratio is None:
        raise InputValidationError("p/z diagnostics apply to gas reservoirs")
    x, y = [0.0], [1.0]
    for h in sim.history:
        if h.pressure_psia is None or h.gp_scf <= 0:
            continue
        x.append(h.gp_scf)
        y.append(ratio(float(h.pressure_psia)))
    fit = _fit(np.asarray(x), np.asarray(y))
    return fit, -fit.intercept / fit.slope if fit.slope != 0 else float("nan")


def drive_indices(result: SimulationResult) -> dict[str, np.ndarray]:
    """Drive indices per report row; each support term divided by total support (sum = 1)."""
    keys = {
        "DDI": "oil_expansion_fraction",          # depletion (oil + dissolved gas) drive
        "GCDI": "gas_cap_expansion_fraction",     # gas-cap drive
        "GDI": "gas_expansion_fraction",          # gas expansion (gas reservoirs / free gas)
        "CDI": "rock_water_fraction",             # compaction + connate-water expansion
        "WDI": "aquifer_fraction",                # water drive
        "IDI": "injection_fraction",              # injection
    }
    return {k: np.asarray([r.extras.get(v, 0.0) for r in result.rows], dtype=float) for k, v in keys.items()}
