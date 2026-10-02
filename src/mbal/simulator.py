"""Forward tank material-balance simulator: solves reservoir pressure at every history time."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .aquifers import AquiferModel, AquiferTrial, NoAquifer
from .data import HistoryRecord, SimulationResult, SimulationRow, validate_history
from .exceptions import MaterialBalanceError, PressureSolveError
from .material_balance import BalanceTerms, MaterialBalanceModel


@dataclass(frozen=True)
class PressureSolverSettings:
    """Numerical settings of the pressure solve.

    ``max_step_days`` limits the internal time step used for aquifer influx.  Cumulative
    production is interpolated linearly between history rows and results are reported only at the
    history dates.  Sub-stepping is skipped for volumetric cases (no aquifer), where the solution
    at a date does not depend on the path.  ``0`` disables sub-stepping (reproduces textbook
    calculations made at the report dates only).
    """

    pressure_tolerance_psi: float = 1e-5
    closure_relative_tolerance: float = 1e-7
    min_pressure_psia: float | None = None
    max_pressure_psia: float | None = None
    bracket_points: int = 240
    max_step_days: float = 30.0


def substep_history(history: list[HistoryRecord], max_step_days: float) -> list[tuple[HistoryRecord, int | None]]:
    """Split history intervals longer than ``max_step_days``.

    Returns ``(record, report_index)`` pairs; ``report_index`` is the index of the original row
    for report rows and ``None`` for interpolated sub-steps (which carry no observation).
    """
    out: list[tuple[HistoryRecord, int | None]] = [(history[0], 0)]
    for i, (a, b) in enumerate(zip(history[:-1], history[1:]), start=1):
        n = 1 if max_step_days <= 0 else max(1, math.ceil((b.time_days - a.time_days) / max_step_days - 1e-9))
        for j in range(1, n):
            f = j / n

            def lerp(x: float, y: float, f: float = f) -> float:
                return x + (y - x) * f

            out.append((HistoryRecord(
                time_days=lerp(a.time_days, b.time_days), np_stb=lerp(a.np_stb, b.np_stb),
                gp_scf=lerp(a.gp_scf, b.gp_scf), wp_stb=lerp(a.wp_stb, b.wp_stb),
                winj_stb=lerp(a.winj_stb, b.winj_stb), ginj_scf=lerp(a.ginj_scf, b.ginj_scf),
                pressure_psia=None, pressure_weight=0.0,
            ), None))
        out.append((b, i))
    return out


class Simulator:
    """Forward tank material-balance simulator.

    Production and injection are imposed as cumulative histories.  Reservoir pressure is solved at
    every (sub-)step so that material balance closes for the selected reservoir-fluid and aquifer
    models.  ``run()`` works on a private copy of the aquifer, so a simulator can be run repeatedly
    (and concurrently from different threads) without hidden state.
    """

    def __init__(
        self,
        reservoir: MaterialBalanceModel,
        aquifer: AquiferModel,
        history: list[HistoryRecord],
        *,
        pressure_solver: PressureSolverSettings | None = None,
    ):
        validate_history(history)
        self.reservoir = reservoir
        self.aquifer = aquifer
        self.history = list(history)
        self.settings = pressure_solver or PressureSolverSettings()
        self.input_warnings: list[str] = list(getattr(reservoir, "input_warnings", []))
        self._validate_hydrocarbon_inventory()

    # ----------------------------------------------------------------------------- checks
    def _validate_hydrocarbon_inventory(self) -> None:
        """Reject impossible inventories: cumulative production cannot exceed the initial component
        inventory (plus injection of that component)."""
        oil_initial = float(self.reservoir.initial_oil_stb)
        gas_initial = float(self.reservoir.initial_gas_scf)
        max_oil = max((float(h.np_stb) for h in self.history), default=0.0)
        max_net_gas = max((float(h.gp_scf - h.ginj_scf) for h in self.history), default=0.0)

        tol = 1e-10
        if max_oil > 0.0 and max_oil > oil_initial * (1.0 + tol):
            raise MaterialBalanceError(
                f"Initial stock-tank-oil component inventory ({oil_initial:g} STB) is smaller than "
                f"cumulative oil production ({max_oil:g} STB)."
            )
        if max_net_gas > 0.0 and max_net_gas > gas_initial * (1.0 + tol):
            raise MaterialBalanceError(
                f"Initial surface-gas component inventory ({gas_initial:g} scf) is smaller than "
                f"net cumulative gas production ({max_net_gas:g} scf)."
            )

    def _volume_floor(self) -> float:
        return max(1e-12 * abs(float(self.reservoir.initial_hcpv_rb)), 1e-300)

    def _pressure_limits(self, previous_pressure: float) -> tuple[float, float]:
        pi = self.reservoir.initial_pressure_psia
        pvt = self.reservoir.pvt
        if self.settings.min_pressure_psia is not None:
            lo = self.settings.min_pressure_psia
        elif pvt.interpolation.extrapolation == "error":
            lo = float(pvt.pressure[0])
        else:
            lo = max(0.1, min(0.05 * pi, 0.25 * previous_pressure))
        if self.settings.max_pressure_psia is not None:
            hi = self.settings.max_pressure_psia
        elif pvt.interpolation.extrapolation == "error":
            hi = float(pvt.pressure[-1])
        else:
            hi = max(pi + 2000.0, 1.5 * pi, 1.25 * previous_pressure)
        if not 0 < lo < hi:
            raise PressureSolveError(f"Invalid pressure search interval [{lo}, {hi}] psia")
        return float(lo), float(hi)

    # ----------------------------------------------------------------------- pressure solve
    def _solve_pressure(
        self,
        aquifer: AquiferModel,
        row: HistoryRecord,
        previous_pressure: float,
        previous_time: float,
    ) -> tuple[float, AquiferTrial, BalanceTerms]:
        lo, hi = self._pressure_limits(previous_pressure)

        def f(p: float) -> float:
            trial = aquifer.preview(p, previous_pressure, row.time_days, previous_time)
            return self.reservoir.balance(p, row, trial.cumulative_influx_rb).residual_rb

        def root_in(a: float, b: float) -> float:
            if a == b:
                return a
            return float(brentq(f, a, b, xtol=self.settings.pressure_tolerance_psi, rtol=1e-12, maxiter=200))

        def evaluate(pressure: float) -> tuple[AquiferTrial, BalanceTerms, float]:
            trial = aquifer.preview(pressure, previous_pressure, row.time_days, previous_time)
            terms = self.reservoir.balance(pressure, row, trial.cumulative_influx_rb)
            return trial, terms, self._closure_relative(terms)

        def accept(pressure: float) -> tuple[float, AquiferTrial, BalanceTerms, float]:
            """Check closure; polish the root once before declaring it a discontinuity.

            For very small steps the pressure tolerance alone can leave a relative residual above the
            closure tolerance.  A genuine root closes to round-off when bracketed more tightly, a
            discontinuity does not.
            """
            trial, terms, relative = evaluate(pressure)
            if relative <= self.settings.closure_relative_tolerance:
                return pressure, trial, terms, relative
            h = max(20.0 * self.settings.pressure_tolerance_psi, 1e-9 * pressure)
            a, b = pressure - h, pressure + h
            try:
                fa, fb = f(a), f(b)
                if math.isfinite(fa) and math.isfinite(fb) and fa * fb <= 0:
                    polished = a if fa == 0 else b if fb == 0 else float(
                        brentq(f, a, b, xtol=1e-13 * pressure, rtol=1e-15, maxiter=200))
                    trial, terms, relative = evaluate(polished)
                    pressure = polished
            except (ArithmeticError, MaterialBalanceError, ValueError):
                pass
            return pressure, trial, terms, relative

        rejected: list[str] = []

        # 1) Local bracket: expand geometrically from the previous pressure in the downhill
        #    direction.  Steps are small relative to the pressure range, so this usually brackets the
        #    root within a few psi and Brent's method then converges in a handful of evaluations.
        try:
            p0 = min(max(previous_pressure, lo), hi)
            f0 = f(p0)
            if f0 == 0.0:
                pressure, trial, terms, relative = accept(p0)
                if relative <= self.settings.closure_relative_tolerance:
                    return pressure, trial, terms
            elif math.isfinite(f0):
                direction = -1.0 if f0 < 0 else 1.0   # residual = support - withdrawal rises as p falls
                delta = max(1.0, 1e-3 * p0)
                a, fa = p0, f0
                while True:
                    b = min(max(p0 + direction * delta, lo), hi)
                    fb = f(b)
                    if not math.isfinite(fb):
                        break
                    if fa * fb <= 0:
                        lo_b, hi_b = (b, a) if b < a else (a, b)
                        pressure = b if fb == 0 else root_in(lo_b, hi_b)
                        pressure, trial, terms, relative = accept(pressure)
                        if relative <= self.settings.closure_relative_tolerance:
                            return pressure, trial, terms
                        rejected.append(f"{pressure:.6g} psia (relative residual {relative:.2e})")
                        break
                    if b in (lo, hi):
                        break
                    a, fa = b, fb
                    delta *= 4.0
        except (ArithmeticError, MaterialBalanceError, ValueError):
            pass

        # 2) One broad bracket.  Brent's method can converge onto a discontinuity, so
        #    material-balance closure is always checked before accepting.
        try:
            flo, fhi = f(lo), f(hi)
            if math.isfinite(flo) and math.isfinite(fhi) and flo * fhi <= 0:
                pressure = lo if flo == 0 else hi if fhi == 0 else root_in(lo, hi)
                pressure, trial, terms, relative = accept(pressure)
                if relative <= self.settings.closure_relative_tolerance:
                    return pressure, trial, terms
                rejected.append(f"{pressure:.6g} psia (relative residual {relative:.2e})")
        except (ArithmeticError, MaterialBalanceError):
            pass

        # 3) Scan for every sign change and try the brackets nearest the previous pressure first.
        grid = np.linspace(lo, hi, max(20, self.settings.bracket_points))
        values: list[float] = []
        failures: dict[str, int] = {}
        for p in grid:
            try:
                value = float(f(float(p)))
            except (ArithmeticError, MaterialBalanceError) as exc:
                value = math.nan
                failures[str(exc)] = failures.get(str(exc), 0) + 1
            values.append(value)
        if failures and all(not math.isfinite(v) for v in values):
            first = next(iter(failures))
            raise PressureSolveError(
                f"At t={row.time_days:g} days the material balance could not be evaluated at any pressure "
                f"between {lo:g} and {hi:g} psia: {first}"
            )
        candidates: list[tuple[float, float]] = []
        for i in range(len(grid) - 1):
            a, b = values[i], values[i + 1]
            if not (math.isfinite(a) and math.isfinite(b)):
                continue
            if a == 0.0:
                candidates.append((float(grid[i]), float(grid[i])))
            elif a * b < 0:
                candidates.append((float(grid[i]), float(grid[i + 1])))
        if math.isfinite(values[-1]) and values[-1] == 0.0:
            candidates.append((float(grid[-1]), float(grid[-1])))
        if not candidates:
            detail = ""
            if failures:
                n_failed = sum(failures.values())
                detail = f" {n_failed} of {len(grid)} trial pressures failed, e.g.: {next(iter(failures))}"
            raise PressureSolveError(
                f"No material-balance pressure root could be bracketed at t={row.time_days:g} days. "
                f"Search interval was {lo:g} to {hi:g} psia.{detail}"
            )
        candidates.sort(key=lambda ab: abs(0.5 * (ab[0] + ab[1]) - previous_pressure))
        for a, b in candidates:
            try:
                pressure = root_in(a, b)
            except (ArithmeticError, MaterialBalanceError, ValueError):
                continue
            pressure, trial, terms, relative = accept(pressure)
            if relative <= self.settings.closure_relative_tolerance:
                return pressure, trial, terms
            rejected.append(f"{pressure:.6g} psia (relative residual {relative:.2e})")
        raise PressureSolveError(
            f"No pressure between {lo:g} and {hi:g} psia closes the material balance at t={row.time_days:g} days; "
            f"sign changes at {', '.join(rejected) or 'none'} are discontinuities, not roots. "
            "Check the PVT table for jumps or 1 - Rs*Rv approaching zero, or narrow solver.min/max_pressure_psia."
        )

    # ------------------------------------------------------------------------- reporting
    @staticmethod
    def _support(terms: BalanceTerms) -> float:
        return (terms.hydrocarbon_expansion_rb + terms.rock_water_expansion_rb + terms.water_influx_rb
                + terms.water_injection_rb + terms.gas_injection_rb)

    def _closure_relative(self, terms: BalanceTerms) -> float:
        scale = max(abs(terms.underground_withdrawal_rb), abs(self._support(terms)), self._volume_floor())
        return abs(terms.residual_rb) / scale

    def _row_from_terms(self, row: HistoryRecord, pressure: float, trial: AquiferTrial,
                        terms: BalanceTerms) -> SimulationRow:
        support = self._support(terms)
        floor = self._volume_floor()

        def fraction(value: float) -> float:
            return value / support if abs(support) > floor else 0.0

        n_total = float(self.reservoir.initial_oil_stb)
        g_total = float(self.reservoir.initial_gas_scf)
        extras = {
            "aquifer_rate_rb_per_day": 0.0 if trial.influx_rate_rb_per_day is None else trial.influx_rate_rb_per_day,
            # Drive indices (Pletcher 2002): each support term over total support (= F at closure).
            "oil_expansion_fraction": fraction(terms.oil_expansion_rb),
            "gas_expansion_fraction": fraction(terms.gas_expansion_rb),
            "gas_cap_expansion_fraction": fraction(terms.gas_cap_expansion_rb),
            "rock_water_fraction": fraction(terms.rock_water_expansion_rb),
            "aquifer_fraction": fraction(terms.water_influx_rb),
            "injection_fraction": fraction(terms.water_injection_rb + terms.gas_injection_rb),
            "oil_recovery_factor": row.np_stb / n_total if n_total > 0 else 0.0,
            "gas_recovery_factor": row.gp_scf / g_total if g_total > 0 else 0.0,
        }
        return SimulationRow(
            time_days=row.time_days,
            pressure_psia=pressure,
            pressure_observed_psia=row.pressure_psia,
            water_influx_rb=trial.cumulative_influx_rb,
            aquifer_pressure_psia=trial.aquifer_pressure_psia,
            underground_withdrawal_rb=terms.underground_withdrawal_rb,
            hydrocarbon_expansion_rb=terms.hydrocarbon_expansion_rb,
            oil_expansion_rb=terms.oil_expansion_rb,
            gas_expansion_rb=terms.gas_expansion_rb,
            gas_cap_expansion_rb=terms.gas_cap_expansion_rb,
            rock_water_expansion_rb=terms.rock_water_expansion_rb,
            water_injection_rb=terms.water_injection_rb,
            gas_injection_rb=terms.gas_injection_rb,
            material_balance_residual_rb=terms.residual_rb,
            closure_relative=self._closure_relative(terms),
            extras=extras,
        )

    def consistency_warnings(self, rows: list[SimulationRow]) -> list[str]:
        """Check produced ratios against the single-phase state implied by the solved pressure."""
        out: list[str] = []
        reservoir = self.reservoir
        sat = reservoir.saturation_pressure_psia
        pvt = reservoir.pvt
        slack = max(10.0 * self.settings.pressure_tolerance_psi, 1e-9 * pvt.p_max)
        outside = sum(1 for r in rows if r.pressure_psia < pvt.p_min - slack or r.pressure_psia > pvt.p_max + slack)
        if outside:
            out.append(
                f"Solved pressure leaves the PVT table [{pvt.p_min:g}, {pvt.p_max:g}] psia at "
                f"{outside} step(s); properties were extrapolated"
            )
        ratio = reservoir.single_phase_ratio()
        if ratio is None:
            return out
        kind, expected = ratio
        samples: list[float] = []
        for r, h in zip(rows, self.history):
            if sat is not None and r.pressure_psia < sat:
                continue
            if kind == "GOR" and h.np_stb > 0:
                samples.append(h.gp_scf / h.np_stb)
            elif kind == "CGR" and h.gp_scf > 0:
                samples.append(h.np_stb / h.gp_scf)
        if samples and expected > 0:
            deviations = [abs(v - expected) / expected for v in samples]
            worst = max(deviations)
            if worst > 0.01:
                label = "cumulative GOR Gp/Np" if kind == "GOR" else "cumulative condensate-gas ratio Np/Gp"
                out.append(
                    f"{sum(d > 0.01 for d in deviations)} single-phase step(s) have a {label} differing from the "
                    f"initial solution ratio by up to {100 * worst:.1f}%; produced fluid is inconsistent with the PVT "
                    "description (the excess component is withdrawn at saturation-state volumes)"
                )
        return out

    # ------------------------------------------------------------------------------- run
    def _steps(self) -> list[tuple[HistoryRecord, int | None]]:
        if isinstance(self.aquifer, NoAquifer) or self.settings.max_step_days <= 0:
            return [(h, i) for i, h in enumerate(self.history)]
        return substep_history(self.history, self.settings.max_step_days)

    def run(self) -> SimulationResult:
        """Solve pressure at every history date; returns one row per history record."""
        pi = self.reservoir.initial_pressure_psia
        aquifer = copy.deepcopy(self.aquifer)
        aquifer.reset(pi, 0.0)
        previous_pressure, previous_time = pi, 0.0
        output: list[SimulationRow] = []

        for k, (row, report) in enumerate(self._steps()):
            zero_state = (
                row.time_days == 0.0 and row.np_stb == 0.0 and row.gp_scf == 0.0 and row.wp_stb == 0.0
                and row.winj_stb == 0.0 and row.ginj_scf == 0.0
            )
            if k == 0 and zero_state:
                trial = aquifer.preview(pi, pi, 0.0, 0.0)
                terms = self.reservoir.balance(pi, row, trial.cumulative_influx_rb)
                aquifer.commit(trial, pi, 0.0)
                output.append(self._row_from_terms(row, pi, trial, terms))
                continue

            pressure, trial, terms = self._solve_pressure(aquifer, row, previous_pressure, previous_time)
            aquifer.commit(trial, pressure, row.time_days)
            if report is not None:
                output.append(self._row_from_terms(row, pressure, trial, terms))
            previous_pressure, previous_time = pressure, row.time_days

        warnings = self.input_warnings + self.consistency_warnings(output)
        return SimulationResult(
            output,
            metadata={
                "initial_pressure_psia": pi,
                "n_steps": len(output),
                "max_closure_relative": max((r.closure_relative for r in output), default=0.0),
                "warnings": list(dict.fromkeys(warnings)),
            },
        )
