"""Analytical aquifer models: none, Fetkovich, Carter-Tracy (CT) and van Everdingen-Hurst (HVE).

All models follow the same two-phase protocol used by :class:`mbal.simulator.Simulator`:

* ``preview(p, p_prev, t, t_prev)`` returns the cumulative influx ``We`` [rb] that *would* exist
  at time ``t`` [days] if the reservoir boundary pressure were ``p`` [psia].  It has no side effects
  on the committed state and is called many times by the pressure root finder.
* ``commit(trial, p, t)`` accepts the solved step and advances the aquifer state.

Influx is signed: when the reservoir is re-pressurised above the aquifer pressure, water flows back
into the aquifer and ``We`` decreases.  Set ``allow_backflow=False`` to forbid decreasing ``We``.

Field units throughout: k [md], mu [cp], ct [1/psi], h, r [ft], t [days], We [rb].
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import j0, j1, y0, y1

from .exceptions import InputValidationError


@dataclass(frozen=True)
class AquiferTrial:
    """Result of one ``preview`` call."""

    cumulative_influx_rb: float
    aquifer_pressure_psia: float | None = None
    influx_rate_rb_per_day: float | None = None


class AquiferModel(Protocol):
    """Interface implemented by every aquifer model."""

    def reset(self, initial_pressure_psia: float, initial_time_days: float = 0.0) -> None: ...

    def preview(self, pressure_psia: float, previous_pressure_psia: float,
                time_days: float, previous_time_days: float) -> AquiferTrial: ...

    def commit(self, trial: AquiferTrial, pressure_psia: float, time_days: float) -> None: ...


def _check_time(time_days: float, last_time_days: float) -> float:
    dt = float(time_days) - float(last_time_days)
    if dt < 0:
        raise InputValidationError("Aquifer time cannot decrease")
    return dt


# ---------------------------------------------------------------------------------------- none
class NoAquifer:
    """Volumetric reservoir: no water influx."""

    def reset(self, initial_pressure_psia: float, initial_time_days: float = 0.0) -> None:
        pass

    def preview(self, pressure_psia: float, previous_pressure_psia: float,
                time_days: float, previous_time_days: float) -> AquiferTrial:
        return AquiferTrial(0.0, None, 0.0)

    def commit(self, trial: AquiferTrial, pressure_psia: float, time_days: float) -> None:
        pass


# ----------------------------------------------------------------------------------- Fetkovich
class FetkovichAquifer:
    """Fetkovich pseudo-steady-state aquifer.

    Parameters
    ----------
    J_rb_day_psi:
        Aquifer productivity index J [rb/day/psi].
    Wei_rb:
        Maximum encroachable water ``Wei = ct * Wi * pi`` [rb].
    allow_backflow:
        Permit negative influx increments when reservoir pressure exceeds aquifer pressure.

    Over a step of length dt the influx increment is::

        dWe = (Wei / pi) * (paq_{n-1} - pbar_n) * (1 - exp(-J * pi * dt / Wei))

    with ``pbar_n`` the average boundary pressure over the step and ``paq = pi (1 - We/Wei)``.
    This is Dake Chapter 9, Eqs. (9.28)-(9.29), for a finite pseudo-steady-state aquifer.
    """

    def __init__(self, J_rb_day_psi: float, Wei_rb: float, *, allow_backflow: bool = True):
        if not (math.isfinite(J_rb_day_psi) and math.isfinite(Wei_rb)) or J_rb_day_psi < 0 or Wei_rb < 0:
            raise InputValidationError("Fetkovich J and Wei must be finite and non-negative")
        self.J = float(J_rb_day_psi)
        self.Wei = float(Wei_rb)
        self.allow_backflow = bool(allow_backflow)
        self.reset(1.0)

    def reset(self, initial_pressure_psia: float, initial_time_days: float = 0.0) -> None:
        if initial_pressure_psia <= 0:
            raise InputValidationError("Initial pressure must be positive")
        self.pi = float(initial_pressure_psia)
        self.we = 0.0
        self.paq = self.pi

    def preview(self, pressure_psia: float, previous_pressure_psia: float,
                time_days: float, previous_time_days: float) -> AquiferTrial:
        dt = _check_time(time_days, previous_time_days)
        if dt == 0 or self.J == 0 or self.Wei == 0:
            return AquiferTrial(self.we, self.paq, 0.0)

        p = float(pressure_psia)
        pbar = 0.5 * (float(previous_pressure_psia) + p)
        response = -math.expm1(max(-self.J * self.pi * dt / self.Wei, -700.0))
        dwe = (self.Wei / self.pi) * (self.paq - pbar) * response
        if not self.allow_backflow:
            dwe = max(dwe, 0.0)
        we = self.we + dwe
        # paq_new - pbar = (paq_old - pbar) (1 - response): the aquifer relaxes towards the average
        # boundary pressure without ever crossing it, so no equilibrium clamp is needed.
        paq = self.pi * (1.0 - we / self.Wei)
        return AquiferTrial(we, paq, (we - self.we) / dt)

    def commit(self, trial: AquiferTrial, pressure_psia: float, time_days: float) -> None:
        self.we = float(trial.cumulative_influx_rb)
        if trial.aquifer_pressure_psia is not None:
            self.paq = float(trial.aquifer_pressure_psia)


# ------------------------------------------------------------------------- radial properties
@dataclass(frozen=True)
class RadialAquiferProperties:
    """Physical inputs shared by HVE and Carter-Tracy radial aquifers.

    ``radius_ratio`` is re/ri; ``None`` means infinite-acting.
    """

    permeability_md: float
    porosity: float
    total_compressibility_1psi: float
    water_viscosity_cp: float
    thickness_ft: float
    inner_radius_ft: float
    radius_ratio: float | None = None
    angle_deg: float = 360.0

    def __post_init__(self) -> None:
        positive = {
            "permeability_md": self.permeability_md,
            "porosity": self.porosity,
            "total_compressibility_1psi": self.total_compressibility_1psi,
            "water_viscosity_cp": self.water_viscosity_cp,
            "thickness_ft": self.thickness_ft,
            "inner_radius_ft": self.inner_radius_ft,
            "angle_deg": self.angle_deg,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise InputValidationError(f"Aquifer {name} must be positive")
        if self.porosity >= 1:
            raise InputValidationError("Aquifer porosity must be smaller than 1")
        if self.angle_deg > 360:
            raise InputValidationError("Aquifer angle_deg cannot exceed 360")
        if self.radius_ratio is not None and (not math.isfinite(self.radius_ratio) or self.radius_ratio <= 1):
            raise InputValidationError("Aquifer radius_ratio = re/ri must exceed 1")

    @property
    def aquifer_constant_rb_psi(self) -> float:
        """U = 1.119 * phi * ct * h * ri^2 * (theta/360) [rb/psi]; Dake Eq. (9.8)."""
        return (1.119 * self.porosity * self.total_compressibility_1psi
                * self.thickness_ft * self.inner_radius_ft**2 * self.angle_deg / 360.0)

    @property
    def time_factor_per_day(self) -> float:
        """tD = alpha*t[days], alpha=0.006328 k/(phi mu ct ri^2); Dake Eq. (9.7)."""
        return (0.006328 * self.permeability_md /
                (self.porosity * self.water_viscosity_cp * self.total_compressibility_1psi
                 * self.inner_radius_ft**2))


# ------------------------------------------------------------- HVE influence function W_D(tD)
def _wd_infinite(t_d: np.ndarray) -> np.ndarray:
    """Dimensionless cumulative influx W_D for an infinite radial aquifer (constant terminal pressure).

    * tD < 0.01       : short-time series ``2 sqrt(t/pi) + t/2 - t sqrt(t/pi)/6 + t^2/16``
    * 0.01 <= tD < 200: Edwardson et al. (1962) rational approximation
    * tD >= 200       : Edwardson et al. long-time form ``(2.02566 tD - 4.29881) / ln tD``

    Checked against Stehfest inversion of the exact Laplace solution: error < 0.1 %, and the jumps
    at the two break points are below 0.02 %, so the objective stays smooth in k and ri.
    """
    t = np.asarray(t_d, dtype=float)
    out = np.zeros_like(t)
    s = (t > 0) & (t < 0.01)
    x = t[s]
    out[s] = 2*np.sqrt(x/np.pi) + x/2 - x*np.sqrt(x/np.pi)/6 + x*x/16
    m = (t >= 0.01) & (t < 200.0)
    x = t[m]
    r = np.sqrt(x)
    out[m] = (1.12838*r + 1.19328*x + 0.269872*x*r + 0.00855294*x*x) / (1.0 + 0.616599*r + 0.0413008*x)
    lg = t >= 200.0
    x = t[lg]
    out[lg] = (2.02566*x - 4.29881) / np.log(x)
    return out


@lru_cache(maxsize=64)
def _finite_roots(radius_ratio: float, n_terms: int) -> tuple[float, ...]:
    """First ``n_terms`` roots of ``J1(a R) Y0(a) - Y1(a R) J0(a) = 0``."""
    R = float(radius_ratio)

    def f(a):
        return j1(a*R)*y0(a) - y1(a*R)*j0(a)

    upper = max(20.0, 10.0*n_terms/max(R - 1.0, 0.2))
    for _ in range(8):
        x = np.linspace(1e-7, upper, max(8000, n_terms*1000))
        y = f(x)
        idx = np.where(np.isfinite(y[:-1]) & np.isfinite(y[1:]) & (np.signbit(y[:-1]) != np.signbit(y[1:])))[0]
        roots: list[float] = []
        for i in idx:
            try:
                root = brentq(f, float(x[i]), float(x[i+1]))
            except ValueError:
                continue
            if not roots or abs(root - roots[-1]) > 1e-7:
                roots.append(root)
            if len(roots) == n_terms:
                return tuple(roots)
        upper *= 2
    raise InputValidationError("Could not evaluate the finite radial aquifer influence function")


#: Below tD = FINITE_SWITCH * (R - 1)^2 the outer boundary has not been felt.  The infinite-acting
#: W_D is used there (the eigen-series converges slowly at small tD); both agree to < 0.02 %.
FINITE_SWITCH = 0.05


def _wd_finite(t_d: np.ndarray, radius_ratio: float, n_terms: int = 20) -> np.ndarray:
    """W_D for a closed finite radial aquifer (van Everdingen-Hurst eigen-series)."""
    t = np.asarray(t_d, dtype=float)
    R = float(radius_ratio)
    switch = FINITE_SWITCH * (R - 1.0)**2
    out = np.empty_like(t)
    early = t < switch
    out[early] = _wd_infinite(t[early])
    late = ~early
    if np.any(late):
        tl = t[late]
        series = np.full_like(tl, 0.5*(R*R - 1.0))
        for a in _finite_roots(round(R, 10), n_terms):
            jar = j1(a*R)
            den = a*a * (j0(a)**2 - jar**2)
            series -= 2.0*np.exp(-a*a*tl) * jar**2 / den
        out[late] = series
    return np.maximum(out, 0.0)


class HVEAquifer:
    """van Everdingen-Hurst radial aquifer with pressure-step superposition.

    ``We_n = U * sum_j dp_j * W_D(tD_n - tD_j)`` with midpoint pressure steps
    ``dp_0 = (p_0 - p_1)/2`` and ``dp_j = (p_{j-1} - p_{j+1})/2``. This is the
    van Everdingen-Timmerman-McMahon step construction used by Dake in Eqs. (9.15)-(9.17).

    Only the last superposition term depends on the trial pressure ``p_n``, so ``We`` is linear in
    ``p_n``: the history part is evaluated once per time step (O(n)) and every root-finder trial is
    O(1).
    """

    def __init__(self, properties: RadialAquiferProperties, *, finite_terms: int = 20,
                 allow_backflow: bool = True):
        self.props = properties
        self.B = properties.aquifer_constant_rb_psi
        self.alpha = properties.time_factor_per_day
        self.finite_terms = int(finite_terms)
        self.allow_backflow = bool(allow_backflow)
        self.reset(1.0)

    def reset(self, initial_pressure_psia: float, initial_time_days: float = 0.0) -> None:
        self.pi = float(initial_pressure_psia)
        self.times: list[float] = [float(initial_time_days)]
        self.pressures: list[float] = [self.pi]
        self.we = 0.0
        self._prepared_for: float | None = None

    def _wd(self, t_d: np.ndarray) -> np.ndarray:
        if self.props.radius_ratio is None:
            return _wd_infinite(t_d)
        return _wd_finite(t_d, self.props.radius_ratio, self.finite_terms)

    def _prepare(self, time_days: float) -> None:
        t = np.asarray(self.times, dtype=float)
        p = np.asarray(self.pressures, dtype=float)
        wd = self._wd(self.alpha * (time_days - t))       # one vectorised call per step
        n = p.size
        if n >= 2:
            dp = np.empty(n - 1)
            dp[0] = 0.5 * (p[0] - p[1])
            dp[1:] = 0.5 * (p[:-2] - p[2:])
            self._fixed = float(dp @ wd[:n - 1])
            self._p_ref = float(p[-2])
        else:
            self._fixed = 0.0
            self._p_ref = float(p[0])
        self._wd_last = float(wd[n - 1])
        self._prepared_for = float(time_days)

    def preview(self, pressure_psia: float, previous_pressure_psia: float,
                time_days: float, previous_time_days: float) -> AquiferTrial:
        dt = _check_time(time_days, self.times[-1])
        if dt == 0:
            return AquiferTrial(self.we, None, 0.0)
        if self._prepared_for != float(time_days):
            self._prepare(float(time_days))
        we = self.B * (self._fixed + 0.5 * (self._p_ref - float(pressure_psia)) * self._wd_last)
        if not self.allow_backflow:
            we = max(we, self.we)
        step = float(time_days) - float(previous_time_days)
        return AquiferTrial(we, None, (we - self.we) / step if step > 0 else 0.0)

    def commit(self, trial: AquiferTrial, pressure_psia: float, time_days: float) -> None:
        self.we = float(trial.cumulative_influx_rb)
        if time_days == self.times[-1]:
            self.pressures[-1] = float(pressure_psia)
        else:
            self.times.append(float(time_days))
            self.pressures.append(float(pressure_psia))
        self._prepared_for = None

    def influx_brute_force(self, times: list[float], pressures: list[float]) -> float:
        """Reference superposition over a full pressure history (used to test the fast path)."""
        t = np.asarray(times, dtype=float)
        p = np.asarray(pressures, dtype=float)
        if t.size < 2:
            return 0.0
        dp = np.empty(p.size - 1)
        dp[0] = 0.5*(p[0] - p[1])
        if p.size > 2:
            dp[1:] = 0.5*(p[:-2] - p[2:])
        return self.B * float(np.sum(dp * self._wd(self.alpha * (t[-1] - t[:-1]))))


# ---------------------------------------------------------- Carter-Tracy influence p_D(tD)
def _pd_infinite(t_d: float) -> tuple[float, float]:
    """Infinite radial constant-terminal-rate ``p_D`` and ``dp_D/dt_D``.

    tD <= 100: Edwardson et al. rational fit.  tD > 100: the two-term asymptotic expansion
    ``pD = 0.5 (ln tD + 0.80907) + (ln tD + 1.80907) / (4 tD)``, which removes the 0.6 % jump the
    leading-order log approximation has at tD = 100.
    """
    t = float(t_d)
    if t <= 0:
        return 0.0, math.inf
    if t > 100:
        lt = math.log(t)
        p = 0.5*(lt + 0.80907) + (lt + 1.80907)/(4.0*t)
        dp = 0.5/t + (1.0 - lt - 1.80907)/(4.0*t*t)
        return p, dp
    root = math.sqrt(t)
    t15 = t*root
    p = (370.529*root + 137.582*t + 5.69549*t15) / (328.834 + 265.488*root + 45.2157*t + t15)
    e = 716.441 + 46.7984*root + 270.038*t + 71.0098*t15
    f = 1296.86*root + 1204.73*t + 618.618*t15 + 538.072*t*t + 142.41*t*t*root
    return p, e/f


class _FiniteRadialPressure:
    """Numerical constant-rate influence function for a closed finite radial aquifer.

    Method-of-lines solution of the dimensionless radial diffusivity equation with inner flux
    ``dpD/drD = -1`` and a closed outer boundary.  Only used to build a cached table.
    """

    def __init__(self, radius_ratio: float, n_nodes: int | None = None):
        self.R = float(radius_ratio)
        n = n_nodes or int(np.clip(30*self.R, 100, 350))
        self.r = np.linspace(1.0, self.R, n)
        self.dr = float(self.r[1] - self.r[0])

    def _rhs(self, _t: float, p: np.ndarray) -> np.ndarray:
        dr, r = self.dr, self.r
        out = np.empty_like(p)
        ghost_inner = p[1] + 2.0*dr             # dpD/drD = -1 at rD = 1
        out[0] = ((p[1] - 2*p[0] + ghost_inner)/dr**2 + (p[1] - ghost_inner)/(2*dr*r[0]))
        out[1:-1] = ((p[2:] - 2*p[1:-1] + p[:-2])/dr**2 + (p[2:] - p[:-2])/(2*dr*r[1:-1]))
        out[-1] = 2.0*(p[-2] - p[-1])/dr**2     # dpD/drD = 0 at rD = R (ghost node = p[-2])
        return out

    def table(self, t_d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sol = solve_ivp(self._rhs, (0.0, float(t_d[-1])), np.zeros_like(self.r), method="BDF",
                        rtol=1e-8, atol=1e-11, t_eval=t_d)
        if not sol.success:
            raise InputValidationError(f"Finite Carter-Tracy influence solve failed: {sol.message}")
        pd = sol.y[0]
        dpd = np.array([self._rhs(0.0, sol.y[:, i])[0] for i in range(sol.y.shape[1])])
        return pd, dpd


#: Cached finite-aquifer table spans tD in [_CT_TD_MIN, _CT_TD_MAX].
_CT_TD_MIN, _CT_TD_MAX, _CT_N = 1e-3, 1e7, 500
#: Below tD = CT_SWITCH * (R - 1)^2 the infinite-acting p_D is used for finite aquifers.
CT_SWITCH = 0.02


@lru_cache(maxsize=32)
def _finite_pd_table(radius_ratio: float) -> tuple[PchipInterpolator, PchipInterpolator]:
    """``p_D`` and ``ln(dp_D/dt_D)`` of a closed radial aquifer, tabulated once per radius ratio.

    The table depends on R only, so it is shared by every forward run and every history-match
    trial.  Interpolation is monotone-cubic in ln tD.
    """
    td = np.logspace(math.log10(_CT_TD_MIN), math.log10(_CT_TD_MAX), _CT_N)
    pd, dpd = _FiniteRadialPressure(radius_ratio).table(td)
    x = np.log(td)
    return PchipInterpolator(x, pd), PchipInterpolator(x, np.log(np.maximum(dpd, 1e-300)))


def _pd_finite(t_d: float, radius_ratio: float) -> tuple[float, float]:
    t = float(t_d)
    R = float(radius_ratio)
    if t <= 0:
        return 0.0, math.inf
    if t < max(_CT_TD_MIN, CT_SWITCH * (R - 1.0)**2):
        return _pd_infinite(t)
    pd_tab, lndpd_tab = _finite_pd_table(round(R, 10))
    if t > _CT_TD_MAX:   # pseudo-steady state: pD grows linearly with slope 2/(R^2 - 1)
        slope = 2.0 / (R**2 - 1.0)
        return float(pd_tab(math.log(_CT_TD_MAX))) + slope*(t - _CT_TD_MAX), slope
    x = math.log(t)
    return float(pd_tab(x)), float(math.exp(lndpd_tab(x)))


class CTAquifer:
    """Carter-Tracy aquifer (recursive constant-terminal-rate approximation to HVE).

    ``We_n = We_{n-1} + (tD_n - tD_{n-1}) [U (pi - p_n) - We_{n-1} pD'(tD_n)]
    / [pD(tD_n) - tD_{n-1} pD'(tD_n)]``
    """

    def __init__(self, properties: RadialAquiferProperties, *, allow_backflow: bool = True):
        self.props = properties
        self.B = properties.aquifer_constant_rb_psi
        self.alpha = properties.time_factor_per_day
        self.allow_backflow = bool(allow_backflow)
        self.reset(1.0)

    def reset(self, initial_pressure_psia: float, initial_time_days: float = 0.0) -> None:
        self.pi = float(initial_pressure_psia)
        self.t0 = float(initial_time_days)
        self.we = 0.0
        self.previous_td = 0.0
        self._pd_cache: tuple[float, tuple[float, float]] = (-1.0, (0.0, 0.0))

    def _pd(self, t_d: float) -> tuple[float, float]:
        # tD is fixed within a time step, so every root-finder trial reuses the same lookup.
        if self._pd_cache[0] != t_d:
            value = _pd_infinite(t_d) if self.props.radius_ratio is None else _pd_finite(t_d, self.props.radius_ratio)
            self._pd_cache = (t_d, value)
        return self._pd_cache[1]

    def preview(self, pressure_psia: float, previous_pressure_psia: float,
                time_days: float, previous_time_days: float) -> AquiferTrial:
        td = self.alpha * (float(time_days) - self.t0)
        if td <= self.previous_td:
            return AquiferTrial(self.we, None, 0.0)
        pd, dpd = self._pd(td)
        den = pd - self.previous_td*dpd
        if abs(den) < 1e-14:
            raise InputValidationError("Carter-Tracy influence-function denominator is zero")
        dtd = td - self.previous_td
        we = self.we + dtd*(self.B*(self.pi - float(pressure_psia)) - self.we*dpd)/den
        if not self.allow_backflow:
            we = max(self.we, we)
        dt = float(time_days) - float(previous_time_days)
        return AquiferTrial(float(we), None, (we - self.we)/dt if dt > 0 else 0.0)

    def commit(self, trial: AquiferTrial, pressure_psia: float, time_days: float) -> None:
        self.we = float(trial.cumulative_influx_rb)
        self.previous_td = self.alpha * (float(time_days) - self.t0)


# Descriptive aliases.
CarterTracyAquifer = CTAquifer
VanEverdingenHurstAquifer = HVEAquifer
