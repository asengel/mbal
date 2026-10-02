"""Nonlinear pressure history matching of selected physical parameters."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .config import Case, ModelFactory, get_dotted
from .exceptions import InputValidationError, MaterialBalanceError

#: Parameters whose physical lower bound follows from cumulative production.
_INVENTORY_PARAMETERS = {"reservoir.N_stb": "oil", "reservoir.G_scf": "gas"}


@dataclass(frozen=True)
class Parameter:
    path: str
    initial: float
    lower: float
    upper: float
    log: bool

    def encode(self, value: float) -> float:
        return math.log(value) if self.log else value

    def decode(self, value: float) -> float:
        return math.exp(value) if self.log else value


@dataclass
class MatchResult:
    initial_parameters: dict[str, float]
    parameters: dict[str, float]
    initial_rmse_psia: float
    rmse_psia: float
    weighted_rmse: float
    success: bool
    nfev: int
    message: str
    warnings: list[str]
    parameter_correlation: dict[str, dict[str, float]]
    parameter_uncertainty: dict[str, dict[str, float | str]] = field(default_factory=dict)
    failed_trials: list[dict] = field(default_factory=list)

    @property
    def improvement_percent(self) -> float:
        if self.initial_rmse_psia <= 0:
            return 0.0
        return 100.0 * (1.0 - self.rmse_psia / self.initial_rmse_psia)

    @property
    def parameter_changes(self) -> dict[str, dict[str, float]]:
        out = {}
        for path, final in self.parameters.items():
            initial = self.initial_parameters[path]
            out[path] = {
                "initial": initial,
                "matched": final,
                "ratio": final / initial if initial != 0 else math.nan,
                "change_percent": 100.0 * (final - initial) / initial if initial != 0 else math.nan,
            }
        return out

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "initial_rmse_psia": self.initial_rmse_psia,
            "rmse_psia": self.rmse_psia,
            "weighted_rmse": self.weighted_rmse,
            "improvement_percent": self.improvement_percent,
            "nfev": self.nfev,
            "message": self.message,
            "warnings": self.warnings,
            "parameter_correlation": self.parameter_correlation,
            "parameter_uncertainty": self.parameter_uncertainty,
            "initial_parameters": self.initial_parameters,
            "matched_parameters": self.parameters,
            "parameter_changes": self.parameter_changes,
            "failed_trials": self.failed_trials[:20],
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def parameters_from_case(case: Case) -> list[Parameter]:
    raw = (case.config.get("history_match") or {}).get("parameters") or {}
    if not isinstance(raw, dict) or not raw:
        raise InputValidationError("history_match.parameters must be a mapping of dotted path to [lower, upper]")
    out = []
    for path, spec in raw.items():
        initial = float(get_dotted(case.config, path))
        if isinstance(spec, dict):
            bounds = spec.get("bounds")
            transform = spec.get("transform", "auto")
        else:
            bounds, transform = spec, "auto"
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise InputValidationError(f"{path}: bounds must be [lower, upper]")
        lo, hi = map(float, bounds)
        if not lo < hi or not lo <= initial <= hi:
            raise InputValidationError(f"{path}: initial value must lie inside increasing bounds")
        use_log = (lo > 0) if transform == "auto" else transform == "log"
        if use_log and lo <= 0:
            raise InputValidationError(f"{path}: log transform requires positive bounds")
        out.append(Parameter(path, initial, lo, hi, use_log))
    return out


class HistoryMatcher:
    """Weighted nonlinear least-squares match of observed pressures.

    Residuals are ``sqrt(w_i) (p_calc - p_obs) / sigma_i``.  Positive parameters are optimised in
    log space.  Inventory parameters (N, G) get a lower bound just above cumulative production, so
    the optimiser never enters a region where the model is undefined.
    """

    def __init__(self, case: Case):
        self.case = case
        self.factory = ModelFactory(case)
        h = case.config.get("history_match") or {}
        self.loss = str(h.get("loss", "soft_l1"))
        self.sigma_default = float(h.get("pressure_sigma_psia", 1.0))
        self.max_nfev = int(h.get("max_nfev", 500))
        sim = self.factory.build()
        self.parameters = self._with_inventory_bounds(parameters_from_case(case), sim)
        self.indices = [i for i, x in enumerate(sim.history) if x.pressure_psia is not None and x.pressure_weight > 0]
        if len(self.indices) <= len(self.parameters):
            raise InputValidationError("History matching needs more pressure observations than fitted parameters")
        self.obs = np.array([sim.history[i].pressure_psia for i in self.indices], float)
        self.sig = np.array([sim.history[i].pressure_sigma_psia or self.sigma_default for i in self.indices], float)
        self.w = np.sqrt(np.array([sim.history[i].pressure_weight for i in self.indices], float))
        self.failures: list[dict] = []
        self._penalty: float | None = None

    @staticmethod
    def _with_inventory_bounds(params: list[Parameter], sim) -> list[Parameter]:
        max_np = max((h.np_stb for h in sim.history), default=0.0)
        max_gp = max((h.gp_scf - h.ginj_scf for h in sim.history), default=0.0)
        out = []
        for p in params:
            component = _INVENTORY_PARAMETERS.get(p.path)
            floor = 1.001 * (max_np if component == "oil" else max_gp) if component else 0.0
            if component and p.lower < floor:
                if p.upper <= floor or p.initial <= floor:
                    raise InputValidationError(
                        f"{p.path}: bounds/initial value must exceed cumulative production ({floor / 1.001:g})"
                    )
                p = Parameter(p.path, p.initial, floor, p.upper, p.log)
            out.append(p)
        return out

    def _decode(self, x: np.ndarray) -> dict[str, float]:
        return {p.path: p.decode(float(v)) for p, v in zip(self.parameters, x)}

    def _simulate(self, x: np.ndarray) -> np.ndarray:
        result = self.factory.build(self._decode(x)).run()
        return np.fromiter((result.rows[i].pressure_psia for i in self.indices), float, len(self.indices))

    def _residuals(self, x: np.ndarray) -> np.ndarray:
        try:
            calc = self._simulate(x)
        except MaterialBalanceError as exc:              # physics failures only; bugs propagate
            self.failures.append({"parameters": self._decode(x), "error": str(exc)})
            return np.full(self.obs.size, self._penalty or 1e6)
        return self.w * (calc - self.obs) / self.sig

    def _rmse(self, overrides: dict[str, float]) -> tuple[float, float]:
        calc = self._simulate(np.array([p.encode(overrides[p.path]) for p in self.parameters]))
        plain = float(np.sqrt(np.mean((calc - self.obs) ** 2)))
        weighted = float(np.sqrt(np.mean((self.w * (calc - self.obs) / self.sig) ** 2)))
        return plain, weighted

    def fit(self) -> MatchResult:
        x0 = np.array([p.encode(p.initial) for p in self.parameters])
        lo = np.array([p.encode(p.lower) for p in self.parameters])
        hi = np.array([p.encode(p.upper) for p in self.parameters])
        initial_parameters = {p.path: p.initial for p in self.parameters}
        r0 = self._residuals(x0)
        self._penalty = 10.0 * max(1.0, float(np.max(np.abs(r0))))
        initial_rmse, _ = self._rmse(initial_parameters)

        opt = least_squares(self._residuals, x0, bounds=(lo, hi), loss=self.loss,
                            max_nfev=self.max_nfev, x_scale="jac")
        params = self._decode(opt.x)
        rmse, weighted = self._rmse(params)

        warnings: list[str] = []
        correlation: dict[str, dict[str, float]] = {}
        uncertainty: dict[str, dict[str, float | str]] = {}
        names = [p.path for p in self.parameters]
        m, n = self.obs.size, len(self.parameters)
        if opt.jac.size and m > n:
            jtj = opt.jac.T @ opt.jac
            s2 = float(np.sum(opt.fun ** 2)) / (m - n)        # residual variance (reduced chi-square)
            covariance = s2 * np.linalg.pinv(jtj)
            sigma = np.sqrt(np.maximum(np.diag(covariance), 0.0))
            for i, p in enumerate(self.parameters):
                xi = float(opt.x[i])
                uncertainty[p.path] = {
                    "sigma_transformed": float(sigma[i]),
                    "transform": "log" if p.log else "linear",
                    "lower_1sigma": p.decode(xi - sigma[i]),
                    "upper_1sigma": p.decode(xi + sigma[i]),
                }
            if n > 1:
                denom = np.outer(sigma, sigma)
                corr = np.divide(covariance, denom, out=np.zeros_like(covariance), where=denom > 0)
                correlation = {a: {b: float(corr[i, j]) for j, b in enumerate(names)} for i, a in enumerate(names)}
                pairs = [f"{names[i]} / {names[j]} ({corr[i, j]:+.3f})"
                         for i in range(n) for j in range(i + 1, n) if abs(corr[i, j]) >= 0.95]
                if pairs:
                    warnings.append(
                        "Strong fitted-parameter correlation: " + "; ".join(pairs)
                        + ". The pressure match may not uniquely identify these parameters."
                    )
        for path, p in zip(names, self.parameters):
            value = params[path]
            if math.isclose(value, p.lower, rel_tol=1e-6) or math.isclose(value, p.upper, rel_tol=1e-6):
                warnings.append(f"{path} finished on a bound ({value:g}); widen the bounds or check the model")
        if self.failures:
            warnings.append(f"{len(self.failures)} trial(s) failed with a material-balance error (see failed_trials)")

        return MatchResult(
            initial_parameters=initial_parameters,
            parameters=params,
            initial_rmse_psia=initial_rmse,
            rmse_psia=rmse,
            weighted_rmse=weighted,
            success=bool(opt.success),
            nfev=int(opt.nfev),
            message=str(opt.message),
            warnings=warnings,
            parameter_correlation=correlation,
            parameter_uncertainty=uncertainty,
            failed_trials=list(self.failures),
        )
