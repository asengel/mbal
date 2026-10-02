"""YAML case files -> model construction, with caching of PVT and history for repeated builds."""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .aquifers import CTAquifer, FetkovichAquifer, HVEAquifer, NoAquifer, RadialAquiferProperties
from .data import HistoryRecord, read_history_csv
from .exceptions import InputValidationError
from .material_balance import (
    BlackOilMaterialBalance,
    GasMaterialBalance,
    GeneralizedGasMaterialBalance,
    GeneralizedOilMaterialBalance,
    GeneralizedTwoPhaseMaterialBalance,
)
from .pvt import PVTInterpolation, PVTTable, check_pvt_table, read_pvt_csv
from .simulator import PressureSolverSettings, Simulator


@dataclass(frozen=True)
class Case:
    config: dict[str, Any]
    path: Path

    @property
    def base_dir(self) -> Path:
        return self.path.parent


def load_case(path: str | Path) -> Case:
    path = Path(path).resolve()
    if not path.is_file():
        raise InputValidationError(f"Case file not found: {path}")
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise InputValidationError(f"Cannot parse {path.name}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise InputValidationError("Case YAML must contain a mapping")
    return Case(cfg, path)


def _resolve(case: Case, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (case.base_dir / p).resolve()


def get_dotted(mapping: dict[str, Any], path: str) -> Any:
    obj: Any = mapping
    for key in path.split("."):
        if not isinstance(obj, dict) or key not in obj:
            raise InputValidationError(f"Unknown parameter path: {path}")
        obj = obj[key]
    return obj


def set_dotted(mapping: dict[str, Any], path: str, value: float) -> None:
    keys = path.split(".")
    obj = mapping
    for key in keys[:-1]:
        if key not in obj or not isinstance(obj[key], dict):
            raise InputValidationError(f"Unknown parameter path: {path}")
        obj = obj[key]
    if keys[-1] not in obj:
        raise InputValidationError(f"Unknown parameter path: {path}")
    obj[keys[-1]] = float(value)


def config_with_overrides(case: Case, overrides: dict[str, float] | None = None) -> dict[str, Any]:
    cfg = deepcopy(case.config)
    for path, value in (overrides or {}).items():
        set_dotted(cfg, path, value)
    return cfg


def _number(section: dict[str, Any], key: str, default=None, *, positive=False, nonnegative=False):
    value = section.get(key, default)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise InputValidationError(f"{key} must be numeric") from None
    if not math.isfinite(value):
        raise InputValidationError(f"{key} must be finite")
    if positive and value <= 0:
        raise InputValidationError(f"{key} must be positive")
    if nonnegative and value < 0:
        raise InputValidationError(f"{key} cannot be negative")
    return value


def _interpolation_from(cfg: dict[str, Any]) -> PVTInterpolation:
    i = (cfg.get("pvt") or {}).get("interpolation") or {}
    return PVTInterpolation(
        bo=i.get("Bo", "linear_pressure"), rs=i.get("Rs", "linear_pressure"),
        bg=i.get("Bg", "reciprocal_pressure"), rv=i.get("Rv", "linear_pressure"),
        bw=i.get("Bw", "linear_pressure"),
        bt_below_saturation=i.get("Bt_below_saturation", "linear_pressure"),
        bt_above_saturation=i.get("Bt_above_saturation", "linear_pressure"),
        extrapolation=i.get("extrapolation", "error"),
        split_at_saturation=bool(i.get("split_at_saturation", True)),
    )


def _build_reservoir(case: Case, cfg: dict[str, Any], warnings: list[str], factory: "ModelFactory"):
    r = cfg.get("reservoir") or {}
    kind = str(r.get("type", "")).lower()
    aliases = {
        "gas": "dry_gas", "single_phase_gas": "dry_gas", "dry_gas": "dry_gas",
        "undersaturated_oil": "black_oil", "saturated_oil": "black_oil",
        "oil_with_gas_cap": "black_oil", "black_oil": "black_oil",
        "volatile_oil": "volatile_oil", "wet_gas": "wet_gas",
        "gas_condensate": "gas_condensate", "generalized_two_phase": "generalized_two_phase",
    }
    if kind not in aliases:
        raise InputValidationError(f"Unsupported reservoir.type: {kind!r}")
    kind = aliases[kind]
    pi = _number(r, "Pi_psia", positive=True)
    if pi is None:
        raise InputValidationError("reservoir.Pi_psia is required")
    sat = None
    if kind in {"black_oil", "volatile_oil"}:
        sat = _number(r, "Pb_psia", positive=True)
        if sat is None:
            raise InputValidationError("reservoir.Pb_psia is required")
    elif kind == "gas_condensate":
        sat = _number(r, "Pd_psia", positive=True)
        if sat is None:
            raise InputValidationError("reservoir.Pd_psia is required")
    elif kind == "wet_gas":
        sat = _number(r, "Pd_psia", None, nonnegative=True) or None  # 0 / blank = never two-phase
    elif kind == "generalized_two_phase":
        sat = pi

    temperature_f = _number(r, "T_degF", None)
    pvt, pvt_warnings = factory.pvt(cfg, sat, temperature_f, kind_for_check=kind, pi=pi)
    warnings.extend(pvt_warnings)
    common = dict(
        initial_pressure_psia=pi,
        cf_psi_inv=_number(r, "cf_1psi", 0.0, nonnegative=True),
        cw_psi_inv=_number(r, "cw_1psi", 0.0, nonnegative=True),
        swi=_number(r, "Swi", 0.0, nonnegative=True),
        bw_rb_per_stb=_number(r, "Bw_rb_stb", 1.0, positive=True),
        bw_inj_rb_per_stb=_number(r, "Bw_inj_rb_stb", r.get("Bw_rb_stb", 1.0), positive=True),
        bg_inj_rb_per_scf=_number(r, "Bg_inj_rb_scf", None, positive=True),
        connate_water_expansion=str(r.get("connate_water_expansion", "compressibility")),
    )
    if common["swi"] >= 1:
        raise InputValidationError("reservoir.Swi must be < 1")

    if kind == "dry_gas":
        G = _number(r, "G_scf", positive=True)
        if G is None:
            raise InputValidationError("reservoir.G_scf is required")
        return GasMaterialBalance(pvt, G_scf=G, **common)

    if kind == "black_oil":
        assert sat is not None
        N = _number(r, "N_stb", positive=True)
        if N is None:
            raise InputValidationError("reservoir.N_stb is required")
        rsi = _number(r, "Rsi_scf_stb", None, nonnegative=True)
        if rsi is None:
            if pvt.rs is None:
                raise InputValidationError("Provide Rsi_scf_stb or an Rs PVT column")
            rsi = pvt.saturated_value("rs", sat)
        return BlackOilMaterialBalance(
            pvt, N_stb=N, m=_number(r, "m", 0.0, nonnegative=True),
            saturation_pressure_psia=sat, rsi_scf_per_stb=rsi, **common,
        )

    if kind == "volatile_oil":
        assert sat is not None
        N = _number(r, "N_stb", positive=True)
        if N is None:
            raise InputValidationError("reservoir.N_stb is required")
        return GeneralizedOilMaterialBalance(
            pvt, N_stb=N, saturation_pressure_psia=sat,
            rsi_scf_per_stb=_number(r, "Rsi_scf_stb", None, nonnegative=True), **common,
        )

    if kind in {"wet_gas", "gas_condensate"}:
        G = _number(r, "G_scf", positive=True)
        if G is None:
            raise InputValidationError("reservoir.G_scf is required")
        rvi = _number(r, "Rvi_stb_MMscf", None, nonnegative=True)
        rvi = None if rvi is None else rvi / 1e6
        if kind == "wet_gas" and rvi is None and pvt.rv is None:
            rvi = 0.0
        return GeneralizedGasMaterialBalance(
            pvt, G_scf=G, saturation_pressure_psia=sat, rvi_stb_per_scf=rvi, **common,
        )

    N = _number(r, "N_foi_stb", positive=True)
    if N is None:
        raise InputValidationError("reservoir.N_foi_stb is required")
    rvi = _number(r, "Rvi_stb_MMscf", None, nonnegative=True)
    return GeneralizedTwoPhaseMaterialBalance(
        pvt, N_foi_stb=N,
        G_fgi_scf=_number(r, "G_fgi_scf", None, nonnegative=True),
        m=_number(r, "m", None, nonnegative=True),
        rsi_scf_per_stb=_number(r, "Rsi_scf_stb", None, nonnegative=True),
        rvi_stb_per_scf=None if rvi is None else rvi / 1e6,
        **common,
    )


def _build_aquifer(cfg: dict[str, Any]):
    a = cfg.get("aquifer") or {"model": "none"}
    model = str(a.get("model", "none")).lower().replace("-", "_")
    aliases = {"none": "none", "fetkovich": "fetkovich", "ct": "ct", "carter_tracy": "ct",
               "hve": "hve", "van_everdingen_hurst": "hve"}
    if model not in aliases:
        raise InputValidationError("aquifer.model must be one of: none, fetkovich, ct, hve")
    model = aliases[model]
    backflow = bool(a.get("allow_backflow", True))
    if model == "none":
        return NoAquifer()
    if model == "fetkovich":
        J = _number(a, "J_rb_day_psi", nonnegative=True)
        Wei = _number(a, "Wei_rb", nonnegative=True)
        if J is None or Wei is None:
            raise InputValidationError("Fetkovich aquifer requires J_rb_day_psi and Wei_rb")
        return FetkovichAquifer(J, Wei, allow_backflow=backflow)

    needed = ("k_md", "phi", "ct_1psi", "muw_cp", "h_ft", "ri_ft")
    missing = [k for k in needed if a.get(k) is None]
    if missing:
        raise InputValidationError(f"{model.upper()} aquifer requires {', '.join(missing)}")
    props = RadialAquiferProperties(
        permeability_md=_number(a, "k_md", positive=True),
        porosity=_number(a, "phi", positive=True),
        total_compressibility_1psi=_number(a, "ct_1psi", positive=True),
        water_viscosity_cp=_number(a, "muw_cp", positive=True),
        thickness_ft=_number(a, "h_ft", positive=True),
        inner_radius_ft=_number(a, "ri_ft", positive=True),
        radius_ratio=_number(a, "re_ri", None, positive=True),
        angle_deg=_number(a, "angle_deg", 360.0, positive=True),
    )
    if model == "ct":
        return CTAquifer(props, allow_backflow=backflow)
    return HVEAquifer(props, allow_backflow=backflow)


_FLUID_SYSTEM = {
    "dry_gas": "dry_gas", "black_oil": "black_oil", "volatile_oil": "volatile_oil",
    "wet_gas": "gas_condensate", "gas_condensate": "gas_condensate",
    "generalized_two_phase": "two_phase",
}


def _solver_settings(cfg: dict[str, Any]) -> PressureSolverSettings:
    s = cfg.get("solver") or {}
    return PressureSolverSettings(
        pressure_tolerance_psi=_number(s, "pressure_tolerance_psi", 1e-5, positive=True),
        closure_relative_tolerance=_number(s, "closure_relative_tolerance", 1e-7, positive=True),
        min_pressure_psia=_number(s, "min_pressure_psia", None, positive=True),
        max_pressure_psia=_number(s, "max_pressure_psia", None, positive=True),
        bracket_points=int(_number(s, "bracket_points", 240, positive=True)),
        max_step_days=_number(s, "max_step_days", 30.0, nonnegative=True),
    )


class ModelFactory:
    """Builds simulators for one case, caching everything that does not depend on the overrides.

    PVT tables (and their sanity checks) are cached per (file, saturation pressure, temperature,
    interpolation settings) and the history per file, so a history match re-creates only the
    lightweight reservoir and aquifer objects for each trial.
    """

    def __init__(self, case: Case):
        self.case = case
        self._pvt: dict[tuple, tuple[PVTTable, list[str]]] = {}
        self._history: dict[Path, tuple[list[HistoryRecord], list[str]]] = {}

    def pvt(self, cfg: dict[str, Any], sat: float | None, temperature_f: float | None, *,
            kind_for_check: str, pi: float) -> tuple[PVTTable, list[str]]:
        p = cfg.get("pvt") or {}
        if "file" not in p:
            raise InputValidationError("pvt.file is required")
        path = _resolve(self.case, str(p["file"]))
        interp = _interpolation_from(cfg)
        key = (path, sat, temperature_f, interp, kind_for_check, pi)
        if key not in self._pvt:
            warnings: list[str] = []
            table = read_pvt_csv(path, interp, saturation_pressure_psia=sat, warnings=warnings,
                                 temperature_f=temperature_f)
            warnings.extend(check_pvt_table(
                table, fluid_system=_FLUID_SYSTEM[kind_for_check], initial_pressure_psia=pi,
                saturation_pressure_psia=sat, needs_two_phase_properties=(kind_for_check == "gas_condensate"),
            ))
            self._pvt[key] = (table, warnings)
        return self._pvt[key]

    def history(self, cfg: dict[str, Any]) -> tuple[list[HistoryRecord], list[str]]:
        h = cfg.get("history") or {}
        if "file" not in h:
            raise InputValidationError("history.file is required")
        path = _resolve(self.case, str(h["file"]))
        if path not in self._history:
            warnings: list[str] = []
            self._history[path] = (read_history_csv(path, warnings=warnings), warnings)
        return self._history[path]

    def build(self, overrides: dict[str, float] | None = None) -> Simulator:
        cfg = config_with_overrides(self.case, overrides)
        warnings: list[str] = []
        reservoir = _build_reservoir(self.case, cfg, warnings, self)
        history, history_warnings = self.history(cfg)
        warnings.extend(history_warnings)
        sim = Simulator(reservoir, _build_aquifer(cfg), history, pressure_solver=_solver_settings(cfg))
        sim.input_warnings = list(dict.fromkeys(warnings + sim.input_warnings))
        return sim


def build_simulator(case: Case, overrides: dict[str, float] | None = None,
                    factory: ModelFactory | None = None) -> Simulator:
    """Build a :class:`Simulator` for ``case`` with dotted-path ``overrides`` applied."""
    return (factory or ModelFactory(case)).build(overrides)
