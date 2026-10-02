"""Reservoir-fluid material-balance formulations (field units, SPE nomenclature).

Every model evaluates, at a trial pressure ``p``, the terms of

    F = E_HC + E_fw + We + Winj*Bw_inj + Ginj*Bg_inj

and returns them as :class:`BalanceTerms` with ``residual = support - withdrawal``.

Models
------
* :class:`GasMaterialBalance`              dry / single-phase gas
* :class:`BlackOilMaterialBalance`         black oil, optional initial gas cap (m)
* :class:`GeneralizedOilMaterialBalance`   volatile oil (Walsh generalized, Rs + Rv)
* :class:`GeneralizedGasMaterialBalance`   gas condensate / wet gas (Walsh generalized)
* :class:`GeneralizedTwoPhaseMaterialBalance` initially saturated oil + free gas (Walsh GMBE)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .data import HistoryRecord
from .exceptions import InputValidationError
from .pvt import PVTTable

#: Relative tolerance on Rsi vs Rs(Pb) (and Rvi vs Rv(Pd)) before the input is rejected / flagged.
SATURATION_RATIO_TOLERANCE = 0.01


@dataclass(frozen=True)
class BalanceTerms:
    """Material-balance terms at one pressure, all in reservoir barrels [rb]."""

    residual_rb: float
    underground_withdrawal_rb: float
    hydrocarbon_expansion_rb: float
    oil_expansion_rb: float = 0.0
    gas_expansion_rb: float = 0.0
    gas_cap_expansion_rb: float = 0.0
    rock_water_expansion_rb: float = 0.0
    water_influx_rb: float = 0.0
    water_injection_rb: float = 0.0
    gas_injection_rb: float = 0.0


class MaterialBalanceModel(Protocol):
    """Interface the simulator, diagnostics and plotting rely on."""

    pvt: PVTTable
    initial_pressure_psia: float
    saturation_pressure_psia: float | None
    initial_hcpv_rb: float
    initial_oil_stb: float
    initial_gas_scf: float
    input_warnings: list[str]

    def inventory(self) -> tuple[float, str]:
        """Primary fitted inventory and its symbol: ``(N [STB], "N")`` or ``(G [scf], "G")``."""
        ...

    def single_phase_ratio(self) -> tuple[str, float] | None:
        """``("GOR", Rsi)`` or ``("CGR", Rvi)`` expected for production in the single-phase region."""
        ...

    def balance(self, pressure_psia: float, history: HistoryRecord, water_influx_rb: float) -> BalanceTerms: ...


CONNATE_WATER_EXPANSION_MODES = {"compressibility", "bw_table"}


def effective_rock_water_compressibility(cf_psi_inv: float, cw_psi_inv: float, swi: float) -> float:
    """``c_fw = (cf + Swi cw) / (1 - Swi)`` [1/psi]."""
    if not (0.0 <= swi < 1.0):
        raise InputValidationError("Swi must satisfy 0 <= Swi < 1")
    return (cf_psi_inv + swi * cw_psi_inv) / (1.0 - swi)


def _require_finite(**values: float | None) -> None:
    for name, value in values.items():
        if value is not None and not math.isfinite(float(value)):
            raise InputValidationError(f"{name} must be a finite number, got {value!r}")


def _relative_difference(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


class _BaseModel:
    """Shared rock/connate-water expansion, injection terms and inventory bookkeeping.

    Connate-water expansion is ``Swi cw (pi - p)`` (``compressibility``) or
    ``Swi (Bw(p)/Bw(pi) - 1)`` from the PVT water column (``bw_table``).
    """

    #: Saturation pressure used by the formulation, or None for single-phase gas.
    saturation_pressure_psia: float | None = None
    #: Total initial stock-tank oil (free + vaporized) and surface gas (free + dissolved).
    initial_oil_stb: float = 0.0
    initial_gas_scf: float = 0.0
    initial_hcpv_rb: float = 0.0

    def __init__(
        self,
        pvt: PVTTable,
        *,
        initial_pressure_psia: float,
        cf_psi_inv: float = 0.0,
        cw_psi_inv: float = 0.0,
        swi: float = 0.0,
        bw_rb_per_stb: float = 1.0,
        bw_inj_rb_per_stb: float = 1.0,
        bg_inj_rb_per_scf: float | None = None,
        connate_water_expansion: str = "compressibility",
    ):
        _require_finite(
            initial_pressure_psia=initial_pressure_psia, cf_psi_inv=cf_psi_inv, cw_psi_inv=cw_psi_inv,
            swi=swi, bw_rb_per_stb=bw_rb_per_stb, bw_inj_rb_per_stb=bw_inj_rb_per_stb,
            bg_inj_rb_per_scf=bg_inj_rb_per_scf,
        )
        if initial_pressure_psia <= 0:
            raise InputValidationError("Initial pressure must be positive")
        if cf_psi_inv < 0 or cw_psi_inv < 0:
            raise InputValidationError("Compressibilities must be non-negative")
        if bw_rb_per_stb <= 0 or bw_inj_rb_per_stb <= 0:
            raise InputValidationError("Water formation volume factors must be positive")
        if bg_inj_rb_per_scf is not None and bg_inj_rb_per_scf <= 0:
            raise InputValidationError("Bg_inj_rb_per_scf must be positive")
        if connate_water_expansion not in CONNATE_WATER_EXPANSION_MODES:
            raise InputValidationError(
                f"connate_water_expansion must be one of {sorted(CONNATE_WATER_EXPANSION_MODES)}"
            )
        self.pvt = pvt
        self.initial_pressure_psia = float(initial_pressure_psia)
        self.cf = float(cf_psi_inv)
        self.cw = float(cw_psi_inv)
        self.swi = float(swi)
        self.c_eff = effective_rock_water_compressibility(self.cf, self.cw, self.swi)
        self.bw = float(bw_rb_per_stb)
        self.bw_inj = float(bw_inj_rb_per_stb)
        self.bg_inj = None if bg_inj_rb_per_scf is None else float(bg_inj_rb_per_scf)
        self.connate_water_expansion = connate_water_expansion
        self.input_warnings: list[str] = []
        if connate_water_expansion == "bw_table":
            if pvt.bw is None:
                raise InputValidationError("connate_water_expansion: bw_table requires a Bw_rb_per_stb PVT column")
            self.bwi = self.pvt.Bw(self.initial_pressure_psia)
        else:
            self.bwi = self.pvt.Bw(self.initial_pressure_psia, default=self.bw) if pvt.bw is not None else self.bw

    # -------------------------------------------------------------------------- interface
    def inventory(self) -> tuple[float, str]:
        if self.initial_oil_stb > 0 and not hasattr(self, "G_scf"):
            return self.initial_oil_stb, "N"
        return self.initial_gas_scf, "G"

    def single_phase_ratio(self) -> tuple[str, float] | None:
        return None

    # ------------------------------------------------------------------------- shared terms
    def _bw_at(self, p: float) -> float:
        return self.pvt.Bw(p, default=self.bw)

    def rock_water_fraction(self, p: float) -> float:
        """Rock + connate-water expansion as a fraction of the initial HCPV (dimensionless)."""
        dp = self.initial_pressure_psia - p
        if self.connate_water_expansion == "bw_table":
            water = self.swi * (self.pvt.Bw(p) / self.bwi - 1.0)
        else:
            water = self.swi * self.cw * dp
        return (self.cf * dp + water) / (1.0 - self.swi)

    def _rock_water_expansion(self, p: float) -> float:
        return self.initial_hcpv_rb * self.rock_water_fraction(p)

    def _injection(self, p: float, h: HistoryRecord) -> tuple[float, float]:
        water = h.winj_stb * self.bw_inj
        if h.ginj_scf <= 0:
            gas = 0.0
        else:
            bg = self.bg_inj if self.bg_inj is not None else self.pvt.Bg(p)
            gas = h.ginj_scf * bg
        return water, gas

    def _terms(self, withdrawal: float, oil: float, gas: float, gas_cap: float, p: float,
               h: HistoryRecord, we: float) -> BalanceTerms:
        rock_water = self._rock_water_expansion(p)
        water_inj, gas_inj = self._injection(p, h)
        hc = oil + gas + gas_cap
        support = hc + rock_water + we + water_inj + gas_inj
        return BalanceTerms(
            residual_rb=support - withdrawal,
            underground_withdrawal_rb=withdrawal,
            hydrocarbon_expansion_rb=hc,
            oil_expansion_rb=oil,
            gas_expansion_rb=gas,
            gas_cap_expansion_rb=gas_cap,
            rock_water_expansion_rb=rock_water,
            water_influx_rb=we,
            water_injection_rb=water_inj,
            gas_injection_rb=gas_inj,
        )


# --------------------------------------------------------------------------------- dry gas
class GasMaterialBalance(_BaseModel):
    """Dry / single-phase gas: ``Gp Bg + Wp Bw = G (Bg - Bgi) + G Bgi c_fw dp + We + ...``."""

    def __init__(self, pvt: PVTTable, *, G_scf: float, **kwargs):
        super().__init__(pvt, **kwargs)
        _require_finite(G_scf=G_scf)
        if G_scf <= 0:
            raise InputValidationError("G_scf must be positive")
        self.G_scf = float(G_scf)
        self.bgi = self.pvt.Bg(self.initial_pressure_psia)
        self.initial_hcpv_rb = self.G_scf * self.bgi
        self.initial_gas_scf = self.G_scf
        self.initial_oil_stb = 0.0

    def inventory(self) -> tuple[float, str]:
        return self.G_scf, "G"

    def gas_fvf_ratio(self, p: float) -> float:
        """``Bgi / Bg(p)`` = ``(p/z) / (p/z)_i`` for a single-phase gas."""
        return self.bgi / self.pvt.Bg(p)

    def balance(self, pressure_psia: float, history: HistoryRecord, water_influx_rb: float) -> BalanceTerms:
        bg = self.pvt.Bg(pressure_psia)
        gas_expansion = self.G_scf * (bg - self.bgi)
        withdrawal = history.gp_scf * bg + history.wp_stb * self._bw_at(pressure_psia)
        return self._terms(withdrawal, 0.0, gas_expansion, 0.0, pressure_psia, history, water_influx_rb)


# ------------------------------------------------------------------------------- black oil
class BlackOilMaterialBalance(_BaseModel):
    """Conventional black-oil material balance with an optional initial gas cap.

    ``F = Np Bt + (Gp - Np Rsi) Bg + Wp Bw``,
    ``Eo = Bt - Bti``,  ``Eg = Bti (Bg/Bgi - 1)``,  ``Efw = (1 + m) Bti c_fw dp``,
    with ``Bt = Bo + (Rsi - Rs) Bg`` below the bubble point and ``Bt = Bo`` above it.
    """

    def __init__(
        self,
        pvt: PVTTable,
        *,
        N_stb: float,
        saturation_pressure_psia: float,
        rsi_scf_per_stb: float,
        m: float = 0.0,
        **kwargs,
    ):
        super().__init__(pvt, **kwargs)
        _require_finite(N_stb=N_stb, m=m, Pb_psia=saturation_pressure_psia, Rsi_scf_per_stb=rsi_scf_per_stb)
        if N_stb <= 0:
            raise InputValidationError("N_stb must be positive")
        if m < 0:
            raise InputValidationError("Gas-cap ratio m cannot be negative")
        if saturation_pressure_psia <= 0 or rsi_scf_per_stb < 0:
            raise InputValidationError("Saturation pressure and Rsi must be physically valid")
        pi, pb = self.initial_pressure_psia, float(saturation_pressure_psia)
        if pb > pi * (1.0 + 1e-9):
            raise InputValidationError(
                f"Pb={pb:g} psia exceeds Pi={pi:g} psia; an initially saturated oil must have Pb = Pi"
            )
        if m > 0 and abs(pb - pi) > 1e-6 * pi:
            raise InputValidationError(
                f"An initial gas cap (m={m:g}) requires Pi = Pb (got Pi={pi:g}, Pb={pb:g} psia)"
            )
        self.N_stb = float(N_stb)
        self.m = float(m)
        self.pb: float = pb
        self.saturation_pressure_psia = pb
        self.rsi = float(rsi_scf_per_stb)
        if pvt.rs is not None:
            rs_pb = pvt.saturated_value("rs", pb)
            if _relative_difference(self.rsi, rs_pb) > SATURATION_RATIO_TOLERANCE:
                raise InputValidationError(
                    f"Rsi={self.rsi:g} scf/STB disagrees with Rs(Pb)={rs_pb:g} scf/STB from the PVT table by more "
                    f"than {100 * SATURATION_RATIO_TOLERANCE:g}%; Bt would be discontinuous at Pb"
                )
        self.bti = self.pvt.Bt(self.initial_pressure_psia, saturation_pressure_psia=pb, rsi_scf_per_stb=self.rsi)
        self.bgi = self.pvt.Bg(self.initial_pressure_psia)
        self.initial_hcpv_rb = (1.0 + self.m) * self.N_stb * self.bti
        self.initial_oil_stb = self.N_stb
        self.initial_gas_scf = self.N_stb * self.rsi + self.m * self.N_stb * self.bti / self.bgi

    def inventory(self) -> tuple[float, str]:
        return self.N_stb, "N"

    def single_phase_ratio(self) -> tuple[str, float] | None:
        return ("GOR", self.rsi)

    def unit_expansions(self, p: float) -> dict[str, float]:
        """Havlena-Odeh expansions per STB of initial oil: ``Eo``, ``Eg`` and ``Efw`` (oil-zone) [rb/STB]."""
        bt = self.pvt.Bt(p, saturation_pressure_psia=self.pb, rsi_scf_per_stb=self.rsi)
        bg = self.pvt.Bg(p)
        return {
            "Eo": bt - self.bti,
            "Eg": self.bti * (bg / self.bgi - 1.0),
            "Efw": self.bti * self.rock_water_fraction(p),
        }

    def balance(self, pressure_psia: float, history: HistoryRecord, water_influx_rb: float) -> BalanceTerms:
        bt = self.pvt.Bt(pressure_psia, saturation_pressure_psia=self.pb, rsi_scf_per_stb=self.rsi)
        bg = self.pvt.Bg(pressure_psia)
        oil_expansion = self.N_stb * (bt - self.bti)
        gas_cap = self.N_stb * self.m * self.bti * (bg - self.bgi) / self.bgi if self.m > 0.0 else 0.0
        # Np*Bt + (Gp - Np*Rsi)*Bg == Np*[Bo + (Rp - Rs)Bg], but stays well behaved when Np = 0.
        withdrawal = (
            history.np_stb * bt
            + (history.gp_scf - history.np_stb * self.rsi) * bg
            + history.wp_stb * self._bw_at(pressure_psia)
        )
        return self._terms(withdrawal, oil_expansion, 0.0, gas_cap, pressure_psia, history, water_influx_rb)


# ---------------------------------------------------------------------------- volatile oil
class GeneralizedOilMaterialBalance(_BaseModel):
    """Walsh generalized material balance for a volatile oil initially at or above Pb.

    Below Pb: ``Eo = Bto - Boi`` with ``Bto = [Bo (1 - Rsi Rv) + Bg (Rsi - Rs)] / (1 - Rs Rv)`` and
    ``F = [Np (Bo - Rs Bg) + Gp (Bg - Rv Bo)] / (1 - Rs Rv) + Wp Bw``.

    Above Pb the same withdrawal expression is used with Rs = Rsi and Bg, Rv frozen at their
    saturated values at Pb.  For produced GOR = Rsi it reduces exactly to ``Np Bo``; any excess gas
    is withdrawn at the saturation-state gas volume, so F is continuous across Pb.
    """

    def __init__(
        self,
        pvt: PVTTable,
        *,
        N_stb: float,
        saturation_pressure_psia: float,
        rsi_scf_per_stb: float | None = None,
        **kwargs,
    ):
        super().__init__(pvt, **kwargs)
        _require_finite(N_stb=N_stb, Pb_psia=saturation_pressure_psia, Rsi_scf_per_stb=rsi_scf_per_stb)
        if N_stb <= 0:
            raise InputValidationError("N_stb must be positive")
        if saturation_pressure_psia <= 0:
            raise InputValidationError("Bubble-point pressure must be positive")
        if saturation_pressure_psia > self.initial_pressure_psia * (1.0 + 1e-9):
            raise InputValidationError(
                "volatile_oil requires Pi >= Pb (an initially two-phase system should use generalized_two_phase)"
            )
        self.N_stb = float(N_stb)
        self.pb: float = float(saturation_pressure_psia)
        self.saturation_pressure_psia = self.pb
        pb = self.pb
        rs_pb = self.pvt.saturated_value("rs", pb)
        self.rsi = float(rsi_scf_per_stb) if rsi_scf_per_stb is not None else rs_pb
        if self.rsi < 0:
            raise InputValidationError("Rsi cannot be negative")
        if _relative_difference(self.rsi, rs_pb) > SATURATION_RATIO_TOLERANCE:
            self.input_warnings.append(
                f"Rsi={self.rsi:g} scf/STB differs from the saturated-branch Rs(Pb)={rs_pb:g} scf/STB by "
                f"{100 * _relative_difference(self.rsi, rs_pb):.1f}%"
            )
        self._bg_sat = self.pvt.saturated_value("bg", pb)
        self._rv_sat = self.pvt.saturated_value("rv", pb)
        if 1.0 - self.rsi * self._rv_sat <= 1e-12:
            raise InputValidationError("1 - Rsi*Rv(Pb) <= 0: the volatile-oil PVT table is inconsistent at Pb")
        self.boi = self.pvt.Bo(self.initial_pressure_psia)
        self.initial_hcpv_rb = self.N_stb * self.boi
        self.initial_oil_stb = self.N_stb
        self.initial_gas_scf = self.N_stb * self.rsi

    def inventory(self) -> tuple[float, str]:
        return self.N_stb, "N"

    def single_phase_ratio(self) -> tuple[str, float] | None:
        return ("GOR", self.rsi)

    def balance(self, pressure_psia: float, history: HistoryRecord, water_influx_rb: float) -> BalanceTerms:
        p = pressure_psia
        if p >= self.pb:
            bo = self.pvt.Bo(p)
            oil_expansion = self.N_stb * (bo - self.boi)
            den = 1.0 - self.rsi * self._rv_sat
            oil_coeff = (bo - self.rsi * self._bg_sat) / den
            gas_coeff = (self._bg_sat - self._rv_sat * bo) / den
        else:
            bto = self.pvt.generalized_Bto(p, rsi_scf_per_stb=self.rsi)
            oil_expansion = self.N_stb * (bto - self.boi)
            oil_coeff, gas_coeff = self.pvt.generalized_withdrawal_coefficients(p)
        withdrawal = history.np_stb * oil_coeff + history.gp_scf * gas_coeff + history.wp_stb * self._bw_at(p)
        return self._terms(withdrawal, oil_expansion, 0.0, 0.0, p, history, water_influx_rb)


# ------------------------------------------------------------------- gas condensate / wet gas
class GeneralizedGasMaterialBalance(_BaseModel):
    """Walsh generalized material balance for gas condensate and wet gas.

    ``saturation_pressure_psia`` is the dew point; ``None`` means a wet gas that never forms a
    reservoir liquid (produced condensate is carried by Rvi).

    Below Pd: ``Eg = Btg - Bgi`` with ``Btg = [Bg (1 - Rvi Rs) + Bo (Rvi - Rv)] / (1 - Rs Rv)``.
    Above Pd the generalized withdrawal is used with Rv = Rvi and Bo, Rs frozen at their saturated
    values at Pd: for a produced CGR = Rvi it reduces exactly to ``Gp Bg`` and stays continuous
    across Pd otherwise.
    """

    def __init__(
        self,
        pvt: PVTTable,
        *,
        G_scf: float,
        saturation_pressure_psia: float | None,
        rvi_stb_per_scf: float | None = None,
        **kwargs,
    ):
        super().__init__(pvt, **kwargs)
        _require_finite(G_scf=G_scf, Pd_psia=saturation_pressure_psia, Rvi_stb_per_scf=rvi_stb_per_scf)
        if G_scf <= 0:
            raise InputValidationError("G_scf must be positive")
        if saturation_pressure_psia is not None and saturation_pressure_psia <= 0:
            raise InputValidationError("Dew-point pressure must be positive (omit Pd_psia for a wet gas)")
        if saturation_pressure_psia is not None and saturation_pressure_psia > self.initial_pressure_psia * (1.0 + 1e-9):
            raise InputValidationError(
                "gas_condensate requires Pi >= Pd (an initially two-phase system should use generalized_two_phase)"
            )
        self.G_scf = float(G_scf)
        self.saturation_pressure_psia = None if saturation_pressure_psia is None else float(saturation_pressure_psia)
        if rvi_stb_per_scf is not None:
            self.rvi = float(rvi_stb_per_scf)
        elif self.pvt.rv is not None:
            self.rvi = self.pvt.Rv(self.initial_pressure_psia)
        else:
            raise InputValidationError("Provide Rvi or an Rv PVT column for gas_condensate / wet_gas")
        if self.rvi < 0:
            raise InputValidationError("Rvi cannot be negative")
        self.bgi = self.pvt.Bg(self.initial_pressure_psia)
        if self.saturation_pressure_psia is not None:
            pd = self.saturation_pressure_psia
            rv_pd = self.pvt.saturated_value("rv", pd)
            if _relative_difference(self.rvi, rv_pd) > SATURATION_RATIO_TOLERANCE:
                self.input_warnings.append(
                    f"Rvi={self.rvi * 1e6:g} STB/MMscf differs from the saturated-branch Rv(Pd)={rv_pd * 1e6:g} "
                    f"STB/MMscf by {100 * _relative_difference(self.rvi, rv_pd):.1f}%"
                )
            self._bo_sat = self.pvt.saturated_value("bo", pd)
            self._rs_sat = self.pvt.saturated_value("rs", pd)
            if 1.0 - self._rs_sat * self.rvi <= 1e-12:
                raise InputValidationError("1 - Rs(Pd)*Rvi <= 0: the gas-condensate PVT table is inconsistent at Pd")
        self.initial_hcpv_rb = self.G_scf * self.bgi
        self.initial_gas_scf = self.G_scf
        self.initial_oil_stb = self.G_scf * self.rvi

    def inventory(self) -> tuple[float, str]:
        return self.G_scf, "G"

    def single_phase_ratio(self) -> tuple[str, float] | None:
        return ("CGR", self.rvi)

    def gas_fvf_ratio(self, p: float) -> float:
        """``Bgi/Bg`` above Pd and ``Bgi/Btg`` (two-phase equivalent) below it."""
        if self.saturation_pressure_psia is None or p >= self.saturation_pressure_psia:
            return self.bgi / self.pvt.Bg(p)
        return self.bgi / self.pvt.generalized_Btg(p, rvi_stb_per_scf=self.rvi)

    def balance(self, pressure_psia: float, history: HistoryRecord, water_influx_rb: float) -> BalanceTerms:
        p = pressure_psia
        if self.saturation_pressure_psia is None:
            bg = self.pvt.Bg(p)
            gas_expansion = self.G_scf * (bg - self.bgi)
            withdrawal = history.gp_scf * bg + history.wp_stb * self._bw_at(p)
        elif p >= self.saturation_pressure_psia:
            bg = self.pvt.Bg(p)
            gas_expansion = self.G_scf * (bg - self.bgi)
            den = 1.0 - self._rs_sat * self.rvi
            oil_coeff = (self._bo_sat - self._rs_sat * bg) / den
            gas_coeff = (bg - self.rvi * self._bo_sat) / den
            withdrawal = history.np_stb * oil_coeff + history.gp_scf * gas_coeff + history.wp_stb * self._bw_at(p)
        else:
            btg = self.pvt.generalized_Btg(p, rvi_stb_per_scf=self.rvi)
            gas_expansion = self.G_scf * (btg - self.bgi)
            oil_coeff, gas_coeff = self.pvt.generalized_withdrawal_coefficients(p)
            withdrawal = history.np_stb * oil_coeff + history.gp_scf * gas_coeff + history.wp_stb * self._bw_at(p)
        return self._terms(withdrawal, 0.0, gas_expansion, 0.0, p, history, water_influx_rb)


# ------------------------------------------------------------------------ two-phase (GMBE)
class GeneralizedTwoPhaseMaterialBalance(_BaseModel):
    """Walsh GMBE for an initially saturated oil + free-gas system (Pi = Psat).

    ``F = N_foi (Bto - Boi) + G_fgi (Btg - Bgi) + Efw + We + ...``
    """

    def __init__(
        self,
        pvt: PVTTable,
        *,
        N_foi_stb: float,
        G_fgi_scf: float | None = None,
        m: float | None = None,
        rsi_scf_per_stb: float | None = None,
        rvi_stb_per_scf: float | None = None,
        **kwargs,
    ):
        super().__init__(pvt, **kwargs)
        _require_finite(N_foi_stb=N_foi_stb, G_fgi_scf=G_fgi_scf, m=m, Rsi_scf_per_stb=rsi_scf_per_stb,
                        Rvi_stb_per_scf=rvi_stb_per_scf)
        if N_foi_stb <= 0:
            raise InputValidationError("N_foi_stb must be positive")
        self.saturation_pressure_psia = self.initial_pressure_psia
        self.N_foi_stb = float(N_foi_stb)
        self.boi = self.pvt.Bo(self.initial_pressure_psia)
        self.bgi = self.pvt.Bg(self.initial_pressure_psia)
        self.rsi = self.pvt.Rs(self.initial_pressure_psia) if rsi_scf_per_stb is None else float(rsi_scf_per_stb)
        self.rvi = self.pvt.Rv(self.initial_pressure_psia) if rvi_stb_per_scf is None else float(rvi_stb_per_scf)

        if G_fgi_scf is None:
            if m is None or m < 0:
                raise InputValidationError("Provide either G_fgi_scf or a non-negative m")
            G_fgi_scf = float(m) * self.N_foi_stb * self.boi / self.bgi
        if G_fgi_scf < 0:
            raise InputValidationError("G_fgi_scf cannot be negative")
        self.G_fgi_scf = float(G_fgi_scf)
        self.initial_hcpv_rb = self.N_foi_stb * self.boi + self.G_fgi_scf * self.bgi
        self.initial_oil_stb = self.N_foi_stb + self.G_fgi_scf * self.rvi
        self.initial_gas_scf = self.G_fgi_scf + self.N_foi_stb * self.rsi

    def inventory(self) -> tuple[float, str]:
        return self.N_foi_stb, "N"

    def balance(self, pressure_psia: float, history: HistoryRecord, water_influx_rb: float) -> BalanceTerms:
        p = pressure_psia
        bto = self.pvt.generalized_Bto(p, rsi_scf_per_stb=self.rsi)
        btg = self.pvt.generalized_Btg(p, rvi_stb_per_scf=self.rvi)
        oil_expansion = self.N_foi_stb * (bto - self.boi)
        gas_expansion = self.G_fgi_scf * (btg - self.bgi)
        oil_coeff, gas_coeff = self.pvt.generalized_withdrawal_coefficients(p)
        withdrawal = history.np_stb * oil_coeff + history.gp_scf * gas_coeff + history.wp_stb * self._bw_at(p)
        return self._terms(withdrawal, oil_expansion, gas_expansion, 0.0, p, history, water_influx_rb)
