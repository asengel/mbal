"""Engineering plots: pressure history and a 2 x 3 diagnostic dashboard.

Dashboard panels
----------------
1. Pressure history: measured, simulated (and optionally a reference run, e.g. the initial guess)
2. Campbell plot: F/Et vs cumulative production, with N (or G) and N + We/Et lines
3. Model-specific straight line, evaluated at measured pressure and independent of fitted m:
   * gas reservoirs: normalised p/z = Bgi/Bg vs Gp
   * black oil with a gas cap: F/(Eo+Efw) vs (Eg+Efw)/(Eo+Efw)  (Havlena-Odeh, intercept N, slope mN)
   * other oils: F - We vs Et (Havlena-Odeh, slope N)
4. Dake plot (F/Et vs We/Et) for water-drive cases, otherwise pressure residuals
5. Drive indices: DDI, GCDI, GDI, CDI, WDI, IDI
6. Cumulative aquifer influx (water drive) or recovery factors (volumetric)

Diagnostic points use the *measured* pressures throughout, including the aquifer influx, which is
computed by driving the aquifer model with the measured pressure history.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import diagnostics as dg
from .aquifers import FetkovichAquifer, NoAquifer
from .data import SimulationResult
from .exceptions import MaterialBalanceError
from .material_balance import BlackOilMaterialBalance
from .simulator import Simulator


def _plt():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def _aquifer_title_line(simulator: Simulator) -> str:
    aq = simulator.aquifer
    if hasattr(aq, "props"):
        props = aq.props
        model = "CT" if aq.__class__.__name__.lower().startswith("ct") else "HVE"
        rr = "∞" if props.radius_ratio is None else f"{props.radius_ratio:g}"
        return (
            f"{model}: k={props.permeability_md:g} md, φ={props.porosity:g}, "
            f"ct={props.total_compressibility_1psi:.2e} psi⁻¹, μw={props.water_viscosity_cp:g} cp, "
            f"h={props.thickness_ft:g} ft, ri={props.inner_radius_ft:g} ft, re/ri={rr}, θ={props.angle_deg:g}°"
        )
    if isinstance(aq, FetkovichAquifer):
        return f"Fetkovich: J={aq.J:.3g} rb/d/psi, Wei={aq.Wei / 1e6:.2f} MMrb"
    return "Aquifer: none"


def _plain(ax, x: bool = True, y: bool = True) -> None:
    if x:
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    if y:
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def _nearly_constant_limits(ax, values) -> None:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    center, spread = 0.5 * (vals.min() + vals.max()), vals.max() - vals.min()
    ref = max(abs(center), 1.0)
    if spread <= 1e-6 * ref:
        margin = max(1e-3 * ref, 1e-6)
        ax.set_ylim(center - margin, center + margin)


def diagnostic_dashboard_figure(
    result: SimulationResult,
    simulator: Simulator,
    *,
    title_prefix: str = "Diagnostics",
    reference: SimulationResult | None = None,
    reference_label: str = "Initial guess",
):
    """Build the 2 x 3 diagnostic dashboard for one simulation result."""
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    res = simulator.reservoir
    inventory, symbol = res.inventory()
    unit, scale = ("MMSTB", 1e6) if symbol == "N" else ("Bscf", 1e9)
    m = float(getattr(res, "m", 0.0) or 0.0)
    has_aquifer = not isinstance(simulator.aquifer, NoAquifer)
    fig.suptitle(
        f"{title_prefix}\n{symbol}={inventory / scale:.3f} {unit}" + (f", m={m:.3f}" if m else "")
        + f"\n{_aquifer_title_line(simulator)}",
        fontsize=14,
    )

    times = np.asarray([r.time_days for r in result.rows])
    p_calc = np.asarray([r.pressure_psia for r in result.rows])
    p_meas = np.asarray([np.nan if r.pressure_observed_psia is None else r.pressure_observed_psia
                         for r in result.rows], dtype=float)
    we_calc = np.asarray([r.water_influx_rb for r in result.rows])
    pts = dg.material_balance_points(simulator)

    # 1. pressure history
    ax = axes[0, 0]
    ok = np.isfinite(p_meas)
    ax.plot(times[ok], p_meas[ok], "ro", label="Measured")
    if reference is not None:
        ax.plot([r.time_days for r in reference.rows], [r.pressure_psia for r in reference.rows],
                color="0.5", ls="--", lw=1.5, label=reference_label)
    ax.plot(times, p_calc, "b-", lw=2, label="Simulation")
    ax.set(title="1. Pressure history", xlabel="Time (days)", ylabel="Pressure (psia)")
    ax.legend()
    ax.grid(True)

    # 2. Campbell
    ax = axes[0, 1]
    if pts.Et.size:
        y = pts.F_net / pts.Et / scale
        x = pts.production / scale
        ax.plot(x, y, "bo", label="Data F/Et (measured p)")
        ax.plot(x, np.full_like(x, inventory / scale), "r--", label=f"{symbol} = {inventory / scale:.3f} {unit}")
        values = [y, [inventory / scale]]
        if has_aquifer:
            y_aq = (inventory + pts.We / pts.Et) / scale
            ax.plot(x, y_aq, color="orange", lw=2, label=f"{symbol} + We/Et")
            values.append(y_aq)
        _nearly_constant_limits(ax, np.concatenate([np.atleast_1d(v) for v in values]))
    ax.set(title=f"2. Campbell plot (F/Et vs {'Np' if symbol == 'N' else 'Gp'})",
           xlabel=f"Cumulative {'oil (MMSTB)' if symbol == 'N' else 'gas (Bscf)'}", ylabel=f"F/Et ({unit})")
    _plain(ax)
    ax.legend()
    ax.grid(True)

    # 3. model-specific straight line
    ax = axes[0, 2]
    try:
        if symbol == "G" and hasattr(res, "gas_fvf_ratio"):
            fit, g_pz = dg.pz_regression(simulator)
            ax.plot(fit.x / 1e9, fit.y, "ko", label="Data (measured p)")
            xl = np.linspace(0, max(fit.x.max(), g_pz) / 1e9, 20)
            ax.plot(xl, fit.intercept + fit.slope * xl * 1e9, "g-", label=f"Regression: G={g_pz / 1e9:.3f} Bscf")
            ax.plot(xl, 1 - xl * 1e9 / inventory, "r--", label=f"Model G={inventory / 1e9:.3f} Bscf (no Efw)")
            ax.set(title="3. Normalised p/z plot", xlabel="Gp (Bscf)", ylabel="(p/z)/(p/z)ᵢ = Bgi/Bg")
            ax.set_ylim(bottom=0)
        elif isinstance(res, BlackOilMaterialBalance) and res.m > 0:
            fit, n_ho, m_ho = dg.havlena_odeh_gas_cap(simulator, pts)
            ax.plot(fit.x, fit.y / 1e6, "ko", label="Data (measured p)")
            xl = np.linspace(0, fit.x.max() * 1.05, 20)
            ax.plot(xl, (fit.intercept + fit.slope * xl) / 1e6, "g-",
                    label=f"Regression: N={n_ho / 1e6:.2f} MM, m={m_ho:.3f}, R²={fit.r_squared:.3f}")
            ax.plot(xl, (res.N_stb + res.m * res.N_stb * xl) / 1e6, "r--",
                    label=f"Model: N={res.N_stb / 1e6:.2f} MM, m={res.m:.3f}")
            ax.set(title="3. Havlena-Odeh gas cap", xlabel="(Eg+Efw)/(Eo+Efw)",
                   ylabel="(F−We)/(Eo+Efw) (MMSTB)")
        else:
            fit, n_ho = dg.havlena_odeh_oil(pts)
            ax.plot(fit.x, fit.y / 1e6, "ko", label="Data (measured p)")
            xl = np.linspace(0, fit.x.max(), 10)
            ax.plot(xl, fit.slope * xl / 1e6, "g-", label=f"Regression: {symbol}={n_ho / scale:.3f} {unit}")
            ax.plot(xl, inventory * xl / 1e6, "r--", label=f"Model {symbol}={inventory / scale:.3f} {unit}")
            ax.set(title="3. Havlena-Odeh (F − We vs Et)", xlabel="Et (rb per unit inventory)",
                   ylabel="F − We (MMrb)")
        ax.legend(fontsize=8)
    except MaterialBalanceError as exc:
        ax.text(0.5, 0.5, f"Diagnostic unavailable:\n{exc}", ha="center", va="center", transform=ax.transAxes, fontsize=8)
    _plain(ax)
    ax.grid(True)

    # 4. Dake plot or residuals
    ax = axes[1, 0]
    if has_aquifer and pts.Et.size:
        x = pts.We / pts.Et / scale
        y = pts.F_net / pts.Et / scale
        ax.plot(x, y, "go", label="Data (measured p)")
        xl = np.linspace(0, max(float(x.max()), 1e-12), 10)
        ax.plot(xl, xl + inventory / scale, "r--", label=f"Slope 1, intercept {symbol}")
        ax.set(title="4. Dake plot", xlabel=f"We/Et ({unit})", ylabel=f"F/Et ({unit})")
        ax.legend()
    else:
        resid = p_calc[ok] - p_meas[ok]
        ax.bar(times[ok], resid, width=max(np.ptp(times) / 60, 1.0), color="tab:blue")
        ax.axhline(0, color="k", lw=0.8)
        rmse = float(np.sqrt(np.mean(resid ** 2))) if resid.size else 0.0
        ax.set(title=f"4. Pressure residuals (RMSE {rmse:.2f} psi)", xlabel="Time (days)",
               ylabel="Simulated − measured (psi)")
    _plain(ax)
    ax.grid(True)

    # 5. drive indices
    ax = axes[1, 1]
    di = dg.drive_indices(result)
    labels = {"DDI": "Depletion (oil)", "GCDI": "Gas cap", "GDI": "Gas expansion",
              "CDI": "Compaction + connate water", "WDI": "Aquifer", "IDI": "Injection"}
    active = (times > 0) & (sum(np.abs(v) for v in di.values()) > 0)
    series = [(labels[k], v[active]) for k, v in di.items() if np.any(np.abs(v[active]) > 1e-4)]
    if series and np.any(active):
        ax.stackplot(times[active], *[v for _, v in series], labels=[n for n, _ in series], alpha=0.7)
        ax.legend(loc="lower left", fontsize=8)
    ax.set(title="5. Drive indices", xlabel="Time (days)", ylabel="Fraction of withdrawal", ylim=(0, 1))

    # 6. aquifer influx or recovery
    ax = axes[1, 2]
    if has_aquifer:
        # Show both the forward-model influx and an independent diagnostic obtained by driving
        # the same aquifer with the measured pressure history.  This makes aquifer mismatch visible
        # without changing the pressure-history solve.
        we_measured = dg.influx_along_measured_pressure(simulator)
        ax.plot(times, we_calc / 1e6, "b-", lw=2, label="We (simulated p)")
        ax.plot(times, we_measured / 1e6, "k--", lw=1.5, label="We (measured p)")
        ax.set(title="6. Cumulative aquifer influx", xlabel="Time (days)", ylabel="We (MMrb)")
        ax.legend()
    else:
        orf = np.asarray([r.extras.get("oil_recovery_factor", 0.0) for r in result.rows])
        grf = np.asarray([r.extras.get("gas_recovery_factor", 0.0) for r in result.rows])
        if np.any(orf > 0):
            ax.plot(times, 100 * orf, "g-", lw=2, label="Oil (stock-tank component)")
        ax.plot(times, 100 * grf, "r-", lw=2, label="Gas (surface component)")
        ax.set(title="6. Recovery factors", xlabel="Time (days)", ylabel="Recovery (% of initial)")
        ax.legend()
    _plain(ax)
    ax.grid(True)

    fig.tight_layout()
    return fig


def pressure_history_figure(result: SimulationResult, *, title: str = "Pressure history"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot([r.time_days for r in result.rows], [r.pressure_psia for r in result.rows], "b-", lw=2, label="Simulation")
    obs = [(r.time_days, r.pressure_observed_psia) for r in result.rows if r.pressure_observed_psia is not None]
    if obs:
        ax.plot([x for x, _ in obs], [y for _, y in obs], "ro", label="Measured")
    ax.set(xlabel="Time (days)", ylabel="Pressure (psia)", title=title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def save_figure(fig, path: str | Path, *, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    _plt().close(fig)
    return path
