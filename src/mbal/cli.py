"""Command-line interface: ``mbal run`` and ``mbal match``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ModelFactory, load_case
from .exceptions import MaterialBalanceError
from .history_match import HistoryMatcher, parameters_from_case
from .plotting import diagnostic_dashboard_figure, pressure_history_figure, save_figure


def _run(args: argparse.Namespace) -> int:
    case = load_case(args.case)
    sim = ModelFactory(case).build()
    result = sim.run()
    result.to_csv(args.output)
    title = (case.config.get("project") or {}).get("name", "Pressure history")
    plot = save_figure(pressure_history_figure(result, title=title), args.plot)
    payload = {"results": str(Path(args.output).resolve()), "plot": str(plot.resolve()),
               "warnings": result.metadata["warnings"]}
    if args.dashboard:
        dash = save_figure(diagnostic_dashboard_figure(result, sim, title_prefix=title), args.dashboard)
        payload["dashboard"] = str(dash.resolve())
    print(json.dumps(payload, indent=2))
    return 0


def _match(args: argparse.Namespace) -> int:
    case = load_case(args.case)
    matcher = HistoryMatcher(case)
    factory = matcher.factory
    initial_sim = factory.build({p.path: p.initial for p in parameters_from_case(case)})
    initial = initial_sim.run()

    fit = matcher.fit()
    fit.to_json(args.match_json)
    matched_sim = factory.build(fit.parameters)
    matched = matched_sim.run()
    matched.to_csv(args.output)

    initial_plot = Path(args.plot).with_name("initial_guess.png")
    save_figure(diagnostic_dashboard_figure(initial, initial_sim, title_prefix="INITIAL GUESS"), initial_plot)
    plot = save_figure(
        diagnostic_dashboard_figure(matched, matched_sim, title_prefix="OPTIMIZED MATCH", reference=initial),
        args.plot,
    )
    print(json.dumps({**fit.to_dict(), "results": str(Path(args.output).resolve()),
                      "plot": str(plot.resolve()), "initial_plot": str(initial_plot.resolve())}, indent=2))
    return 0 if fit.success else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mbal", description="Generalized reservoir material balance")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a forward material-balance case")
    run.add_argument("case")
    run.add_argument("--output", default="results.csv")
    run.add_argument("--plot", default="pressure_history.png")
    run.add_argument("--dashboard", default=None, help="Optional path for the 2x3 diagnostic dashboard")
    run.set_defaults(func=_run)

    match = sub.add_parser("match", help="History-match selected parameters")
    match.add_argument("case")
    match.add_argument("--output", default="matched_results.csv")
    match.add_argument("--match-json", default="match.json")
    match.add_argument("--plot", default="history_match.png")
    match.set_defaults(func=_match)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except MaterialBalanceError as exc:
        print(f"mbal: {exc}", file=sys.stderr)
        return 1
