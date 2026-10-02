"""Regenerate the bundled example history-match outputs into ``outputs/``."""
import json
from pathlib import Path

from mbal.config import load_case
from mbal.history_match import HistoryMatcher, parameters_from_case
from mbal.plotting import diagnostic_dashboard_figure, save_figure

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def run_case(case_path: Path) -> dict:
    name = case_path.parent.name
    out = OUTPUTS / name
    out.mkdir(parents=True, exist_ok=True)
    case = load_case(case_path)
    matcher = HistoryMatcher(case)

    initial_sim = matcher.factory.build({p.path: p.initial for p in parameters_from_case(case)})
    initial = initial_sim.run()
    initial.to_csv(out / "initial_results.csv")

    fit = matcher.fit()
    fit.to_json(out / "match.json")
    matched_sim = matcher.factory.build(fit.parameters)
    matched = matched_sim.run()
    matched.to_csv(out / "matched_results.csv")

    save_figure(diagnostic_dashboard_figure(initial, initial_sim, title_prefix="INITIAL GUESS"),
                out / "initial_guess.png")
    save_figure(diagnostic_dashboard_figure(matched, matched_sim, title_prefix="OPTIMIZED MATCH", reference=initial),
                out / "history_match.png")
    return {"case": name, **fit.to_dict(), "simulation_warnings": matched.metadata["warnings"]}


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    summaries = []
    for case_path in sorted(EXAMPLES.glob("*/case.yaml")):
        print(f"Running {case_path.parent.name} ...")
        summaries.append(run_case(case_path))
    (OUTPUTS / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote outputs for {len(summaries)} cases to {OUTPUTS}")


if __name__ == "__main__":
    main()
