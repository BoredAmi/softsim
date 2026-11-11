"""Leave-one-out validation of the whole-bounce drop model: fit on one weight, predict the other.

    softsim calibrate calibrations/loo_ecoflex_00_10_fit_w020.yaml    # ~1.5 h each, run both
    softsim calibrate calibrations/loo_ecoflex_00_10_fit_w050.yaml
    python tools/leave_one_out.py                                    # ~5 min

Each fold fits all seven parameters of the MuJoCo whole-bounce model (stiffness,
damping, tack, cloth support) to one recording, from neutral starting values.
This runs the fitted model on the other recording with that recording's
observables and windows, and prints the error next to the in-sample errors
(the fold on its own recording, and the model fitted on both). Writes
runs/leave_one_out/report.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from softsim.experiments.calibrate import Calibration, _evaluate_case

ROOT = Path(__file__).resolve().parents[1]
FOLDS = {"w020": ROOT / "calibrations" / "loo_ecoflex_00_10_fit_w020.yaml",
         "w050": ROOT / "calibrations" / "loo_ecoflex_00_10_fit_w050.yaml"}
BOTH = ROOT / "calibrations" / "ecoflex_00_10_drop_adhesion_support.yaml"
OUT = ROOT / "runs" / "leave_one_out"
KEYS = ("nrmse", "indentation_rel.nrmse", "specimen_lift.nrmse", "peak_sim", "peak_exp")


def _report(cal: Calibration) -> dict:
    return yaml.safe_load((cal.output / "report.yaml").read_text())


def _score(fit: Calibration, case: str) -> dict:
    """Run ``fit``'s best model on ``case`` with the joint calibration's observables for that case."""
    from softsim.config.loader import apply_overrides, load_yaml, set_dotted

    both = Calibration(BOTH)
    raw = apply_overrides(load_yaml(both.dataset.config_path(case)), fit.overrides)
    for k, v in _report(fit)["best"].items():
        raw = set_dotted(raw, k, v)
    m = _evaluate_case((raw, str(both.dataset.config_path(case).parent), both.tracks, case, both.observables[case]))
    return {k: m.get(k) for k in KEYS} | ({"error": m["error"]} if "error" in m else {})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    both = _report(Calibration(BOTH))
    rows = []
    for fold, path in FOLDS.items():
        cal = Calibration(path)
        other = next(c for c in FOLDS if c != fold)
        rows.append({"fitted_on": fold, "values": _report(cal)["best"],
                     "in_sample": {fold: _score(cal, fold)}, "predicted": {other: _score(cal, other)},
                     "both_fit": {c: both["per_case"][c].get("nrmse") for c in FOLDS}})
    (OUT / "report.json").write_text(json.dumps(rows, indent=1, default=float))
    print("| Fitted on | Error on its own recording | Error predicting the other | The model fitted on both, same recording |")
    print("| --- | --- | --- | --- |")
    for r in rows:
        (own, s_own), = r["in_sample"].items()
        (other, s_pred), = r["predicted"].items()
        print(f"| {r['fitted_on']} | {100 * s_own['nrmse']:.0f}% | **{other}: {100 * s_pred['nrmse']:.0f}%** | "
              f"{100 * r['both_fit'][other]:.0f}% |")
    for r in rows:
        print(r["fitted_on"], {k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r["values"].items()})


if __name__ == "__main__":
    main()
