"""Fit E and one Prony term on the reference FEM to the first contact of the centred drops.

    python tools/fit_reference_drop.py                 # 50 and 20 g; ~3 h on 2 cores (12 x 8 mesh)
    python tools/fit_reference_drop.py w050            # a subset

Neo-Hookean, nu = 0.4995 (nearly incompressible), sticking contact, Prony
(g, tau = 2 ms) on the isochoric part. 50 g lands centred; 20 g (6 mm off
centre, spinning) is run centred, so it is an approximation. 500 g is left
out: on the reference its full-face load builds so slowly that the
experiment's contact rule (0.1 mm departure from free fall) fires only after
several mm of travel, so its first-contact metric is not comparable. Evaluations go to
runs/reference_fit/<cases>/evaluations.jsonl (resumes if re-run).
"""

from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from softsim.experiments.compare import ExperimentCase, compare_case, simulated_indentation
from softsim.fem.drop import FemSettings, run_drop

ROOT = Path(__file__).resolve().parents[1]
TRACKS = ROOT / "data" / "processed" / "czopek2026_ecoflex_drop"
OUT = ROOT / "runs" / "reference_fit"
CASES = tuple(sys.argv[1:]) or ("w050", "w020")
OUT = OUT / "_".join(CASES)
DURATION = {"w500": 0.018}  # its first contact lasts ~15 ms
TAU = 0.002
FEM = FemSettings(cells_r=12, cells_z=8)


def evaluate(args):
    case, e, g = args
    exp = ExperimentCase.load(TRACKS, case)
    mat = (f"{{density: 1040, elastic: {{model: neo_hookean, youngs_modulus: {e}, poisson_ratio: 0.4995}}, "
           f"damping: {{model: prony, g: [{g}], tau: [{TAU}]}}}}")
    try:
        ts, _ = run_drop(ROOT / "scenarios" / "drop_test" / "cases" / f"{case}.yaml",
                         [*exp.sim_overrides(), f"specimen.material={mat}", f"simulation.duration={DURATION.get(case, 0.011)}",
                          "simulation.output_rate=10000"], FEM, ignore_asymmetry=True)
        return compare_case(simulated_indentation(ts), exp)
    except Exception as exc:  # noqa: BLE001 - a failed run costs the maximum
        return {"nrmse": float("nan"), "error": str(exc)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "evaluations.jsonl"
    done = {}
    if log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line)
            done[(round(r["E"], 3), round(r["g"], 6))] = r["results"]
    pool = ProcessPoolExecutor(len(CASES))

    def cost(x):
        e, g = math.exp(x[0]), 1 / (1 + math.exp(-x[1]))
        key = (round(e, 3), round(g, 6))
        if key not in done:
            done[key] = list(pool.map(evaluate, [(c, e, g) for c in CASES]))
            with log.open("a") as f:
                f.write(json.dumps({"E": e, "g": g, "tau": TAU, "results": done[key]}, default=float) + "\n")
        n = [r.get("nrmse", float("nan")) for r in done[key]]
        c = float(np.mean([v if v == v else 10.0 for v in n]))
        print(f"E={e/1e3:7.1f} kPa g={g:.3f}: cost {c:.4f}  " + "  ".join(
            f"{k} {r.get('nrmse', float('nan')):.3f} (peak {r.get('peak_sim', float('nan'))*1e3:.2f} mm @ "
            f"{r.get('t_peak_sim', float('nan'))*1e3:.1f} ms)" for k, r in zip(CASES, done[key])), flush=True)
        return c

    x0 = np.array([math.log(110e3), 0.0])  # 110 kPa, g = 0.5
    res = minimize(cost, x0, method="Nelder-Mead",
                   options={"maxfev": 30, "initial_simplex": [x0, x0 + [0.25, 0], x0 + [0, 1.0]], "xatol": 0.01})
    e, g = math.exp(res.x[0]), 1 / (1 + math.exp(-res.x[1]))
    print(f"best: E = {e/1e3:.1f} kPa, g = {g:.3f}, tau = {TAU*1e3:.0f} ms, cost {res.fun:.4f}")
    (OUT / "best.json").write_text(json.dumps({"E": e, "g": g, "tau": TAU, "cost": res.fun}))
    pool.shutdown()


if __name__ == "__main__":
    main()
