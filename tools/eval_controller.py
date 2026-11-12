"""Closed-loop evaluation of softsim.control.controller against the MuJoCo simulation (docs/surrogate.md).

    python tools/eval_controller.py cantilever [surrogate.json]
    python tools/eval_controller.py arm [surrogate.json]

The surrogate is fitted on the nominal body; the plant is the same scenario with the stiffness 20 % off (and, for the
cantilever, another mesh). Prints settling time and steady-state error with feedforward only and with feedback, for
step targets and a slow ramp, at the control rate of the Gymnasium environments (50 Hz and 20 Hz).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from softsim.control.controller import Controller
from softsim.control.inverse import build_simulation
from softsim.control.surrogate import Surrogate

ROOT = Path(__file__).parents[1]

CASES = {
    "cantilever": {
        "config": ROOT / "scenarios/actuation/cable_cantilever.yaml", "outputs": ["bend_angle"], "actuators": ["cable"],
        "base": ["body.mesh.size=4mm", "simulation.timestep=1ms", "body.material.damping={model: rayleigh, beta: 0.01}"],
        "E": "body.material.elastic.youngs_modulus", "E0": 500e3, "period": 0.02, "scales": {"bend_angle": 0.005},
        "variants": {"nominal": [], "E +20%": ["body.material.elastic.youngs_modulus=600kPa"],
                     "E -20%": ["body.material.elastic.youngs_modulus=400kPa"],
                     "mesh 6 mm": ["body.mesh.size=6mm"]},
        "steps": [[0.03], [0.06], [0.04]], "hold": 2.0, "ramp": ([0.02], [0.06], 4.0), "band": 0.05,
        "surrogate": ROOT / "runs/surrogate/cable_cantilever/surrogate.json", "ki": 6.0,
    },
    "arm": {
        "config": ROOT / "scenarios/actuation/cable_arm.yaml", "outputs": ["tip_x", "tip_z"],
        "actuators": ["cable_a", "cable_b"],
        "base": ["body.mesh.size=6mm", "simulation.timestep=0.5ms", "simulation.iterations=100",
                 "body.material.damping={model: rayleigh, beta: 0.15}",
                 "body.zones[0].material.damping={model: rayleigh, beta: 0.15}",
                 "body.zones[1].material.damping={model: rayleigh, beta: 0.15}"],
        "period": 0.05, "scales": {"tip_x": 0.5e-3, "tip_z": 0.5e-3},
        "variants": {"nominal": [], "E +20%": ["body.material.elastic.youngs_modulus=0.0"],
                     "E -20%": ["body.material.elastic.youngs_modulus=0.0"]},
        "steps": None, "hold": 3.0, "ramp": None, "band": 0.05,
        "surrogate": ROOT / "runs/surrogate/cable_arm/surrogate.json", "ki": 3.0,
    },
}


def run(case, variant, mode, targets, hold, sur_path, ki, ramp=None):
    """Return (time, outputs, targets, actuation, compute seconds per step, rest outputs)."""
    sim = build_simulation(case["config"], case["base"] + variant, "mujoco", None)
    ctl = Controller.from_json(sur_path, dt=case["period"], ki=ki, mode=mode, scales=case["scales"],
                               outputs=case["outputs"])
    sim.reset()
    sub = int(round(case["period"] / sim.config.simulation.timestep))
    outs = case["outputs"]
    if ramp:
        a, b, dur = map(np.asarray, (ramp[0], ramp[1], ramp[2]))
        n_steps = int(round(float(dur) / case["period"]))
        sched = [a + (b - a) * min(k / n_steps, 1.0) for k in range(n_steps + int(1.0 / case["period"]))]
    else:
        per = int(round(hold / case["period"]))
        sched = [np.asarray(t) for t in targets for _ in range(per)]
    T, Y, R, U, cpu = [], [], [], [], []
    y = rest = np.array([sim.metrics()[o] for o in outs])
    for k, r in enumerate(sched):
        t0 = time.perf_counter()
        u = ctl.step(y, r)
        cpu.append(time.perf_counter() - t0)
        sim.apply_controls(dict(zip(case["actuators"], map(float, u))))
        sim.step(sub)
        m = sim.metrics()
        y = np.array([m[o] for o in outs])
        T.append((k + 1) * case["period"])
        Y.append(y)
        R.append(r)
        U.append(u)
    return np.array(T), np.array(Y), np.array(R), np.array(U), float(np.mean(cpu)), rest


def step_metrics(T, Y, R, period, hold, band, rest):
    """Per target step: settling time (|e| stays within band*|step| of the target; nan if never) and steady error."""
    per = int(round(hold / period))
    rows, prev = [], rest
    for i in range(len(T) // per):
        s = slice(i * per, (i + 1) * per)
        e = np.linalg.norm(Y[s] - R[s], axis=1)
        amp = np.linalg.norm(R[s][0] - prev)
        ok = e <= band * amp
        idx = np.where(~ok)[0]
        settle = 0.0 if len(idx) == 0 else (np.nan if idx[-1] == per - 1 else (idx[-1] + 1) * period)
        rows.append((settle, float(np.linalg.norm((Y[s] - R[s])[-int(per * 0.2):], axis=1).mean()), float(amp)))
        prev = R[s][0]
    return rows


if __name__ == "__main__":
    name = sys.argv[1]
    case = CASES[name]
    sur_path = Path(sys.argv[2]) if len(sys.argv) > 2 else case["surrogate"]
    sur = Surrogate.load(sur_path)
    if name == "arm":
        pts = [sur.predict([0.45, 0.15])[0], sur.predict([0.3, 0.3])[0], sur.predict([0.1, 0.4])[0]]
        case["steps"] = [p.tolist() for p in pts]
        case["ramp"] = (pts[0].tolist(), pts[2].tolist(), 5.0)
        # the Ecoflex modulus is in the preset: give it explicitly (card value 0.1 MPa class); see docs
        base = build_simulation(case["config"], case["base"], "mujoco", None)
        E = next(b.material.elastic.youngs_modulus for b in base.scene.soft_bodies if b.name == "arm")
        case["variants"] = {"nominal": [], "E +20%": [f"body.material.elastic.youngs_modulus={1.2 * E!r}"],
                            "E -20%": [f"body.material.elastic.youngs_modulus={0.8 * E!r}"]}
    results = []
    for vname, variant in case["variants"].items():
        for mode in ("ff", "integral"):
            T, Y, R, U, cpu, rest = run(case, variant, mode, case["steps"], case["hold"], sur_path, case["ki"])
            rows = step_metrics(T, Y, R, case["period"], case["hold"], case["band"], rest)
            Tr, Yr, Rr, Ur, _, _ = run(case, variant, mode, None, 0, sur_path, case["ki"], ramp=case["ramp"])
            half = len(Tr) // 2
            ramp_err = float(np.linalg.norm((Yr - Rr)[half:], axis=1).mean())
            res = {"plant": vname, "mode": mode, "steps": [{"settle_s": r[0], "steady_err": r[1], "amp": r[2]} for r in rows],
                   "ramp_err_second_half": ramp_err, "controller_ms_per_step": 1e3 * cpu}
            results.append(res)
            print(json.dumps(res), flush=True)
    out = ROOT / f"runs/eval_controller_{name}.json"
    out.write_text(json.dumps(results, indent=1))
