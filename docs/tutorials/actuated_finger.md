# Tutorial: an actuated body and its transfer to MuJoCo

## 1. A cable-bent beam

`scenarios/actuation/cable_cantilever.yaml` is a 10 x 10 x 60 mm beam clamped at its base, with a cable 3 mm off
its axis pulled to 0.7 N. A cable puts a pure moment on the beam, so Euler-Bernoulli gives the tip angle exactly:
0.270 rad.

```bash
softsim run scenarios/actuation/cable_cantilever.yaml --backend mujoco
softsim run scenarios/actuation/cable_cantilever.yaml --backend fem --set simulation.timestep=2ms
```

`summary.yaml` has `final_bend_angle`. The reference lands within 1% of the beam theory. MuJoCo bends about 30%
less at this 2 mm mesh, because linear tetrahedra lock in bending. That isn't a bug in either backend: it's the
error bar of linear tetrahedra at this resolution, and the [benchmark table](../benchmarks.md) states it.

Actuators are YAML entries with a schedule. From Python, a controller can drive them in closed loop instead. It's
called before every output interval and returns the actuator values; here a PI loop holds the tip at 0.1 rad:

```python
from softsim import Simulation

sim = Simulation("scenarios/actuation/cable_cantilever.yaml",
                 ["body.mesh.size=2.5mm", "simulation.duration=600ms",
                  "body.material.damping={model: rayleigh, alpha: 0, beta: 2 ms}"])
state = {"integral": 0.0}

def controller(s):
    error = 0.1 - s.metrics()["bend_angle"]
    state["integral"] += error * 1e-3          # output interval: 1 ms
    return {"cable": max(0.0, 0.5 * error + 40.0 * state["integral"])}

result = sim.run(controller=controller)
```

## 2. A pneumatic finger

`scenarios/actuation/pneunet.yaml` is a PneuNet finger: chambers, a channel and a stiffer strain-limiting layer
(a material zone), inflated 0-8 kPa. The same finger can be read from CAD:
`scenarios/actuation/pneunet_from_step.yaml` loads `cad/pneunet.step` and finds the cavity on its own.

```bash
softsim run scenarios/actuation/pneunet.yaml --backend mujoco          # about 6 minutes
softsim export scenarios/actuation/pneunet.yaml --cad finger.step      # the parametric shape as STEP
```

## 3. Transfer: make MuJoCo agree with the reference

MuJoCo at 1.5 mm bends this finger 24% less than the locking-free reference. `softsim transfer` fits the
*effective* moduli that make MuJoCo reproduce the reference, then checks them on load cases it never saw:

```bash
softsim transfer calibrations/transfer_pneunet_mujoco.yaml       # about 2.5 hours on a laptop
```

The result for that file: body 114 → 91 kPa, layer 1 → 1.76 MPa, error 17.7% → 2.9%, and on the unseen cases
3.2% (11 kPa) and 4.3% (a 4-chamber finger). The fitted card is written with `kind: effective` and the backend,
mesh and element it belongs to, so it isn't mistaken for a material constant.
