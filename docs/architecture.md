# Architecture

```
 YAML config ──► Scenario (validated config model)
                    │ build_scene()
                    ▼
                  Scene  (backend-neutral: tet meshes, materials, rigid parts,
                    │     poses, velocities, contact + simulation settings)
        checks ◄────┤
                    ▼
                 Backend.check(scene)   ── UnsupportedFeatureError listing all gaps
                 Backend.build(scene)   ── e.g. MJCF + MjSpec → MjModel
                 Backend.step / state   ── SimState: rigid poses/twists,
                    │                      soft vertices, contact wrenches
                    ▼
       Scenario.metrics(state) per output sample ──► timeseries
       Scenario.summarize(timeseries)            ──► summary
```

## Principles and where they live

| Principle | Mechanism |
| --- | --- |
| Core is ROS-agnostic | `softsim/` imports no ROS; the ROS nodes (M2) wrap `Simulation`. |
| Pluggable backends | `backends/base.py: Backend`; discovered through the `softsim.backends` entry-point group. |
| Fail loudly on unsupported physics | `capabilities.py: Capability`. Scene elements declare `requires`, backends declare `capabilities` and `extra_checks()`. |
| Data-driven | Pydantic models with `extra="forbid"` (`config/schema.py`), units in `config/units.py`, `extends:` and `--set` in `config/loader.py`. |
| New phenomenon = new scenario | `scenarios/base.py: Scenario`, discovered through the `softsim.scenarios` entry-point group. |
| Comparable to experiments | Scenarios emit named metrics in SI units at a configurable output rate. Datasets and calibration come in M3. |

## Key decisions

- **Model generation uses MJCF and MjSpec, not templates.** The skeleton is
  generated MJCF (readable, saved with every run). The MuJoCo Python spec then
  applies what MJCF cannot express: Neo-Hookean flex elasticity
  (`elastic3d`, spec-only in MuJoCo 3.17) and mass-consistent lumped vertex
  masses (flexcomp spreads mass equally over vertices).
- **Soft bodies are passed as our own tetrahedral meshes** (`flexcomp
  type="direct"`). MuJoCo's `flexcomp type="cylinder" dim=3` is a radial-slider
  shell with tetrahedra fanned to the centre, not a solid FEM body.
- **The structured mesher is the default.** It is deterministic, symmetric,
  dependency-free, has a vertex on the axis for measurements, and its elements
  are better quality than gmsh's (min quality 0.5–0.6 vs 0.25). gmsh is
  optional for arbitrary geometry.
- **Body frames are at the base centre** (soft bodies) or bottom-face centre
  (rigid bodies), so placing things on surfaces needs no arithmetic.
- **Contact compliance is a physical parameter.** `contact.timeconst` is fixed
  in scenario configs (not tied to the timestep), because it changes results
  about as much as the material does.

## State and units

All values are SI: m, kg, s, Pa, N, rad. Quaternions are (w, x, y, z).
`ContactWrench.force` is the force exerted by `body_a` on `body_b` in the world
frame; the static ground is called `ground`.
