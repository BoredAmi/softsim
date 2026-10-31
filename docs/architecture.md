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
| Core is ROS-agnostic | `softsim/` imports no ROS; `ros2/softsim_ros` wraps `Simulation` (see [ros2.md](ros2.md)). |
| Pluggable backends | `backends/base.py: Backend`; discovered through the `softsim.backends` entry-point group. |
| Fail loudly on unsupported physics | `capabilities.py: Capability`. Scene elements declare `requires`, backends declare `capabilities` and `extra_checks()`. |
| Data-driven | Pydantic models with `extra="forbid"` (`config/schema.py`), units in `config/units.py`, `extends:` and `--set` in `config/loader.py`. |
| New phenomenon = new scenario | `scenarios/base.py: Scenario`, discovered through the `softsim.scenarios` entry-point group. |
| Comparable to experiments | Scenarios emit named metrics in SI units at a configurable output rate. `experiments/`: datasets, video tracking, comparison over any observable and window, calibration. |
| Physics shared across backends | Backend-neutral models a backend applies as external forces: `contact/adhesion.py` (cohesive tack from analytic signed distances) and `materials/viscoelastic.py` (Prony Maxwell overstress; the backend's own elasticity carries the long-term law). |
| Materials are physics, backends are approximations | A material card (`materials/models.py`) is `physical` or `effective`. `materials/homogeneous.py` gives its exact response in standard tests, `materials/fit.py` fits cards to public test data (`experiments/material_tests.py`), and the locking-free reference (`fem/`, backend `fem`) is the referee for the tetrahedral backends. |
| Every value is a parameter | `config/params.py` lists every dotted key with unit and range (`softsim params`). The same keys work in `--set`, sweeps, `Material.with_overrides`, `Simulation.set_material` and the ROS `set_material` service. |

## Key decisions

- **Model generation uses MJCF and MjSpec, not templates.** The skeleton is
  generated MJCF (readable, saved with every run). The MuJoCo Python spec then
  applies what MJCF cannot express: Stable Neo-Hookean flex elasticity
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
- **One Prony implementation for every backend.** SOFA ships a
  visco-hyperelastic force field (MJED), but it relaxes the volumetric part too
  and uses non-isochoric invariants. Using it would silently change the model,
  so MuJoCo and SOFA both get the same explicit overstress, with the reference
  solver's update.
- **Model the experiment, not only the material.** What happens after impact
  in the drop recordings is set by the specimen's tack and the cloth under it
  as much as by its elasticity, so both are scene elements (`adhesion`,
  `support`) with their own fitted parameters, not hidden in the material.

## State and units

All values are SI: m, kg, s, Pa, N, rad. Quaternions are (w, x, y, z).
`ContactWrench.force` is the force exerted by `body_a` on `body_b` in the world
frame; the static ground is called `ground`.
