# softsim

Data-driven simulation of soft materials and soft robots, with pluggable physics
backends and validation against real experiments. The core is a pure Python
package; ROS 2 integration is a thin layer on top (milestone M2).

**Status: M1 (core + MuJoCo drop test), M2 (ROS 2 layer), M3 (video
tracking + calibration) and M4 (SOFA backend) done.** Backends: MuJoCo flex
(default, fast) and SOFA (more material laws, optional install), plus an
axisymmetric reference FEM (locking-free, for verification; in progress). See
[docs/roadmap.md](docs/roadmap.md) for what's next.

![20 g weight dropped on an Ecoflex 00-10 cylinder: high-speed camera vs MuJoCo flex simulation](docs/media/drop_20g_sim_vs_experiment.gif)

*20 g weight dropped from 38 cm onto an Ecoflex 00-10 cylinder (D 40 × H 17 mm),
aligned at the tracked contact: ~100× slow motion through the impact, then
2 ms per frame. Left: high-speed camera recording from
[Czopek et al., doi:10.5281/zenodo.23185808](https://doi.org/10.5281/zenodo.23185808)
(CC-BY-4.0, cropped and denoised). Right: softsim + MuJoCo flex with the
**calibrated** material and the measured impact speed, offset and spin. The
weight sticks to the tacky silicone and the specimen hops, dragging the cloth
under it (green plate), as in the recording. The fit reproduces the whole first
100 ms of the 20 g and 50 g drops (12–25% error); without the tack the weight
flies 7 cm high. The 500 g impact is not reproduced; see
[docs/experiments.md](docs/experiments.md) for what fits, what doesn't, and why.*

![All five dataset weights simulated side by side](docs/media/drop_all_weights.gif)

*The five weights of the dataset (20–500 g) from the same YAML scenario,
switching one preset each (nominal centred impacts; fitted tack and cloth
support). Regenerate both GIFs with
`python tools/make_readme_media.py` (needs `020gram.zip` in `data/raw/czopek2026/`
for the left panel).*

## Why

Gazebo has no deformable bodies, SOFA is powerful but hard to use from ROS,
and MuJoCo's new volumetric flex is fast but low-level. softsim adds:

- **Everything as data.** Materials, bodies, scenarios and sweeps are YAML with
  units (`"17 mm"`, `"55 kPa"`), strictly validated, so typos are errors.
  Backend models (MJCF) are generated, never hand-written.
- **Honest backends.** Each backend declares what it can simulate. A scene that
  needs, e.g., Mooney-Rivlin on MuJoCo is refused with the full list of gaps
  instead of silently running different physics.
- **Physics checks before every run**: contact compliance vs. impact speed,
  mesh resolution, volumetric locking, StVK under large compression, explicit
  stability limits, and density/mass consistency.
- **Validation first.** Every scenario produces metrics that are meant to be
  compared with experimental data (see [docs/experiments.md](docs/experiments.md)).

## Install

Python ≥ 3.10. On ROS 2 Humble, use a venv that can see the ROS packages:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e ".[dev]"      # mujoco 3.17, gmsh, scipy, pytest
.venv/bin/python -m pytest              # ~20 s; add -m "not slow" to skip the slow ones
```

`numpy<2` is pinned so ROS 2 Humble's compiled Python modules keep working.

**Reference FEM (optional).** `pip install -e ".[fem]"` adds JAX (CPU,
float64) for `softsim.fem`, the axisymmetric locking-free solver used to
cross-check the backends ([docs/reference_fem.md](docs/reference_fem.md)).

**SOFA backend (optional).** Unzip the SOFA v26.06 Linux release (Python 3.10)
and point `SOFA_ROOT` at it; then add `--backend sofa` to any command. Without
it, the SOFA tests skip. See [docs/backends/sofa.md](docs/backends/sofa.md).

## Quick start

```bash
softsim materials                                   # list material presets
softsim check scenarios/drop_test/cases/w020.yaml   # mesh stats, physics findings, backend support
softsim run   scenarios/drop_test/cases/w020.yaml   # results in runs/<name>_<timestamp>/
softsim run   scenarios/drop_test/cases/w020.yaml \
    --set specimen.material.elastic.youngs_modulus="45 kPa" --set drop.tilt="[5 deg, 0]"
softsim view  scenarios/drop_test/cases/w050.yaml   # interactive MuJoCo viewer (slowed 200x)
softsim sweep scenarios/drop_test/sweeps/all_weights.yaml --jobs 5
softsim export scenarios/drop_test/cases/w020.yaml --out model.xml
softsim track datasets/czopek2026_ecoflex_drop.yaml --jobs 5       # video tracking (needs the dataset zips)
softsim calibrate calibrations/ecoflex_00_10_drop.yaml --jobs 3    # fit E and damping to the recordings
softsim run   scenarios/drop_test/cases/w020.yaml --backend sofa \
    --set specimen.material=ecoflex_00_10_drop_fit_sofa_ogden --set specimen.mesh.size=5mm --set simulation.timestep=5e-5
```

From Python:

```python
from softsim import Simulation

sim = Simulation("scenarios/drop_test/cases/w020.yaml")
result = sim.run()
print(result.summary["peak_indentation"])
sim.set_material("specimen", {"preset": "ecoflex_00_10", "elastic": {"youngs_modulus": "40 kPa"}})
sim.action("drop", preset="czopek2026_w100", height=0.2)
state = sim.state()            # rigid poses/twists, soft vertices, contact wrenches
```

### ROS 2

```bash
source /opt/ros/humble/setup.bash && source .venv/bin/activate
python -m colcon build --base-paths ros2 && source install/setup.bash
ros2 launch softsim_bringup softsim.launch.py config:=drop_test/cases/w050.yaml   # RViz
ros2 service call /softsim/action softsim_interfaces/srv/RunAction "{action: drop, arguments: '{mass: 0.1}'}"
```

The node publishes `/clock`, TF, poses and twists, soft-body point clouds and
meshes, contact wrenches, ground reaction and scenario metrics. Its services
cover reset, pause, save, material changes and scenario actions. See
[docs/ros2.md](docs/ros2.md).

A run directory contains `config.yaml` (fully resolved and re-runnable),
`timeseries.parquet`, `summary.yaml` (summary, findings, versions, wall time)
and `model.xml`.

## Layout

```text
softsim/            core library (no ROS)
  config/           YAML loading (extends, --set overrides), units, strict schemas
  materials/        constitutive models + presets/*.yaml
  geometry/         soft and rigid primitives + rigid body presets
  meshing/          tetrahedral meshing (structured built-in, gmsh optional)
  scene.py          backend-neutral scene description
  backends/         Backend interface, MuJoCo flex backend, SOFA backend
  scenarios/        Scenario base class + drop_test
  checks/           pre-run physics checks
  experiments/      datasets, video tracking, sim-vs-experiment comparison, calibration, plots
  runner.py         Simulation / run_config
  sweep.py, cli.py
ros2/               softsim_interfaces, softsim_ros (generic node), softsim_bringup (launch, RViz)
scenarios/          scenario configs, cases and sweeps (data, not code)
datasets/           experimental dataset descriptions (DOI, camera, cases)
calibrations/       calibration configs
data/               raw/ (not versioned) and processed/ experimental data
tools/              helper scripts (README media generation)
docs/               architecture, how-to guides, backend notes, media/
tests/
```

## Documentation

- [Architecture](docs/architecture.md)
- How to add a [material](docs/adding_a_material.md), [body](docs/adding_a_body.md),
  [scenario](docs/adding_a_scenario.md), [backend](docs/adding_a_backend.md)
- [MuJoCo backend: physics notes and limits](docs/backends/mujoco.md)
- [SOFA backend: setup, physics notes and limits](docs/backends/sofa.md)
- [Reference FEM (axisymmetric, locking-free) and backend cross-checks](docs/reference_fem.md)
- [Drop test scenario and dataset](docs/scenarios/drop_test.md)
- [ROS 2 integration](docs/ros2.md)
- [Experiments: video tracking and calibration](docs/experiments.md)

## Data and citation

The first validation scenario uses:

> Czopek P., Superczynska P., Marcinkowska A., Gajewski P., Bernat J.
> *Records of falling mass on Ecoflex 00-10 cylinder.* Zenodo.
> doi:[10.5281/zenodo.23185808](https://doi.org/10.5281/zenodo.23185808). CC-BY-4.0.

The dataset is not redistributed here. Download it into `data/raw/czopek2026/`.
To cite softsim itself, see [CITATION.cff](CITATION.cff).

## License

Apache License 2.0; see [LICENSE](LICENSE).
