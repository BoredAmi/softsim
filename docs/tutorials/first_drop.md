# Tutorial: a first drop simulation

A 20 g brass weight falls 38 cm onto an Ecoflex 00-10 cylinder (40 mm across, 17 mm tall). This is the first
case of the Czopek et al. drop dataset ([drop test scenario](../scenarios/drop_test.md)); you don't need the
dataset to run the simulation.

## 1. Check the scene before running it

```bash
softsim check scenarios/drop_test/cases/w020.yaml
```

This prints the mesh (vertices, tetrahedra, quality), the physics findings (is the contact stiff enough for
the impact speed? is the time step stable? does the material lock?) and whether the chosen backend supports
everything the scene needs. Warnings are worth reading; errors stop a run.

## 2. Run it

```bash
softsim run scenarios/drop_test/cases/w020.yaml --duration 0.05
```

`--duration` shortens the run to the first 50 ms (the impact takes about 5 ms). The output goes to
`runs/drop_test_w020_<timestamp>/`:

| File | Content |
|---|---|
| `config.yaml` | the fully resolved config: re-run it with `softsim run <dir>/config.yaml` |
| `timeseries.parquet` | one row per output sample: `weight_z`, `weight_vz`, `indentation`, `contact_force_z`, ... |
| `summary.yaml` | scalars (`peak_indentation`, `rebound_speed`, `restitution`, ...), the check findings, versions, wall time |

## 3. Change something

Any value in the YAML can be changed from the command line with `--set`, units included:

```bash
softsim run scenarios/drop_test/cases/w020.yaml --duration 0.05 \
    --set drop.height="20 cm" --set specimen.mesh.size="5 mm"
softsim params scenarios/drop_test/cases/w020.yaml      # every key you can set, with unit and range
```

## 4. From Python

```python
from softsim import Simulation

sim = Simulation("scenarios/drop_test/cases/w020.yaml", ["simulation.duration=50ms"])
result = sim.run()
print(result.summary["peak_indentation"])           # m
ts = result.timeseries                               # a pandas DataFrame
print(ts[["t", "weight_z"]].head())
```

## 5. Other backends

```bash
softsim run scenarios/drop_test/cases/w020.yaml --duration 0.012 --backend fem    # the 3D reference: slow, exact
softsim run scenarios/drop_test/cases/w020.yaml --duration 0.05 --backend sofa    # needs SOFA_ROOT
```

The same material card gives different answers on MuJoCo and on the reference, because linear tetrahedra
lock and MuJoCo's contact loses energy. [Experiments](../experiments.md) shows which models reproduce the
real recordings and by how much, and [materials](../materials.md) explains how a card is carried to a fast
backend with a stated error bar.
