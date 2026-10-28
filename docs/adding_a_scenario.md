# Adding a scenario (a new phenomenon)

A scenario is a config schema plus a small class. The core does not change.

## 1. Config schema

Subclass `ScenarioConfig`. It already contains `scenario`, `name`, `backend`,
`simulation`, `contact`, `ground`, `output` and `material_paths`.

```python
class IndentationConfig(ScenarioConfig):
    scenario: Literal["indentation"] = "indentation"
    specimen: SpecimenConfig            # reuse drop_test's or write your own
    indenter: IndenterConfig
```

Use the unit types (`Length`, `Mass`, `Pressure`, ...) from
`softsim.config.units`, so YAML can say `"5 mm"`.

## 2. Scenario class

```python
class IndentationScenario(Scenario):
    name = "indentation"
    config_model = IndentationConfig
    actions = {"press": "move the indenter to depth [m]"}

    def build_scene(self) -> Scene: ...                 # meshes, materials, poses, velocities
    def metrics(self, state: SimState) -> dict: ...      # one row per output sample, SI units
    def summarize(self, timeseries) -> dict: ...         # optional scalar summary
    def estimates(self) -> dict: ...                     # optional: feeds the physics checks
    def material_key(self, body) -> str: ...             # optional: enables set_material
    def apply_action(self, action, raw_config, **kw): ...  # optional: returns a new raw config
```

Guidelines:

- Start the simulation where the physics starts, as the drop test does by
  starting just above contact with the free-fall speed.
- Report metrics that can be measured in the experiment, under the same names
  and sign conventions as the dataset loader will use.
- Use `self.load_material(ref)` so presets and relative paths resolve correctly.

## 3. Register

Built-ins go in `softsim/scenarios/registry.py`. External packages register an
entry point instead, with no change to softsim:

```toml
[project.entry-points."softsim.scenarios"]
indentation = "my_package.indentation:IndentationScenario"
```

## 4. Configs and tests

Put YAML in `scenarios/<name>/` (a base file plus `cases/*.yaml` that use
`extends:`). Add tests: all shipped configs validate, a short headless run, and
at least one analytic or limiting-case check.
