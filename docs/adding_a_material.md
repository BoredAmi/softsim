# Adding a material

## A new preset (no code)

Create `<name>.yaml` in `softsim/materials/presets/` (shipped presets), in any
directory listed in `$SOFTSIM_MATERIAL_PATH`, or in a directory listed under
`material_paths:` in a scenario config.

```yaml
name: my_silicone
description: What it is.
source: Where every number comes from. Say "uncalibrated" or "fitted to X" honestly.
density: 1070 kg/m^3
elastic:
  model: stable_neo_hookean   # linear | stvk | neo_hookean | stable_neo_hookean | mooney_rivlin | yeoh | ogden | magneto_active
  youngs_modulus: 60 kPa      # small-strain equivalent
  poisson_ratio: 0.45
damping:                       # optional: rayleigh | kelvin_voigt | prony
  model: rayleigh
  beta: 2 ms
friction: {sliding: 1.0, torsional: 0, rolling: 0}
provenance: {}                 # e.g. the mesh size/backend a calibrated value is tied to
```

Use it by name (`material: my_silicone`), override part of it
(`material: {preset: my_silicone, elastic: {youngs_modulus: 50 kPa}}`), or
write it inline. Run `softsim materials` to see what is visible.

## A new constitutive model (code)

1. Add a pydantic class to `softsim/materials/models.py` with a unique
   `model: Literal["..."]` discriminator, validators for its parameters,
   `initial_youngs_modulus()` / `initial_poisson_ratio()` (small-strain
   equivalents for checks), and `requires` (a capability set).
2. Add a `Capability` member in `softsim/capabilities.py` if it is a new kind
   of physics, and add the class to the `ElasticModel` (or `DampingModel`) union.
3. In each backend that can simulate it, add the capability and the mapping
   (e.g. `backends/mujoco/builder.py`). Backends that cannot will refuse it
   automatically.
4. Add tests: parameter validation, the small-strain equivalents, and the
   backend rejecting or accepting it.

Do not map an unsupported model silently to a "close" one. If an approximation
is useful, make it an explicit, documented model or option.
