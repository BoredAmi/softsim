# Adding a material

See [materials.md](materials.md) for the full card format (laws, Prony
viscoelasticity, validity, uncertainty, physical vs effective cards), for how
to change any value, and for the homogeneous test curves of a card.

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
  model: stable_neo_hookean   # linear | stvk | neo_hookean | stable_neo_hookean | mooney_rivlin | yeoh | ogden | gent | arruda_boyce | magneto_active
  youngs_modulus: 60 kPa      # small-strain equivalent
  poisson_ratio: 0.45
damping:                       # optional: rayleigh | kelvin_voigt | prony
  model: rayleigh
  beta: 2 ms
friction: {sliding: 1.0, torsional: 0, rolling: 0}
adhesion: {strength: 14 kPa, range: 0.36 mm}   # optional tack against rigid bodies (MuJoCo and SOFA)
provenance: {}                 # e.g. the mesh size/backend a calibrated value is tied to
```

`adhesion` is a cohesive law (peak tensile traction `strength` at an opening
of `range`; work of separation e·strength·range per area). Leave it out for
non-tacky materials. The ground has its own `ground.adhesion`.

Use it by name (`material: my_silicone`), override part of it
(`material: {preset: my_silicone, elastic: {youngs_modulus: 50 kPa}}`), or
write it inline. Run `softsim materials` to see what is visible.

## A new constitutive model (code)

1. Add a pydantic class to `softsim/materials/models.py` with a unique
   `model: Literal["..."]` discriminator, validators for its parameters,
   `initial_shear_modulus()` / `initial_bulk_modulus()` (small-strain
   equivalents for checks; laws with a bulk modulus derive from
   `_BulkHyperelastic` and also implement `_raw_shear_modulus` for the
   `poisson_ratio` input), and `requires` (a capability set).
2. Add a `Capability` member in `softsim/capabilities.py` if it is a new kind
   of physics, and add the class to the `ElasticModel` (or `DampingModel`) union.
3. In each backend that can simulate it, add the capability and the mapping
   (e.g. `backends/mujoco/builder.py`). Backends that cannot will refuse it
   automatically.
4. Add its isochoric energy to `softsim/fem/materials.py` (a JAX pytree; see
   the registration at the end of the file). The reference FEM and the
   homogeneous tests then support it.
5. Add tests: parameter validation, the closed-form uniaxial stress
   (`tests/test_homogeneous.py`), the small-strain equivalents, and the
   backend rejecting or accepting it.

Do not map an unsupported model silently to a "close" one. If an approximation
is useful, make it an explicit, documented model or option.
