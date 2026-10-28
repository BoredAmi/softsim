# Adding a body

## Rigid body preset (no code)

Add an entry to `softsim/geometry/presets/rigid_bodies.yaml`:

```yaml
my_weight:
  description: 30 g steel cylinder
  source: measured
  mass: 30 g
  shape: {type: cylinder, diameter: 20 mm, height: 12 mm}   # cylinder | weight_with_knob | plate
```

Reference it from a scenario (`weight: {body: my_weight}`), or inline
`{mass: ..., shape: {...}}`, or override it (`{preset: my_weight, mass: 35 g}`).

## New rigid shape (code)

In `softsim/geometry/rigid.py`, add a pydantic class with a `type` literal, a
`bottom_radius` property and `parts()`. `parts()` returns a list of `Part`
primitives (cylinder or box, with centres in the body frame, origin at the
centre of the bottom face). Add it to the `RigidShape` union. Backends build
parts generically and distribute mass over them by volume.

## New soft shape (code)

In `softsim/geometry/soft.py`, add a class with a `type` literal, `top_height`,
`min_feature`, `volume()` and `mesh(settings) -> TetMesh`, and add it to
`SoftShape`. Meshes must:

- put the base centre at the origin with +z up,
- have positively oriented, conforming tetrahedra (`meshing.tet._orient` fixes
  orientation),
- pass `tests/test_meshing.py::_check_mesh` (closed outward surface, every
  vertex used).

For CAD geometry, add a gmsh builder in `softsim/meshing/tet.py`. For
structured meshes, extrude a 2D triangulation with `_extrude`, which splits
prisms consistently.
