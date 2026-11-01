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

## Soft body from a file (no code)

Your own design, from CAD or a mesh (`geometry: {type: mesh_file, path: ...}`,
`softsim/geometry/soft.py::SoftMeshFile`; needs gmsh, `pip install softsim[mesh]`):

| File | What happens |
| --- | --- |
| `.step`/`.stp`, `.iges`/`.igs` | meshed at `mesh.size`; units from the file |
| `.brep` | meshed at `mesh.size`; no units: `scale` |
| `.stl`, `.obj` | a closed surface (inner closed shells are cavities) turned into a volume; remeshed at `mesh.size`, or `remesh: false` keeps its triangles; no units: `scale: 1e-3` for millimetres |
| `.msh`, `.vtk`, `.mesh` (Medit) | volume mesh, used as it is (linear tetrahedra); no units: `scale` |
| `.geo` | gmsh script, meshed at `mesh.size` if it holds no mesh |

Several solids in one CAD file (a body and a layer of another material) are
fragmented into one conforming, bonded body. `path` is relative to the config
file.

**Regions.** Physical groups named in gmsh become regions: 2D groups name
boundary faces (`fixed`, pressure), 3D groups name element sets (`zones:
[{region: <group>}]`). A file with several volumes and no 3D groups (the
solids of a STEP, the references of a Medit mesh) names them `volume_<tag>`.
Every file body also has `surface` (the whole boundary)
and `cavity` for its closed inner shell (`cavity_1`, `cavity_2`... if there
are several, largest first), so a pressure actuator needs no group at all. For
files without groups (STEP, STL), name faces by box: boundary triangles with
all three vertices inside it, after placement. A zero-thickness box picks a
flat face:

```yaml
body:
  geometry:
    type: mesh_file
    path: cad/my_finger.step
    rotate: [0, 0, 0]        # degrees about x, then y, then z
    center: none             # base: bounding box bottom centre to the origin
    translate: [0, 0, 0]
    box_regions:
      base: [[0, -8 mm, 0], [0, 8 mm, 16 mm]]
  mesh: {size: 1.5 mm, method: gmsh}
  fixed: [base]
  zones:
    - {name: layer, box: [[0, -8 mm, 0], [54 mm, 8 mm, 1.5 mm]], material: {preset: ..., elastic: {...}}}
actuators: [{name: cavity, region: cavity}]
```

`scenarios/actuation/pneunet_from_step.yaml` is this for the PneuNet finger.
Its STEP file was written from the parametric shape by `softsim export
scenarios/actuation/pneunet.yaml --cad scenarios/actuation/cad/pneunet.step`
(`box`, `cylinder`, `closed_tube` and `pneunet` can be written, in metres,
body frame). Use the same command to start a design in CAD from a parametric
one. Read back, it is the same body: total, cavity and layer volumes, and the
bonded base, agree exactly with the parametric mesh (`tests/test_mesh_files.py`).

Check the mesh before a long run: `softsim check <config>` prints its size and
minimum quality.

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
