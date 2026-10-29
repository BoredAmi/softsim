# Adding a backend

Subclass `softsim.backends.base.Backend`:

```python
class MyBackend(Backend):
    name = "mine"
    capabilities = frozenset({Capability.STABLE_NEO_HOOKEAN, Capability.TET_MESH, ...})

    @classmethod
    def extra_checks(cls, scene) -> list[str]: ...   # settings this backend rejects

    def build(self, scene): ...       # call self.check(scene) first; compile; set initial condition
    def step(self, n=1): ...
    def state(self) -> SimState: ...  # copies; soft vertices in mesh order; contact forces by a on b
    def reset(self): ...
    time: float                       # property
    scene: Scene                      # property
    # optional: is_finite(), export_model(), diagnostics(), close()
```

Contract:

- **Declare only what is simulated faithfully.** If a material model is
  approximated, do not list its capability; refusing is better than silently
  different physics.
- **Vertex order** in `SoftState.vertices` must match `SoftBodySpec.mesh.points`.
- **Initial conditions** come from the scene: body poses and rigid-body linear
  and angular velocities (world frame).
- **Contacts**: aggregate per body pair into `ContactWrench` (force by `body_a`
  on `body_b`, centre of pressure, residual torque). Name the static ground
  `ground`.
- **Divergence**: `is_finite()` must return False after a blow-up, even if the
  engine resets itself (MuJoCo does).

Register through the entry-point group `softsim.backends` (or `_BUILTIN` in
`backends/registry.py`). Then reuse the backend-agnostic tests: run
`tests/test_drop_test.py` with `--set backend=mine` configs, plus the static
uniaxial compression check, which every backend must pass.

## Example: the SOFA backend

`softsim/backends/sofa/backend.py` is a complete second implementation and a
good template. It shows:

- locating an external engine (`$SOFA_ROOT`) lazily, so the capability
  declaration works without it;
- mapping each material model to engine parameters, each mapping checked
  against an analytic result (`tests/test_sofa_backend.py`);
- reconstructing contact forces when the engine doesn't expose them per pair
  (momentum balance);
- documenting where results differ from other backends
  ([backends/sofa.md](backends/sofa.md)).
