"""SOFA fast path for quasi-static inverse control, with the SoftRobots.Inverse plugin.

    res = solve_inverse_sofa("scenarios/actuation/cable_cantilever.yaml", {"tip_x": "10 mm"}, ["cable"],
                             {"cable": (0, "0.7 N")}, overrides=["body.mesh.size=4mm"])

The repeated-solve method of :mod:`softsim.control.inverse` finds the actuator values by running an equilibrium
(a settling simulation) per trial value. SOFA's inverse solver does the inversion inside the constraint solver:
``QPInverseProblemSolver`` replaces the Gauss-Seidel solver, every cable is a ``CableActuator`` and every target a
``PositionEffector``, and at each time step the solver finds the actuator forces that bring the effectors to their
goals at the end of the step, as a quadratic program with the force bounds as constraints. Stepping until the effector
has arrived (and the forces have stopped changing) gives the equilibrium solution in one run, not ``1 + n`` per
iteration.

What is built (the scene is the scenario's body with the same mesh, material and fixed vertices as the ordinary SOFA
backend, see ``docs/backends/sofa.md``; only the actuation differs):

* cables: a child node of the body whose ``MechanicalObject`` holds the cable's via points (a ``BarycentricMapping``
  child, mapForces/mapMasses off) and a ``CableActuator`` with ``minForce`` / ``maxForce`` from the bounds and its
  pull point a little behind the first via point (``docs/roadmap.md``, M6). SoftRobots' constraint value is an impulse;
  the force read back is ``lambda / dt`` (checked: the force-type cable equals the explicit one);
* pressure actuators: a child node of the body holding the actuator's ``cavity`` triangles (a ``MeshTopology``, a
  ``MechanicalObject`` on their vertices, a ``BarycentricMapping`` to the body) and a ``SurfacePressureActuator`` with
  ``minPressure`` / ``maxPressure``. Same impulse convention as the cables: bounds go in as ``pressure * dt``, the
  pressure read back is ``lambda / dt`` (checked: the bounded run equals the ordinary backend's explicit follower
  pressure to 7 digits). Two things differ from the cables: the cavity triangles are given the opposite winding to
  softsim's (SOFA's normals point into the cavity; the other way the volume shrinks and the pressure clamps at 0), and
  the QP is badly scaled in SI units (the pressure's ``Q`` entry is ~1e-12, below qpOASES' tolerance, so it returns
  ``lambda = 0`` for any reachable goal), hence the ``PositionEffector`` ``weight`` of 1e6 (``qp_weight``; 1e5 to 1e7
  give the same pressure to 0.1%);
* effectors: one ``PositionEffector`` on a barycentric point at the tip face's rest centroid (the ``tip`` / ``top``
  region), with ``useDirections`` selecting the targeted axes among ``tip_x``, ``tip_y``, ``tip_z``;
* the angle targets ``bend_angle`` / ``chord_angle`` are not a position. They are reached by moving the effector
  goal along ``x`` (secant iteration on the angle read from the vertices: a handful of inverse solves).

Outputs reported are the scenario's own (``Scenario.metrics`` on the final vertices), so they are comparable with
:func:`softsim.control.inverse.solve_inverse`.

Checked against: the repeated-solve method on SOFA and the explicit forward cable (``tests/test_sofa_inverse.py``).

Limits: quasi-static; the body is SOFA's linear tetrahedra (locking like MuJoCo's, see ``docs/actuation.md``); only
tension-controlled cables and pressures (no angle targets with a pressure); contact, adhesion, viscoelasticity, prescribed motion and moving mounts are not built; a goal
that needs a negative force is infeasible (the cable can only pull).
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from softsim.actuation import CableActuator, PressureActuator
from softsim.backends.base import SimState, SoftState
from softsim.backends.sofa.backend import PLUGINS, force_field_args, import_sofa, rayleigh
from softsim.config.loader import apply_overrides, load_yaml
from softsim.config.units import parse_quantity
from softsim.control.inverse import InverseResult, actuator_dimension, output_dimension
from softsim.errors import ConfigError
from softsim.runner import Simulation

INVERSE_PLUGINS = ["SoftRobots", "SoftRobots.Inverse"]
POSITION_OUTPUTS = {"tip_x": 0, "tip_y": 1, "tip_z": 2}
ANGLE_OUTPUTS = ("bend_angle", "chord_angle")
_SCALES = {"length": 1e-3, "angle": 1e-2}
# SurfacePressureActuator's normals must point into the cavity (out of the solid is the other way round: the volume
# then shrinks and the actuator clamps at 0): softsim's cavity triangles are outward from the solid, so swap two corners
PRESSURE_WINDING = [0, 2, 1]


class InverseScene:
    """The SOFA scene with ``QPInverseProblemSolver``, cables as ``CableActuator`` and the tip as ``PositionEffector``.

    Args:
        sim: A built :class:`~softsim.runner.Simulation` (only its scene, scenario and config are used; its
            body should carry damping, see ``settle_damping`` of :func:`solve_inverse_sofa`).
        cables: Names of the cable actuators to solve for, with their ``(lower, upper)`` force bounds [N].
        axes: Effector axes (0, 1, 2 for x, y, z) that are targeted.
        tolerance: QP solver tolerance.
        max_iterations: QP solver iteration cap.
    """

    def __init__(self, sim: Simulation, cables: Mapping[str, tuple[float, float]], axes: Sequence[int],
                 tolerance: float = 1e-8, max_iterations: int = 1000, epsilon: float = 1e-3, weight: float = 1.0):
        scene = sim.scene
        if len(scene.soft_bodies) != 1 or scene.rigid_bodies:
            raise ConfigError("sofa-qp: needs a scene with one soft body and no rigid bodies")
        if scene.mount is not None or scene.spin is not None or scene.soft_bodies[0].prescribed:
            raise ConfigError("sofa-qp: moving mounts and prescribed motion are not supported")
        self.sim, self.scene, self.body = sim, scene, scene.soft_bodies[0]
        tip = getattr(sim.scenario, "_tip", None)
        if tip is None:
            raise ConfigError("sofa-qp: the body has no 'tip' / 'top' region for the effector")
        self.sofa = Sofa = import_sofa()
        body, dt = self.body, scene.simulation.timestep
        self.dt = dt
        rot = _rotation(body.pose.quaternion)
        self.offset = np.asarray(body.pose.position, float)
        pts = body.mesh.points @ rot.T + self.offset
        self.points0 = pts
        root = Sofa.Core.Node("root")
        root.addObject("RequiredPlugin", pluginName=PLUGINS + INVERSE_PLUGINS)
        root.gravity = list(scene.simulation.gravity)
        root.dt = dt
        root.addObject("FreeMotionAnimationLoop")
        root.addObject("QPInverseProblemSolver", maxIterations=max_iterations, tolerance=tolerance, epsilon=epsilon,
                        printLog=False)
        node = root.addChild(body.name)
        alpha, beta = rayleigh(body)
        node.addObject("EulerImplicitSolver", rayleighMass=alpha, rayleighStiffness=beta)
        node.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixMat3x3d")
        node.addObject("TetrahedronSetTopologyContainer", position=pts.tolist(), tetrahedra=body.mesh.tets.tolist())
        self.dofs = node.addObject("MechanicalObject", template="Vec3d", position=pts.tolist())
        density = body.effective_density if body.uniform_density else body.element_densities().tolist()
        node.addObject("MeshMatrixMass", massDensity=density, lumping=True)
        groups = body.material_groups()
        if len(groups) == 1:
            ff, args = force_field_args(body)
            node.addObject(ff, **args)
        else:  # zones: one child node per material, as in the ordinary backend
            for k, (material, elements) in enumerate(groups):
                if not len(elements):
                    continue
                child = node.addChild(f"material{k}")
                child.addObject("TetrahedronSetTopologyContainer", position=pts.tolist(),
                                tetrahedra=body.mesh.tets[elements].tolist())
                ff, args = force_field_args(dataclasses.replace(body, material=material, zones=()))
                child.addObject(ff, **args)
        if body.pinned:
            node.addObject("FixedProjectiveConstraint", indices=list(body.pinned))
        node.addObject("LinearSolverConstraintCorrection")
        self.actuators: dict[str, Any] = {}
        self.kinds: dict[str, str] = {}
        for a in scene.actuators:
            if a.name not in cables:
                continue
            lo, hi = cables[a.name]
            if isinstance(a, PressureActuator):
                tris = np.asarray(body.region(a.region))
                used, local = np.unique(tris, return_inverse=True)
                pnode = node.addChild(f"pressure_{a.name}")
                pnode.addObject("MeshTopology", name="topo", position=pts[used].tolist(),
                                triangles=local.reshape(-1, 3)[:, PRESSURE_WINDING].tolist())
                pnode.addObject("MechanicalObject", position=pts[used].tolist())
                self.actuators[a.name] = pnode.addObject(
                    "SurfacePressureActuator", name=a.name, triangles="@topo.triangles",
                    minPressure=float(lo) * dt, maxPressure=float(hi) * dt)
                pnode.addObject("BarycentricMapping")
                self.kinds[a.name] = "pressure"
                continue
            if not isinstance(a, CableActuator):
                raise ConfigError(f"sofa-qp: actuator {a.name!r} is not a cable or a pressure")
            nodes, bary = body.cable_rows(a)
            via = np.einsum("kj,kji->ki", bary, body.mesh.points[nodes]) @ rot.T + self.offset
            behind = via[0] + (via[0] - via[1]) / np.linalg.norm(via[1] - via[0]) * 1e-3
            cnode = node.addChild(f"cable_{a.name}")
            cnode.addObject("MechanicalObject", position=via.tolist())
            self.actuators[a.name] = cnode.addObject(
                "CableActuator", name=a.name, indices=list(range(len(via))), pullPoint=behind.tolist(),
                hasPullPoint=True, minForce=float(lo) * dt, maxForce=float(hi) * dt)
            cnode.addObject("BarycentricMapping", mapForces=False, mapMasses=False)
            self.kinds[a.name] = "force"
        missing = set(cables) - set(self.actuators)
        if missing:
            raise ConfigError(f"sofa-qp: no actuators {sorted(missing)}; have {[a.name for a in scene.actuators]}")
        # effector at the tip face's rest centroid
        from softsim.scenarios.inflation import _face_centre_normal
        self.tip_rest = _face_centre_normal(pts, sim.scenario._regions[tip])[0]
        goal = root.addChild("goal")
        self.goal = goal.addObject("MechanicalObject", name="goalMO", position=[self.tip_rest.tolist()])
        enode = node.addChild("effector")
        enode.addObject("MechanicalObject", position=[self.tip_rest.tolist()])
        use = [int(k in axes) for k in range(3)]
        self.effector = enode.addObject("PositionEffector", template="Vec3d", indices=[0], useDirections=use,
                                        effectorGoal="@../../goal/goalMO.position",
                                        weight=weight)
        enode.addObject("BarycentricMapping", mapForces=False, mapMasses=False)
        Sofa.Simulation.init(root)
        self.root = root
        self.tip_region = tip

    def set_bounds(self, name: str, lower: float, upper: float) -> None:
        """Change an actuator's bounds (values, not impulses) while the scene runs."""
        a = self.actuators[name]
        key = ("minPressure", "maxPressure") if self.kinds[name] == "pressure" else ("minForce", "maxForce")
        a.findData(key[0]).value = float(lower) * self.dt
        a.findData(key[1]).value = float(upper) * self.dt
        a.reinit()

    def set_goal(self, position: np.ndarray) -> None:
        self.goal.position.value = [list(map(float, position))]

    def vertices(self) -> np.ndarray:
        return np.array(self.dofs.position.value, float)

    def outputs(self) -> dict[str, float]:
        """The scenario's own outputs at the current vertices."""
        state = SimState(self.time, soft={self.body.name: SoftState(self.vertices(), np.zeros_like(self.points0))},
                         controls=self.forces())
        return self.sim.scenario.metrics(state)

    @property
    def time(self) -> float:
        return float(self.root.time.value)

    def forces(self) -> dict[str, float]:
        """Cable tensions [N] and pressures [Pa] now: the solver's lambda is an impulse, so value = lambda / dt."""
        return {n: float(np.asarray(a.force.value if self.kinds[n] == "force" else a.pressure.value).ravel()[0])
                / self.dt
                for n, a in self.actuators.items()}

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.sofa.Simulation.animate(self.root, self.dt)

    def close(self) -> None:
        try:
            self.sofa.Simulation.unload(self.root)
        except Exception:  # pragma: no cover
            pass


def _rotation(q) -> np.ndarray:
    from softsim.backends.sofa.backend import _quat_to_mat
    return _quat_to_mat(q)


def _settled(forces: np.ndarray, outs: np.ndarray, force_tol: np.ndarray, out_tol: np.ndarray) -> bool:
    """Forces and outputs sampled over the window (rows) have each moved less than their tolerance."""
    return bool(np.all(np.ptp(forces, axis=0) <= force_tol) and np.all(np.ptp(outs, axis=0) <= out_tol))


def solve_inverse_sofa(config: str | Path | Mapping[str, Any], targets: Mapping[str, Any], actuators: Sequence[str],
                       bounds: Mapping[str, tuple[Any, Any]], overrides: Sequence[str] = (), tolerance: float = 0.05,
                       scales: Mapping[str, float] | None = None, timestep: float | str = "2 ms",
                       settle_damping: Mapping[str, Any] | None = None, settle_tol: float = 1e-3,
                       settle_window: float | str = 0.05, max_settle: float | str = 2.0, max_corrections: int = 8,
                       qp_weight: float | None = None, angle_direction: str | None = None, **_ignored: Any) -> InverseResult:
    """Find the constant cable tensions / pressures that bring ``targets`` to equilibrium, with SoftRobots.Inverse.

    Position targets (``tip_x``, ``tip_y``, ``tip_z``) are the effector's goal: SOFA's QP finds the tensions in one
    run per goal. The effector sits at a barycentric point at the tip face's rest centroid while the output is the
    area-weighted centroid of the face, so a few corrections of the goal by the remaining error follow the first
    solve (``max_corrections``). An angle target (``bend_angle`` or ``chord_angle``) is not a position: it is
    reached by moving the effector goal along a free axis (``angle_direction``) with a Broyden iteration on the
    goals of all targets (a secant for one angle), which also handles angle and position targets together and any
    mix of cables and pressures. Any other target is refused.

    Args:
        config: Scenario YAML path or mapping (cable or pressure actuators on one soft body with a ``tip`` / ``top`` region).
        targets: ``{output: value}``, SI or ``"<n> <unit>"`` strings.
        actuators: Actuator names to solve for (as many as there are effector axes, or one cable for an angle target).
        bounds: ``{actuator: (lower, upper)}``: tensions [N] (``minForce`` / ``maxForce``) or pressures [Pa]
            (``minPressure`` / ``maxPressure``) of the QP.
        overrides: Dotted ``key=value`` config overrides.
        tolerance: Stop when the largest scaled residual is below this (scale: 1 mm, 0.01 rad; ``scales`` overrides).
        timestep: SOFA step [s]: it is implicit, so far larger than the scenario's explicit-solver step.
        settle_damping: Damping card for the body (default: Rayleigh beta = 10 ms, as the repeated-solve method).
        settle_tol, settle_window, max_settle: Settled when every output moved less than ``settle_tol`` of its scale
            and every force less than ``settle_tol`` of its range over the window [s]; hitting ``max_settle`` [s] is
            counted in ``unsettled``.
        max_corrections: Cap on goal corrections / secant steps.
        qp_weight: ``PositionEffector`` weight (default 1; 1e6 with pressure actuators, see the module doc).
        angle_direction: Axis (``"+x"``, ``"-x"``, ``"+y"``, ...) along which the effector goal is moved to reach an
            angle target, i.e. the direction the tip moves as it bends (the bending tube curls to ``-x``). Default
            ``None``: ``+x`` first (an axis no position target uses), ``-x`` if that leaves every actuator at its
            lower bound with the angle short.

    Returns:
        :class:`~softsim.control.inverse.InverseResult` (``simulations`` = QP runs to a settled state, ``steps`` =
        SOFA time steps). A target beyond the bounds gives the bounded best with ``status == "bounded_best"``.
    """
    from softsim.control.inverse import DEFAULT_DAMPING, _with_damping

    t0 = time.perf_counter()
    max_settle, settle_window = parse_quantity(max_settle, "time"), parse_quantity(settle_window, "time")
    dt = parse_quantity(timestep, "time")
    if isinstance(config, (str, Path)):
        base, raw = Path(config).resolve().parent, load_yaml(config)
    else:
        base, raw = Path.cwd(), dict(config)
    raw = apply_overrides(raw, [f"simulation.timestep={dt}", *overrides])
    raw = _with_damping(raw, settle_damping if settle_damping is not None else DEFAULT_DAMPING)
    names, outs = list(actuators), list(targets)
    missing = [n for n in names if n not in bounds]
    if missing:
        raise ConfigError(f"no bounds for actuators {missing}")
    if not names or not outs:
        raise ConfigError("need at least one actuator and one target")
    unknown = [o for o in outs if o not in POSITION_OUTPUTS and o not in ANGLE_OUTPUTS]
    if unknown:
        raise ConfigError(f"sofa-qp: targets {unknown} are not effector outputs; use {sorted(POSITION_OUTPUTS)} "
                          f"or one of {list(ANGLE_OUTPUTS)}")
    if len(outs) < len(names):
        raise ConfigError(f"{len(names)} actuators but only {len(outs)} targets: the solution would not be unique")
    sim = Simulation(raw, backend="sofa", base_dir=base)
    scene = None
    try:
        dims = {n: actuator_dimension(sim, n) for n in names}
        for n in names:
            if dims[n] not in ("force", "pressure", "length"):
                raise ConfigError(f"sofa-qp: actuator {n!r} is neither a cable nor a pressure")
        lo = {n: parse_quantity(bounds[n][0], dims[n]) for n in names}
        hi = {n: parse_quantity(bounds[n][1], dims[n]) for n in names}
        for n in names:
            if not (np.isfinite(lo[n]) and np.isfinite(hi[n]) and hi[n] > lo[n]):
                raise ConfigError(f"actuator {n!r}: bounds must be finite with upper > lower, got {bounds[n]!r}")
        target = {o: parse_quantity(targets[o], output_dimension(o)) for o in outs}
        scale = {o: parse_quantity((scales or {}).get(o, _SCALES[output_dimension(o)]), output_dimension(o))
                 for o in outs}
        goal_axes, goal_signs, auto_sign = _goal_axes(outs, angle_direction)
        axes = sorted(set(goal_axes))
        weight = qp_weight if qp_weight is not None else (1e6 if "pressure" in dims.values() else 1.0)
        # a stroke-controlled cable (tension = k (length - rest + stroke)) is solved for its tension, with tension bounds
        # re-derived from the stroke bounds at the current length after each run; the stroke is read back from it
        strokes = {n: (a.stiffness, sim.scenario._cables[n][3]) for n in names if dims[n] == "length"
                   for a in sim.scene.actuators if a.name == n}
        tlo = {n: (strokes[n][0] * max(lo[n], 0.0) if n in strokes else lo[n]) for n in names}
        thi = {n: (strokes[n][0] * hi[n] if n in strokes else hi[n]) for n in names}
        scene = InverseScene(sim, {n: (tlo[n], thi[n]) for n in names}, axes, weight=weight)
        return _run(scene, names, outs, target, scale, lo, hi, tolerance, settle_tol, settle_window, max_settle,
                    max_corrections, t0, goal_axes, goal_signs, auto_sign, strokes)
    finally:
        if scene is not None:
            scene.close()
        sim.close()


def _goal_axes(outs, direction: str | None) -> tuple[list[int], list[float], bool]:
    """The effector axis and sign each target's goal is moved along (positions: their own axis; angles: free axes)."""
    axes: list[int | None] = [POSITION_OUTPUTS.get(o) for o in outs]
    used = {a for a in axes if a is not None}
    sign, auto = 1.0, direction is None
    free = [a for a in range(3) if a not in used]
    first = None
    if direction is not None:
        d = str(direction).strip().lower()
        if len(d) != 2 or d[0] not in "+-" or d[1] not in "xyz":
            raise ConfigError(f"angle_direction must be '+x', '-x', '+y', ... not {direction!r}")
        sign, first = (1.0 if d[0] == "+" else -1.0), "xyz".index(d[1])
        if first in used:
            raise ConfigError(f"angle_direction {direction!r}: the {d[1]} axis is already a position target")
    signs = []
    for k, a in enumerate(axes):
        if a is None:
            cand = [first] if first is not None and first in free else free
            if not cand:
                raise ConfigError("sofa-qp: no free effector axis for the angle target (angles need an axis each "
                                  "that no position target uses)")
            axes[k] = a = cand[0]
            free.remove(a)
        signs.append(1.0 if outs[k] in POSITION_OUTPUTS else sign)
    return [int(a) for a in axes], signs, auto


def _run(scene: InverseScene, names, outs, target, scale, lo, hi, tolerance, settle_tol, settle_window, max_settle,
         max_corrections, t0, axes=None, signs=None, auto_sign=False, strokes=None) -> InverseResult:
    """The goal / correction loop around the settled QP runs (see :func:`solve_inverse_sofa`)."""
    chunk = 10
    window = max(int(round(settle_window / (chunk * scene.dt))), 2)
    max_chunks = max(int(round(max_settle / (chunk * scene.dt))), window + 1)
    strokes = strokes or {}
    # tension bounds of the run (a stroke-controlled cable's follow its stroke bounds at the current length)
    tlo = {n: (strokes[n][0] * max(lo[n], 0.0) if n in strokes else lo[n]) for n in names}
    thi = {n: (strokes[n][0] * hi[n] if n in strokes else hi[n]) for n in names}
    force_tol = np.array([settle_tol * (thi[n] - tlo[n]) for n in names])
    last: dict[str, Any] = {}
    secant: dict[tuple[str, str], tuple[float, float]] = {}   # (bound used, bound it should be) of the last run
    out_tol = np.array([settle_tol * scale[o] for o in outs])
    sc = np.array([scale[o] for o in outs])
    tgt = np.array([target[o] for o in outs])
    if axes is None:
        axes, signs, auto_sign = _goal_axes(outs, None)
    signs = list(signs)
    is_angle = np.array([o in ANGLE_OUTPUTS for o in outs])
    stats = {"steps": 0, "unsettled": 0, "runs": 0}
    history: list[dict[str, Any]] = []

    def settle() -> tuple[np.ndarray, np.ndarray]:
        f_hist, o_hist = [], []
        for _ in range(max_chunks):
            scene.step(chunk)
            stats["steps"] += chunk
            m, f = scene.outputs(), scene.forces()
            last["m"] = m
            f_hist.append([f[n] for n in names])
            o_hist.append([m[o] for o in outs])
            if len(f_hist) >= window and _settled(np.array(f_hist[-window:]), np.array(o_hist[-window:]), force_tol,
                                                  out_tol):
                break
        else:
            stats["unsettled"] += 1
        stats["runs"] += 1
        history.append({"values": dict(zip(names, map(float, stroke_of(np.array(f_hist[-1]), last["m"])))),
                        "outputs": dict(zip(outs, map(float, o_hist[-1])))})
        return np.array(f_hist[-1]), np.array(o_hist[-1])

    rest = scene.tip_rest
    height = float(rest[2] - scene.points0[:, 2].min())

    def first_goals() -> np.ndarray:
        # positions: the target itself; angles: the tip of a circular arc of the target angle (x ~ theta L / 2)
        return np.array([float(np.clip(target[o], -np.pi, np.pi)) * height / 2 if a else target[o] - rest[ax]
                         for o, a, ax in zip(outs, is_angle, axes)])

    def slope0() -> np.ndarray:  # d(output) / d(goal) at the start: 1 for a position, 2 / L for an angle
        return np.diag(np.where(is_angle, 2.0 / height, 1.0) / sc)

    goals, jac = first_goals(), slope0()
    best: dict[str, Any] = {"res": np.inf}
    def stroke_of(f: np.ndarray, m: dict[str, float]) -> np.ndarray:
        """Values as the job gives them: tensions, pressures and, for a stroke cable, stroke = T / k - (L - L0)."""
        return np.array([f[i] if n not in strokes else
                         f[i] / strokes[n][0] - (m[f"{n}_length"] - strokes[n][1]) for i, n in enumerate(names)])

    def tension_bounds(m: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        out = [], []
        for n in names:
            if n in strokes:
                k, rest = strokes[n]
                stretch = m[f"{n}_length"] - rest
                tl, th = max(k * (lo[n] + stretch), 0.0), max(k * (hi[n] + stretch), 0.0)
            else:
                tl, th = lo[n], hi[n]
            out[0].append(tl)
            out[1].append(th)
        return np.array(out[0]), np.array(out[1])

    low, high = np.array([tlo[n] for n in names]), np.array([thi[n] for n in names])
    span = high - low
    prev = None                      # (goals, residual) of the last run, for Broyden's update
    flipped = False
    for k in range(max_corrections + 1):
        goal = rest.copy()
        for ax, sg, d in zip(axes, signs, goals):
            goal[ax] = rest[ax] + sg * d
        scene.set_goal(goal)
        f, o = settle()
        r = (o - tgt) / sc
        worst = float(np.abs(r).max())
        at_low = bool(np.any(f <= low + 1e-3 * span))
        saturated = bool(at_low or np.any(f >= low + span * (1 - 1e-3)))
        if worst < best["res"] or (strokes and saturated):  # a saturated stroke cable: the latest run has the consistent bounds
            best.update(res=worst, f=f.copy(), o=o.copy(), m=dict(last["m"]), low=low.copy(), span=span.copy())
        if (auto_sign and not flipped and is_angle.any() and at_low and worst > tolerance):
            # the angle goal pushed the wrong way (every actuator can only push one way): go the other way
            flipped = True
            signs = [-sg if a else sg for sg, a in zip(signs, is_angle)]
            goals, jac, prev = first_goals(), slope0(), None
            best = {"res": np.inf}
            continue
        moved = False
        if strokes and saturated:  # the next run's tension bounds follow the stroke bounds at the length reached now
            target_low, target_high = tension_bounds(last["m"])   # what the bounds are at the length reached now
            new_low, new_high = target_low.copy(), target_high.copy()
            for i, n in enumerate(names):  # the length depends on the tension: secant on bound = target(bound)
                if n not in strokes:
                    continue
                for tag, used, target, new in (("lo", low, target_low, new_low), ("hi", high, target_high, new_high)):
                    if (n, tag) in secant:
                        (b1, t1), (b2, t2) = secant[n, tag], (used[i], target[i])
                        den = (t2 - b2) - (t1 - b1)
                        if abs(den) > 1e-12:
                            new[i] = max(b2 - (t2 - b2) * (b2 - b1) / den, 0.0)
                    secant[n, tag] = (used[i], target[i])
            moved = bool(np.any(np.abs(new_low - low) + np.abs(new_high - high) > 1e-3 * span))
            if moved:
                low, high = new_low, new_high
                span = np.maximum(high - low, 1e-12)
                for i, n in enumerate(names):
                    if n in strokes:
                        scene.set_bounds(n, low[i], high[i])
        if worst <= tolerance or (saturated and k >= 1 and not moved) or k == max_corrections:
            break
        if is_angle.any():  # Broyden's update of d(residual) / d(goal), then a Newton step
            if prev is not None:
                dg, dr = goals - prev[0], r - prev[1]
                if np.dot(dg, dg) > 0:
                    jac = jac + np.outer(dr - jac @ dg, dg) / np.dot(dg, dg)
            prev = (goals.copy(), r.copy())
            try:
                step = np.linalg.solve(jac, r)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(jac, r, rcond=None)[0]
            goals = goals - step
        else:
            goals = goals - (o - tgt)
    f, o = best["f"], best["o"]
    low, span = best["low"], best["span"]       # the bounds of the run that gave the best result
    at_bounds = [n for i, n in enumerate(names) if f[i] <= low[i] + 1e-3 * span[i] or f[i] >= low[i] + span[i] * (1 - 1e-3)]
    values = {n: float(v) for n, v in zip(names, stroke_of(f, best["m"]))}
    if best["res"] <= tolerance:
        status = "converged"
    elif at_bounds:
        status = "bounded_best"
    elif stats["runs"] > max_corrections:
        status = "max_evaluations"
    else:
        status = "stalled"
    return InverseResult(values, dict(zip(outs, map(float, o))), dict(zip(outs, map(float, tgt))),
                         dict(zip(outs, map(float, o - tgt))), best["res"], status, at_bounds, stats["runs"],
                         stats["steps"], stats["unsettled"], time.perf_counter() - t0, history)


def run_job_sofa(path: str | Path, job: Mapping[str, Any]) -> dict[str, Any]:
    """Run a parsed inverse job file with the SOFA QP method (see :func:`softsim.control.inverse.run_job`)."""
    path = Path(path).resolve()
    own = {"overrides", "tolerance", "scales", "timestep", "settle_damping", "settle_tol", "settle_window",
           "max_settle", "max_corrections", "qp_weight", "angle_direction"}
    shared = {"config", "targets", "actuators", "backend", "method", "initial", "max_evaluations", "warm_start",
              "jacobian_step"}  # the repeated-solve method's keys are accepted and ignored
    extra = set(job) - own - shared
    if extra:
        raise ConfigError(f"{path}: unknown keys {sorted(extra)}")
    bounds = {n: tuple(b) for n, b in job["actuators"].items()}
    result = solve_inverse_sofa(path.parent / job["config"], job["targets"], list(bounds), bounds,
                                **{k: v for k, v in job.items() if k in own})
    return result.to_dict()
