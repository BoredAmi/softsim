"""A run stopped and continued is the run that was never stopped, sample for sample, as closely as a run repeats at all.

Two identical uninterrupted grasps differ by up to 5e-12 in their metrics (parallel sums on the GPU), and so does a resumed
one: ``same`` allows ``REPEATABLE``, a thousand times less than the smallest effect of leaving any state out.

Stopping is done by raising from the progress callback after a save, as a crash would. Between them the two cases touch
every part of the solver's state: the drop has viscous memory and a sliding weight, the grasp has friction, a free body,
moved regions read through the residual, and pressure.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from softsim.checkpoint import Saved, read_checkpoint, write_checkpoint
from softsim.errors import ConfigError
from softsim.runner import Simulation

DROP = ("scenarios/drop_test/cases/w050.yaml", [
    "specimen.material.damping={model: prony, g: [0.5], tau: [2 ms]}", "specimen.material.elastic.poisson_ratio=0.49",
    "specimen.mesh.size=0.008", "simulation.timestep=1e-4", "simulation.duration=0.0012", "simulation.output_rate=10000",
    "drop.gap=0.0002", "output.save_model=false"])


GRASP = ("scenarios/grasp/box.yaml", [
    "output.save_model=false", "body.mesh.size=3mm", "body.geometry.finger.tooth_height=0", "body.geometry.finger.wall=3mm",
    "body.geometry.finger.side_wall=3mm", "body.geometry.finger.floor=3mm", "body.material.friction.sliding=0.8",
    "object.size=[24.5mm,12mm,30mm]", "simulation.duration=0.05"])  # a box that nearly fills the gap: contact in two steps


REPEATABLE = 1e-10  # SI


class Stop(Exception):
    pass


def run(case, path=None, stop=None, duration=None, overrides=()):
    file, over = case
    over = [*over, *overrides, *([f"output.checkpoint={path}", "output.checkpoint_every=1e-9"] if path else [])]
    sim = Simulation(file, over, backend="fem")

    def progress(fraction):
        if stop is not None and fraction >= stop:
            raise Stop

    try:
        return sim.run(duration=duration, record_vertices=True, progress=progress)
    finally:
        sim.close()


@pytest.fixture(scope="module")
def whole():
    return run(DROP)


def same(a, b):
    assert list(a.timeseries.columns) == list(b.timeseries.columns)
    np.testing.assert_allclose(a.timeseries.to_numpy(), b.timeseries.to_numpy(), rtol=0, atol=REPEATABLE)
    for name in a.vertices:
        np.testing.assert_allclose(a.vertices[name], b.vertices[name], rtol=0, atol=REPEATABLE)


def test_a_stopped_run_ends_as_if_it_never_stopped(whole, tmp_path):
    path = tmp_path / "run.npz"
    with pytest.raises(Stop):
        run(DROP, path, stop=0.5)
    assert 0 < sample(path) < len(whole.timeseries) - 1
    same(whole, run(DROP, path))


def sample(path: Path) -> int:
    with np.load(path) as d:
        return json.loads(str(d["head"]))["sample"]


@pytest.mark.slow
def test_friction_memory_survives_a_stop(tmp_path):
    """Coulomb contact on a free box, with the grip read from the residual: stopped once the fingers are on the box."""
    path, whole = tmp_path / "run.npz", run(GRASP)
    assert whole.timeseries.grip_force.iloc[2] > 0 and whole.timeseries.object_x.abs().max() > 0
    with pytest.raises(Stop):
        run(GRASP, path, stop=0.6)
    assert sample(path) >= 2
    same(whole, run(GRASP, path))


def test_a_finished_run_can_be_extended(whole, tmp_path):
    path = tmp_path / "run.npz"
    run(DROP, path, duration=0.0006)
    same(whole, run(DROP, path))


def test_a_checkpoint_is_refused_by_a_different_run(tmp_path):
    path = tmp_path / "run.npz"
    with pytest.raises(Stop):
        run(DROP, path, stop=0.5)
    with pytest.raises(ConfigError, match="different run"):
        run(DROP, path, overrides=["simulation.timestep=5e-5"])
    with pytest.raises(ConfigError, match="different run"):
        run(DROP, path, overrides=["specimen.mesh.size=0.007"])


def test_a_write_that_dies_leaves_the_last_checkpoint_whole(tmp_path, monkeypatch):
    path = tmp_path / "run.npz"
    first = Saved("x", 1, 1.0, [{"t": 0.1 + 0.2}], {}, {"y": np.arange(3.0)})
    write_checkpoint(path, first)
    monkeypatch.setattr(os, "fsync", lambda _: (_ for _ in ()).throw(OSError("disk gone")))
    with pytest.raises(OSError):
        write_checkpoint(path, Saved("x", 2, 2.0, [], {}, {"y": np.zeros(3)}))
    back = read_checkpoint(path, "x")
    assert back.sample == 1 and back.rows == first.rows  # a double survives exactly
    np.testing.assert_array_equal(back.state["y"], first.state["y"])


def test_a_backend_without_checkpoints_says_so():
    sim = Simulation("scenarios/drop_test/cases/w050.yaml", ["output.checkpoint=x.npz", "simulation.duration=0.001"], backend="mujoco")
    with pytest.raises(NotImplementedError, match="mujoco"):
        sim.backend.checkpoint()
    sim.close()


@pytest.mark.slow
def test_everything_a_step_changes_is_saved_and_restored():
    """Structural, so that state added later cannot be forgotten: every array a few steps change (friction memory included,
    which the runs above barely load: a symmetric squeeze needs no friction) is in the checkpoint and comes back exactly."""
    from softsim.fem.solid3d import _SOLVER_STATE

    def build():
        sim = Simulation(GRASP[0], GRASP[1], backend="fem")
        return sim, sim.backend.solver

    def arrays(solver):
        out = {}
        for name, value in vars(solver).items():
            items = value if isinstance(value, list) else [value]
            if all(isinstance(v, np.ndarray) for v in items) and items:
                out[name] = [v.copy() for v in items]
        return out

    a, s = build()
    before = arrays(s)
    for _ in range(4):
        a.step()
    changed = {k for k, v in arrays(s).items() if k in before and any(not np.array_equal(x, y) for x, y in zip(v, before[k]))}
    assert {"_lam", "_anchor", "_bond", "y"} <= changed  # friction was engaged
    assert changed <= set(_SOLVER_STATE), changed - set(_SOLVER_STATE)
    b, fresh = build()
    b.backend.restore(a.backend.checkpoint())
    for name in _SOLVER_STATE:
        for x, y in zip(*(v if isinstance(v, list) else [v] for v in (getattr(s, name), getattr(fresh, name)))):
            np.testing.assert_array_equal(x, y, err_msg=name)
    assert fresh.t == s.t
    a.close(), b.close()
