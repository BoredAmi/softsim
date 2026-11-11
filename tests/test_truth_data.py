"""The Truth tab's data (tools/truth_data.py): the page shows the run it says it shows, and the fast finger runs the same
grasp. docs/gripper.md."""

import base64
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from softsim.reduced import ReducedFinger  # noqa: E402
from truth_data import QUANTUM, chapters, face_nodes, fast_run, pack, pressure_schedule  # noqa: E402


@pytest.fixture(scope="module")
def finger():
    return ReducedFinger.from_json(ROOT / "softsim" / "reduced" / "wang_finger.json")


def _palm(t):
    u = np.clip((t - 0.7) / 0.5, 0, 1)
    return 0.03 * (10 * u**3 - 15 * u**4 + 6 * u**5)


def _run(frames=151):
    """A grasp in the run file's layout: a two-triangle face on y = 0.012, one back node, the box on the table. The pad
    node touches the box until 0.9 s, the box leaves the table at 1.0 s."""
    t = np.linspace(0, 1.5, frames)
    rest = np.array([[0.016, 0.012, 0.03], [0.03, 0.012, 0.03], [0.03, 0.012, 0.14], [0.016, 0.012, 0.14], [0.03, -0.012, 0.14]])
    tris = np.array([[0, 1, 2], [0, 2, 3], [1, 2, 4]])
    x = np.tile(rest, (frames, 1, 1))
    x[:, :, 2] += _palm(t)[:, None]
    x[t > 0.9, 0, 0] += 0.003
    z = 0.03 + np.clip(t - 1.0, 0, None) * 0.03
    obj = np.column_stack([0 * t, 0 * t, z])
    return dict(t=t, x=x, rest=rest, tris=tris, object=obj, grip=0.1 + 0 * t, palm=_palm(t))


def test_the_face_is_the_side_at_full_width():
    nodes, tris = face_nodes(_run()["rest"], _run()["tris"])
    assert nodes.tolist() == [0, 1, 2, 3]
    assert tris.tolist() == [[0, 1, 2], [0, 2, 3]]


def test_packed_positions_are_the_run_to_half_a_quantum(finger):
    d = _run()
    p = pack(d, finger, "test")
    q = np.frombuffer(base64.b64decode(p["xz"]), "<i2").reshape(len(d["t"]), p["nodes"], 2) * p["quantum"]
    assert np.abs(q - d["x"][:, :4][:, :, [0, 2]]).max() <= QUANTUM / 2 + 1e-12
    assert p["touch"][0] == [[0.016, 0.03]]  # a touching node, wherever it is on the surface
    assert p["touch"][-1] == []              # it moved away at 0.9 s


def test_chapters_are_found_in_the_run():
    d = _run()
    marks = {c["label"]: d["t"][c["frame"]] for c in chapters(d, d["object"][:, 2] - 0.03)}
    assert marks["Closed"] == pytest.approx(0.7, abs=0.01)
    assert marks["Tips let go"] == pytest.approx(0.9, abs=0.02)
    assert marks["Off the table"] == pytest.approx(1.0, abs=0.02)
    assert marks["End"] == 1.5


def test_the_schedule_is_the_scenarios():
    t, p = pressure_schedule()
    assert t[0] == 0 and t[-1] == pytest.approx(1.5)
    assert np.interp(0.6, t, p) == pytest.approx(25e3)
    assert np.interp(0.15, t, p) == pytest.approx(2.5e3)


def test_the_fast_finger_runs_the_same_grasp(finger):
    """On the finite-element clock: on the table while closing, carried with the palm after."""
    t = np.linspace(0, 1.5, 151)
    r = fast_run(finger, t, _palm(t))
    assert np.abs(r["rise"][:70]).max() < 1e-3
    assert r["held"] and 0.02 < r["rise"][-1] < 0.03
    assert r["palm"][-1] == pytest.approx(0.03, abs=2e-3)
    assert r["rects"].shape == (151, 2 * finger.links, 4, 2)
    assert (r["rects"][-1, :finger.links, :, 0] > 0).all() != (r["rects"][-1, finger.links:, :, 0] > 0).all()  # one finger each side
    assert r["grip"][70] > 0.05
    assert r["grip"][:20].max() < 0.01  # under 4 kPa for the first 0.2 s: it has not closed yet
