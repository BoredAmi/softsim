"""Experiment pipeline tests on synthetic data (the real recordings are not needed)."""

import math
import zipfile

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")

from softsim.experiments import FrameSource, load_dataset  # noqa: E402
from softsim.experiments.calibrate import _parse_parameters  # noqa: E402
from softsim.experiments.compare import ExperimentCase, compare_case, simulated_indentation  # noqa: E402
from softsim.experiments.video.segment import SegmentationParams, largest_component, weight_mask  # noqa: E402
from softsim.experiments.video.track import (  # noqa: E402
    TrackingParams,
    _fill_holes,
    camera_elevation,
    specimen_geometry,
    track_rigid,
)

from conftest import ROOT  # noqa: E402

BACKGROUND = (30, 90, 0)  # BGR: saturated green, no red, like the recordings


def _frame(center, theta_deg, size=(240, 240), oversample=8):
    """A rigid dark 'weight' (body + knob + light mark) rotated by theta (clockwise on screen).

    Rendered at ``oversample`` x resolution and area-averaged down, like a camera
    integrating light over each pixel; plain anti-aliased polygon filling does
    not represent sub-pixel motion faithfully.
    """
    k = oversample
    img = np.zeros((size[0] * k, size[1] * k, 3), np.uint8)
    img[:] = BACKGROUND
    th = math.radians(theta_deg)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    bits = 4  # fractional vertex bits at the oversampled resolution
    place = lambda pts: np.round((np.asarray(pts, float) @ rot.T + center) * k * 2**bits).astype(np.int32)  # noqa: E731
    cv2.fillPoly(img, [place([(-30, -12), (30, -12), (30, 12), (-30, 12)]),
                       place([(-6, -24), (6, -24), (6, -12), (-6, -12)])], (20, 25, 18), shift=bits)
    cv2.fillPoly(img, [place([(-18, -3), (-12, -3), (-12, 3), (-18, 3)])], (200, 200, 200), shift=bits)
    return cv2.resize(img, size[::-1], interpolation=cv2.INTER_AREA)


def test_track_rigid_recovers_translation_and_rotation():
    centers = [(100 + 0.3 * i, 60 + 2.5 * i) for i in range(30)]
    thetas = [0.4 * i for i in range(30)]
    frames = [_frame(c, th) for c, th in zip(centers, thetas)]
    reference = np.zeros_like(frames[0])
    reference[:] = BACKGROUND
    p = TrackingParams(edge_dilate_px=5)  # default upper-part mask, as used on the recordings
    seg = lambda img: _fill_holes(largest_component(  # noqa: E731
        weight_mask(img, reference, "dark", SegmentationParams()), 50))
    m0 = seg(frames[0])
    ys, xs = np.nonzero(m0)
    out = track_rigid(lambda i: frames[i], seg, 0, len(frames), np.array([xs.mean(), ys.mean()]),
                      float(ys.max() - ys.min()), p)
    assert max(out) == len(frames) - 1
    com, theta, cc = out[29]
    # the tracked point is the material point at the initial centroid, which also turns about the body centre
    p0 = np.array([xs.mean(), ys.mean()])
    th = math.radians(thetas[29])
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    expected = np.asarray(centers[29]) + rot @ (p0 - np.asarray(centers[0]))
    assert com == pytest.approx(expected, abs=1.5)  # ~2 % of a 72 px displacement
    # rotation is the less precise output (sub-pixel edge motion per frame); position is what calibration uses
    assert math.degrees(theta) == pytest.approx(thetas[29], rel=0.2)
    assert cc > 0.9


def test_camera_elevation_roundtrip():
    d, h = 40.0, 17.0
    for e_deg in (0.0, 10.0, 18.0, 30.0):
        e = math.radians(e_deg)
        s = 5.5
        width, height = s * d, s * (h * math.cos(e) + d * math.sin(e))
        assert math.degrees(camera_elevation(width, height, d, h)) == pytest.approx(e_deg, abs=1e-6)


def test_specimen_geometry_ignores_flash():
    mask = np.zeros((200, 300), bool)
    mask[50:150, 100:200] = True       # 100 px wide body
    mask[140:150, 80:220] = True       # wider casting flash at the base
    g = specimen_geometry(mask, flash_fraction=0.15)
    assert g["width"] == 100 and g["top"] == 50 and g["bottom"] == 149 and g["center_x"] == pytest.approx(149.5)


def _synthetic_drop(t, z0=0.03, v0=-2.7, t_contact=0.005, k_over_m=4e6, g=9.81):
    """Free fall, then a damped-spring bounce: returns COM height."""
    z = z0 + v0 * t - 0.5 * g * t**2
    after = t > t_contact
    w = math.sqrt(k_over_m)
    zc = z0 + v0 * t_contact - 0.5 * g * t_contact**2
    vc = v0 - g * t_contact
    tau = t[after] - t_contact
    z[after] = zc + vc / w * np.sin(w * tau)
    return z


def test_simulated_indentation_applies_contact_rule():
    t = np.arange(0, 0.012, 1e-4)
    z = _synthetic_drop(t)
    contact_force = np.where(t > 0.005, 1.0, 0.0)
    ts = pd.DataFrame({"t": t, "weight_com_z": z, "specimen_min_z": 0.0, "contact_force_z": contact_force})
    sim = simulated_indentation(ts)
    # contact = last sample before the COM is 0.1 mm above free fall: close to (slightly after) the onset
    c = int(np.argmin(np.abs(sim.t.to_numpy())))
    assert 0.005 <= t[c] <= 0.0055
    # indentation is the COM drop since that sample
    assert sim.indentation_rel.to_numpy() == pytest.approx(z[c] - z, abs=1e-12)


def test_compare_identical_curves(tmp_path):
    t = np.arange(-0.002, 0.02, 1 / 3000)
    y = np.where(t > 0, 0.004 * np.sin(np.clip(t, 0, None) * math.pi / 0.008), 0.0)
    y[t > 0.008] = -0.0005
    pd.DataFrame({"t": t, "indentation_rel": y}).to_parquet(tmp_path / "c.parquet")
    (tmp_path / "c.yaml").write_text("impact_speed: 2.7\noffset_at_contact_mm: 1.5\nrotation_rate_at_contact_deg_s: 30\n")
    exp = ExperimentCase.load(tmp_path, "c")
    assert exp.window()[1] == pytest.approx(0.008, abs=1 / 3000)
    m = compare_case(pd.DataFrame({"t": t, "indentation_rel": y}), exp)
    assert m["rmse"] == pytest.approx(0.0, abs=1e-12) and m["peak_sim"] == pytest.approx(m["peak_exp"])
    ov = exp.sim_overrides()
    assert "drop.offset=[0.0015, 0.0]" in ov and any(o.startswith("drop.angular_velocity=[0.0, 0.52") for o in ov)


def test_dataset_and_frames(tmp_path):
    ds = load_dataset(ROOT / "datasets" / "czopek2026_ecoflex_drop.yaml")
    assert ds.doi == "10.5281/zenodo.23185808" and set(ds.cases) == {"w020", "w050", "w100", "w200", "w500"}
    assert ds.config_path("w020").exists()
    archive = tmp_path / "x.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for i in range(3):
            ok, buf = cv2.imencode(".png", np.full((8, 8, 3), i * 10, np.uint8))
            z.writestr(f"rec/frame_{i:06d}.png", buf.tobytes())
    with FrameSource(archive, "rec") as f:
        assert len(f) == 3 and f.read(2)[0, 0, 0] == 20


def test_calibration_parameters():
    params = _parse_parameters({
        "specimen.material.elastic.youngs_modulus": {"initial": "50 kPa", "bounds": ["5 kPa", "1 MPa"]},
        "specimen.material.damping.beta": {"initial": "2 ms", "bounds": ["0.1 ms", "20 ms"], "log": False},
    })
    e, b = params
    assert (e.initial, e.lower, e.upper, e.log) == (50e3, 5e3, 1e6, True)
    assert e.from_unit(e.to_unit(7e3)) == pytest.approx(7e3)
    assert e.from_unit(e.to_unit(1e9)) == 1e6  # clipped to bounds
    assert b.initial == pytest.approx(2e-3) and not b.log


def _fake_case(task):
    """Stand-in for a case simulation: w_crash kills its worker like a native segfault."""
    import os

    if task[3] == "w_crash":
        os._exit(11)
    return {"rmse": 0.001, "nrmse": 0.5}


def _bare_calibration(tmp_path, cases, jobs):
    from softsim.experiments.calibrate import Calibration

    cal = object.__new__(Calibration)
    cal.cases = {c: 1.0 for c in cases}
    cal._raw = {c: ({}, str(tmp_path)) for c in cases}
    cal.history, cal.output, cal._workers, cal.tracks = [], tmp_path, jobs, str(tmp_path)
    cal._spec_hash, cal._log, cal._done = "test", tmp_path / "evaluations.jsonl", {}
    cal._pool = None
    return cal


def test_calibration_survives_worker_crash_and_resumes(tmp_path, monkeypatch):
    from concurrent.futures import ProcessPoolExecutor

    from softsim.experiments import calibrate

    monkeypatch.setattr(calibrate, "_evaluate_case", _fake_case)
    cal = _bare_calibration(tmp_path, ["w_ok", "w_crash"], 2)
    cal._pool = ProcessPoolExecutor(2)
    try:
        cost = cal.evaluate({"x": 1.0})
        assert cal.evaluate({"x": 2.0}) == cost  # the rebuilt pool keeps working
    finally:
        cal._pool.shutdown()
    assert cost == pytest.approx((0.5 + 10.0) / 2)  # crashed case costs like an unstable one
    assert "crashed" in cal.history[0]["w_crash.error"]

    # a second run of the same calibration replays the log without simulating
    monkeypatch.setattr(calibrate, "_evaluate_case", lambda task: pytest.fail("should replay"))
    again = _bare_calibration(tmp_path, ["w_ok", "w_crash"], 1)
    again._done = again._load_log()
    assert again.evaluate({"x": 1.0}) == cost and len(again._done) == 2
