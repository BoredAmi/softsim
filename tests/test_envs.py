"""Gymnasium environments (softsim.envs), MuJoCo, coarse meshes and short episodes."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
gym = pytest.importorskip("gymnasium")

from softsim.config.loader import load_yaml  # noqa: E402
from softsim.errors import StabilityError  # noqa: E402
from softsim.runner import Simulation  # noqa: E402

import softsim.envs  # noqa: E402,F401  (registers the ids)
from softsim.envs import make_env  # noqa: E402

ROOT = Path(__file__).parents[1]
BEAM = ROOT / "scenarios" / "envs" / "cable_beam_reach.yaml"
ARM = ROOT / "scenarios" / "envs" / "cable_arm_reach.yaml"
BEAM_ID = "softsim/CableBeamReach-v0"


@pytest.fixture
def beam():
    env = gym.make(BEAM_ID)
    yield env
    env.close()


def test_check_env_passes(beam):
    """Gymnasium's own checker (spaces, reset/step signatures, seeding, determinism of reset) on the beam env."""
    from gymnasium.utils.env_checker import check_env

    check_env(beam.unwrapped, skip_render_check=True)


def test_registered_ids_and_spaces(beam):
    """Both example ids are registered; the action box is the cable tension range in N."""
    assert "softsim/CableArmReach-v0" in gym.registry
    box = beam.action_space
    assert box.shape == (1,) and box.low[0] == 0.0 and box.high[0] == pytest.approx(0.7)
    obs, info = beam.reset(seed=0)
    assert beam.observation_space.contains(obs)
    assert info["target"].shape == (1,) and 0.02 <= info["target"][0] <= 0.07


def test_same_seed_same_trajectory():
    """Two environments with the same seed and the same actions produce identical observations and rewards."""
    runs = []
    for _ in range(2):
        env = gym.make(BEAM_ID, randomize=True)
        obs, _ = env.reset(seed=5)
        traj = [obs]
        for a in (0.2, 0.5, 0.35):
            obs, r, *_ = env.step(np.array([a], dtype=np.float32))
            traj += [obs, r]
        env.close()
        runs.append(np.concatenate([np.atleast_1d(x) for x in traj]))
    assert np.array_equal(runs[0], runs[1])


def test_constant_action_equals_the_scenario_with_that_constant_actuator(beam):
    """Holding 0.5 N for 4 env steps gives the metrics of the Simulation driven directly with the same
    overrides and a 0.5 N control for the same number of physics steps (the env adds nothing physical)."""
    cfg = load_yaml(BEAM)
    sim = Simulation(BEAM.parent / cfg["scenario"], overrides=cfg["overrides"])
    sim.apply_controls({"cable": 0.5})
    sim.step(4 * beam.unwrapped.substeps)
    want = sim.metrics()
    sim.close()
    beam.reset(seed=1)
    for _ in range(4):
        _, _, _, _, info = beam.step(np.array([0.5], dtype=np.float32))
    got = info["metrics"]
    assert got["cable_tension"] == pytest.approx(0.5)
    assert want["bend_angle"] > 0.03  # it did bend
    for key in ("t", "bend_angle", "tip_x", "tip_z", "cable_length"):
        assert got[key] == pytest.approx(want[key], rel=1e-6, abs=1e-12), key


def test_reset_really_resets(beam):
    """After bending the beam, reset returns the rest state: zero time, angle, tension and the rest observation."""
    obs0, _ = beam.reset(seed=2, options={"target": [0.05]})
    for _ in range(3):
        beam.step(np.array([0.7], dtype=np.float32))
    assert beam.unwrapped._metrics["bend_angle"] > 0.05
    obs, info = beam.reset(options={"target": [0.05]})
    m = info["metrics"]
    assert m["t"] == 0.0 and m["bend_angle"] == 0.0 and m["cable_tension"] == 0.0
    assert np.array_equal(obs, obs0)
    # and the next step starts from rest again: same as the first episode's first step
    first, *_ = beam.step(np.array([0.3], dtype=np.float32))
    beam.reset(options={"target": [0.05]})
    again, *_ = beam.step(np.array([0.3], dtype=np.float32))
    assert np.array_equal(first, again)


def test_proportional_controller_reaches_the_target(beam):
    """A hand-written integrating controller (tension += 5 N/rad x error) reaches a 0.05 rad tip angle within 10
    steps: terminated by success, within the 0.004 rad tolerance, reward positive from the success bonus."""
    beam.reset(seed=3, options={"target": [0.05]})
    a = np.zeros(1)
    for k in range(10):
        angle = beam.unwrapped._metrics["bend_angle"]
        a = np.clip(a + 5.0 * (0.05 - angle), 0.0, 0.7)
        _, reward, terminated, truncated, info = beam.step(a.astype(np.float32))
        if terminated or truncated:
            break
    assert terminated and not truncated and info["success"], f"no success in 10 steps (error {info['error']})"
    assert abs(info["error"][0]) <= 0.004
    assert reward > 0.0


def test_truncation_after_max_steps_and_smooth_reward(beam):
    """Doing nothing never reaches a 0.05 rad target: truncated at max_steps, reward the negative scaled distance."""
    beam.reset(seed=4, options={"target": [0.05]})
    for k in range(beam.unwrapped.max_steps):
        _, reward, terminated, truncated, _ = beam.step(np.zeros(1, dtype=np.float32))
        assert reward == pytest.approx(-5.0)  # |0 - 0.05| / 0.01 rad
        assert not terminated
    assert truncated and k + 1 == beam.unwrapped.max_steps


def test_backend_failure_terminates_with_info(beam, monkeypatch):
    """A StabilityError from the simulation ends the episode with the failure penalty and the message in info."""
    beam.reset(seed=0)

    def boom(n=None):
        raise StabilityError("diverged for test")

    monkeypatch.setattr(beam.unwrapped.sim, "step", boom)
    _, reward, terminated, truncated, info = beam.step(np.array([0.3], dtype=np.float32))
    assert terminated and not truncated and "diverged for test" in info["failure"]
    assert reward == beam.unwrapped.failure_penalty


def test_range_randomisation_within_bounds_and_reproducible():
    """With randomize=True the Young's modulus is redrawn within +-20 % of the card (rule in the config), the sim's
    material really changes, and the same seed gives the same draw and target."""
    env = gym.make(BEAM_ID, randomize=True).unwrapped
    nominal = env._nominal.elastic.youngs_modulus
    draws, angles = [], []
    for seed in (1, 2, 3):
        _, info = env.reset(seed=seed)
        e = info["material"]["elastic.youngs_modulus"]
        assert abs(e / nominal - 1.0) <= 0.2 + 1e-12
        assert env.sim.scene.soft_bodies[0].material.elastic.youngs_modulus == pytest.approx(e)
        draws.append(e)
        for _ in range(3):
            env.step(np.array([0.4], dtype=np.float32))
        angles.append(env._metrics["bend_angle"])
    assert len(set(draws)) == 3
    assert np.all(np.diff(sorted(angles)) > 0)  # a different material bends a different amount
    _, again = env.reset(seed=2)
    assert again["material"]["elastic.youngs_modulus"] == draws[1]
    env.close()


def test_card_uncertainty_randomisation():
    """With a card that states uncertainty (E +-5 % normal, nu +-0.01 normal) the redraw follows ``uncertainty``:
    draws within 5 sigma, different between seeds, equal for equal seeds, and the simulation's material is the draw."""
    card = {"density": "1070 kg/m^3", "damping": {"model": "rayleigh", "beta": 0.01},
            "elastic": {"model": "stable_neo_hookean", "youngs_modulus": "500 kPa", "poisson_ratio": 0.45},
            "uncertainty": {"elastic.youngs_modulus": {"rel": 0.05, "dist": "normal"},
                            "elastic.poisson_ratio": {"abs": 0.01, "dist": "normal"}}}
    cfg = load_yaml(BEAM)
    cfg["scenario"] = str(BEAM.parent / cfg["scenario"])
    cfg["overrides"] = cfg["overrides"] + [f"body.material={json.dumps(card)}"]
    cfg["randomize"] = {"card_uncertainty": True}
    env = make_env(cfg)
    try:
        e0, nu0 = 500e3, 0.45
        _, a = env.reset(seed=11)
        _, b = env.reset(seed=12)
        _, c = env.reset(seed=11)
        assert set(a["material"]) == {"elastic.youngs_modulus", "elastic.poisson_ratio"}
        assert a["material"] == c["material"] and a["material"] != b["material"]
        for d in (a, b):
            assert abs(d["material"]["elastic.youngs_modulus"] / e0 - 1) <= 5 * 0.05
            assert abs(d["material"]["elastic.poisson_ratio"] - nu0) <= 5 * 0.01
        assert env.sim.scene.soft_bodies[0].material.elastic.youngs_modulus == pytest.approx(
            c["material"]["elastic.youngs_modulus"])
    finally:
        env.close()


def test_randomisation_off_keeps_the_card():
    """Without randomize the material is the card's, and a reset after a randomised one restores it."""
    env = gym.make(BEAM_ID).unwrapped
    _, info = env.reset(seed=0)
    assert info["material"] == {}
    assert env.sim.scene.soft_bodies[0].material.elastic.youngs_modulus == env._nominal.elastic.youngs_modulus
    env.close()


def test_sync_vector_env():
    """gymnasium.vector.SyncVectorEnv runs two beam envs: batched spaces, batched reset/step, autoreset on success."""
    from gymnasium.vector import SyncVectorEnv

    venv = SyncVectorEnv([lambda: gym.make(BEAM_ID) for _ in range(2)])
    try:
        obs, info = venv.reset(seed=[1, 2])
        assert obs.shape == (2, 5) and venv.action_space.shape == (2, 1)
        assert obs[0, -1] != obs[1, -1]  # different targets from different seeds
        obs, reward, term, trunc, info = venv.step(np.array([[0.1], [0.6]], dtype=np.float32))
        assert obs.shape == (2, 5) and reward.shape == (2,)
        assert obs[1, 0] > obs[0, 0] > 0  # more tension, more bend
    finally:
        venv.close()


def test_arm_env_two_actions_reach_observation():
    """The arm env: 2 tensions in, tip position in the observation; the known tensions (0.45 N, 0.15 N) held
    2 s bring the tip within 2 mm of the (reachable) default target. Two steps here are enough to check wiring."""
    env = gym.make("softsim/CableArmReach-v0").unwrapped
    try:
        assert env.action_space.shape == (2,) and env.actuators == ["cable_a", "cable_b"]
        obs, info = env.reset(seed=0)
        assert obs.shape == (8,) and obs[-2:] == pytest.approx([0.003994, 0.118515], abs=1e-6)
        for _ in range(2):
            obs, reward, term, trunc, info = env.step(np.array([0.45, 0.15], dtype=np.float32))
        m = info["metrics"]
        assert m["cable_a_tension"] == pytest.approx(0.45) and m["cable_b_tension"] == pytest.approx(0.15)
        assert m["tip_z"] < 0.12 and np.isfinite(reward)
    finally:
        env.close()


def test_bad_configs_are_refused():
    """Unknown actuator, unknown metric and an inverted bound each raise a ConfigError that names the problem."""
    from softsim.errors import ConfigError

    base = load_yaml(BEAM)
    base["scenario"] = str(BEAM.parent / base["scenario"])
    for patch, text in (({"actuators": {"nope": ["0 N", "1 N"]}}, "no actuator"),
                        ({"observation": ["nope"]}, "unknown metric"),
                        ({"actuators": {"cable": ["1 N", "0 N"]}}, "low < high")):
        with pytest.raises(ConfigError, match=text):
            make_env({**base, **patch})


def test_missing_gymnasium_gives_a_clear_error():
    """With gymnasium unimportable, `import softsim.envs` still works and make_env raises an ImportError that
    names the extra to install."""
    code = ("import sys; sys.modules['gymnasium'] = None\n"
            "import softsim.envs as e\n"
            "try:\n    e.make_env\nexcept ImportError as exc:\n    print('ERR', exc)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert "ERR" in out.stdout and "softsim[rl]" in out.stdout


# -- pneumatic finger, reduced gripper, baseline (M12 groundwork, second batch) ---------------------------------

PNEU = ROOT / "scenarios" / "envs" / "pneunet_bend.yaml"
PNEU_ID = "softsim/PneuNetBend-v0"
GRIP_ID = "softsim/ReducedGripperLift-v0"


@pytest.fixture
def pneu():
    env = gym.make(PNEU_ID)
    yield env
    env.close()


@pytest.fixture
def grip():
    env = gym.make(GRIP_ID)
    yield env
    env.close()


def test_pneunet_env_check_and_spaces(pneu):
    """check_env passes; the action is the chamber pressure in Pa, the observation bend angle, tip x/z, its rate, the
    held pressure and the target."""
    from gymnasium.utils.env_checker import check_env

    check_env(pneu.unwrapped, skip_render_check=True)
    assert pneu.action_space.shape == (1,) and pneu.action_space.high[0] == pytest.approx(25e3)
    obs, info = pneu.reset(seed=0)
    assert obs.shape == (6,) and 0.1 <= info["target"][0] <= 0.3
    assert obs[3] == 0.0 and obs[4] == 0.0  # at rest: no rate, no pressure


def test_pneunet_constant_action_equals_the_scenario(pneu):
    """15 kPa held for 3 env steps equals the Simulation driven directly with the same overrides; more pressure
    bends more, and the rate observation is the bend angle's change over the control period."""
    cfg = load_yaml(PNEU)
    sim = Simulation(PNEU.parent / cfg["scenario"], overrides=cfg["overrides"], backend="mujoco")
    sim.apply_controls({"cavity": 15e3})
    sim.step(3 * pneu.unwrapped.substeps)
    want = sim.metrics()["bend_angle"]
    sim.close()
    pneu.reset(seed=1)
    angles = [0.0]
    for _ in range(3):
        obs, _, _, _, info = pneu.step(np.array([15e3], dtype=np.float32))
        angles.append(info["metrics"]["bend_angle"])
    assert info["metrics"]["bend_angle"] == pytest.approx(want)
    assert want > 0.05
    assert obs[3] == pytest.approx((angles[3] - angles[2]) / 0.05, rel=1e-4)


def test_pneunet_randomisation_changes_the_response_reproducibly():
    """Modulus +-20 % changes the bend after one step, and the same seed gives the same material and the same bend."""
    runs = []
    for seed in (3, 3, 4):
        env = gym.make(PNEU_ID, randomize=True)
        env.reset(seed=seed)
        _, _, _, _, info = env.step(np.array([20e3], dtype=np.float32))
        runs.append((info["material"]["elastic.youngs_modulus"], info["metrics"]["bend_angle"]))
        env.close()
    assert runs[0] == runs[1] and runs[0][0] != runs[2][0] and runs[0][1] != runs[2][1]


def test_gripper_env_check_and_spaces(grip):
    from gymnasium.utils.env_checker import check_env

    check_env(grip.unwrapped, skip_render_check=True)
    assert grip.action_space.shape == (2,) and grip.observation_space.shape == (7,)
    obs, info = grip.reset(seed=0)
    assert 0.008 <= info["object"]["mass"] <= 0.020 and 0.7 <= info["object"]["friction"] <= 1.2
    assert obs[3] == pytest.approx(0.0, abs=1e-6) and obs[0] == 0.0  # the object starts on the table


def test_gripper_constant_action_equals_direct_simulation(grip):
    """A constant (pressure, lift) action over 3 steps equals stepping ReducedGripper directly with the same
    object and the env's linear ramp from zero during the first control period."""
    from softsim.reduced import ReducedFinger, ReducedGripper
    from softsim.reduced.mjcf import OBJECTS

    env = grip.unwrapped
    env.reset(seed=2, options={"mass": 0.012, "friction": 0.9})
    for _ in range(3):
        obs, *_ = env.step(np.array([30e3, 0.01], dtype=np.float32))
    g = ReducedGripper(ReducedFinger.from_json(ROOT / "softsim" / "reduced" / "wang_finger.json"), shape="box",
                       size=OBJECTS["box"], mass=0.012, friction=0.9)
    n = env.substeps
    for i in range(1, n + 1):
        g.step(30e3 * i / n, 0.01 * i / n)
    for _ in range(2 * n):
        g.step(30e3, 0.01)
    assert env.gripper.object_height == pytest.approx(g.object_height, abs=1e-9)
    assert obs[2] == pytest.approx(g.palm_rise, abs=1e-6)


def test_gripper_seeding_and_randomisation():
    """Same seed, same object and same trajectory; options fix the object; randomize=False uses the range means."""
    outs = []
    for seed in (7, 7, 8):
        env = gym.make(GRIP_ID)
        _, info = env.reset(seed=seed)
        for _ in range(5):
            obs, r, *_ = env.step(np.array([20e3, 0.0], dtype=np.float32))
        outs.append((info["object"], obs, r))
        env.close()
    assert outs[0][0] == outs[1][0] and np.array_equal(outs[0][1], outs[1][1]) and outs[0][2] == outs[1][2]
    assert outs[0][0] != outs[2][0]
    env = gym.make(GRIP_ID, randomize=False)
    _, info = env.reset(seed=1)
    assert info["object"] == {"mass": pytest.approx(0.014), "friction": pytest.approx(0.95)}
    _, info = env.reset(options={"mass": 0.02, "friction": 1.1})
    assert info["object"] == {"mass": 0.02, "friction": 1.1}
    env.close()


def _gripper_episode(env, pressure: float, lift: float, **options):
    env.reset(seed=0, options=options)
    total = 0.0
    for k in range(env.unwrapped.max_steps):
        a = [pressure * min(k / 12, 1), lift * min(max(k - 40, 0) / 10, 1)]
        _, r, term, trunc, info = env.step(np.array(a, dtype=np.float32))
        total += r
        if term or trunc:
            return total, info
    raise AssertionError("episode did not end")


def test_gripper_lift_rewards_holding_and_punishes_dropping(grip):
    """Scripted grasp of a 10 g box (mu 1): squeezing and lifting is held (is_held) with a high return; the same
    lift without pressure leaves the object on the table; reward is bounded by 1 per step."""
    held, info_h = _gripper_episode(grip, 30e3, 0.04, mass=0.010, friction=1.0)
    free, info_f = _gripper_episode(grip, 0.0, 0.04, mass=0.010, friction=1.0)
    assert info_h["held"] and info_h["success"] and info_h["rise"] > 0.03
    assert not info_f["held"] and info_f["rise"] == pytest.approx(0.0, abs=1e-3)
    assert held > 15 and free < 0 and held <= grip.unwrapped.max_steps


def test_gripper_bad_inputs():
    with pytest.raises(ValueError, match="unknown object shape"):
        gym.make(GRIP_ID, shape="sphere")
    env = gym.make(GRIP_ID)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="finite"):
        env.unwrapped.step(np.array([np.nan, 0.0]))
    env.close()


def test_train_baseline_script_runs(tmp_path):
    """tools/train_baseline.py end to end on a tiny budget (the gripper env is the fast one): it writes a result with
    the random and CEM returns. The real budget and numbers are in docs/envs.md."""
    out = tmp_path / "r.json"
    cmd = [sys.executable, str(ROOT / "tools" / "train_baseline.py"), "--env", GRIP_ID, "--generations", "1", "--pop", "4",
           "--episodes", "1", "--eval-episodes", "2", "--workers", "2", "--json", str(out)]
    done = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    res = json.loads(out.read_text())
    assert res["train_episodes"] == 4 and np.isfinite(res["random"]["mean"]) and np.isfinite(res["cem"]["mean"])
