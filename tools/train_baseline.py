"""Training baseline without heavy dependencies: cross-entropy method (CEM) on a linear policy, numpy only.

The policy is ``action = low + (high - low) * sigmoid(W @ z + b)`` with ``z`` the observation standardised by the
mean/std of a few random rollouts. CEM keeps a Gaussian over the (W, b) parameters, scores each sample by its mean
return over a few episodes (the same seeds for the whole population in a generation) and refits the Gaussian on
the best fraction. The result is compared with uniformly random actions on held-out seeds.

    python tools/train_baseline.py                                  # CableBeamReach-v0, a few minutes on 2 cores
    python tools/train_baseline.py --env softsim/PneuNetBend-v0 --generations 4 --pop 8   # ~7 steps/s: slow
    python tools/train_baseline.py --json out.json

It exists to show the environments are learnable and to give a number to beat; for real training use
stable-baselines3 / torch on the same ids (optional, not a dependency of softsim).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

_ENV = None


def _init(env_id: str) -> None:
    global _ENV
    import gymnasium as gym
    import softsim.envs  # noqa: F401  (registers the ids)
    from softsim.envs import make_env  # noqa: F401

    _ENV = gym.make(env_id, randomize=True) if "Gripper" not in env_id else gym.make(env_id)


class Policy:
    def __init__(self, low, high, mean, std):
        self.low, self.high, self.mean, self.std = low, high, mean, std
        self.n_obs, self.n_act = len(mean), len(low)
        self.size = self.n_act * (self.n_obs + 1)

    def act(self, theta, obs):
        w = theta.reshape(self.n_act, self.n_obs + 1)
        z = (np.asarray(obs, float) - self.mean) / self.std
        s = 1 / (1 + np.exp(-np.clip(w[:, :-1] @ z + w[:, -1], -30, 30)))
        return (self.low + (self.high - self.low) * s).astype(np.float32)


def rollout(policy_args, theta, seed: int):
    """One episode; theta None = uniformly random actions. Returns (return, success, steps)."""
    env = _ENV
    pol = Policy(*policy_args)
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    total, info, steps = 0.0, {}, 0
    while True:
        a = env.action_space.sample() if theta is None else pol.act(theta, obs)
        obs, r, term, trunc, info = env.step(a)
        total += r
        steps += 1
        if term or trunc:
            return total, bool(info.get("success", False)), steps


def _roll(args):
    return rollout(*args)


def evaluate(pool, policy_args, theta, seeds):
    out = pool.map(_roll, [(policy_args, theta, s) for s in seeds])
    r = np.array([o[0] for o in out])
    return float(r.mean()), float(r.std()), float(np.mean([o[1] for o in out]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="softsim/CableBeamReach-v0")
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--elite", type=float, default=0.25)
    ap.add_argument("--episodes", type=int, default=3, help="training episodes per candidate")
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="write the result to this file")
    a = ap.parse_args()

    import gymnasium as gym
    import softsim.envs  # noqa: F401
    from softsim.envs import make_env  # noqa: F401

    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, _init, (a.env,)) as pool:
        probe = gym.make(a.env)
        low, high = probe.action_space.low.astype(float), probe.action_space.high.astype(float)
        probe.close()
        # observation statistics from random rollouts (also the random-action baseline's observations)
        _init(a.env)
        obs_all = []
        for s in range(3):
            o, _ = _ENV.reset(seed=1000 + s)
            obs_all.append(o)
            for _ in range(10):
                o, *_ = _ENV.step(_ENV.action_space.sample())
                obs_all.append(o)
        obs_all = np.array(obs_all, float)
        mean, std = obs_all.mean(0), obs_all.std(0) + 1e-6
        pa = (low, high, mean, std)
        pol = Policy(*pa)
        rng = np.random.default_rng(a.seed)
        mu, sigma = np.zeros(pol.size), np.full(pol.size, 1.0)
        n_elite = max(int(a.pop * a.elite), 2)
        history = []
        for g in range(a.generations):
            thetas = mu + sigma * rng.standard_normal((a.pop, pol.size))
            thetas[0] = mu
            seeds = [int(x) for x in rng.integers(0, 10**6, a.episodes)]
            out = pool.map(_roll, [(pa, th, s) for th in thetas for s in seeds])
            score = np.array([np.mean([o[0] for o in out[i * a.episodes:(i + 1) * a.episodes]]) for i in range(a.pop)])
            elite = thetas[np.argsort(score)[-n_elite:]]
            mu, sigma = elite.mean(0), elite.std(0) + 0.05
            history.append(dict(generation=g, best=float(score.max()), mean=float(score.mean())))
            print(f"gen {g}: best {score.max():8.2f} mean {score.mean():8.2f}  ({time.time() - t0:.0f} s)", flush=True)
        seeds = list(range(5000, 5000 + a.eval_episodes))  # held out: never used in training
        rnd = evaluate(pool, pa, None, seeds)
        cem = evaluate(pool, pa, mu, seeds)
    res = dict(env=a.env, eval_episodes=a.eval_episodes, random=dict(mean=rnd[0], std=rnd[1], success=rnd[2]),
               cem=dict(mean=cem[0], std=cem[1], success=cem[2]), train_episodes=a.generations * a.pop * a.episodes,
               seconds=time.time() - t0, history=history)
    print(f"\n{a.env}: held-out return over {a.eval_episodes} episodes")
    print(f"  random actions : {rnd[0]:8.2f} +- {rnd[1]:.2f}   success {rnd[2]:.0%}")
    print(f"  CEM linear     : {cem[0]:8.2f} +- {cem[1]:.2f}   success {cem[2]:.0%}   "
          f"({res['train_episodes']} training episodes, {res['seconds']:.0f} s)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
