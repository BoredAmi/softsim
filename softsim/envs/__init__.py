"""Gymnasium environments on :class:`softsim.runner.Simulation` (optional extra ``softsim[rl]``).

``gymnasium`` is imported on first use: ``import softsim.envs`` works without it, and touching ``SoftsimEnv``,
``make_env`` or ``register_envs`` then raises an ImportError saying how to install it.
"""

from __future__ import annotations

from typing import Any

__all__ = ["SoftsimEnv", "ReducedGripperLiftEnv", "make_env", "register_envs", "ENV_IDS", "GRIPPER_ID"]

ENV_IDS = {
    "softsim/CableBeamReach-v0": "cable_beam_reach.yaml",
    "softsim/CableArmReach-v0": "cable_arm_reach.yaml",
    "softsim/PneuNetBend-v0": "pneunet_bend.yaml",
}
GRIPPER_ID = "softsim/ReducedGripperLift-v0"  # softsim.envs.gripper.ReducedGripperLiftEnv (not a SoftsimEnv config)


def require_gymnasium():
    """Import and return gymnasium, or raise an ImportError that says how to get it."""
    try:
        import gymnasium
    except ImportError as exc:
        raise ImportError("softsim.envs needs the optional package 'gymnasium'; install it with "
                          "`pip install \"softsim[rl]\"` (or `pip install \"gymnasium>=0.29\"`)") from exc
    return gymnasium


def __getattr__(name: str) -> Any:
    if name in ("SoftsimEnv", "make_env", "register_envs"):
        require_gymnasium()
        from softsim.envs import env

        return getattr(env, name)
    if name == "ReducedGripperLiftEnv":
        require_gymnasium()
        from softsim.envs import gripper

        return gripper.ReducedGripperLiftEnv
    raise AttributeError(f"module 'softsim.envs' has no attribute {name!r}")
