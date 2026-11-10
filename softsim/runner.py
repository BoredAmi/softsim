"""Run scenarios: config -> scenario -> scene -> checks -> backend -> results.

:class:`Simulation` is the stateful object used by scripts, the CLI and the
ROS node (step, state, reset, actions, material changes). :func:`run_config`
is the one-call batch entry point.
"""

from __future__ import annotations

import datetime as _dt
import math
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

from softsim.backends import Backend, SimState, get_backend
from softsim.checkpoint import Saved, fingerprint, read_checkpoint, write_checkpoint
from softsim.checks import Finding, check_scene
from softsim.config.loader import apply_overrides, load_yaml, set_dotted
from softsim.config.schema import validate
from softsim.errors import ConfigError, StabilityError
from softsim.io import save_table
from softsim.scenarios import Scenario, get_scenario


@dataclass
class RunResult:
    """Outcome of a run.

    Attributes:
        config: Fully resolved raw config (re-runnable).
        timeseries: One row per output sample (scenario metrics).
        summary: Scenario summary scalars.
        findings: Pre-run check findings.
        metadata: Versions, wall time, backend diagnostics.
        vertices: Optional (samples, n_vertices, 3) array per soft body.
        model_text: Backend-native model (e.g. MJCF) if available.
    """

    config: dict[str, Any]
    timeseries: pd.DataFrame
    summary: dict[str, float]
    findings: list[Finding]
    metadata: dict[str, Any]
    vertices: dict[str, np.ndarray] = field(default_factory=dict)
    model_text: str | None = None

    def save(self, directory: str | Path, fmt: str = "parquet") -> Path:
        """Write everything to ``directory`` and return it."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.yaml").write_text(yaml.safe_dump(self.config, sort_keys=False))
        save_table(self.timeseries, out / "timeseries", fmt)
        report = {
            "summary": {k: float(v) for k, v in self.summary.items()},
            "findings": [{"level": f.level, "code": f.code, "message": f.message} for f in self.findings],
            "metadata": self.metadata,
        }
        (out / "summary.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
        if self.vertices:
            np.savez_compressed(out / "vertices.npz", t=self.timeseries["t"].to_numpy(), **self.vertices)
        if self.model_text:
            (out / "model.xml").write_text(self.model_text)
        return out


def _versions(backend: Backend) -> dict[str, str]:
    import softsim

    out = {"softsim": softsim.__version__, "python": platform.python_version()}
    try:
        from importlib.metadata import version

        out[backend.name] = version(backend.name)
    except Exception:  # backend distribution name may differ; not critical
        pass
    return out


class Simulation:
    """A live scenario simulation.

    Args:
        config: Path to a scenario YAML or an already loaded mapping.
        overrides: Dotted ``key=value`` overrides.
        backend: Backend name overriding the config's ``backend``.
        base_dir: Directory for relative paths when ``config`` is a mapping.
        strict: If True, warnings are promoted to errors.
    """

    def __init__(self, config: str | Path | Mapping[str, Any], overrides: Iterable[str] = (),
                 backend: str | None = None, base_dir: str | Path | None = None, strict: bool = False):
        if isinstance(config, (str, Path)):
            self.base_dir = Path(config).resolve().parent
            raw = load_yaml(config)
        else:
            self.base_dir = Path(base_dir or Path.cwd()).resolve()
            raw = dict(config)
        raw = apply_overrides(raw, overrides)
        if backend is not None:
            raw["backend"] = backend
        self.strict = strict
        self.backend: Backend | None = None
        self._configure(raw)

    # -- setup ---------------------------------------------------------------

    def _configure(self, raw: dict[str, Any]) -> None:
        if "scenario" not in raw:
            raise ConfigError("config has no 'scenario' key")
        scenario_cls = get_scenario(raw["scenario"])
        config = validate(scenario_cls.config_model, raw, where=f"{raw['scenario']} config")
        scenario: Scenario = scenario_cls(config, self.base_dir)
        scene = scenario.build_scene()
        backend_cls = get_backend(config.backend)
        findings = check_scene(scene, scenario.estimates(), locking_free=backend_cls.locking_free,
                               implicit_actuation=backend_cls.implicit_actuation,
                               implicit_prony=backend_cls.implicit_prony)
        fatal = [f for f in findings if f.level == "error" or (self.strict and f.level == "warning")]
        if fatal:
            raise StabilityError("refusing to run:\n" + "\n".join(f"  {f}" for f in fatal))
        backend_cls.check(scene)
        if self.backend is None or not isinstance(self.backend, backend_cls):
            if self.backend is not None:
                self.backend.close()
            self.backend = backend_cls()
        self.backend.build(scene)
        self.raw_config, self.config, self.scenario, self.scene, self.findings = raw, config, scenario, scene, findings

    # -- stepping ------------------------------------------------------------

    @property
    def time(self) -> float:
        return self.backend.time

    def state(self) -> SimState:
        return self.backend.state()

    def metrics(self) -> dict[str, float]:
        return self.scenario.metrics(self.backend.state())

    def step(self, n: int | None = None) -> None:
        """Advance ``n`` physics steps (default: one output interval); raise if diverged."""
        t0 = self.backend.time
        n = n or self.config.simulation.output_every
        self.backend.step(n)
        if not self.backend.is_finite() or self.backend.time < t0:  # MuJoCo resets its state (and clock) on NaN
            raise StabilityError(f"simulation diverged between t = {t0:.6f} s and "
                                 f"{t0 + n * self.config.simulation.timestep:.6f} s (often inverted elements: "
                                 f"too soft for the load, or too large a timestep)")

    def reset(self) -> None:
        self.backend.reset()

    def run(self, duration: float | None = None, record_vertices: bool | None = None,
            progress: Callable[[float], None] | None = None,
            controller: Callable[["Simulation"], Mapping[str, float] | None] | None = None) -> RunResult:
        """Run from the current state for ``duration`` (default: config duration).

        ``controller(sim)`` (closed-loop control) is called before every output interval with this
        simulation (``sim.time``, ``sim.metrics()``, ``sim.state()``) and returns actuator values to
        hold from then on, ``{name: value}`` in SI (NaN: back to the schedule), or None.

        With ``output.checkpoint`` the run saves itself there every ``output.checkpoint_every`` of wall-clock time and,
        if the file exists, goes on from it (:mod:`softsim.checkpoint`). A longer ``duration`` extends a finished run. A
        ``controller``'s own memory is not saved: only what it set."""
        sim = self.config.simulation
        duration = sim.duration if duration is None else duration
        record_vertices = self.config.output.save_vertices if record_vertices is None else record_vertices
        n_samples = int(round(duration / (sim.output_every * sim.timestep)))
        rows, verts = [], {b.name: [] for b in self.scene.soft_bodies} if record_vertices else {}
        path = self.config.output.checkpoint
        path = None if path is None else self.base_dir / path
        mark = fingerprint(self.config, self.scene, record_vertices) if path else ""
        first, spent = 0, 0.0
        if path and path.exists():
            saved = read_checkpoint(path, mark)
            if saved.sample > n_samples:
                raise ConfigError(f"{path} is at sample {saved.sample}, past this run's {n_samples}")
            self.backend.restore(saved.state)
            rows, spent, first = saved.rows, saved.wall_time, saved.sample + 1
            verts = {k: list(v) for k, v in saved.vertices.items()} if record_vertices else {}
        start = last_save = time.perf_counter()
        for i in range(first, n_samples + 1):
            if i:
                self.step()
            if controller is not None and i < n_samples:
                self.apply_controls(controller(self) or {})
            state = self.backend.state()
            rows.append(self.scenario.metrics(state))
            for name in verts:
                verts[name].append(state.soft[name].vertices.astype(np.float32))
            if progress is not None:
                progress((i) / max(n_samples, 1))
            now = time.perf_counter()
            if path and (now - last_save >= self.config.output.checkpoint_every or i == n_samples):
                write_checkpoint(path, Saved(mark, i, spent + now - start, rows, {k: np.stack(v) for k, v in verts.items()},
                                             self.backend.checkpoint()))
                last_save = now
        return self.make_result(rows, spent + time.perf_counter() - start, verts)

    def make_result(self, rows: list[dict[str, float]], wall_time: float,
                    vertices: Mapping[str, list[np.ndarray]] | None = None) -> RunResult:
        """Assemble a RunResult from metric rows recorded by any driver (run(), ROS node, ...)."""
        ts = pd.DataFrame(rows)
        sim_time = float(ts["t"].iloc[-1] - ts["t"].iloc[0]) if "t" in ts and len(ts) > 1 else 0.0
        metadata = {
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "backend": self.backend.name,
            "versions": _versions(self.backend),
            "wall_time_s": wall_time,
            "realtime_factor": sim_time / wall_time if wall_time > 0 else float("inf"),
            "mesh": {b.name: {k: float(v) for k, v in b.mesh.quality().items()} for b in self.scene.soft_bodies},
        }
        findings = list(self.findings)
        if hasattr(self.backend, "diagnostics"):
            metadata["diagnostics"] = self.backend.diagnostics()
            if metadata["diagnostics"]:
                findings.append(Finding("warning", "backend-warnings",
                                        f"{self.backend.name} reported {metadata['diagnostics']} during the run "
                                        "(e.g. raise simulation.iterations or lower the timestep)"))
        return RunResult(
            config=self.raw_config,
            timeseries=ts,
            summary=self.scenario.summarize(ts) if len(ts) else {},
            findings=findings,
            metadata=metadata,
            vertices={k: np.stack(v) for k, v in (vertices or {}).items() if len(v)},
            model_text=self.backend.export_model() if self.config.output.save_model else None,
        )

    # -- runtime changes -----------------------------------------------------

    def apply_controls(self, values: Mapping[str, float]) -> None:
        """Hold each actuator ``name`` at ``value`` (SI) from the next step on; NaN returns it to its schedule."""
        for name, value in values.items():
            if math.isnan(value):
                self.backend.release_control(name)
            else:
                self.backend.set_control(name, float(value))

    def set_material(self, body: str, material: str | Mapping[str, Any] | None,
                     overrides: Iterable[str] = ()) -> None:
        """Replace a soft body's material (preset name, file or mapping) and rebuild.

        ``overrides`` (dotted card keys, e.g. ``["elastic.youngs_modulus=40 kPa"]``)
        are applied on top; with ``material=None`` they change the current material.
        The scenario config is updated too, so saved configs stay reproducible.
        """
        key = self.scenario.material_key(body)
        if material is None:
            current = {b.name: b.material for b in self.scene.soft_bodies}
            if body not in current:
                raise ConfigError(f"no soft body {body!r}; soft bodies: {list(current)}")
            new_material = current[body]
        else:
            new_material = self.scenario.load_material(material)
        overrides = list(overrides)
        if overrides:
            new_material = new_material.with_overrides(overrides)
        self.backend.set_material(body, new_material)
        self.scene = self.backend.scene
        self.raw_config = set_dotted(self.raw_config, key, new_material.model_dump(mode="json"))

    def action(self, name: str, **kwargs: Any) -> None:
        """Run a scenario action (e.g. ``drop``); rebuilds and resets the simulation."""
        self._configure(self.scenario.apply_action(name, self.raw_config, **kwargs))

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()


def run_config(config: str | Path | Mapping[str, Any], overrides: Iterable[str] = (),
               backend: str | None = None, out_dir: str | Path | None = None, **kwargs: Any) -> RunResult:
    """Load, check, run and optionally save a scenario in one call."""
    sim = Simulation(config, overrides, backend=backend, **kwargs)
    try:
        result = sim.run()
    finally:
        sim.close()
    if out_dir is not None:
        result.save(out_dir, sim.config.output.format)
    return result
