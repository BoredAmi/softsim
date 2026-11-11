"""The commands the README, the docs' landing page and the tutorials tell a reader to type are real: each
``softsim`` line parses, the files it names exist, and a scenario config with its ``--set`` overrides passes the
schema (no simulation is run). The Python snippets compile; the first tutorial's snippet runs."""

import re
import shlex
from pathlib import Path

import pytest

from softsim.cli import build_parser
from softsim.config.loader import apply_overrides, load_yaml
from softsim.config.schema import validate
from softsim.scenarios import get_scenario

ROOT = Path(__file__).parents[1]
PAGES = [ROOT / "README.md", ROOT / "docs" / "index.md", *sorted((ROOT / "docs" / "tutorials").glob("*.md"))]
BLOCK = re.compile(r"```(\w+)\n(.*?)```", re.S)
# commands whose first positional argument is a file in the repository
FILE_ARG = {"run", "check", "params", "export", "view", "sweep", "sensitivity", "calibrate", "transfer", "fit",
            "fetch", "fit-material", "track", "optimize", "inverse"}
SCENARIO = {"run", "check", "params", "export", "view", "sensitivity"}


def _blocks(lang: str) -> list[tuple[str, str]]:
    out = []
    for page in PAGES:
        for found, body in BLOCK.findall(page.read_text()):
            if found == lang:
                out.append((page.name, body))
    return out


def _commands() -> list[tuple[str, list[str]]]:
    cmds = []
    for page, body in _blocks("bash"):
        for line in body.replace("\\\n", " ").splitlines():
            tokens = shlex.split(line, comments=True)
            if "|" in tokens:  # a pipe into grep etc.: the softsim part only
                tokens = tokens[:tokens.index("|")]
            if tokens and tokens[0] == "softsim":
                cmds.append((page, tokens[1:]))
    return cmds


COMMANDS = _commands()


def test_the_pages_have_commands():
    assert len(COMMANDS) > 25 and {c[1][0] for c in COMMANDS} >= {"run", "check", "params", "material", "bench"}


@pytest.mark.parametrize("page, argv", COMMANDS, ids=[f"{p}:{' '.join(a)[:60]}" for p, a in COMMANDS])
def test_command_parses_and_its_config_validates(page, argv):
    args = build_parser().parse_args(argv)
    if args.command not in FILE_ARG:
        return
    target = getattr(args, "config", None) or getattr(args, "sweep", None) or getattr(args, "fit", None) \
        or next((v for v in vars(args).values() if isinstance(v, str) and v.endswith(".yaml")), None)
    assert target, argv
    path = ROOT / target
    assert path.is_file(), f"{page}: {target} does not exist"
    if args.command in SCENARIO:
        raw = apply_overrides(load_yaml(path), getattr(args, "overrides", []) or [])
        if getattr(args, "backend", None):
            raw["backend"] = args.backend
        validate(get_scenario(raw.get("scenario", "")).config_model, raw, where=f"{page}: {' '.join(argv)}")


@pytest.mark.parametrize("page, code", _blocks("python"))
def test_python_snippets_compile(page, code):
    compile(code, page, "exec")


@pytest.mark.slow
def test_first_tutorial_python_runs(monkeypatch):
    pytest.importorskip("mujoco")
    monkeypatch.chdir(ROOT)
    code = next(c for p, c in _blocks("python") if p == "first_drop.md")
    exec(compile(code.replace("simulation.duration=50ms", "simulation.duration=5ms"), "first_drop.md", "exec"), {})
