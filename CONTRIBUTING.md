# Contributing

Thanks for helping. softsim's rule is **validation first**: a feature comes with a capability flag (and a
refusal where it is unsupported), a test against an analytic solution or the reference, and, where possible,
a comparison with published data.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,fem]"
.venv/bin/python -m pytest -m "not slow"     # the fast suite; drop -m to run everything
.venv/bin/python -m ruff check .
```

SOFA tests skip unless `SOFA_ROOT` points at a SOFA v26.06 install. Mark a test `@pytest.mark.slow` if it takes
more than about 30 s.

## What goes where

- A new material law or preset: [docs/adding_a_material.md](docs/adding_a_material.md)
- A new shape: [docs/adding_a_body.md](docs/adding_a_body.md)
- A new scenario: [docs/adding_a_scenario.md](docs/adding_a_scenario.md)
- A new physics engine: [docs/adding_a_backend.md](docs/adding_a_backend.md)
- A new dataset: [docs/adding_a_dataset.md](docs/adding_a_dataset.md)
- A new validated case: a YAML entry in `benchmarks/` (see `softsim/bench.py`), so it shows in
  [docs/benchmarks.md](docs/benchmarks.md)

## Style

- Line length 120; `ruff check .` must pass.
- SI units inside the code; units as strings in YAML (`"17 mm"`, `"55 kPa"`).
- Docstrings say what a function computes and against what it was checked.
- Results in the docs give the command that reproduces them.

## Data

Public data only, with DOI and licence. Data that may not be redistributed is downloaded by `softsim fetch`,
never committed.
