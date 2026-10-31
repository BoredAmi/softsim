# Moving to a faster machine (GPU box)

What to install, what to copy from the laptop, and which long jobs to run
there. Git carries the code, datasets files and calibration files; it does
**not** carry `data/` or `runs/` (both gitignored).

## 1. Install

```bash
git clone <repo> soft_package && cd soft_package
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,fem,fast,calib,mesh,video]"
# NVIDIA GPU: JAX with CUDA (same version range as the fem extra)
pip install -U "jax[cuda12]>=0.4.30,<0.7"
python -c "import jax; print(jax.devices())"        # should list a CUDA device
```

- **SOFA** (optional): unpack the SOFA v26.06 binary and
  `export SOFA_ROOT=.../SOFA_v26.06.00_Linux` ([backends/sofa.md](backends/sofa.md)).
- **ROS 2 Humble** (optional): [ros2.md](ros2.md).

## 2. Copy from the laptop

```bash
# from the laptop's repository root; NEW = user@gpu-box:path/to/soft_package
rsync -a data/processed/ NEW/data/processed/          # tracked drop recordings (372 KB)
rsync -a runs/calibration/ runs/reference_tracks/ NEW/runs/ --relative   # finished and running fits
```

- **Raw drop videos:** the five `*gram.zip` files from Zenodo
  (doi:10.5281/zenodo.23185808) go into `data/raw/czopek2026/`. They are only
  needed to re-track; the tracks in `data/processed/` are enough for
  simulations and fits.
- **Material-test data:** `softsim fetch datasets/roels2025_elastomers.yaml`
  and `softsim fetch datasets/marechal2021_srmd.yaml`. Both are downloaded and
  checked by checksum.

A calibration resumes from its `runs/calibration/<name>/evaluations.jsonl`:
copied logs are replayed, not re-simulated.

## 3. Check

```bash
python -m pytest -q                       # fast suite, ~6 min on the laptop
python -m pytest -q -m slow               # Ogden in 3D and other slow checks
python tools/bench_fem.py                 # per-part timings of the 3D solver
JAX_PLATFORMS=cpu python tools/bench_fem.py   # the same on the CPU, for comparison
```

On the laptop (8-core CPU, no GPU), a 5 mm 3D step takes about 1.1 s:
about 45% element kernels and 30% PARDISO
([reference_fem.md](reference_fem.md#performance)).

## 4. Long jobs, in order

| Job | Command | Laptop time | What it answers |
| --- | --- | --- | --- |
| Drop-only physical card on the 3D reference | `softsim calibrate calibrations/ecoflex_00_10_drop_fem_prony.yaml --jobs 2` | ~6 h (40 evaluations) | E0, g, τ of Ecoflex 00-10 from 20 and 50 g, free of backend artefacts |
| Joint fit: drop + datasheet | `softsim calibrate calibrations/ecoflex_00_10_drop_fem_joint.yaml --jobs 2` | ~6 h | whether one card fits impacts and the quasi-static stiffness ([materials.md](materials.md#joint-fits-lab-tests-and-dynamic-data-together)) |
| Reference tracks of the fitted card | `python tools/make_reference_tracks.py --calibration calibrations/ecoflex_00_10_drop_fem_joint.yaml --force` | ~20 min | the target for the backend transfers |
| MuJoCo and SOFA transfers | `softsim calibrate calibrations/transfer_ecoflex_00_10_{mujoco,sofa}.yaml --jobs 2` | 5 min / 35 min | each backend's effective card and its error bar |
| All-weights validation | `python tools/validate_drop_bounces.py` | ~10 min | the acceptance table in [experiments.md](experiments.md) |

With `--jobs 2` each case runs in its own process. More jobs don't help a
two-case calibration. On a many-core machine, run two calibrations side by
side instead.
