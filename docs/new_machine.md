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
python -m pytest -q                       # the whole suite (slow checks included): 278 tests, 17 min on the RTX 3060 box
python -m pytest -q -m "not slow"         # without the slow checks (Ogden in 3D, the rotating mount's)
python tools/bench_fem.py                 # per-part timings of the 3D solver
JAX_PLATFORMS=cpu python tools/bench_fem.py   # the same on the CPU, for comparison
```

On the laptop (8-core CPU, no GPU), a 5 mm 3D step takes about 1.1 s:
about 45% element kernels and 30% PARDISO
([reference_fem.md](reference_fem.md#performance)).

**Measured on an RTX 3060 (12 GB, 8 cores, JAX 0.6.2):** the GPU's float64 rate is
only ~160 GFLOP/s (the CPU's 90; its float32 is 11 TFLOP/s), and PARDISO stays
on the CPU, so the GPU helps only the element tangents. `tools/bench_fem.py`,
per Newton iteration, GPU / CPU:

| Mesh | Tangents | Tangents + assembly | Solve |
| --- | --- | --- | --- |
| 8 mm (1 215 dofs) | 1.3 / 4.1 ms | 4.8 / 5.6 ms | 2.5 / 1.6 ms |
| 5 mm (4 788 dofs) | 5.4 / 17.4 ms | 15.3 / 19.8 ms | 12.8 / 12.7 ms |
| 3 mm (24 609 dofs) | 29.9 / 130.5 ms | 76.7 / 183.5 ms | 195 / 228 ms |

A real calibration evaluation (20 g and 50 g, 5 mm, 11 ms) takes 58–69 s on the
GPU and 73–83 s on the CPU, against ~8 min on the laptop. The two cases of
`--jobs 2` share the GPU. The bench's own drop line (about 3 s per 0.1 ms step) is
slower than a real case because its short drop takes ~28 Newton iterations per
step; ignore it.

- Use the GPU anyway (a 15–20% gain), but expect the big win from the
  8 cores and PARDISO, not CUDA.
- Calibration logs replay: replaying a laptop evaluation here reproduces its numbers
  to 5e-8, so a run can continue across machines. Track files are part of the log's
  key (a digest), so a log fitted to other tracks is ignored.

## 4. Long jobs, in order

| Job | Command | Laptop time | What it answers |
| --- | --- | --- | --- |
| Drop-only physical card on the 3D reference | `softsim calibrate calibrations/ecoflex_00_10_drop_fem_prony.yaml --jobs 2` | ~6 h (40 evaluations) | E0, g, τ of Ecoflex 00-10 from 20 and 50 g, free of backend artefacts. **Done on the laptop** (147 kPa, 0.70, 2.24 ms; docs/experiments.md) |
| Joint fit: drop + datasheet | `softsim calibrate calibrations/ecoflex_00_10_drop_fem_joint.yaml --jobs 2` | ~6 h (40 evaluations; **~50 min here**, with other jobs running, plus ~25 min for the plots of all five weights) | whether one card fits impacts and the quasi-static stiffness ([materials.md](materials.md#joint-fits-lab-tests-and-dynamic-data-together)). **Done**: 80 evaluations, 190 kPa / 0.60 / 0.58 ms ([experiments.md](experiments.md#joint-fit-the-drop-and-the-datasheet-gpu-box-2026-11-02)) |
| Reference tracks of the fitted card | `python tools/make_reference_tracks.py --calibration calibrations/ecoflex_00_10_drop_fem_joint.yaml --force` | ~20 min | the target for the backend transfers |
| MuJoCo and SOFA transfers | `softsim calibrate calibrations/transfer_ecoflex_00_10_{mujoco,sofa}.yaml --jobs 2` | 5 min / 35 min | each backend's effective card and its error bar |
| All-weights validation | `python tools/validate_drop_bounces.py` | ~10 min | the acceptance table in [experiments.md](experiments.md) |

With `--jobs 2` each case runs in its own process. More jobs don't help a
two-case calibration. On a many-core machine, run two calibrations side by
side instead.
