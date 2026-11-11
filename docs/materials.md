# Materials: cards, parameters and what a card does

A softsim material is a **card**: one YAML file (or mapping) that holds
everything about a soft material that a simulation needs, plus where the
numbers come from and how far they can be trusted. Every number on a card can
be changed the same way as any other config value.

## A card

```yaml
name: my_silicone
description: What it is.
source: Where every number comes from. Say "uncalibrated" or "fitted to X" honestly.
kind: physical                # physical | effective (see below)
density: 1070 kg/m^3
elastic:
  model: yeoh                 # linear | stvk | neo_hookean | stable_neo_hookean | mooney_rivlin | yeoh
  c10: 12 kPa                 #   | ogden | gent | arruda_boyce | magneto_active
  c20: 0.8 kPa
  poisson_ratio: 0.49         # or bulk_modulus: 1.2 MPa
damping:                      # optional
  model: prony                # prony (a material model) | rayleigh | kelvin_voigt (numerical damping)
  g: [0.5, 0.2]
  tau: [2 ms, 50 ms]
friction: {sliding: 1.0}
adhesion: {strength: 14 kPa, range: 0.3 mm}   # optional tack (cohesive law)
validity:                     # where the data behind the card was taken
  strain: [-0.5, 3.0]         # nominal uniaxial strain range
  strain_rate: [0 1/s, 300 1/s]
  temperature: 23 degC
uncertainty:                  # one standard deviation (normal, lognormal) or half-width (uniform)
  elastic.c10: {rel: 0.08}
  damping.g[0]: {abs: 0.05, dist: uniform}
references:
  - {doi: 10.1089/soro.2019.0115, note: uniaxial tension data}
```

**Elastic laws.** All laws accept unit strings ("12 kPa"). Their parameters:

| model | parameters | notes |
| --- | --- | --- |
| `linear`, `stvk`, `neo_hookean`, `stable_neo_hookean` | any two of `youngs_modulus`, `poisson_ratio`, `shear_modulus`, `bulk_modulus` | stored as E and ν |
| `mooney_rivlin` | `c10`, `c01` | |
| `yeoh` | `c10`, `c20`, `c30` | often the best simple fit for silicones |
| `ogden` | `mu: [...]`, `alpha: [...]` | N terms |
| `gent` | `mu`, `jm` | stiffens without bound as I1 − 3 → J_m |
| `arruda_boyce` | `mu`, `lambda_m` | five-term series (Abaqus form); locking stretch λ_m |
| `magneto_active` | `matrix` (any hyperelastic law above), `susceptibility` or `particle_volume_fraction`, `along`, `normal`, `demagnetization` | iron-filled (MRE): soft-magnetic in the scene's `magnetic_field`; reference backend only ([magnetic.md](magnetic.md)) |

The laws with a bulk modulus take `bulk_modulus` or `poisson_ratio`.
`softsim material show <card>` prints the small-strain equivalents (E0, μ0,
κ, ν0), wave speeds and explicit time-step estimates of any card.

**Rate dependence.** `prony` is a material model: generalised Maxwell
relaxation of the shear (isochoric) part, G(t) = G0 (1 − Σ g_i (1 − e^(−t/τ_i))).
The elastic constants are the instantaneous ones, so the long-term shear
modulus is G0 (1 − Σ g). `rayleigh` and `kelvin_voigt` are numerical damping:
their loss grows without bound with frequency. They help stabilise or match
one experiment, but they are not a description of a silicone.

Prony runs on every solver with the same update and energies:

- the reference FEM and the homogeneous tests;
- MuJoCo and SOFA. There the backend's own elasticity carries the long-term
  law, and the decaying Maxwell overstresses are added as vertex forces each
  step (`softsim/materials/viscoelastic.py`). These forces are explicit, so
  `softsim check` verifies that the time step resolves them.

Prony needs a law with an isochoric/volumetric split, so not `linear` or
`stvk`.

**Physical and effective cards.** `kind: physical` means that the numbers
describe the material and should hold on any solver that represents the law
well. `kind: effective` means the numbers reproduce data only with one
backend, mesh and contact model. Calibrating a backend against an experiment
produces such cards (the `*_drop_fit*` presets). An effective card must name
`backend` and `mesh_size` in its `provenance`.

## The library

`softsim materials` lists every card.

- **Preferred physical cards:** the `*_roels2025` cards for 7 silicones
  (Ecoflex 00-20/00-30/00-50, Dragon Skin 20/30, Mold Max 14NV,
  Mold Star 30). They are fitted to tension and compression specimens together
  (Roels et al. 2025, CC-BY-4.0), with a measured slow Prony series (1 s to
  3 h). Ecoflex 00-30 and 00-50 also carry measured **fast** relaxation for
  impacts (see below).
- **Older physical cards:** the `*_marechal2021` cards for 18 silicones
  (Marechal et al. 2021), from tension only. Their moduli are probably ~3×
  too low, because the strain was measured at the crosshead.
- All are listed with their fit quality and caveats in
  [material_library.md](material_library.md).
- **Datasheet estimates:** `ecoflex_00_10`, `ecoflex_00_30`, `dragon_skin_10`.
- **Effective drop-test fits:** `ecoflex_00_10_drop_fit*`.

To fit your own data, write a `kind: material_tests` dataset file and run the
commands below. See `datasets/roels2025_elastomers.yaml` for several test
kinds and specimens per material, or `datasets/marechal2021_srmd.yaml` for
one curve.

```bash
softsim fetch datasets/marechal2021_srmd.yaml                     # downloads + checks the archive
softsim fit-material datasets/marechal2021_srmd.yaml ecoflex_00_30 --strain-max 1.0
# every law: cards with uncertainty, report.yaml, fit.png in runs/material_fits/...
softsim material project ecoflex_00_30_marechal2021 --to neo_hookean --strain-range=-0.3,0.6 --out eco30_nh.yaml
```

`fit-material` fits neo_hookean, mooney_rivlin, yeoh, ogden1, ogden2, gent
and arruda_boyce. It ranks them by AIC, reports parameter uncertainties, and
flags laws that lose stability (stress decreasing with stretch) in uniaxial,
equibiaxial or planar loading within the fitted range.

`project` fits another law to a card's own response, for backends that lack
the card's law. MuJoCo has only stable Neo-Hookean, and its refusal message
points to this command.

### Fast rate dependence (impacts)

Tension tests run at about 0.4/s, while an impact loads the material within
milliseconds. Silicones are much stiffer and lossier there. Ecoflex 00-30, for
example, is 1.65× stiffer at 100 Hz and 2.2× at 1 kHz than in its tension
test. A card without fast-rate data is therefore too soft and too bouncy in
impacts.

If you have a frequency-domain fit (DMA, rheometer, guided waves) as a
fractional Kelvin–Voigt model μ0 (1 + (iωτ)ⁿ), convert it to Prony branches
and add them to a card. The response at the test rate stays as fitted:

```python
from softsim.materials import load_material, save_material
from softsim.materials.viscoelastic import prony_from_fractional, with_fast_relaxation

a, taus, rms = prony_from_fractional(n=0.20, tau=2.9e-3, band=(1.0, 2000.0))   # Hz
card = with_fast_relaxation(load_material("my_silicone"), a, taus)
save_material(card, "my_silicone_rate.yaml")
```

- **Band:** branches are placed only inside the band. Keep its top below
  1/(2π·dt) of the backend you run, and its bottom where your measured
  relaxation starts, so nothing is counted twice.
- **Source of the library's values:** Croquette et al. 2026
  (arXiv:2604.27722), who measured Ecoflex 00-30 and 00-50 with guided elastic
  waves and checked them against a rheometer from 1 to 100 Hz.
- **Assumption:** only the shape of their curve relative to the test rate is
  used, applied to the isochoric part at large strain. Their samples were
  about 2× softer than Roels', so their static modulus is not used.

## Changing values

Every value is a dotted key, the same everywhere:

```bash
softsim params scenarios/drop_test/cases/w050.yaml --grep material   # every key, value, unit, allowed range
softsim run scenarios/drop_test/cases/w050.yaml \
    --set specimen.material.elastic.youngs_modulus='40 kPa' \
    --set specimen.material.damping.model=prony \
    --set 'specimen.material.damping.g=[0.6]' --set 'specimen.material.damping.tau=[2 ms]'
softsim material show ecoflex_00_10 --set elastic.model=gent --set elastic.mu='30 kPa' --set elastic.jm=20
```

- **Switching a law:** setting `elastic.model` (or `damping.model`) keeps
  only the parameters the new law understands. Going from an E/ν law to one
  with a bulk modulus carries ν over.
- **List entries** are indexed: `elastic.mu[1]`, `damping.tau[0]`.
- **Sweeps:** the same keys work in sweep grids (`softsim sweep`).
- **Which values matter:** `softsim sensitivity` runs a Morris screening over
  value ranges and reports each parameter's influence (μ*) and
  non-linearity/interaction (σ) on any summary metric:
  ```bash
  softsim sensitivity scenarios/drop_test/cases/w050.yaml --metric peak_indentation --jobs 8 \
      --param "specimen.material.elastic.youngs_modulus=30 kPa:150 kPa" \
      --param "specimen.material.damping.beta=0.1 ms:3 ms" --param "drop.height=0.2:0.4"
  ```
- **Uncertainty propagation:** `softsim sweep sweep.yaml --sample 32 --seed 0`
  (or a `sample:` block in the sweep file) crosses the grid with 32
  variants of the material, drawn from its card's `uncertainty`. The drawn
  values become `sample.<key>` columns of `results.parquet`.
- **Python:**
  ```python
  from softsim.materials import load_material, sample
  m = load_material("ecoflex_00_10", overrides=["elastic.youngs_modulus=40 kPa"])
  m2 = m.with_overrides(["elastic.model=yeoh", "elastic.c10=12 kPa"])
  variants = sample(card_with_uncertainty, 64, seed=0)  # draws from `uncertainty`
  ```
- **At run time:**
  - `Simulation.set_material(body, material, overrides)` rebuilds the model;
  - `set_material(body, None, overrides)` changes the current material;
  - ROS: the `set_material` service takes the same `overrides` list.

## What a card does: homogeneous tests

`softsim.materials.homogeneous` computes the exact response of a card in the
standard laboratory tests, without a mesh or a backend. It needs the `fem`
extra (JAX). The energies and the Prony update are those of the reference FEM.

```python
from softsim.materials.homogeneous import curve, relaxation, constant_rate, dma
curve(m, "uniaxial", [0.5, 1.5, 2.0])        # also equibiaxial, planar, simple_shear, volumetric, confined
curve(m, "uniaxial", [1.5], "instantaneous") # infinitely fast; default "relaxed" = equilibrium
relaxation(m, strain=0.05)                   # ramp-and-hold
constant_rate(m, "uniaxial", 2.0, rate=100)  # strain rate in 1/s
dma(m, [1, 10, 100])                         # storage and loss modulus vs frequency
```

```bash
softsim material plot ecoflex_00_10 my_silicone.yaml --tests uniaxial,simple_shear,relaxation,dma --out cards.png
```

These are checked against closed forms for every law (`tests/test_homogeneous.py`):

- uniaxial, biaxial and planar stresses;
- small-strain moduli;
- Prony relaxation;
- DMA against the analytic G*(ω).

## From a physical card to a backend (transfer)

A physical card holds on any solver that represents its law well. The
tetrahedral backends don't fully do so: linear elements lock, and MuJoCo's
contact loses energy. To find the parameters with which a backend reproduces
a physical card in a given scenario:

1. **Run the card on the locking-free reference** with the scenario's
   conditions, and save the runs as tracks. `tools/make_reference_tracks.py`
   does this for the drop test with each recording's measured impact
   (`softsim.experiments.reference_tracks`).
2. **Calibrate the backend against those tracks.** This is an ordinary
   `softsim calibrate` (e.g. `calibrations/transfer_ecoflex_00_10_mujoco.yaml`).
   The result is the backend's `effective` card, and the remaining NRMSE is
   its error bar against the physical card.

For Ecoflex 00-10 in the drop test, the physical values are 143 kPa, 0.69
and 2 ms. With a 5 mm mesh:

- MuJoCo reproduces the card within 3–10% using E0 = 66 kPa, g = 0.94,
  τ = 5.5 ms;
- SOFA within 3–7% using 79.6 kPa, 0.77 and 2.7 ms
([experiments.md](experiments.md#a-physical-card-on-the-3d-reference-fem-backend)).

A transferred card is only valid for loads like the ones it was transferred
on. The MuJoCo card above (fitted on 20 and 50 g) relaxes 94% within a few
milliseconds. Under 500 g it crushes the specimen until MuJoCo's tetrahedra
invert, and the run stops with "simulation diverged".

## Joint fits: lab tests and dynamic data together

Lab tests fix the quasi-static stiffness. An impact sees the instantaneous
one and how fast it relaxes. Neither alone pins down a viscoelastic card:

- tension data can't tell how much of the stiffness relaxes within
  milliseconds;
- the drop data can't tell what the material does a second later.

A calibration file can therefore include material-test terms next to its
simulated cases (`softsim.experiments.calibrate`). They are evaluated on the
card's homogeneous-test curves, take milliseconds, and add to the same cost:

```yaml
cases: {w020: 1.0, w050: 1.0}
material_tests:
  # specimens from a dataset, one term per test kind and strain window
  - {dataset: ../datasets/roels2025_elastomers.yaml, material: ecoflex_00_30, kind: tension, strain: [0, 1.0]}
  # relaxation: the normalised hold, so it tests rate dependence only
  - {dataset: ../datasets/roels2025_elastomers.yaml, material: ecoflex_00_30, kind: relaxation}
  # or inline points: a datasheet's "100% modulus" is often all there is
  - {name: datasheet_100pct, points: {strain: [1.0], stress: [55 kPa]}, weight: 0.5}
```

A term's cost is its RMSE relative to the largest stress, averaged over
specimens. The term is compared with the card's **fully relaxed** curve,
unless it has a test rate: the dataset's `strain_rate`, or `rate:` in the
entry. With a rate, a Prony card is loaded at that rate.

A `relaxation` term ramps the card to each recording's hold strain at
`rate` (default 0.4/s) and holds it. Both curves are divided by their value
at the start of the recorded hold, which begins `hold_offset` (default
1.3 s) after the ramp, as in Roels et al. 2025. Ecoflex 00-30's library card
matches its 3-hour hold within 0.35%; without damping it is off by 1.4%.

**Example: Ecoflex 00-10.** The drop-only fit on the 3D reference
(E0 ≈ 150 kPa, g ≈ 0.7) relaxes to E∞ ≈ 44 kPa. The datasheet's 100% modulus
(55 kPa) implies about 94 kPa. Every point that drop-only fit has evaluated
misses the datasheet by 40–60%. `calibrations/ecoflex_00_10_drop_fem_joint.yaml`
fits both together. After 80 evaluations: E0 = 190 kPa, g = 0.60, τ = 0.58 ms,
E∞ = 76 kPa, the datasheet term 20% off (from 40–60%), the 20 g first contact
14% and the 50 g one 3%
([experiments.md](experiments.md#joint-fit-the-drop-and-the-datasheet-gpu-box-2026-11-02)).
