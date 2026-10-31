# Validation: predictions on data the model never saw

Fitting a model to data and then showing it matches that data says how well
the fit converged, not how good the model is. This page lists only
**predictions**: each result compares a model with data that played no part in
building it.

| Check | Data the model saw | Data it predicts | Result |
| --- | --- | --- | --- |
| [Drop card → independent tension test](#1-a-drop-fitted-card-predicts-an-independent-tension-test) | the 20 and 50 g drop videos | quasi-static tension of Ecoflex 00-10, another lab, optical strain | **3–22%** up to 100% strain |
| [Tension → compression](#2-tension-only-cards-predict-compression) | tension specimens | compression specimens, same lab | **4–17%** |
| [Compression → tension](#3-compression-only-cards-predict-tension) | compression specimens (to 25%) | tension specimens to 100% | **4.5–20%** (Mold Max 32%) |
| [Leave one weight out](#4-leave-one-weight-out-whole-bounce) | one drop recording | the other drop recording | weight trajectory **22–28%**; the specimen's hop **fails** (51–93%) |
| [Labs against each other](#5-how-much-do-labs-disagree) | – | the same product measured by different labs | up to **3×** apart |

Regenerate: `python tools/validate_materials.py` (sections 2, 3 and 5) and
`python tools/leave_one_out.py` (section 4).

## 1. A drop-fitted card predicts an independent tension test

The physical Ecoflex 00-10 card (Neo-Hookean, E0 = 143 kPa, Prony g = 0.69 at
τ = 2 ms) was fitted only to the first contact of the 20 and 50 g drop videos,
on the locking-free 3D reference ([experiments.md](experiments.md)). Its fully
relaxed response is a prediction for a slow tension test.

Shojaeifard and Bacca 2025 ([arXiv:2506.05371](https://arxiv.org/abs/2506.05371),
Table 1) measured Ecoflex 00-10 in quasi-static tension, with optical strain
tracking, three specimens: a one-term Ogden law with E = 34 kPa and α = 2.8.
Their deep-indentation tests agree with it (E = 38 kPa).

Nominal stress of each card over theirs:

| Stretch | 1.25 | 1.5 | 2.0 | 2.5 | 3.0 |
| --- | --- | --- | --- | --- | --- |
| **Physical card from the drop (relaxed)** | 1.22 | 1.13 | **0.97** | 0.84 | 0.74 |
| MuJoCo effective drop fit | 1.50 | 1.38 | 1.14 | 0.96 | 0.81 |
| Datasheet card (`ecoflex_00_10`) | 2.49 | 2.28 | 1.90 | 1.59 | 1.35 |
| Marechal 2021 card (crosshead strain) | 0.85 | 0.72 | 0.62 | 0.58 | 0.56 |

- **The drop-fitted card predicts it within 3–22% up to 100% strain.**
- Above that it falls short because Neo-Hookean does not stiffen, and the
  drops never strain the material that far.
- The datasheet's 100% modulus (55 kPa) is about **2× too stiff** for this
  lab's material. A joint fit of the drop and that datasheet value was
  therefore stopped: it would have pulled the card the wrong way.

## 2. Tension-only cards predict compression

Roels et al. 2025: several tension and compression specimens per material. A
card fitted to the tension specimens only (0–100%, law chosen by BIC as in the
library) predicts the compression specimens (0 to −25%). Error = RMS / the
specimen's largest stress, averaged over specimens:

| Material | Law | Fit (tension) | **Predicted (compression)** | Joint fit, compression |
| --- | --- | --- | --- | --- |
| Dragon Skin 20 | Gent | 4.7% | **13.5%** | 11.0% |
| Dragon Skin 30 | Yeoh | 1.0% | **12.2%** | 4.9% |
| Ecoflex 00-20 | Arruda–Boyce | 10.8% | **7.3%** | 2.0% |
| Ecoflex 00-30 | Yeoh | 4.1% | **5.6%** | 1.4% |
| Ecoflex 00-50 | Yeoh | 3.6% | **5.5%** | 1.6% |
| Mold Max 14NV | Yeoh | 0.8% | **3.9%** | 3.5% |
| Mold Star 30 | Yeoh | 0.7% | **16.7%** | 5.1% |

## 3. Compression-only cards predict tension

The reverse is harder. 25% compression barely constrains a law's higher-order
terms.

| Material | Law | Fit (compression) | **Predicted (tension, to 100%)** | With Neo-Hookean |
| --- | --- | --- | --- | --- |
| Dragon Skin 20 | Arruda–Boyce | 10.9% | **20.6%** | 19.3% |
| Dragon Skin 30 | Arruda–Boyce | 4.5% | **16.0%** | 21.2% |
| Ecoflex 00-20 | Neo-Hookean | 0.8% | **11.7%** | 11.7% |
| Ecoflex 00-30 | Mooney–Rivlin | 1.4% | **4.5%** | 10.0% |
| Ecoflex 00-50 | Neo-Hookean | 1.6% | **7.8%** | 7.8% |
| Mold Max 14NV | Mooney–Rivlin | 3.3% | **32.0%** | 18.9% |
| Mold Star 30 | Arruda–Boyce | 4.7% | **8.5%** | 26.0% |

**This check found a bug, now fixed.** Yeoh fits to compression alone were
marked stable, because stability was checked only over the fitted range (up to
a stretch of 1). In tension they diverged (errors of 10⁸% and more). Fits are
now checked for stability up to a stretch of at least 2, so such laws are
rejected. Every card still records the strain range it was fitted on
(`validity`), and `softsim check` warns when a run leaves it.

## 4. Leave one weight out (whole bounce)

The MuJoCo whole-bounce model (stiffness, damping, tack, cloth support: 7
parameters) is refitted to one recording only, from neutral starting values,
and then predicts the other recording's first 100 ms
(`calibrations/loo_ecoflex_00_10_fit_w0x0.yaml`, `tools/leave_one_out.py`).

Error = NRMSE over the first 100 ms: the mean of the weight's trajectory
(`indentation`) and the specimen's hop (`specimen_lift`).

| Fitted on | Own recording | **Predicting the other** | Fitted on both, same recording |
| --- | --- | --- | --- |
| 20 g | 13% | **50 g: 60%** (weight 28%, hop 93%) | 8% |
| 50 g | 15% | **20 g: 37%** (weight 22%, hop 51%) | 18% |

**The material transfers; the setup model does not.**
- **The material agrees between the folds:** E = 45 vs 53 kPa, β = 0.87 vs
  1.0 ms (the MuJoCo effective values). Each fold also predicts the other
  weight's trajectory nearly as well as it fits its own (22–28% against
  18–20%), and the first-contact peak within 2% (4.39 vs 4.31 mm, fitted on
  20 g) and 23% (2.89 vs 3.74 mm, fitted on 50 g).
- **The cloth support is not identified by one recording.** The folds put its
  mass at 2 g and 77 g, its damping at 1.25 and 0.32 N s/m, and the tack range
  at 1.2 and 0.3 mm. Each fits its own hop (8–10%), but predicts the other's
  poorly (51–93%).
- So the support and tack values in the drop cards describe the two recordings
  together. They aren't properties of the material or the table, and the
  specimen-hop results in [experiments.md](experiments.md) hold for these two
  weights, not as predictions for others.

## 5. How much do labs disagree?

The same product, measured by different labs (stress ratios as in section 1,
against Shojaeifard and Bacca's Ecoflex 00-30: E = 81 kPa, α = 2.4):

| Ecoflex 00-30 card | Stretch 1.25 | 1.5 | 2.0 | 3.0 |
| --- | --- | --- | --- | --- |
| Roels et al. 2025 (preferred library card) | 1.90 | 1.79 | 1.57 | 1.74 |
| Datasheet card | 1.35 | 1.29 | 1.16 | 0.95 |
| Marechal et al. 2021 (crosshead strain) | 0.61 | 0.59 | 0.56 | 0.52 |

Small-strain moduli of Ecoflex 00-30 from five sources: 54 kPa (Marechal),
68 kPa (Croquette et al. 2026, 3μ0), 81 kPa (Shojaeifard and Bacca), about
118 kPa (datasheet 100% modulus, Neo-Hookean) and 166 kPa (Roels).

**Batch, cure and test method spread a nominal silicone over a factor of 3.**
No simulation can be more accurate than that for a part made from an
unmeasured batch. For accurate work, measure your own material (one tension
test, `softsim fit-material`). A library card is a starting point with an
uncertainty of about ±50%, not a fact.
