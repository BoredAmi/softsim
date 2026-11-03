# Open decisions

Each entry: the question, the options, what I recommend and why, and what I did
meanwhile so nothing waits on it. Answer by editing the **Decision** line (or
tell me in the chat); I will then do what the entry says.

## 1. MuJoCo timing fix: keep or revert

Since the prescribed-motion work, MuJoCo's `step` is split `mj_step1` → explicit
forces → `mj_step2`, and `state()` recomputes kinematics, because after
`mj_step` MuJoCo's positions are one step stale (the forces and the reported
state were a step old). It moves documented MuJoCo numbers by 1.5–3% (nrmse)
and one 0.1 ms step in the time of a peak; the committed MuJoCo fits and
transfers used the old timing.

- **Keep (recommended).** The old timing is a bug (a force computed from last
  step's positions); the new numbers are the correct ones. Cost: the committed
  MuJoCo fits/transfers should be re-run when convenient (hours of MuJoCo time, not measured here); nothing else changes.
- **Revert.** Hand-edit `softsim/backends/mujoco/backend.py` (it shares the file
  with prescribed motion). Keeps old numbers, keeps the bug.

**Decision:** Keep the fix (my recommendation, applied 2026-11-03 under your "follow your recommendations"). Why: the old timing used last step's positions, a bug. Re-run on 2026-11-03 from a frozen copy of the package: the bending, inflation and PneuNet transfers reproduce the committed numbers (bending unchanged; inflation E 23.66 vs 23.7 kPa, errors 0.5/1.2/2.2% as before; PneuNet body E 91.0 kPa, layer 1.76 MPa, cost 2.9%, checks 3.2% and 4.3%, as before), so the timing fix moves the transfers by less than their rounding and the documented tables stand. The Ecoflex 00-10 drop transfer (`transfer_ecoflex_00_10_mujoco.yaml`, 40 evaluations, fresh output folder) reproduces the committed run to every digit (E 64.89 kPa, g 0.678, tau 7.82 ms, cost 0.0593, 8.6% / 3.3%), so the numbers in docs/experiments.md are those of the current code. `ecoflex_00_10_drop.yaml` (fit against the recordings, 50 evaluations) is not re-run.

## 2. Joint drop + datasheet fit: stop or continue

The 80-evaluation joint fit is nearly flat (E0 190 kPa, g 0.60, τ 0.58 ms,
cost 0.108) but not proven converged; at 40 evaluations it clearly was not.

- **Continue** to 160 evaluations (80 more evaluations at 60–70 s each on the GPU box, so 1.5–2 h):
  settles whether the datasheet term and the drop cases are compatible.
- **Stop (recommended).** The cost is flat to the third digit; the remaining
  mismatch (the 20 g first contact 14%) looks structural (the card has one
  Prony term) rather than unconverged, so more evaluations buy little. A better
  use of the time is a second Prony term, which changes the card, so it is your
  call.

**Decision:** Stop (recommendation applied 2026-11-03). Why: cost flat to the third digit; the residual looks structural (one Prony term). A second Prony term changes the card, so it waits for you.

## 3. M7 real data for the pneumatic finger (blocks "compare with a real finger")

The brief asks for a PneuNet against real public data with DOI and licence. What
I found (Zenodo API, repositories, search, 2026-11-02):

| Candidate | DOI | Licence | Has | Problem |
| --- | --- | --- | --- | --- |
| Elgeneidy, Lohse, Jackson (2017), *Experimental data – modelling the bending response of soft pneumatic actuators* | 10.17028/rd.lboro.5509528 | **CC BY-NC 4.0** | bending angle θ [deg] against pressure [psi] and tilt, flex-sensor volts; 1108 rows (training + testing) | geometry and printed material are in the paper (Mechatronics 50, 2017, doi 10.1016/j.mechatronics.2017.10.005, open access) which I cannot download: the publisher and the repository mirrors return 403 or a bot challenge. I will not bypass that. |
| Joshi & Paik (2019), *Multi-DoF force characterization of soft actuators* | 10.5281/zenodo.3627901 | CC-BY-4.0 | pressure, position and force for pulling, tip loading and three-point bending | the actuator is a fabric-constrained film bellows (Robertson et al. 2017), not a PneuNet, and neither geometry nor material is in the record |
| Georgopoulou et al. (2021–22) | 10.5281/zenodo.5841687, .5841739 | CC-BY-4.0 | stress–strain and resistance of printed TPS strips | strain-sensor data; no pressure–angle pairs |
| Omnidirectional soft sleeve (2026) | 10.5281/zenodo.18433594 | CC-BY-4.0 | bending angle and force files (PDF) | a sleeve, not a PneuNet; plots as PDFs |

- **Option A (recommended): Elgeneidy.** Right quantity (pressure → bending
  angle) with a published geometry. Needs you to download the paper (open
  access, <https://doi.org/10.1016/j.mechatronics.2017.10.005>) into
  `data/raw/` so I can read the dimensions and the printing material and
  build the finger. The licence is non-commercial: fine for this research use,
  but the table must not be redistributed in the repository (as with the
  drop-test videos: referenced by DOI, downloaded by the user; the fit file
  points at the local copy). The data is also a free-bending angle from a
  flex-sensor-instrumented printed actuator, so the material is likely a
  printed elastomer rather than silicone: the card will have to be fitted, not
  taken from a datasheet.
- **Option B:** build a finger of your own and measure it (angle against
  pressure on camera); gives full control of geometry and material, costs your
  bench time.
- **Option C:** digitise a figure from an open-access paper that gives a full
  geometry (e.g. a CC-BY MDPI four-chamber design). Weaker: digitised points
  carry reading error, and I would flag them as such.

**Meanwhile:** the machinery is done and tested on a twin experiment (`softsim
fit`, `docs/experiments.md`), and the reference's mesh convergence on the
existing PneuNet is measured (`docs/actuation.md`).

**Decision:** Option A chosen by recommendation, but BLOCKED: the paper is not in data/raw/ yet. Needs you to download https://doi.org/10.1016/j.mechatronics.2017.10.005 there.

## 4. M8 rod backend: which library

M8 wants a rod (Cosserat) backend for slender bodies, cross-checked against FEM.

- **SOFA BeamAdapter** (installed with the v26.06 plugins; no new dependency):
  fits the existing SOFA backend; Euler–Bernoulli-like beams with fixed
  stiffness.
- **PyElastica:** a real Cosserat rod (shear, twist, dynamics), pip-installable;
  a new dependency and a new backend to write.
- **Own implementation:** most control, most work, and a verification burden
  (it would need its own analytic checks).

**Recommendation:** BeamAdapter first, because it is already installed and a
cantilever has exact solutions to check it against; PyElastica only if a
quantity BeamAdapter cannot give (large shear, twist) is needed.

**Decision:** SOFA BeamAdapter first (recommendation applied 2026-11-03). Why: already installed, no new dependency, cantilever has exact solutions to check against.

## 5. Rigid-tendon option for cables

A cable whose tendon stiffness tends to infinity (stroke imposed exactly) is
the one case where an implicit SOFA `CableConstraint` (displacement type) would
beat the explicit force, which is unstable for stiff tendons. It changes the
cable model (today `stiffness` must be finite) and is SOFA-only.

- **Add it** (config: `stiffness: rigid`): lets SOFA run stroke-controlled
  cables at any stiffness; the reference already has a stiff-tendon test to
  compare with. Cost: about a day, SOFA-only.
- **Skip (recommended for now).** No scenario needs it; finite tendon stiffness
  is the physical case.

**Decision:** Skip (recommendation applied 2026-11-03). Why: no scenario needs it; finite tendon stiffness is the physical case.
