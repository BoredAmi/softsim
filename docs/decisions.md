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

**Decision:** Option A chosen by recommendation. Status 2026-11-03: the paper is in data/raw/ and the table is downloaded (`data/raw/elgeneidy2017/data_open_access.xlsx`, md5 45e17500b9c3afbe926262670a08a7ac checked against figshare; gitignored, CC BY-NC 4.0). The paper says the finger is moulded from Ecoflex-50 (SmoothOn), with a paper strain-limiting layer and a flex sensor, but gives NO dimensions: they are in Elgeneidy & Lohse, ASME IDETC 2016 (the Loughborough record is under permanent embargo). The table is a 100 Hz time series: P [psi], flex sensor, tilt phi [deg] (-45, 0, 45) and absolute bending angle theta [deg]; supply 8, 10, 12 psi; internal pressure up to about 7 psi (48 kPa), above the 8 kPa where the thin walls of our PneuNet already balloon. BLOCKED again on the geometry: needs the dimensions of that finger (a copy of the 2016 paper, or the user's own measurement of a finger).

**Update 2026-11-03: Option C chosen** (the 2016 paper costs money). Candidate: Wang, Zhang, Cai, Tong, Wei, *Design, Experimental Validation, and Pressure-Control Simulation of a Modular Pneumatic Soft-Finger Gripper*, Actuators 15(9), 488, doi 10.3390/act15090488, **CC BY 4.0** (Crossref); PDF in `data/raw/actuators-15-00488.pdf`. It has: measured chord angle at 5, 10, ..., 40 kPa (Table 5: 5.00, 10.30, 16.90, 23.20, 30.10, 34.40, 36.80, 35.00 deg; one finger, one ascending sequence, no uncertainty stated); Shore A20 silicone (Posilicone) with a third-order Yeoh fit from the same batch (Table 2: C10 0.044163 MPa, C20 5.3071e-4, C30 1.6251e-5, D1 0.226434 /MPa; monotonic uniaxial only); nine chambers, a = 89 mm, h = 10 mm, wall t = 2 mm, chamber width 20 mm, finger width 24 mm, constraint layer c = 3 mm with teeth (3 mm wide, 1 mm high, 2 mm spacing), tip reserve 10 mm; the paper's Abaqus result (Table 4) as a second reference. Chord angle = angle between the initial axis and the root-to-tip line (not the tip rotation our `bend_angle` reports; a chord metric is needed). The finger hangs vertically in the photos.
Open: the total height, the manifold height and the chamber/wall pitch are not stated (a = 89 mm with nine chambers does not match l = 5 mm and t = 2 mm), and Figure 2 is a schematic. The supplement (https://www.mdpi.com/article/10.3390/act15090488/s1) holds the raw tensile data and the Abaqus input files, which fix the geometry. MDPI returns 403 to my fetcher, so the user downloads it. Until then the finger is not built: guessed dimensions would make the comparison meaningless.

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

**Status 2026-11-03, late: finger built, validation NOT yet passing.** Built from the supplement's mesh: shape `ribbed_finger` (`softsim/geometry/soft.py`; solid 19,568 mm^3 and cavity 12,400 mm^3 equal the paper's mesh, tests/test_ribbed_finger.py), card `posilicone_a20_wang2026`, scenario `scenarios/actuation/ribbed_finger.yaml`, metric `chord_angle`, data `datasets/wang2026_finger.yaml`, tool `tools/compare_wang_finger.py`. First reference runs (fem, 2 mm, teeth omitted): the dynamic ramp failed to converge at ~17 kPa after 24 min; a static probe (steps of 2 kPa) gave chord 10.0 deg at 2 kPa, 18.5 at 4, 26.0 at 6, 32.4 at 8, 37.3 at 10, 41.5 at 14 kPa, then a limit point near 16 kPa (volume doubles). The paper measured 5.0 deg at 5 kPa and 10.3 at 10 kPa and its Abaqus gives 5.12 and 10.92: our finger bends about four times too much. Not explained yet. Candidates: teeth omitted (the floor is 2.6 mm thick with them, not 2), the paper's Root_Face and planar constraint, the pressure's direction/sign, or the solver. The decisive next step: load the paper's OWN mesh (the .inp corner nodes, CC BY) as a mesh file in softsim and run it with the same Yeoh card, which separates my geometry from the solver; then the same mesh with the paper's constraints. Do not report any agreement with the measurements until that is understood.

**Update 2026-11-04: the 4x was the missing planar constraint.** The paper's .inp holds `*Boundary AllNodesPlanar, 3, 3`: U3 = 0 on EVERY node (plane strain across the 24 mm width), plus `Root_Face, ENCASTRE`. Our finger had a free width. With u_y = 0 on all nodes (y is our width axis; z is the height, constraining it removes all bending), same Yeoh card, 2 mm mesh, no teeth, static, chord angle by the paper's definition:

| p [kPa] | ours | measured | paper FE |
|---|---|---|---|
| 5 | 5.03 | 5.00 | 5.12 |
| 10 | 10.07 | 10.30 | 10.92 |
| 15 | 14.92 | 16.90 | 17.34 |
| 20 | 19.07 | 23.20 | 24.09 |
| 30 | 20.85 | 34.40 | 35.25 |
| 40 | 4.51 | 35.00 | 35.38 |

Agreement to 10 kPa (within 2%), then ours falls behind and peaks at 21.7 deg (27.5 kPa) before un-bending. Steps of 2.5 kPa give the same curve, so this is the equilibrium path, not a branch jump. Open: the toothless floor (2 mm instead of an effective 2.6 mm) is the leading suspect for the high-pressure gap (the floor is the strain-limiting layer); being run with teeth at 1.5 mm. The planar constraint is not yet an option of the scenario (the probe sets `Problem3D.fixed_dofs` by hand: scratch script, not in the repo). Reproduce: build the scene from scenarios/actuation/ribbed_finger.yaml with `body.mesh.size=2mm body.geometry.tooth_height=0 mm`, set `fixed_dofs` to 3*node+1 for all nodes, `Solid3DSolver(dynamic=False)`, raise `pressures[0]` in steps and read `chord_angle`.

**Update 2026-11-04, later: validation passes with teeth.** Same setup but teeth included (`tooth_height=1 mm`, mesh 1.5 mm, planar u_y = 0, static, steps of 5 kPa), chord angle [deg]:

| p [kPa] | ours | measured | paper FE |
|---|---|---|---|
| 5 | 5.04 | 5.00 | 5.12 |
| 10 | 10.73 | 10.30 | 10.92 |
| 15 | 16.99 | 16.90 | 17.34 |
| 20 | 23.55 | 23.20 | 24.09 |
| 25 | 29.73 | 30.10 | 30.45 |
| 30 | 34.42 | 34.40 | 35.25 |
| 35 | 36.30 | 36.80 | 37.16 |
| 40 | 34.41 | 35.00 | 35.38 |

All eight points are within 0.5 deg of the measurement, and the peak and the fall-off after 35 kPa are reproduced. The toothless floor was the cause of the earlier gap. Still to do: make the planar constraint a scenario option, run through `tools/compare_wang_finger.py`, mesh convergence (1.5 mm vs 1 mm), a test, and the MuJoCo/SOFA transfers. Reproduce: as in the previous update, with `body.mesh.size=1.5mm body.geometry.tooth_height=1 mm`.

**Update 2026-11-04, end-to-end: the scenario file reproduces the paper.** `body.plane_strain: y` is now a scenario option (reference only; tests/test_plane_strain.py: a plane-strain beam deflects 0.8106 of a free one against 1 - nu^2 = 0.7975, and dropping the held dofs fails the test). The committed scenario, run as is on the reference (`softsim run scenarios/actuation/ribbed_finger.yaml --backend fem --set simulation.timestep=10ms`, 1.5 mm, dynamic 1 s ramp), read by `python tools/compare_wang_finger.py runs/wang/fem_1p5mm --plot` (figure docs/media/wang_finger.png): RMS error against the measurements 0.76 deg (the paper's own FE: 0.56 deg), largest error 1.3 deg (30 kPa); the 5 kPa point is 1.15 deg low, the start of the ramp, where the dynamic run lags the static value (static probe: 5.04). Open: mesh convergence (1 mm), transfers to MuJoCo/SOFA (Yeoh must be projected), docs/actuation.md section.

**Update 2026-11-04, plane strain on MuJoCo and SOFA.** `body.plane_strain` runs on all three backends. MuJoCo: the scene is lowered to a zero prescribed motion of every free vertex along the axis (`SoftBodySpec.lowered`). SOFA: the same lowering was tried first and failed the ratio test (tip deflection with the width held / free: 0.17 at nu = 0.45 and 0.14 at nu = 0.3, expected 1 - nu^2 = 0.80), so it is not locking: `PartialLinearMovementProjectiveConstraint` over every vertex with one held axis made the beam about 7x too stiff (the held axis was exactly 0, the free ones barely moved). SOFA now uses `PartialFixedProjectiveConstraint` with the axis as fixed direction: 0.85 at 3 mm in the short check, test passes at 2 mm. Caution for later: a many-vertex prescribed region holding one axis on SOFA may over-constrain the same way; tests/test_prescribed.py only covers boundary grips. tests/test_plane_strain.py: reference 0.8106 (+1.6% vs 1 - nu^2), MuJoCo and SOFA within 6% (slow-marked: 2 and 12 min).

**Update 2026-11-04, Wang finger on MuJoCo: the transfer does not work at 2 mm, the mesh is the limit.** `calibrations/transfer_wang_finger_mujoco.yaml` (stable Neo-Hookean projection of the Yeoh card, nu 0.45, 2 mm, 0.2 ms; output `runs/transfer/wang_finger_mujoco_2mm`): best E 242 kPa, cost 0.25 (raw card 0.28); cavity volume NRMSE 0.008 but chord angle NRMSE 0.49, flat across E (cost 0.25-0.45 for E 219-468 kPa; E 137 kPa diverged), so it is not a modulus problem. Mesh test at the fitted E, run as the scenario with the backend overrides: RMS error against the measurements 16.4 deg at 2 mm, 8.1 deg at 1.5 mm, reference 0.76 deg (table from `python tools/compare_wang_finger.py runs/wang/mujoco_2mm runs/wang/mujoco_1.5mm runs/wang/fem_1p5mm`). The error halves with 25% finer mesh: the 2 mm walls are about one linear tet thick and bend too stiff, and a modulus cannot compensate because the stiffening depends on the load shape. 1.5 mm on MuJoCo takes ~40 min per ramp. A finer mesh (1 mm) is untested and probably too slow. SOFA transfer not run: it uses linear tets too, so the same limit is expected; it was not tested. (compare_wang_finger.py --plot rewrote docs/media/wang_finger.png with the two MuJoCo curves added.)

**Update 2026-11-04, Wang finger on SOFA at 1.5 mm.** `calibrations/transfer_wang_finger_sofa.yaml` (stable Neo-Hookean card, nu 0.45, 1.5 mm, 2 ms, 6 evaluations, 51 min, ~9 min per ramp; output `runs/transfer/wang_finger_sofa_1p5mm`): the unfitted card (E 268 kPa) is the best point, cost 0.18 (chord angle NRMSE 0.25, cavity volume 0.11); E 293k 0.23, 219k 0.27, 370k 0.32, 468k 0.40, 137k diverged at t = 0.75 s. Chambers_5 check 0.16. So the fit finds nothing to gain from E, the remaining error is mesh/element error again, but with half of MuJoCo-2mm's chord error (0.49) and at 2 ms, so SOFA at 1.5 mm is the usable fast backend for this finger; it is still not as close as the reference (0.76 deg RMS). Not run: SOFA at 1 mm, and the chord-angle curve of the SOFA run against the measurements (only NRMSE against the reference is recorded).
