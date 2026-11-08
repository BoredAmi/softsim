# Three-minute gripper demo

A script for showing the soft gripper live, on `docs/viewer/gripper.html`. Every outcome below is rehearsed on the real
page by `tools/rehearse_gripper_demo.py`, and held grasps are checked out to 20 s, so you can talk over them.

## Before you start

```bash
cd ~/personal_projects/softsim
source .venv/bin/activate
python tools/rehearse_gripper_demo.py docs/viewer/gripper.html    # every beat as scripted, about 3 minutes
xdg-open docs/viewer/gripper.html
```

Leave the page on **Play**, with Box, 2 fingers, friction 0.80 and mass 10 g, which is how it opens. Check that the Speed
readout says more than 1x real time on the machine you will present from. The page needs no network or server.

If the page fails, the desktop version has the same finger and the same sliders:
`python tools/demo_gripper.py --finger softsim/reduced/wang_finger.json`.

## The script

### 0:00 The problem (20 s)

> Soft robots are hard to simulate. An accurate solver takes hours; a fast one is usually wrong. This gripper is built from
> a published silicone finger. My own finite-element solver matches the authors' measured bending to 0.76 degrees. That
> solver is far too slow to touch, so it taught a fast model, and the fast model is what's running here, live, in this browser.

### 0:20 Pick something up (30 s)

Set **Finger pressure** to 30 kPa and press **Grasp and lift**. The fingers close, and the badge goes to **Held**.

> Air pressure bends the fingers. Nothing here is animated: that's physics running in the page. Look at the readout: the
> grip gives friction enough to carry 0.20 newtons, and this box weighs 0.10.

### 0:50 Break it on purpose (30 s)

Drag **Mass** up slowly, about 1 g a second, while pointing at the bar under the readout.

> Friction can carry about 20 grams at this pressure, so it should let go right about there.

The badge goes to **Dropped** at 20 g.

> That's the number you get by hand, from grip times friction.

Press **Reset**, set **Mass** to 20 g and **Finger pressure** to 35 kPa, then press **Grasp and lift**. **Held**.

> More pressure, more grip, and the same box stays up.

### 1:20 Show the truth (45 s)

Open **Truth** and press **Closed**, **Off the table**, then **End**.

> A fast model is only worth something if you know where it's wrong. Left is the finite-element grasp, which took an hour
> to compute. Right is the live model, on the same clock. Both pick the box up almost as soon as the palm moves, and both
> carry it. The grip agrees to 8% while holding. What the fast model can't show is the pad squashing around the box, which
> you see on the left. I also ran the 30 gram box on the slow solver: the fingertips slide up the box and it stays on the
> table, which is what the fast model said. That's one check each side of the limit.

Press **Play** at 1/4 speed if you have the time.

### 2:05 What it can hold (25 s)

Open **Grasp map**, and select the cell for 15 g at 30 kPa.

> That's 504 grasps from the fast model, which you could never run on the slow solver. Pick a cell and it loads straight
> into the live demo.

It opens in Play and picks the box up.

### 2:30 What I'd tell you before you trust it (30 s)

> None of these grasps is checked against hardware, only against my solver. The solver is what's checked
> against real measurements. Two finite-element grasps are in, on a coarse mesh, one held and one dropped,
> so the rest of the map is the fast model's own. It also taught me to be careful what "held" means: my first map stopped two seconds after the
> lift, and 48 of its 154 held grasps turned out to fall within 20 seconds, so now every grasp has to last 20. What I'd
> stand behind is the chain: measurements, solver, fast model, with the error at each link written down.

## Stay away from

- **Quoting a payload limit as a property of the gripper.** The 20-26 g is this fast model's, and the real four-finger gripper
  held far more.
- **Calling the map checked.** Two finite-element grasps check one boundary, the box at 25 kPa. The cylinder and bottle
  are unchecked. If someone asks about the earlier no-slip runs, see [the gripper notes](gripper.md#against-the-finite-element-grasp).
- **Cylinder or bottle at 30 kPa or above.** These grip well and then fall within 20 s. If you want a second object, use the
  cylinder at 10 g, 25 kPa, which stays up for the full 20 s.
- **Three and four fingers** as evidence. They are fun to show, but they are extrapolations.
- **Waiting on a held box.** It sinks about 0.4 mm/s. That is MuJoCo's soft friction creeping, not a real slip.

## Likely questions

**Why not use SOFA or MuJoCo's soft bodies?** On this finger, both were 8° or more off the measurements, against under 1°
for my solver. Full MuJoCo flex also ran at about 1/370 of real time.

**How good is the fast finger?** 5-7% on a tip load left out of the fit, and 9-27% on a load spread along the pads, which is
its known weakness.

**Why is the grip so small?** The finger is very soft: 0.2 N moves its tip about 90 mm. This is a gripper for light things.

**Does it work on a real gripper?** Not shown, and I know it's weaker than the real one. The authors measured 0.96 N of
tip force at 35 kPa, which they never modelled. I ran it blind on my solver and got 0.165 N, about 5.5 times too weak,
from the same finger that matches their bending to 0.76 degrees. Releasing the width and stiffening the rubber closes about
half of it, and no setting matches both. Where the tip touched their gauge isn't recorded. So the bending is checked and
the force isn't, and the gripper's limits inherit that.
