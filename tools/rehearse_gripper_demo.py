"""Rehearse the demo in docs/demo_script.md on the real page, through the controls a presenter touches.

    python tools/rehearse_gripper_demo.py docs/viewer/gripper.html

Opens the page in headless Chrome, sets the sliders and clicks the buttons the script names, and fails if any beat ends
differently from what the script promises. It also finds the mass at which a held box slips as the Mass slider rises.
The page runs by hand (``window.softsim.step``), so a beat gives the same answer on any machine and takes seconds.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from check_gripper_page import Page, chrome, opened

FRAME = 16  # steps of 1 ms per drawn frame, about 60 frames/s
SETTLE = 5.0  # s after pressing Grasp and lift: 1.5 s to close and settle, 0.5 s to lift, 3 s to hold or fall
LIFT_AT = 1.5  # s after the press: run_grasp settles this long, and the map ran run_grasp
LONG = 20.0  # s: a held beat must still be held this long after the press, in case the presenter talks over it
BEATS = [("box", 10, 0.8, 30, "Held"), ("box", 10, 0.8, 25, "Held"), ("box", 10, 0.4, 25, "Dropped"),
         ("box", 20, 0.8, 30, "Dropped"), ("box", 20, 0.8, 35, "Held"), ("cylinder", 10, 0.8, 25, "Held")]

CHAPTERS = ("Closed", "Off the table", "End")
MAP_CELL = (3, 4)  # row and column of 15 g, 30 kPa on the box's grid at friction 0.8

SET = """(id, value) => { const e = document.getElementById(id); e.value = value; e.dispatchEvent(new Event('input')); }"""


async def run(p: Page, seconds: float, frame: int = FRAME) -> dict:
    """Advance the page; the readout filters grip, so it has to run frame by frame."""
    n = round(seconds * 1e3 / frame)
    return await p.js(f"(() => {{ let r; for (let i = 0; i < {n}; i++) r = window.softsim.step({frame}); return r; }})()")


async def press(p: Page, shape: str, mass: float, mu: float, pressure: float) -> None:
    await p.js("document.getElementById('reset').click()")
    await p.js(f"document.querySelector('#shape [data-v={shape}]').click()")
    await p.js(f"({SET})('mass', {mass}); ({SET})('mu', {mu}); ({SET})('pressure', {pressure})")
    await p.js("document.getElementById('grasp').click()")


async def rehearse(url: str, port: int) -> None:
    async with opened(url, port) as p:
        wrong = []
        await press(p, "box", 10, 0.8, 30)
        early, late = await run(p, LIFT_AT - 0.05), await run(p, 0.1)
        print(f"palm rise {early['lift'] * 1e3:.1f} mm just before {LIFT_AT} s, {late['lift'] * 1e3:.1f} mm just after")
        if early["lift"] != 0 or late["lift"] <= 0:
            wrong.append(f"the lift does not start at {LIFT_AT} s, so a map cell loaded into Play is a different grasp")
        for shape, mass, mu, pressure, want in BEATS:
            await press(p, shape, mass, mu, pressure)
            r = await run(p, SETTLE)
            late = await run(p, LONG - SETTLE, 200)
            print(f"{shape:9s}{mass:3d} g  friction {mu:.1f}  {pressure:2d} kPa: {r['badge']:8s} {r['height']:5.1f} mm at {SETTLE:.0f} s, "
                  f"{late['badge']:8s} {late['height']:5.1f} mm at {LONG:.0f} s, grip {r['grip']:.3f} N", flush=True)
            if r["badge"] != want or late["badge"] != want:
                wrong.append(f"{shape} {mass} g {pressure} kPa: page says {r['badge']}, then {late['badge']}; script says {want}")
        await press(p, "box", 10, 0.8, 30)
        r = await run(p, SETTLE)
        mass = 10
        while r["badge"] == "Held" and mass < 60:
            mass += 1
            await p.js(f"({SET})('mass', {mass})")
            r = await run(p, 1.0)
        print(f"box held at 30 kPa, Mass slider raised 1 g a second: {r['badge']} at {mass} g")
        if r["badge"] != "Dropped" or abs(mass - 20) > 1:
            wrong.append(f"the Mass slider dropped the box at {mass} g, the script says 20 g")
        await p.js("location.hash='truth'")
        await asyncio.sleep(1)
        chapters = await p.js("[...document.querySelectorAll('#truth-chapters button')].map(b => (b.click(), b.textContent))")
        print("truth chapters:", chapters)
        if [c.split(",")[0] for c in chapters[:3]] != list(CHAPTERS):
            wrong.append(f"the script names the chapters {CHAPTERS}")
        await p.js("location.hash='map'")
        await asyncio.sleep(1)
        await p.js(f"document.querySelector('#map-grid button[data-i=\"{MAP_CELL[0]}\"][data-j=\"{MAP_CELL[1]}\"]').click()")
        r = await run(p, SETTLE)
        title = await p.js("location.hash + ', ' + document.getElementById('mass-o').textContent + ', ' + document.getElementById('pressure-o').textContent")
        print(f"map cell loaded into {title}: {r['badge']}, {r['height']:.1f} mm")
        if r["badge"] != "Held" or title != "#play, 15 g, 30.0 kPa":
            wrong.append(f"the map cell gave {title}, {r['badge']}")
        if p.errors or wrong:
            raise SystemExit("\n".join(wrong + p.errors))
        print(f"all {len(BEATS)} beats as scripted; script errors: none")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("page", type=Path)
    a = ap.parse_args()
    with chrome() as port:
        asyncio.run(rehearse(a.page.resolve().as_uri(), port))


if __name__ == "__main__":
    main()
