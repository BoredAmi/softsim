"""Check the drop page in headless Chrome: it starts, a real drop runs, and it gives the numbers Node gives.

    python tools/check_drop_page.py docs/viewer/drop.html --shots runs/page

Opens the page from the file system, fails on any script error, clicks Drop and watches the slow-motion rate, then
runs the two filmed drops inside the page and compares peaks with the same solver file under Node. Declared settings run
through the page's own fields and are compared with a job written here from the same numbers, so a wrong unit or a
field that does nothing on the page shows as a gap.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

import numpy as np

from check_drop_solver import DIAMETER, HEIGHT, PHYSICAL, drop, mesh, run_js
from check_gripper_page import Page


# page settings, and the same numbers in SI worked out by hand for the job: shear modulus, body size, object
DECLARED = {
    "shear modulus, wider and taller body, steel ball by material":
        (dict(by="G", modulus=60.0, nu=0.49, relax=0.5, tau=3.0, density=1100.0, bodyD=50.0, bodyH=25.0, shape="sphere", objD=30.0, massBy="material",
              objectMaterial="Steel", height=20), dict(G=60e3, nu=0.49, g=0.5, tau=3e-3, density=1100.0, D=0.05, H=0.025, radius=0.015, drop=0.20, density_object=7850.0)),
    "100% modulus, flat body, disc by mass":
        (dict(by="M100", modulus=200.0, nu=0.4995, relax=0.2, tau=5.0, density=1000.0, bodyD=30.0, bodyH=12.0, shape="disc", objD=20.0, objH=8.0, mass=30.0, massBy="mass", height=30),
         dict(G=200e3 / 1.75, nu=0.4995, g=0.2, tau=5e-3, density=1000.0, D=0.03, H=0.012, radius=0.01, thick=0.008, mass=0.03, drop=0.30)),
}


def declared_peak(cells: dict, c: dict, end: float) -> float:
    """Peak indentation [m] of a declared drop, from a job built without the page's helpers."""
    pts = np.array(cells["points"]).reshape(-1, 3) * [c["D"] / DIAMETER, c["D"] / DIAMETER, c["H"] / HEIGHT]
    sphere = "thick" not in c
    low = c["radius"] if sphere else c["thick"] / 2
    obj = dict(shape="sphere", radius=c["radius"], mass=c["density_object"] * 4 / 3 * np.pi * c["radius"]**3) if sphere else \
        dict(shape="disc", radius=c["radius"], height=c["thick"], mass=c["mass"])
    speed = (2 * 9.81 * c["drop"])**0.5
    material = dict(E=2 * c["G"] * (1 + c["nu"]), nu=c["nu"], g=c["g"], tau=c["tau"], mass=c["density"] * np.pi * (c["D"] / 2)**2 * c["H"])
    r = run_js(dict(mesh=dict(cells, points=pts.ravel().tolist()), material=material, friction=dict(object=5, table=5), duration=end, sample=1e-5,
                    object=obj | dict(position=[0, 0, c["H"] + low + 1e-4], velocity=[0, 0, -speed])))
    return float((c["H"] - (np.array(r["z"]) - low)).max())


async def clicks(p: Page) -> None:
    """The real controls: declaring the same stiffness another way changes nothing, a typed value drives the solver,
    a library material does, and a filmed replay puts everything back."""
    press = "(box, v) => document.querySelector(`#${box} [data-v=${v}]`).click()"
    read = "({E: window.softdrop.card().E, G: window.softdrop.card().G, dt: window.softdrop.stats().dt, mass: window.softdrop.stats().bodyMass, status: document.getElementById('status').textContent})"
    await p.js(f"({press})('recorded', 'w050')")
    film = await p.js(read)
    for by, shown in (("G", 47.78), ("M100", 83.62), ("E", 143.3)):
        await p.js(f"({press})('by', '{by}')")
        now, number = await p.js(read), await p.js("+document.getElementById('modulusNumber').value")
        print(f"declared by {by}: field {number} kPa, E {now['E'] / 1e3:.3f} kPa")
        if abs(now["E"] / film["E"] - 1) > 1e-9 or abs(number - shown) > 0.1 or now["dt"] != film["dt"]:
            raise SystemExit(f"declaring by {by} changed the solid")
    await p.js("const n = document.getElementById('modulusNumber'); n.value = 500; n.dispatchEvent(new Event('change'))")
    typed = await p.js(read)
    print(f"typed 500 kPa: E {typed['E'] / 1e3:.1f} kPa, time step {typed['dt'] * 1e6:.2f} us against {film['dt'] * 1e6:.2f}; {typed['status']}")
    if abs(typed["E"] - 5e5) > 1 or abs(typed["dt"] / film["dt"] - (film["E"] / typed["E"])**0.5) > 1e-6:
        raise SystemExit("a typed modulus did not reach the solver")
    await p.js(f"({press})('by', 'material')")
    await p.js("const m = document.getElementById('material'); m.value = 5; m.dispatchEvent(new Event('change'))")
    lib = await p.js(read)
    print(f"Posilicone A20: G {lib['G'] / 1e3:.1f} kPa, body {lib['mass'] * 1e3:.1f} g")
    if abs(lib["G"] - 88.3e3) > 100:
        raise SystemExit("the library material did not reach the solver")
    await p.js(f"({press})('recorded', 'w050')")
    back = await p.js(read)
    if abs(back["E"] / film["E"] - 1) > 1e-12 or back["dt"] != film["dt"] or abs(back["mass"] - 19.49e-3) > 1e-9:
        raise SystemExit(f"the filmed replay did not restore the card: {back}")


async def drive(url: str, port: int, shots: Path | None) -> None:
    target = next(t for t in json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list")) if t["type"] == "page")
    async with websockets.connect(target["webSocketDebuggerUrl"], max_size=2**27) as ws:
        p = Page(ws)
        await p.call("Runtime.enable")
        await p.call("Page.enable")
        await p.call("Page.navigate", url=url)
        for _ in range(120):
            if await p.js("document.documentElement?.dataset.ready === '1'"):
                break
            await asyncio.sleep(0.25)
        else:
            raise SystemExit(f"page did not start: {p.errors}")
        if shots:
            shots.mkdir(parents=True, exist_ok=True)
            await asyncio.sleep(1)
            await p.shot(shots / "drop_idle.png")
        await p.js("document.querySelector('#recorded [data-v=w050]').click(); document.getElementById('drop').click()")
        await asyncio.sleep(12)
        s = await p.js("window.softdrop.stats()")
        print("live:", s, f"= 1/{1 / max(s['rate'], 1e-9):.0f} of real time")
        if shots:
            await p.shot(shots / "drop_impact.png")
        cells = mesh()
        for case in ("w020", "w050"):
            r = await p.js(f"window.softdrop.run('{case}', 0.011)")
            sim = drop(PHYSICAL, case, cells)
            print(f"{case}: page {r['peak'] * 1e3:.3f} mm at {r['peakT'] * 1e3:.2f} ms, {r['force']:.1f} N; Node {sim.indentation.max() * 1e3:.3f} mm; ok {r['ok']}")
            if not r["ok"]:
                raise SystemExit(1)
        for name, (settings, hand) in DECLARED.items():
            r = await p.js(f"window.softdrop.declare({json.dumps(settings)}, 0.008)")
            node = declared_peak(cells, hand, 0.008)
            print(f"{name}: page {r['peak'] * 1e3:.3f} mm, by hand {node * 1e3:.3f} mm; {r['bodyRead']} {r['objectRead']}")
            if not r["ok"] or abs(r["peak"] - node) > 0.05e-3:
                raise SystemExit(f"{name}: the page and the hand-built job differ")
        await clicks(p)
        if shots:
            await p.js("document.getElementById('bodyBox').open = true; document.getElementById('objectBox').open = false; document.getElementById('bodyBox').scrollIntoView()")
            await asyncio.sleep(1)
            await p.shot(shots / "drop_declared.png")
        await p.js("document.querySelector('#thing [data-v=w500]').click(); document.getElementById('height').value = 80;"
                   "document.getElementById('height').dispatchEvent(new Event('input')); document.getElementById('drop').click()")
        await asyncio.sleep(10)
        print("500 g from 80 cm (the solver's known limit):", await p.js("document.getElementById('badge').textContent"))
        print(f"script errors: {p.errors or 'none'}")
        if p.errors:
            raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("page", type=Path)
    ap.add_argument("--shots", type=Path)
    a = ap.parse_args()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        chrome = subprocess.Popen(["google-chrome", "--headless=new", "--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                   f"--remote-debugging-port={port}", f"--user-data-dir={profile}", "--window-size=1280,800", "about:blank"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            time.sleep(3)
            asyncio.run(drive(a.page.resolve().as_uri(), port, a.shots))
        finally:
            chrome.terminate()
            chrome.wait()


if __name__ == "__main__":
    main()
