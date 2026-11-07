"""Check the gripper page in headless Chrome: it starts, it runs fast enough, and it agrees with the Python model.

    python tools/check_gripper_page.py docs/viewer/gripper.html --finger softsim/reduced/wang_finger.json --shots runs/page

Opens the page from the file system (as a presenter would), fails on any script error, then runs the same scripted
grasps in the page's MuJoCo build and in Python and prints the largest difference in object height and fingertip
position. MuJoCo versions differ between the two (WebAssembly 3.1.16 here), so the check is a tolerance, not equality.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import websockets

from softsim.reduced import ReducedFinger, ReducedGripper
from softsim.reduced.mjcf import OBJECTS

CASES = [dict(shape="box", fingers=2, mass=0.01, mu=0.8, pressure=30.0, lift=0.04, steps=3000),
         dict(shape="box", fingers=2, mass=0.05, mu=0.8, pressure=30.0, lift=0.04, steps=3000),
         dict(shape="cylinder", fingers=3, mass=0.01, mu=0.6, pressure=25.0, lift=0.04, steps=3000),
         dict(shape="bottle", fingers=4, mass=0.02, mu=1.0, pressure=30.0, lift=0.04, steps=3000)]


def python_run(f: ReducedFinger, c: dict) -> np.ndarray:
    g = ReducedGripper(f, fingers=c["fingers"], shape=c["shape"], size=OBJECTS[c["shape"]], mass=c["mass"], friction=c["mu"])
    n = f.links
    last = g.model.body(f"f0l{n - 1}").id
    out = []
    for i in range(c["steps"]):
        t = i * 1e-3
        g.step(c["pressure"] * 1e3 * min(t / 0.6, 1), c["lift"] * min(max((t - 1.5) / 0.5, 0), 1))
        if i % 100 == 99:
            out.append([g.object_height, g.data.xpos[last, 0], g.data.xpos[last, 2]])
    return np.array(out)


class Page:
    def __init__(self, ws):
        self.ws, self.id, self.errors = ws, 0, []

    async def call(self, method: str, **params) -> dict:
        self.id += 1
        await self.ws.send(json.dumps(dict(id=self.id, method=method, params=params)))
        while True:
            m = json.loads(await self.ws.recv())
            if m.get("method") == "Runtime.exceptionThrown":
                self.errors.append(m["params"]["exceptionDetails"].get("exception", {}).get("description", "script error"))
            if m.get("id") == self.id:
                return m.get("result", {})

    async def js(self, expr: str):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"])
        return r["result"].get("value")

    async def shot(self, path: Path) -> None:
        path.write_bytes(base64.b64decode((await self.call("Page.captureScreenshot", format="png"))["data"]))


async def drive(url: str, port: int, f: ReducedFinger, shots: Path | None) -> None:
    target = next(t for t in json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list")) if t["type"] == "page")
    async with websockets.connect(target["webSocketDebuggerUrl"], max_size=2**27) as ws:
        p = Page(ws)
        await p.call("Runtime.enable")
        await p.call("Page.enable")
        await p.call("Page.navigate", url=url)
        for _ in range(240):
            if await p.js("document.documentElement?.dataset.ready === '1' || !!document.getElementById('fail')?.textContent"):
                break
            await asyncio.sleep(0.25)
        fail = await p.js("document.getElementById('fail').textContent")
        if fail:
            raise SystemExit(f"page did not start: {fail}")
        await asyncio.sleep(3)
        s = await p.js("window.softsim.stats()")
        print(f"live: {s['rtf']:.1f}x real time, {s['frames'] / 3:.0f} frames/s (software GL in this check)")
        if shots:
            shots.mkdir(parents=True, exist_ok=True)
            await p.shot(shots / "idle.png")
            await p.js("document.getElementById('mass').value=10;document.getElementById('mass').dispatchEvent(new Event('input'));document.getElementById('grasp').click()")
            await asyncio.sleep(8)
            await p.shot(shots / "lifted.png")
            print("lifted state:", await p.js("window.softsim.stats()"))
            await p.js("location.hash='map'")
            await asyncio.sleep(1)
            await p.shot(shots / "map.png")
            await p.js("location.hash='play'")
        worst = 0.0
        for c in CASES:
            page = np.array(await p.js(f"window.softsim.run({json.dumps(c)})"))
            py = python_run(f, c)
            err = np.abs(page - py).max(axis=0)
            worst = max(worst, float(err.max()))
            print(f"{c['shape']:8s} {c['fingers']} fingers {c['mass'] * 1e3:.0f} g: object z page {page[-1, 0] * 1e3:.2f} python "
                  f"{py[-1, 0] * 1e3:.2f} mm; largest gap object {err[0] * 1e3:.3f} mm, fingertip {err[1:].max() * 1e3:.3f} mm")
        print(f"worst gap {worst * 1e3:.3f} mm; script errors: {p.errors or 'none'}")
        if p.errors:
            raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("page", type=Path)
    ap.add_argument("--finger", type=Path, required=True)
    ap.add_argument("--shots", type=Path)
    a = ap.parse_args()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        chrome = subprocess.Popen(["google-chrome", "--headless=new", "--no-sandbox", "--use-angle=swiftshader",
                                   "--enable-unsafe-swiftshader", f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
                                   "--window-size=1280,800", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            time.sleep(3)
            asyncio.run(drive(a.page.resolve().as_uri(), port, ReducedFinger.from_json(a.finger), a.shots))
        finally:
            chrome.terminate()
            chrome.wait()


if __name__ == "__main__":
    main()
