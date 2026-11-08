"""Record the README's gripper GIFs from the real page: a live grasp, and the finite-element replay beside the fast finger.

    python tools/make_gripper_gif.py docs/viewer/gripper.html --play docs/media/gripper_play.gif --truth docs/media/gripper_truth.gif

Drives the page in headless Chrome through its own controls and hands the frames to ffmpeg. The live grasp advances by
``window.softsim.step``, so it plays at real time however slowly software graphics draw it. The replay lingers on the
moments between ``SLOW``, where the fingertips let go and the box leaves the table.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import subprocess
import tempfile
from pathlib import Path

from check_gripper_page import Page, chrome, opened

PLAY_FPS, PLAY_SECONDS, PLAY_END = 20, 3.2, 1.0  # frames/s, s of grasp, s the last frame stays
TRUTH_FPS, COARSE, SLOW, TRUTH_END = 15, 3, (72, 112), 1.5  # frames/s, every 3rd recorded frame outside SLOW, s the last frame stays
WIDTH = 720  # px


async def shoot(p: Page, box: dict, path: Path) -> None:
    r = await p.call("Page.captureScreenshot", format="png", clip=dict(box, scale=1))
    path.write_bytes(base64.b64decode(r["data"]))


async def area(p: Page, *ids: str) -> dict:
    """The rectangle around the named elements, in page pixels."""
    rects = await p.js(f"{list(ids)}.map(i => {{ const r = document.getElementById(i).getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; }})")
    x0, y0 = min(r[0] for r in rects), min(r[1] for r in rects)
    return dict(x=x0, y=y0, width=max(r[2] for r in rects) - x0, height=max(r[3] for r in rects) - y0)


def encode(frames: Path, fps: int, end: float, out: Path) -> None:
    """Frames to a GIF on one shared palette, the last frame held for ``end`` seconds."""
    filters = (f"tpad=stop_mode=clone:stop_duration={end},scale={WIDTH}:-2:flags=lanczos,split[a][b];"
               "[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "%04d.png"), "-filter_complex", filters, str(out)], check=True)
    print(f"{out}: {out.stat().st_size / 1e6:.1f} MB", flush=True)


async def play(p: Page, frames: Path, out: Path) -> None:
    """The first beat of the demo: a 10 g box at 30 kPa, closed, lifted and held."""
    box = await area(p, "stage")
    await p.js("(() => { const e = document.getElementById('pressure'); e.value = 30; e.dispatchEvent(new Event('input')); document.getElementById('grasp').click(); })()")
    steps = round(1e3 / PLAY_FPS)
    for i in range(round(PLAY_SECONDS * PLAY_FPS)):
        r = await p.js(f"window.softsim.step({steps})")
        await asyncio.sleep(0.05)
        await shoot(p, box, frames / f"{i:04d}.png")
    print(f"play: {r['badge']}, {r['height']:.1f} mm", flush=True)
    if r["badge"] != "Held":
        raise SystemExit("the live grasp did not hold")
    encode(frames, PLAY_FPS, PLAY_END, out)


async def truth(p: Page, frames: Path, out: Path) -> None:
    await p.js("location.hash='truth'")
    await asyncio.sleep(1)
    box = await area(p, "truth-view", "truth-trace")
    count = await p.js("window.softsimTruth.count")
    shown = [f for f in range(count) if SLOW[0] <= f < SLOW[1] or f % COARSE == 0] + [count - 1]
    for i, f in enumerate(sorted(set(shown))):
        await p.js(f"window.softsimTruth.goTo({f})")
        await asyncio.sleep(0.05)
        await shoot(p, box, frames / f"{i:04d}.png")
    encode(frames, TRUTH_FPS, TRUTH_END, out)


async def record(url: str, port: int, play_out: Path | None, truth_out: Path | None) -> None:
    async with opened(url, port) as p:
        for job, out in ((play, play_out), (truth, truth_out)):
            if out:
                with tempfile.TemporaryDirectory() as d:
                    await job(p, Path(d), out)
        if p.errors:
            raise SystemExit(f"script errors: {p.errors}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("page", type=Path)
    ap.add_argument("--play", type=Path)
    ap.add_argument("--truth", type=Path)
    a = ap.parse_args()
    with chrome() as port:
        asyncio.run(record(a.page.resolve().as_uri(), port, a.play, a.truth))


if __name__ == "__main__":
    main()
