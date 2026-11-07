"""Write the gripper page: one self-contained HTML file, physics included.

    python tools/make_gripper_page.py --finger softsim/reduced/wang_finger.json --out docs/viewer/gripper.html

The page embeds MuJoCo compiled to WebAssembly (npm ``mujoco``), three.js, the fast finger's tables and, for every
object and finger count, the model text and joint layout written by :mod:`softsim.reduced.mjcf`, so the page and the
Python model run the same model. It opens by double click, with no server and no network. The two libraries are
fetched once into ``runs/vendor`` and checked against the digests below.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path

from softsim.reduced import ReducedFinger, ReducedGripper
from softsim.reduced.mjcf import OBJECTS

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_name("gripper_page.html")
VENDOR = ROOT / "runs" / "vendor"
MUJOCO_TGZ = "https://registry.npmjs.org/mujoco/-/mujoco-3.1.16.tgz"
THREE_JS = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"
DIGESTS = {"three.min.js": "9274bbcec8d96168626c732b5d31c775aa8cfb7eaa0599bec0c175908a2c1ce2",
           "mujoco.js": "e583772f6b17400becf9d48622245fd06c29e52ec0135df7ba4db31646720c2e",
           "mujoco.wasm": "ca22a3c3338c13bcfdbf8d9d19ebe698ac9bcab51af7cbab3d9cf57636998ea1"}
LIFT = 0.04  # m
GAP, HEIGHT = 0.05, 0.14  # m
CAVEAT = ("Live model: a chain of links fitted to the finite-element finger. It follows the finger's bend and stiffness "
          "but does not deform around the object. In one coarse finite-element grasp the same 10 g box was carried too, but "
          "the real fingers peel and re-grip first, using 14 mm of the lift where this model uses almost none. Held or "
          "dropped limits come from this fast model only. Two fingers are the fitted case; three and four are extrapolations.")


def vendor() -> dict[str, bytes]:
    """The three library files, fetched once, refused if they differ from the pinned digests."""
    VENDOR.mkdir(parents=True, exist_ok=True)
    if not all((VENDOR / n).exists() for n in DIGESTS):
        (VENDOR / "three.min.js").write_bytes(urllib.request.urlopen(THREE_JS).read())
        with tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(MUJOCO_TGZ).read())) as tar:
            for n in ("mujoco.js", "mujoco.wasm"):
                (VENDOR / n).write_bytes(tar.extractfile(f"package/{n}").read())
    files = {n: (VENDOR / n).read_bytes() for n in DIGESTS}
    for n, b in files.items():
        if hashlib.sha256(b).hexdigest() != DIGESTS[n]:
            raise ValueError(f"{n} does not match its pinned digest")
    return files


def models(f: ReducedFinger) -> dict:
    out = {}
    for shape, size in OBJECTS.items():
        for n in (2, 3, 4):
            kw = dict(shape=shape, size=size, gap=GAP, height=HEIGHT, friction=0.8, mass=0.01)
            g = ReducedGripper(f, fingers=n, **kw)
            out[f"{shape}{n}"] = dict(xml=g.xml, layout=g.layout())
    return out


def page(f: ReducedFinger, extra: dict | None = None) -> str:
    lib = vendor()
    data = dict(finger={k: getattr(f, k).tolist() for k in ("pressures", "angles", "lengths", "stiffness")}
                | dict(offset=f.offset, width=f.width), models=models(f), objects=OBJECTS, lift=LIFT, height=HEIGHT,
                rest={k: (v[2] if k == "box" else v[1]) + 1e-4 for k, v in OBJECTS.items()}, caveat=CAVEAT, **(extra or {}))
    module = lib["mujoco.js"].decode().replace("export default loadMujoco;", "")
    html = TEMPLATE.read_text()
    for token, value in (("/*@THREE@*/", lib["three.min.js"].decode()), ("/*@DATA@*/null", json.dumps(data)),
                         ("/*@MUJOCO@*/", module), ("/*@WASM@*/", base64.b64encode(lib["mujoco.wasm"]).decode())):
        html = html.replace(token, value.replace("</script>", "<\\/script>"))
    return html


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--finger", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "viewer" / "gripper.html")
    ap.add_argument("--map", type=Path, help="tools/grasp_map.py output: adds the Grasp map tab")
    a = ap.parse_args()
    extra = {}
    if a.map:
        grasps = json.loads(a.map.read_text())
        grasps["count"] = sum(len(sum(g, [])) for s in grasps["grid"].values() for g in s.values())
        extra["map"] = grasps
    html = page(ReducedFinger.from_json(a.finger), extra)
    a.out.write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">{html}</html>')
    print(f"{a.out}: {a.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
