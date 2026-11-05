"""Interactive slider viewer (one self-contained HTML file) from recorded softsim runs.

    python tools/make_viewer.py pneunet                   # docs/viewer/pneunet.html (from runs/media/pneunet/*.npz)
    python tools/make_viewer.py arm stroke --frames 80
    python tools/make_viewer.py --npz runs/x/a.npz:fem runs/x/b.npz:mujoco --title "my run" --out my.html

The runs are the quasi-static ramps of the actuation scenarios (``tools/make_actuation_media.py`` records them:
``runs/media/<case>/<backend>.npz`` with the positions of every mesh vertex over time). The page lets you drag a
slider through the actuation (pressure, tension, stroke) and orbit the body with the mouse; several runs of one case
(e.g. the reference and MuJoCo) are drawn side by side at the same slider position. The vertex positions are
quantised to 16 bits over the bounding box of the whole run (error < box / 65535: 2 um on a 0.13 m box); the page
needs no network and no library (2D canvas, painter's algorithm, camera-lit shading).
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "media"

TITLES = {"pneunet": "PneuNet finger (Ecoflex 00-30), pressure 0-8 kPa", "stroke": "Cable reeled in, stroke 0-3 mm",
          "arm": "Two-segment cable arm: the S-shape", "wang": "Ribbed cast finger (Wang et al. 2026), 0-40 kPa"}


def pack(path: Path, frames: int) -> dict:
    """One run as JSON-able data: quantised vertices of ``frames`` evenly spaced frames, triangles, the slider values."""
    d = np.load(path)
    k = np.unique(np.linspace(0, len(d["t"]) - 1, frames).round().astype(int))
    x = d["x"][k].astype(np.float64)
    lo, hi = x.reshape(-1, 3).min(0), x.reshape(-1, 3).max(0)
    scale = (hi - lo).max()
    q = np.round((x - lo) / scale * 65535).astype("<u2")
    tris = d["tris"].astype("<u4")
    return dict(n=int(x.shape[1]), frames=int(len(k)), lo=lo.tolist(), scale=float(scale),
                q=base64.b64encode(q.tobytes()).decode(), tris=base64.b64encode(tris.tobytes()).decode(),
                value=d["value"][k].tolist(), bend=np.degrees(d["bend"][k]).tolist(), t=d["t"][k].tolist())


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
:root{--bg:#f5f6f4;--fg:#1d2a24;--mut:#66736c;--acc:#2e7d5b;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#161b19;--fg:#e4ebe7;--mut:#8b9a92;--acc:#58c79a;--card:#1f2623}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}h1{font-size:20px;margin:4px 0 12px}
.row{display:flex;gap:12px;flex-wrap:wrap}.card{flex:1 1 320px;max-width:680px;background:var(--card);border-radius:10px;padding:8px}
canvas{width:100%;height:auto;touch-action:none;cursor:grab;display:block}.lab{font-size:13px;color:var(--mut);padding:2px 6px}
.ctl{margin:14px 0;display:flex;align-items:center;gap:12px}input[type=range]{flex:1;accent-color:var(--acc);height:28px}
#val{font:600 18px ui-monospace,monospace;min-width:150px}small{color:var(--mut)}
</style></head><body><main><h1>__TITLE__</h1>
<div class="ctl"><input id="s" type="range" min="0" max="1000" value="0"><span id="val"></span></div>
<div class="row" id="views"></div>
<p><small>Drag the slider to actuate. Drag the picture to orbit, wheel to zoom. Each frame is a result of the simulation
(quasi-static ramp); between frames the vertices are interpolated linearly.</small></p></main>
<script>
const RUNS=__DATA__;
const dec=b=>{const s=atob(b),a=new Uint8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);return a.buffer};
for(const r of RUNS){r.Q=new Uint16Array(dec(r.q));r.T=new Uint32Array(dec(r.tris));}
let yaw=-0.6,pitch=0.35,zoom=1,u=0;
const views=document.getElementById('views');
for(const r of RUNS){const c=document.createElement('div');c.className='card';
 c.innerHTML='<div class="lab">'+r.label+'</div><canvas width="640" height="520"></canvas><div class="lab" id="b'+RUNS.indexOf(r)+'"></div>';
 views.appendChild(c);r.cv=c.querySelector('canvas');r.ctx=r.cv.getContext('2d');r.bl=c.querySelector('.lab:last-child');
 // centre and radius of the whole run
 const P=pos(r,r.frames-1);let m=[0,0,0],M=[0,0,0];r.cen=[0,0,0];
 const lo=[1e9,1e9,1e9],hi=[-1e9,-1e9,-1e9];
 for(let f=0;f<r.frames;f+=Math.max(1,r.frames>>3)){const p=pos(r,f);for(let i=0;i<r.n;i++)for(let k=0;k<3;k++){const v=p[3*i+k];if(v<lo[k])lo[k]=v;if(v>hi[k])hi[k]=v}}
 for(let k=0;k<3;k++)r.cen[k]=(lo[k]+hi[k])/2;r.rad=Math.hypot(hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2])/2;}
function pos(r,f){const o=new Float64Array(3*r.n),b=f*3*r.n;for(let i=0;i<3*r.n;i++)o[i]=r.lo[i%3]+r.Q[b+i]/65535*r.scale;return o}
function mix(r,x){const f=x*(r.frames-1),i=Math.min(Math.floor(f),r.frames-2),a=f-i,A=pos(r,i),B=pos(r,i+1);
 for(let k=0;k<A.length;k++)A[k]+=a*(B[k]-A[k]);return {p:A,f,i,a}}
function interp(arr,f){const i=Math.min(Math.floor(f),arr.length-2),a=f-i;return arr[i]*(1-a)+arr[i+1]*a}
function draw(r){const {p,f}=mix(r,u),cx=Math.cos(yaw),sx=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
 const W=r.cv.width,H=r.cv.height,s=zoom*0.42*Math.min(W,H)/r.rad,n=r.n,X=new Float64Array(n),Y=new Float64Array(n),Z=new Float64Array(n);
 for(let i=0;i<n;i++){const x=p[3*i]-r.cen[0],y=p[3*i+1]-r.cen[1],z=p[3*i+2]-r.cen[2];
  const a=cx*x+sx*y,b=-sx*x+cx*y;X[i]=a;Y[i]=cp*z-sp*b;Z[i]=sp*z+cp*b}   // Z: depth towards the viewer's far side
 const T=r.T,m=T.length/3,idx=[],dep=new Float64Array(m),shade=new Float64Array(m);
 for(let t=0;t<m;t++){const a=T[3*t],b=T[3*t+1],c=T[3*t+2];
  const ux=X[b]-X[a],uy=Y[b]-Y[a],uz=Z[b]-Z[a],vx=X[c]-X[a],vy=Y[c]-Y[a],vz=Z[c]-Z[a];
  let nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;const l=Math.hypot(nx,ny,nz)||1;nx/=l;ny/=l;nz/=l;
  // facing: the view direction is +Z (away), so a visible face has its normal towards -Z; surfaces have outward normals
  if(nz>0)continue;shade[t]=0.25+0.75*Math.max(0,-0.35*nx+0.45*ny-0.82*nz);dep[t]=(Z[a]+Z[b]+Z[c])/3;idx.push(t)}
 idx.sort((a,b)=>dep[b]-dep[a]);
 const g=r.ctx;g.clearRect(0,0,W,H);
 for(const t of idx){const a=T[3*t],b=T[3*t+1],c=T[3*t+2],k=shade[t];
  g.fillStyle='rgb('+(70+150*k|0)+','+(120+120*k|0)+','+(95+110*k|0)+')';g.strokeStyle=g.fillStyle;g.lineWidth=0.6;
  g.beginPath();g.moveTo(W/2+s*X[a],H/2-s*Y[a]);g.lineTo(W/2+s*X[b],H/2-s*Y[b]);g.lineTo(W/2+s*X[c],H/2-s*Y[c]);g.closePath();g.fill();g.stroke()}
 return f}
function render(){let txt='';
 for(const r of RUNS){const f=draw(r);r.bl.textContent=r.bend.length?'bend angle '+interp(r.bend,f).toFixed(1)+'°':'';
  if(r===RUNS[0])txt=r.value.some(v=>v!==0)?interp(r.value,f).toFixed(r.digits)+' '+r.unit:(u*100).toFixed(0)+' % of the ramp'}
 document.getElementById('val').textContent=txt}
const s=document.getElementById('s');s.oninput=()=>{u=s.value/1000;render()};
let drag=null;for(const r of RUNS){r.cv.onpointerdown=e=>{drag=[e.clientX,e.clientY];r.cv.setPointerCapture(e.pointerId)};
 r.cv.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*0.01;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-drag[1])*0.01));drag=[e.clientX,e.clientY];render()};
 r.cv.onpointerup=()=>drag=null;r.cv.onwheel=e=>{e.preventDefault();zoom*=Math.exp(-e.deltaY*0.001);render()}}
render();
</script></body></html>"""


def build(sources: list[tuple[Path, str]], title: str, frames: int, unit: tuple[str, float, int]) -> str:
    """The page for ``sources`` (npz path, label)."""
    data = []
    for path, label in sources:
        r = pack(path, frames)
        r["value"] = [v * unit[1] for v in r["value"]]
        data.append({**r, "label": label, "unit": unit[0], "digits": unit[2]})
    return PAGE.replace("__TITLE__", title).replace("__DATA__", json.dumps(data))


UNITS = {"pneunet": ("kPa", 1e-3, 1), "stroke": ("mm", 1e3, 2), "arm": ("N", 1.0, 2), "wang": ("kPa", 1e-3, 1)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("case", nargs="*", help="cases recorded in runs/media/<case>/")
    p.add_argument("--npz", nargs="*", default=[], metavar="FILE[:LABEL]")
    p.add_argument("--title")
    p.add_argument("--out")
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--unit", default="kPa", help="--npz runs: unit of the slider value (kPa, mm, N)")
    args = p.parse_args()
    jobs = []
    for case in args.case:
        files = sorted((RUNS / case).glob("*.npz"))
        if not files:
            raise SystemExit(f"no recorded runs in {RUNS / case}; run tools/make_actuation_media.py --only {case}")
        jobs.append((case, [(f, f.stem) for f in files], TITLES.get(case, case), UNITS.get(case, ("", 1.0, 2))))
    if args.npz:
        src = [(Path(a.split(":")[0]), a.split(":")[1] if ":" in a else Path(a).stem) for a in args.npz]
        unit = {"kPa": ("kPa", 1e-3, 1), "mm": ("mm", 1e3, 2), "N": ("N", 1.0, 2)}[args.unit]
        jobs.append((args.out or "custom", src, args.title or "softsim run", unit))
    for name, src, title, unit in jobs:
        out = Path(args.out) if args.out and len(jobs) == 1 else ROOT / "docs" / "viewer" / f"{name}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build(src, args.title or title, args.frames, unit))
        print(f"{out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
