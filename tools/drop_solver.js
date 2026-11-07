/* Explicit soft-solid solver for the drop page: a rigid object falls on a soft body on a table.
 *
 * The same file runs in the page and in Node (tools/check_drop_solver.py), so what is checked is what is shown.
 * SI units throughout. Linear tetrahedra with nodal averaged pressure (Bonet & Burton 1998: the volume change is
 * averaged over each node's elements, which keeps a nearly incompressible tetrahedron from locking), the split
 * Neo-Hookean law of softsim.fem.materials (shear modulus mu, bulk modulus kappa) and one Prony term on the isochoric
 * stress, as the physical Ecoflex card has: G(t) = mu (1 - g (1 - exp(-t / tau))). Central differences in time, lumped
 * masses. Contact is penalty on the nodes, stiffness set from each node's mass and the time step; "stick" contact
 * stops tangential motion up to the friction limit. The object is a rigid disc or sphere with six degrees of freedom.
 */
class DropSim {
  /* mesh: {points: [x,y,z,...], tets: [a,b,c,d,...]}; material: {E, nu, g, tau, density | mass};
   * object: {shape: "disc"|"sphere"|"none", mass, radius, height, position: [x,y,z], velocity: [x,y,z],
   *          tilt: rad about y, spin: rad/s about y}; friction: {object, table} */
  constructor(mesh, material, object, friction = {object: 2, table: 0.8}, options = {}) {
    this.gravity = options.gravity ?? 9.81;
    this.cfl = options.cfl ?? 0.35;
    this.n = mesh.points.length / 3;
    this.m = mesh.tets.length / 4;
    this.x = Float64Array.from(mesh.points);
    this.v = new Float64Array(3 * this.n);
    this.f = new Float64Array(3 * this.n);
    this.tets = Int32Array.from(mesh.tets);
    this.friction = friction;
    this.setup(material);
    this.object = object.shape === "none" ? null : this.rigid(object);
    if (options.velocity) for (let i = 0; i < this.n; i++) for (let k = 0; k < 3; k++) this.v[3 * i + k] = options.velocity[k];
    if (options.lift) for (let i = 0; i < this.n; i++) this.x[3 * i + 2] += options.lift;
    this.t = 0;
    this.peakForce = 0;
    this.objectForce = [0, 0, 0];
    this.failed = false;
  }

  setup(mat) {
    const {n, m, x, tets} = this;
    this.mu = mat.E / (2 * (1 + mat.nu));
    this.kappa = mat.E / (3 * (1 - 2 * mat.nu));
    this.g = mat.g ?? 0;
    this.tau = mat.tau ?? 1;
    this.B = new Float64Array(9 * m);   // inverse of the rest edge matrix
    this.V = new Float64Array(m);
    this.nodeV = new Float64Array(n);
    this.Q = new Float64Array(6 * m);   // hereditary stress h, material frame (xx yy zz xy xz yz)
    this.S = new Float64Array(6 * m);   // isochoric stress of the last step
    this.J = new Float64Array(m);
    this.Jold = new Float64Array(m).fill(1);
    this.nodeJ = new Float64Array(n);
    let h = Infinity, total = 0;
    for (let e = 0; e < m; e++) {
      const [a, b, c, d] = [tets[4 * e], tets[4 * e + 1], tets[4 * e + 2], tets[4 * e + 3]];
      const D = [0, 1, 2].map(r => [x[3 * b + r] - x[3 * a + r], x[3 * c + r] - x[3 * a + r], x[3 * d + r] - x[3 * a + r]]);
      const det = D[0][0] * (D[1][1] * D[2][2] - D[1][2] * D[2][1]) - D[0][1] * (D[1][0] * D[2][2] - D[1][2] * D[2][0]) +
                  D[0][2] * (D[1][0] * D[2][1] - D[1][1] * D[2][0]);
      if (det <= 0) throw new Error("tetrahedron inside out at rest");
      const inv = [(D[1][1] * D[2][2] - D[1][2] * D[2][1]) / det, (D[0][2] * D[2][1] - D[0][1] * D[2][2]) / det, (D[0][1] * D[1][2] - D[0][2] * D[1][1]) / det,
                   (D[1][2] * D[2][0] - D[1][0] * D[2][2]) / det, (D[0][0] * D[2][2] - D[0][2] * D[2][0]) / det, (D[0][2] * D[1][0] - D[0][0] * D[1][2]) / det,
                   (D[1][0] * D[2][1] - D[1][1] * D[2][0]) / det, (D[0][1] * D[2][0] - D[0][0] * D[2][1]) / det, (D[0][0] * D[1][1] - D[0][1] * D[1][0]) / det];
      this.B.set(inv, 9 * e);
      const vol = det / 6;
      this.V[e] = vol;
      total += vol;
      for (const i of [a, b, c, d]) this.nodeV[i] += vol / 4;
      for (let r = 0; r < 3; r++) {  // shortest altitude: 1 / largest row norm of the shape gradients
        const q = Math.hypot(inv[3 * r], inv[3 * r + 1], inv[3 * r + 2]);
        h = Math.min(h, 1 / q);
      }
      const q0 = Math.hypot(inv[0] + inv[3] + inv[6], inv[1] + inv[4] + inv[7], inv[2] + inv[5] + inv[8]);
      h = Math.min(h, 1 / q0);
    }
    const density = mat.mass ? mat.mass / total : mat.density;
    this.nodeM = this.nodeV.map(v => v * density);
    this.bodyMass = total * density;
    const wave = Math.sqrt((this.kappa + 4 * this.mu / 3) / density);
    this.dt = this.cfl * h / wave;
    // the averaged pressure leaves a checkerboard of element volumes unresisted: a share of each element's own pressure
    // and a bulk viscosity (density x wave speed x length, the usual explicit-code form) hold it down
    this.blend = mat.blend ?? 0.1;
    this.bulkViscosity = (mat.bulkViscosity ?? 0.1) * density * wave * h;
    this.decay = Math.exp(-this.dt / this.tau);
    this.alpha = (1 - this.decay) / (this.dt / this.tau);
    this.kContact = this.nodeM.map(q => 0.2 * q / (this.dt * this.dt));
    this.cContact = this.nodeM.map(q => 0.1 * q / this.dt);
  }

  rigid(o) {
    const r = o.radius, h = o.height ?? 0, m = o.mass;
    const inertia = o.shape === "sphere" ? [0.4 * m * r * r, 0.4 * m * r * r, 0.4 * m * r * r]
      : [m * (3 * r * r + h * h) / 12, m * (3 * r * r + h * h) / 12, 0.5 * m * r * r];
    const c = Math.cos((o.tilt ?? 0) / 2), s = Math.sin((o.tilt ?? 0) / 2);
    return {shape: o.shape, mass: m, radius: r, height: h, inertia, p: [...o.position], v: [...o.velocity],
            w: [0, o.spin ?? 0, 0], q: [c, 0, s, 0], R: null, f: [0, 0, 0], tq: [0, 0, 0]};
  }

  /* one step: internal forces, contact, then semi-implicit Euler (equal to central differences) */
  step() {
    const o = this.object;
    this.f.fill(0);
    this.internal();
    if (o) { o.f = [0, 0, -o.mass * this.gravity]; o.tq = [0, 0, 0]; o.R = rotation(o.q); }
    this.contact();
    const {n, x, v, f, nodeM, dt} = this, gz = this.gravity;
    for (let i = 0; i < n; i++) {
      const im = dt / nodeM[i];
      v[3 * i] += f[3 * i] * im;
      v[3 * i + 1] += f[3 * i + 1] * im;
      v[3 * i + 2] += f[3 * i + 2] * im - gz * dt;
      x[3 * i] += v[3 * i] * dt;
      x[3 * i + 1] += v[3 * i + 1] * dt;
      x[3 * i + 2] += v[3 * i + 2] * dt;
    }
    if (o) this.moveObject(o);
    this.t += dt;
    if (!(Math.abs(x[0]) < 10)) this.failed = true;
  }

  moveObject(o) {
    const dt = this.dt;
    if (o.shape !== "none") this.objectOnTable(o);
    for (let k = 0; k < 3; k++) { o.v[k] += o.f[k] / o.mass * dt; o.p[k] += o.v[k] * dt; }
    const R = o.R, tb = [0, 1, 2].map(i => R[i] * o.tq[0] + R[3 + i] * o.tq[1] + R[6 + i] * o.tq[2]);
    const wb = [0, 1, 2].map(i => R[i] * o.w[0] + R[3 + i] * o.w[1] + R[6 + i] * o.w[2]);
    const I = o.inertia;
    const dw = [(tb[0] - (I[2] - I[1]) * wb[1] * wb[2]) / I[0], (tb[1] - (I[0] - I[2]) * wb[2] * wb[0]) / I[1], (tb[2] - (I[1] - I[0]) * wb[0] * wb[1]) / I[2]];
    for (let k = 0; k < 3; k++) wb[k] += dw[k] * dt;
    o.w = [0, 1, 2].map(i => R[3 * i] * wb[0] + R[3 * i + 1] * wb[1] + R[3 * i + 2] * wb[2]);
    const [a, b, c, d] = o.q, [p, q, r] = o.w, s = 0.5 * dt;
    const nq = [a - s * (p * b + q * c + r * d), b + s * (p * a + q * d - r * c), c + s * (q * a + r * b - p * d), d + s * (r * a + p * c - q * b)];
    const len = Math.hypot(...nq);
    o.q = nq.map(u => u / len);
  }

  /* the rigid object against the table: a few points on its rim (or its lowest point), by a stiff spring */
  objectOnTable(o) {
    const k = 0.05 * o.mass / (this.dt * this.dt), c = 2 * 0.7 * Math.sqrt(k * o.mass), R = o.R;
    const pts = [];
    if (o.shape === "sphere") pts.push([0, 0, -o.radius, true]);
    else for (let i = 0; i < 16; i++) for (const z of [-o.height / 2, o.height / 2]) {
      pts.push([o.radius * Math.cos(i * Math.PI / 8), o.radius * Math.sin(i * Math.PI / 8), z]);
    }
    for (const q of pts) {
      const r = q[3] ? [0, 0, -o.radius] : [R[0] * q[0] + R[1] * q[1] + R[2] * q[2], R[3] * q[0] + R[4] * q[1] + R[5] * q[2], R[6] * q[0] + R[7] * q[1] + R[8] * q[2]];
      const z = o.p[2] + r[2];
      if (z >= 0) continue;
      const vel = [o.v[0] + o.w[1] * r[2] - o.w[2] * r[1], o.v[1] + o.w[2] * r[0] - o.w[0] * r[2], o.v[2] + o.w[0] * r[1] - o.w[1] * r[0]];
      const N = Math.max(-k * z - c * vel[2], 0), vt = Math.hypot(vel[0], vel[1]);
      const ft = vt > 1e-9 ? Math.min(this.friction.table * N, 0.3 * o.mass * vt / this.dt) / vt : 0;
      addAt(o, r, [-ft * vel[0], -ft * vel[1], N]);
    }
  }

  contact() {
    const {n, x, v, f, kContact, cContact, nodeM, dt, friction} = this, o = this.object;
    let total = 0;
    for (let i = 0; i < n; i++) {
      const j = 3 * i;
      if (x[j + 2] < 0) {  // table
        const N = Math.max(-kContact[i] * x[j + 2] - cContact[i] * v[j + 2], 0);
        const vt = Math.hypot(v[j], v[j + 1]);
        const ft = vt > 1e-9 ? Math.min(friction.table * N, 0.5 * nodeM[i] * vt / dt) / vt : 0;
        f[j] -= ft * v[j]; f[j + 1] -= ft * v[j + 1]; f[j + 2] += N;
      }
      if (!o) continue;
      const R = o.R, d0 = x[j] - o.p[0], d1 = x[j + 1] - o.p[1], d2 = x[j + 2] - o.p[2];
      const reach = o.shape === "sphere" ? o.radius : Math.hypot(o.radius, o.height / 2);
      if (d0 * d0 + d1 * d1 + d2 * d2 > reach * reach) continue;
      const l0 = R[0] * d0 + R[3] * d1 + R[6] * d2, l1 = R[1] * d0 + R[4] * d1 + R[7] * d2, l2 = R[2] * d0 + R[5] * d1 + R[8] * d2;
      let pen, nl;
      if (o.shape === "sphere") {
        const r = Math.hypot(l0, l1, l2);
        pen = o.radius - r;
        nl = [l0 / r, l1 / r, l2 / r];
      } else {
        const r = Math.hypot(l0, l1), side = o.radius - r, low = o.height / 2 + l2, top = o.height / 2 - l2;
        pen = Math.min(side, low, top);
        nl = pen === low ? [0, 0, -1] : pen === top ? [0, 0, 1] : [l0 / r, l1 / r, 0];
      }
      if (pen <= 0) continue;
      const nw = [R[0] * nl[0] + R[1] * nl[1] + R[2] * nl[2], R[3] * nl[0] + R[4] * nl[1] + R[5] * nl[2], R[6] * nl[0] + R[7] * nl[1] + R[8] * nl[2]];
      const pv = [o.v[0] + o.w[1] * d2 - o.w[2] * d1, o.v[1] + o.w[2] * d0 - o.w[0] * d2, o.v[2] + o.w[0] * d1 - o.w[1] * d0];
      const rel = [v[j] - pv[0], v[j + 1] - pv[1], v[j + 2] - pv[2]];
      const vn = rel[0] * nw[0] + rel[1] * nw[1] + rel[2] * nw[2];
      const N = Math.max(kContact[i] * pen - cContact[i] * vn, 0);
      const t = [rel[0] - vn * nw[0], rel[1] - vn * nw[1], rel[2] - vn * nw[2]], vt = Math.hypot(...t);
      const ft = vt > 1e-9 ? Math.min(friction.object * N, 0.5 * nodeM[i] * vt / dt) / vt : 0;
      const F = [N * nw[0] - ft * t[0], N * nw[1] - ft * t[1], N * nw[2] - ft * t[2]];
      f[j] += F[0]; f[j + 1] += F[1]; f[j + 2] += F[2];
      addAt(o, [d0, d1, d2], [-F[0], -F[1], -F[2]]);
      total -= F[2];  // up on the object
    }
    this.contactForce = total;
    this.peakForce = Math.max(this.peakForce, this.contactForce);
  }

  internal() {
    const {m, tets, x, B, V, J: Je, Jold, nodeJ, nodeV, mu, kappa, Q, S, decay, alpha, f, dt, blend, bulkViscosity} = this;
    nodeJ.fill(0);
    const F = this._F ?? (this._F = new Float64Array(9 * m));
    for (let e = 0; e < m; e++) {
      const a = 3 * tets[4 * e], b = 3 * tets[4 * e + 1], c = 3 * tets[4 * e + 2], d = 3 * tets[4 * e + 3], o = 9 * e;
      const e00 = x[b] - x[a], e01 = x[c] - x[a], e02 = x[d] - x[a];
      const e10 = x[b + 1] - x[a + 1], e11 = x[c + 1] - x[a + 1], e12 = x[d + 1] - x[a + 1];
      const e20 = x[b + 2] - x[a + 2], e21 = x[c + 2] - x[a + 2], e22 = x[d + 2] - x[a + 2];
      const b0 = B[o], b1 = B[o + 1], b2 = B[o + 2], b3 = B[o + 3], b4 = B[o + 4], b5 = B[o + 5], b6 = B[o + 6], b7 = B[o + 7], b8 = B[o + 8];
      const f0 = e00 * b0 + e01 * b3 + e02 * b6, f1 = e00 * b1 + e01 * b4 + e02 * b7, f2 = e00 * b2 + e01 * b5 + e02 * b8;
      const f3 = e10 * b0 + e11 * b3 + e12 * b6, f4 = e10 * b1 + e11 * b4 + e12 * b7, f5 = e10 * b2 + e11 * b5 + e12 * b8;
      const f6 = e20 * b0 + e21 * b3 + e22 * b6, f7 = e20 * b1 + e21 * b4 + e22 * b7, f8 = e20 * b2 + e21 * b5 + e22 * b8;
      F[o] = f0; F[o + 1] = f1; F[o + 2] = f2; F[o + 3] = f3; F[o + 4] = f4; F[o + 5] = f5; F[o + 6] = f6; F[o + 7] = f7; F[o + 8] = f8;
      const J = f0 * (f4 * f8 - f5 * f7) - f1 * (f3 * f8 - f5 * f6) + f2 * (f3 * f7 - f4 * f6);
      Je[e] = J;
      const w = V[e] * J / 4;
      nodeJ[a / 3] += w; nodeJ[b / 3] += w; nodeJ[c / 3] += w; nodeJ[d / 3] += w;
    }
    for (let e = 0; e < m; e++) {
      const ia = tets[4 * e], ib = tets[4 * e + 1], ic = tets[4 * e + 2], id = tets[4 * e + 3], o = 9 * e, s = 6 * e;
      const Jbar = 0.25 * (nodeJ[ia] / nodeV[ia] + nodeJ[ib] / nodeV[ib] + nodeJ[ic] / nodeV[ic] + nodeJ[id] / nodeV[id]);
      const f0 = F[o], f1 = F[o + 1], f2 = F[o + 2], f3 = F[o + 3], f4 = F[o + 4], f5 = F[o + 5], f6 = F[o + 6], f7 = F[o + 7], f8 = F[o + 8];
      const J = Math.max(Je[e], 0.05);  // a collapsing element keeps a finite stress instead of ending the run
      // cofactor of F = J F^-T
      const c0 = f4 * f8 - f5 * f7, c1 = f5 * f6 - f3 * f8, c2 = f3 * f7 - f4 * f6;
      const c3 = f2 * f7 - f1 * f8, c4 = f0 * f8 - f2 * f6, c5 = f1 * f6 - f0 * f7;
      const c6 = f1 * f5 - f2 * f4, c7 = f2 * f3 - f0 * f5, c8 = f0 * f4 - f1 * f3;
      const iJ = 1 / J;
      // C^-1 = F^-1 F^-T = (cof^T cof) / J^2
      const k2 = iJ * iJ;
      const i00 = (c0 * c0 + c3 * c3 + c6 * c6) * k2, i11 = (c1 * c1 + c4 * c4 + c7 * c7) * k2, i22 = (c2 * c2 + c5 * c5 + c8 * c8) * k2;
      const i01 = (c0 * c1 + c3 * c4 + c6 * c7) * k2, i02 = (c0 * c2 + c3 * c5 + c6 * c8) * k2, i12 = (c1 * c2 + c4 * c5 + c7 * c8) * k2;
      const I1 = f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3 + f4 * f4 + f5 * f5 + f6 * f6 + f7 * f7 + f8 * f8;
      const sc = mu * Math.pow(J, -2 / 3), t = I1 / 3;
      // isochoric second Piola-Kirchhoff stress S, and its hereditary part h (Simo's recursion)
      const s0 = sc * (1 - t * i00), s1 = sc * (1 - t * i11), s2 = sc * (1 - t * i22), s3 = -sc * t * i01, s4 = -sc * t * i02, s5 = -sc * t * i12;
      const q0 = decay * Q[s] + alpha * (s0 - S[s]), q1 = decay * Q[s + 1] + alpha * (s1 - S[s + 1]), q2 = decay * Q[s + 2] + alpha * (s2 - S[s + 2]);
      const q3 = decay * Q[s + 3] + alpha * (s3 - S[s + 3]), q4 = decay * Q[s + 4] + alpha * (s4 - S[s + 4]), q5 = decay * Q[s + 5] + alpha * (s5 - S[s + 5]);
      Q[s] = q0; Q[s + 1] = q1; Q[s + 2] = q2; Q[s + 3] = q3; Q[s + 4] = q4; Q[s + 5] = q5;
      S[s] = s0; S[s + 1] = s1; S[s + 2] = s2; S[s + 3] = s3; S[s + 4] = s4; S[s + 5] = s5;
      const gi = this.g, ge = 1 - gi;  // stress = (1 - g) S + g h: all of S at once, relaxing to (1 - g) S
      const a0 = ge * s0 + gi * q0, a1 = ge * s1 + gi * q1, a2 = ge * s2 + gi * q2, a3 = ge * s3 + gi * q3, a4 = ge * s4 + gi * q4, a5 = ge * s5 + gi * q5;
      const rate = (Je[e] - Jold[e]) / (dt * J);
      Jold[e] = Je[e];
      const p = kappa * ((1 - blend) * (Jbar - 1) + blend * (Je[e] - 1)) + bulkViscosity * rate;
      // P = F S + p cof(F)
      const P0 = f0 * a0 + f1 * a3 + f2 * a4 + p * c0, P1 = f0 * a3 + f1 * a1 + f2 * a5 + p * c1, P2 = f0 * a4 + f1 * a5 + f2 * a2 + p * c2;
      const P3 = f3 * a0 + f4 * a3 + f5 * a4 + p * c3, P4 = f3 * a3 + f4 * a1 + f5 * a5 + p * c4, P5 = f3 * a4 + f4 * a5 + f5 * a2 + p * c5;
      const P6 = f6 * a0 + f7 * a3 + f8 * a4 + p * c6, P7 = f6 * a3 + f7 * a1 + f8 * a5 + p * c7, P8 = f6 * a4 + f7 * a5 + f8 * a2 + p * c8;
      // forces on nodes b, c, d: -V P B^T columns; node a takes the balance
      const v = V[e], b0 = B[o], b1 = B[o + 1], b2 = B[o + 2], b3 = B[o + 3], b4 = B[o + 4], b5 = B[o + 5], b6 = B[o + 6], b7 = B[o + 7], b8 = B[o + 8];
      const h00 = -v * (P0 * b0 + P1 * b1 + P2 * b2), h01 = -v * (P0 * b3 + P1 * b4 + P2 * b5), h02 = -v * (P0 * b6 + P1 * b7 + P2 * b8);
      const h10 = -v * (P3 * b0 + P4 * b1 + P5 * b2), h11 = -v * (P3 * b3 + P4 * b4 + P5 * b5), h12 = -v * (P3 * b6 + P4 * b7 + P5 * b8);
      const h20 = -v * (P6 * b0 + P7 * b1 + P8 * b2), h21 = -v * (P6 * b3 + P7 * b4 + P8 * b5), h22 = -v * (P6 * b6 + P7 * b7 + P8 * b8);
      const A = 3 * ia, Bn = 3 * ib, C = 3 * ic, D = 3 * id;
      f[Bn] += h00; f[Bn + 1] += h10; f[Bn + 2] += h20;
      f[C] += h01; f[C + 1] += h11; f[C + 2] += h21;
      f[D] += h02; f[D + 1] += h12; f[D + 2] += h22;
      f[A] -= h00 + h01 + h02; f[A + 1] -= h10 + h11 + h12; f[A + 2] -= h20 + h21 + h22;
    }
  }
}

function rotation(q) {
  const [w, x, y, z] = q;
  return [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
          2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
          2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
}

function addAt(o, r, F) {
  for (let k = 0; k < 3; k++) o.f[k] += F[k];
  o.tq[0] += r[1] * F[2] - r[2] * F[1];
  o.tq[1] += r[2] * F[0] - r[0] * F[2];
  o.tq[2] += r[0] * F[1] - r[1] * F[0];
}

/* What the page lets you declare. ``declaredCard``: stiffness by Young's modulus ("E"), shear modulus ("G") or a
 * datasheet's 100% modulus ("M100": the engineering stress at twice the length, 1.75 G for an incompressible
 * Neo-Hookean solid), all in Pa, with Poisson's ratio nu. */
function declaredCard({by, value, nu, g = 0, tau = 1}) {
  const G = {E: value / (2 * (1 + nu)), G: value, M100: value / 1.75}[by];
  if (G === undefined) throw new Error(`cannot declare stiffness by ${by}`);
  const E = 2 * G * (1 + nu);
  return {E, G, K: E / (3 * (1 - 2 * nu)), nu, g, tau};
}

/* The cylinder's mesh stretched ``radial`` times across and ``axial`` times along its axis (z). */
function scaledMesh(mesh, radial, axial) {
  return {tets: mesh.tets, points: mesh.points.map((v, i) => v * (i % 3 === 2 ? axial : radial))};
}

function cylinderMass(density, radius, height) { return density * Math.PI * radius * radius * height; }

function objectMass(object, density) {
  return object.shape === "sphere" ? density * 4 / 3 * Math.PI * object.radius ** 3 : cylinderMass(density, object.radius, object.height);
}

if (typeof module !== "undefined") module.exports = {DropSim, declaredCard, scaledMesh, cylinderMass, objectMass};
