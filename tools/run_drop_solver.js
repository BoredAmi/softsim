/* Run tools/drop_solver.js from a JSON job on stdin, for tools/check_drop_solver.py and tests/test_drop_solver.py:
 * {mesh, material, object, friction, options, duration, sample} -> JSON {t, z, tilt, force, dt, wall, failed} on stdout;
 * z [m] is the object's centre height, sampled every ``sample`` s. */
const {DropSim} = require("./drop_solver.js");
let text = "";
process.stdin.on("data", d => text += d).on("end", () => {
  const job = JSON.parse(text);
  const sim = new DropSim(job.mesh, job.material, job.object, job.friction, job.options);
  const out = {t: [], z: [], force: [], dt: sim.dt, failed: false};
  const start = Date.now();
  let next = 0;
  while (sim.t < job.duration && !sim.failed) {
    sim.step();
    if (sim.t >= next) {
      out.t.push(sim.t);
      out.z.push(sim.object ? sim.object.p[2] : sim.x[2]);
      out.force.push(sim.contactForce ?? 0);
      next += job.sample;
    }
  }
  out.wall = (Date.now() - start) / 1e3;
  out.failed = sim.failed || !sim.x.every(Number.isFinite);
  console.log(JSON.stringify(out));
});
