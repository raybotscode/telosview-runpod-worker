// Validate the bundle adjuster on a synthetic drifted-chain problem.
import { bundleAdjust } from '../../src/sfm/ba.js';
import { rodrigues, m3mul, m3t, m3mulv, so3Log, makeRng }
  from '../../src/sfm/geometry.js';

const rng = makeRng(4242);
const gauss = () => {
  const u = Math.max(1e-9, rng()), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

// ---- ground truth: 25-camera forward walk, 800 points, distorted lens ----
const NC = 25, NP = 800;
const fGT = 620, cx = 320, cy = 180, k1GT = -0.12, k2GT = 0.02;
const gtCams = [];
for (let i = 0; i < NC; i++) {
  const yaw = 0.02 * i, walk = 0.25 * i;
  const R = rodrigues([0, yaw, 0]);
  const C = [0.3 * Math.sin(i * 0.4), 0.05 * Math.sin(i * 0.9), walk]; // camera center
  const t = m3mulv(R, C).map((v) => -v);
  gtCams.push({ R: Array.from(R), t });
}
const gtPts = [];
for (let i = 0; i < NP; i++) {
  gtPts.push([(rng() - 0.5) * 8, (rng() - 0.5) * 4, 2 + rng() * 10]);
}
const projectGT = (cam, p) => {
  const pc = m3mulv(cam.R, p).map((v, k) => v + cam.t[k]);
  if (pc[2] < 0.2) return null;
  const xp = pc[0] / pc[2], yp = pc[1] / pc[2];
  const r2 = xp * xp + yp * yp;
  if (r2 > 0.4) return null; // stay in frame-ish
  const D = 1 + k1GT * r2 + k2GT * r2 * r2;
  return [fGT * xp * D + cx, fGT * yp * D + cy];
};

// observations (0.4px noise), only for cameras that see the point
const obs = [];
for (let j = 0; j < NP; j++) {
  for (let i = 0; i < NC; i++) {
    const uv = projectGT(gtCams[i], gtPts[j]);
    if (!uv) continue;
    if (rng() < 0.5) continue; // sparsify tracks
    obs.push({ ci: i, pi: j, u: uv[0] + gauss() * 0.4, v: uv[1] + gauss() * 0.4 });
  }
}

// ---- corrupted init: accumulated chain drift + wrong f + no distortion ----
const cams = gtCams.map((c, i) => {
  const drift = i / NC;
  const dw = [0.03 * drift * drift, 0.02 * drift, -0.04 * drift * drift]; // grows along chain
  const dt = [0.05 * drift, 0.15 * drift * drift, 0.05 * drift];          // "dives" like the user saw
  return {
    R: Array.from(m3mul(rodrigues(dw), c.R)),
    t: [c.t[0] + dt[0], c.t[1] + dt[1], c.t[2] + dt[2]],
  };
});
cams[0] = { R: Array.from(gtCams[0].R), t: gtCams[0].t.slice() }; // anchor exact
const points = gtPts.map((p) => [p[0] + gauss() * 0.05, p[1] + gauss() * 0.05, p[2] + gauss() * 0.1]);
const fInit = 660; // ~6.5% focal error

const poseErr = () => {
  let rot = 0, tr = 0;
  for (let i = 0; i < NC; i++) {
    const dR = m3mul(m3t(cams[i].R), gtCams[i].R);
    rot += Math.hypot(...so3Log(dR));
    const Cgt = m3mulv(m3t(gtCams[i].R), gtCams[i].t).map((v) => -v);
    const Ce = m3mulv(m3t(cams[i].R), cams[i].t).map((v) => -v);
    tr += Math.hypot(Cgt[0] - Ce[0], Cgt[1] - Ce[1], Cgt[2] - Ce[2]);
  }
  return { rotDeg: (rot / NC) * 180 / Math.PI, trans: tr / NC };
};

console.log(`obs: ${obs.length}, cams: ${NC}, points: ${NP}`);
const e0 = poseErr();
console.log(`before: rot ${e0.rotDeg.toFixed(3)} deg, trans ${e0.trans.toFixed(4)} (scene depth ~6)`);

const t0 = Date.now();
const res = bundleAdjust(
  { cams, points, obs, f: fInit, cx, cy },
  { maxIters: 40, log: (m) => console.log(m) });
console.log(`BA took ${((Date.now() - t0) / 1000).toFixed(1)}s, iters ${res.iters}`);
console.log(`rms: ${res.rmsBefore.toFixed(2)} -> ${res.rmsAfter.toFixed(3)} px`);
console.log(`f: ${res.f.toFixed(1)} (GT ${fGT}), k1: ${res.k1.toFixed(4)} (GT ${k1GT}), k2: ${res.k2.toFixed(4)} (GT ${k2GT})`);
const e1 = poseErr();
console.log(`after: rot ${e1.rotDeg.toFixed(4)} deg, trans ${e1.trans.toFixed(5)}`);

const pass =
  res.rmsAfter < 0.6 &&
  Math.abs(res.f - fGT) / fGT < 0.01 &&
  Math.abs(res.k1 - k1GT) < 0.02 &&
  e1.rotDeg < 0.05 && e1.trans < 0.01;
console.log(pass ? 'BA TEST PASSED' : 'BA TEST FAILED');
process.exit(pass ? 0 : 1);
