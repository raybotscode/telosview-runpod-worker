// Synthetic validation of the SfM geometry toolbox.
import {
  jacobiEigen, rodrigues, so3Log, m3mulv, m3t, m3mul, m3det,
  estimateE, ransacE, decomposeE, selectPose, triangulateN,
  dltPnP, pnpRansac, refinePose, reprojError, makeRng, solveLinear,
} from '../../src/sfm/geometry.js';

let failures = 0;
const check = (name, cond, detail = '') => {
  console.log(`${cond ? 'PASS' : 'FAIL'} ${name} ${detail}`);
  if (!cond) failures++;
};

const rng = makeRng(999);

// ---- jacobiEigen sanity ----
{
  const A = [4, 1, 0, 1, 3, 1, 0, 1, 2];
  const { vals, vecs } = jacobiEigen(A, 3);
  // check A v = lambda v for each column
  let maxErr = 0;
  for (let j = 0; j < 3; j++) {
    const v = [vecs[j], vecs[3 + j], vecs[6 + j]];
    const Av = m3mulv(Float64Array.from(A), v);
    for (let i = 0; i < 3; i++) maxErr = Math.max(maxErr, Math.abs(Av[i] - vals[j] * v[i]));
  }
  check('jacobiEigen 3x3', maxErr < 1e-9, `err=${maxErr.toExponential(2)}`);
  check('jacobiEigen sorted', vals[0] >= vals[1] && vals[1] >= vals[2]);
}

// ---- solveLinear ----
{
  const A = [2, 1, 0, 1, 3, 1, 0, 1, 4];
  const x0 = [1, -2, 3];
  const b = m3mulv(Float64Array.from(A), x0);
  const x = solveLinear(A, b, 3);
  const err = Math.max(...x.map((v, i) => Math.abs(v - x0[i])));
  check('solveLinear', err < 1e-10, `err=${err.toExponential(2)}`);
}

// ---- rodrigues roundtrip ----
{
  const w = [0.3, -0.7, 0.2];
  const R = rodrigues(w);
  const w2 = so3Log(R);
  const err = Math.max(...w.map((v, i) => Math.abs(v - w2[i])));
  check('rodrigues/so3Log roundtrip', err < 1e-10, `err=${err.toExponential(2)}`);
  check('rodrigues det=1', Math.abs(m3det(R) - 1) < 1e-12);
}

// ---- synthetic two-view scene ----
function makeScene(nPts, noise) {
  const pts = [];
  for (let i = 0; i < nPts; i++) {
    pts.push([(rng() - 0.5) * 2, (rng() - 0.5) * 2, 3 + rng() * 2]);
  }
  const Rgt = rodrigues([0.05, 0.35, -0.03]);
  // camera 2 displaced to the right
  const C2 = [1.0, 0.15, 0.1];
  const tgt = m3mulv(Rgt, C2).map((v) => -v);
  const proj = (R, t, X) => {
    const p = m3mulv(R, X);
    const z = p[2] + t[2];
    return [(p[0] + t[0]) / z + (rng() - 0.5) * noise, (p[1] + t[1]) / z + (rng() - 0.5) * noise];
  };
  const I = Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const x1s = pts.map((X) => proj(I, [0, 0, 0], X));
  const x2s = pts.map((X) => proj(Rgt, tgt, X));
  return { pts, Rgt, tgt, x1s, x2s };
}

{
  const { pts, Rgt, tgt, x1s, x2s } = makeScene(120, 0.001);
  const res = ransacE(x1s, x2s, 0.004, rng);
  check('ransacE finds inliers', res && res.inliers.length > 100,
    `inliers=${res ? res.inliers.length : 0}`);
  const cands = decomposeE(res.U, res.V);
  const pose = selectPose(cands, x1s, x2s, res.inliers);
  check('selectPose cheirality', pose.goodFrac > 0.95, `frac=${pose.goodFrac.toFixed(2)}`);
  // compare rotation with ground truth
  const dR = m3mul(m3t(pose.R), Rgt);
  const angErr = Math.hypot(...so3Log(dR));
  check('recovered rotation', angErr < 0.02, `angErr=${(angErr * 180 / Math.PI).toFixed(3)}deg`);
  // translation direction (up to scale)
  const tn = Math.hypot(...tgt);
  const dot = (pose.t[0] * tgt[0] + pose.t[1] * tgt[1] + pose.t[2] * tgt[2]) / tn;
  check('recovered translation dir', Math.abs(Math.abs(dot) - 1) < 0.02, `|cos|=${Math.abs(dot).toFixed(4)}`);

  // triangulation with GT poses
  const cam1 = { R: Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]), t: [0, 0, 0] };
  const cam2 = { R: Rgt, t: tgt };
  let terr = 0;
  for (let i = 0; i < 50; i++) {
    const X = triangulateN([cam1, cam2], [x1s[i], x2s[i]]);
    terr = Math.max(terr, Math.hypot(X[0] - pts[i][0], X[1] - pts[i][1], X[2] - pts[i][2]));
  }
  check('triangulateN', terr < 0.05, `maxErr=${terr.toFixed(4)}`);

  // PnP: recover camera 2 from 3D points + observations
  const pnp = pnpRansac(pts, x2s, 0.01, rng);
  check('pnpRansac inliers', pnp && pnp.inliers.length > 100, `inliers=${pnp ? pnp.inliers.length : 0}`);
  const dR2 = m3mul(m3t(pnp.R), Rgt);
  const angErr2 = Math.hypot(...so3Log(dR2));
  const tErr2 = Math.hypot(pnp.t[0] - tgt[0], pnp.t[1] - tgt[1], pnp.t[2] - tgt[2]);
  check('pnp rotation', angErr2 < 0.01, `angErr=${(angErr2 * 180 / Math.PI).toFixed(3)}deg`);
  check('pnp translation', tErr2 < 0.02, `tErr=${tErr2.toFixed(4)}`);

  // refinePose from a perturbed start
  const Rpert = m3mul(rodrigues([0.03, -0.02, 0.04]), Rgt);
  const tpert = [tgt[0] + 0.05, tgt[1] - 0.04, tgt[2] + 0.03];
  const ref = refinePose(Rpert, tpert, pts, x2s, 15);
  const dR3 = m3mul(m3t(ref.R), Rgt);
  const angErr3 = Math.hypot(...so3Log(dR3));
  const tErr3 = Math.hypot(ref.t[0] - tgt[0], ref.t[1] - tgt[1], ref.t[2] - tgt[2]);
  check('refinePose rotation', angErr3 < 0.005, `angErr=${(angErr3 * 180 / Math.PI).toFixed(3)}deg`);
  check('refinePose translation', tErr3 < 0.01, `tErr=${tErr3.toFixed(4)}`);
}

// ---- with outliers ----
{
  const { Rgt, tgt, x1s, x2s } = makeScene(150, 0.0015);
  // corrupt 30% of matches
  for (let i = 0; i < 45; i++) {
    x2s[i * 3] = [(rng() - 0.5) * 0.8, (rng() - 0.5) * 0.8];
  }
  const res = ransacE(x1s, x2s, 0.004, rng);
  check('ransacE w/ 30% outliers', res && res.inliers.length > 90 && res.inliers.length < 120,
    `inliers=${res ? res.inliers.length : 0}`);
  if (res) {
    const cands = decomposeE(res.U, res.V);
    const pose = selectPose(cands, x1s, x2s, res.inliers);
    const dR = m3mul(m3t(pose.R), Rgt);
    const angErr = Math.hypot(...so3Log(dR));
    check('rotation w/ outliers', angErr < 0.03, `angErr=${(angErr * 180 / Math.PI).toFixed(3)}deg`);
  }
}

console.log(failures ? `\n${failures} FAILURES` : '\nall geometry tests passed');
process.exit(failures ? 1 : 0);
