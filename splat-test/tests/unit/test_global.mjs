// Validate global.js: rotation averaging + global positioning on a noisy
// synthetic orbit (ground truth known).
import { rotationAveraging, globalPositions, projectSO3 }
  from '../../src/sfm/global.js';
import { rodrigues, m3mul, m3t, m3mulv, so3Log, makeRng }
  from '../../src/sfm/geometry.js';

const rng = makeRng(1234);
const gauss = () => {
  const u = Math.max(1e-9, rng()), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

const N = 60;
const gt = [];
for (let i = 0; i < N; i++) {
  const a = (i / N) * Math.PI * 1.2; // 216-degree orbit
  const C = [Math.cos(a) * 5, 0.2 * Math.sin(3 * a), Math.sin(a) * 5];
  const R = m3mul(rodrigues([0.05 * Math.sin(a * 2), -a + Math.PI / 2, 0]), rodrigues([0.1, 0, 0]));
  gt.push({ R: Array.from(R), C });
}

// noisy pairwise edges (neighbors up to +-6, some outliers)
const rotEdges = [], posEdges = [];
for (let i = 0; i < N; i++)
  for (let d = 1; d <= 6; d++) {
    const j = i + d;
    if (j >= N) continue;
    const outlier = rng() < 0.08;
    let Rij = m3mul(gt[j].R, m3t(gt[i].R));
    let noise = rodrigues([gauss() * 0.01, gauss() * 0.01, gauss() * 0.01]);
    if (outlier) noise = rodrigues([gauss() * 0.5, gauss() * 0.5, gauss() * 0.5]);
    Rij = m3mul(noise, Rij);
    rotEdges.push({ i, j, R: Rij, w: 1 });
    let v = [gt[i].C[0] - gt[j].C[0], gt[i].C[1] - gt[j].C[1], gt[i].C[2] - gt[j].C[2]];
    let len = Math.hypot(...v);
    v = v.map((x) => x / len);
    const nz = outlier ? 0.5 : 0.01;
    v = [v[0] + gauss() * nz, v[1] + gauss() * nz, v[2] + gauss() * nz];
    len = Math.hypot(...v);
    posEdges.push({ i, j, d: v.map((x) => x / len), w: 1 });
  }

const { R } = rotationAveraging(N, rotEdges, console.log);
// rotation error after aligning gauge: err_i = angle(R_i (R_gt_i A)^-1) with A the anchor offset
let A = null, rotErr = 0;
for (let i = 0; i < N; i++) {
  if (!R[i]) { console.log('cam', i, 'missing'); continue; }
  if (!A) A = m3mul(m3t(gt[i].R), R[i]); // world-frame gauge: R_i = R_gt_i * A
  const pred = m3mul(gt[i].R, A);
  rotErr += Math.hypot(...so3Log(m3mul(R[i], m3t(pred))));
}
rotErr = (rotErr / N) * 180 / Math.PI;
console.log('mean rotation error:', rotErr.toFixed(3), 'deg');

// positions: directions in WORLD frame use gauge A too: d_world_solver = A^T d_world_gt
const posEdges2 = posEdges.map((e) => ({ ...e, d: Array.from(m3mulv(m3t(A), e.d)) }));
const C = globalPositions(N, posEdges2, 0, console.log);
// similarity-align C to gt centers (scale+R already consistent up to scale + translation)
const gtC = gt.map((g) => m3mulv(m3t(A), g.C)); // rotate gt into solver frame
let mC = [0, 0, 0], mG = [0, 0, 0];
for (let i = 0; i < N; i++) for (let k = 0; k < 3; k++) { mC[k] += C[i][k] / N; mG[k] += gtC[i][k] / N; }
let num = 0, den = 0;
for (let i = 0; i < N; i++) for (let k = 0; k < 3; k++) {
  num += (gtC[i][k] - mG[k]) * (C[i][k] - mC[k]);
  den += (C[i][k] - mC[k]) ** 2;
}
const s = num / den;
let posErr = 0, span = 0;
for (let i = 0; i < N; i++) {
  let e2 = 0, g2 = 0;
  for (let k = 0; k < 3; k++) {
    e2 += ((C[i][k] - mC[k]) * s - (gtC[i][k] - mG[k])) ** 2;
    g2 += (gtC[i][k] - mG[k]) ** 2;
  }
  posErr += Math.sqrt(e2); span = Math.max(span, Math.sqrt(g2));
}
posErr /= N;
console.log('mean position error:', posErr.toFixed(4), '(scene span ~' + (2 * span).toFixed(1) + ') =',
            (100 * posErr / (2 * span)).toFixed(2) + '%');
// thresholds = the synthetic noise floor (1deg rotation noise per edge,
// 0.6deg direction noise): the solver can't beat its inputs
const pass = rotErr < 1.0 && posErr / (2 * span) < 0.015;
console.log(pass ? 'GLOBAL INIT TEST PASSED' : 'GLOBAL INIT TEST FAILED');
process.exit(pass ? 0 : 1);
