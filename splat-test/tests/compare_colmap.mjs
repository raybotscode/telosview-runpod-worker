// Compare our SfM poses (JSON from the browser) against COLMAP ground truth
// (images.bin). Similarity-aligns camera centers (Umeyama) and reports
// per-camera position error + height profiles.
import { readFileSync } from 'fs';

const [,, binPath, oursPath] = process.argv;

// ---- parse COLMAP images.bin ----
function parseImagesBin(path) {
  const buf = readFileSync(path);
  let off = 0;
  const u64 = () => { const v = buf.readBigUInt64LE(off); off += 8; return Number(v); };
  const u32 = () => { const v = buf.readUInt32LE(off); off += 4; return v; };
  const f64 = () => { const v = buf.readDoubleLE(off); off += 8; return v; };
  const str = () => { let s = ''; while (buf[off] !== 0) s += String.fromCharCode(buf[off++]); off++; return s; };
  const n = u64();
  const out = new Map();
  for (let i = 0; i < n; i++) {
    u32(); // image id
    const qw = f64(), qx = f64(), qy = f64(), qz = f64();
    const tx = f64(), ty = f64(), tz = f64();
    u32(); // camera id
    const name = str();
    const np = u64();
    off += np * 24; // 2 doubles + int64 per 2D point
    // quat (w,x,y,z) -> R (world->cam), center = -R^T t
    const R = [
      1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy),
      2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx),
      2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy),
    ];
    const C = [
      -(R[0]*tx + R[3]*ty + R[6]*tz),
      -(R[1]*tx + R[4]*ty + R[7]*tz),
      -(R[2]*tx + R[5]*ty + R[8]*tz),
    ];
    out.set(name, { C, R });
  }
  return out;
}

// ---- parse COLMAP images.txt (text export) ----
function parseImagesTxt(path) {
  const out = new Map();
  const lines = readFileSync(path, 'utf8').split('\n');
  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i].trim();
    if (!ln || ln.startsWith('#')) continue;
    const p = ln.split(/\s+/);
    if (p.length < 10) continue;
    // IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME  (followed by a points2D line)
    const [qw, qx, qy, qz, tx, ty, tz] = p.slice(1, 8).map(Number);
    if ([qw, qx, qy, qz, tx, ty, tz].some(Number.isNaN)) continue;
    const name = p[9];
    const R = [
      1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy),
      2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx),
      2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy),
    ];
    const C = [
      -(R[0]*tx + R[3]*ty + R[6]*tz),
      -(R[1]*tx + R[4]*ty + R[7]*tz),
      -(R[2]*tx + R[5]*ty + R[8]*tz),
    ];
    out.set(name, { C, R });
    i++; // skip the points2D line
  }
  return out;
}

const gt = binPath.endsWith('.txt') ? parseImagesTxt(binPath) : parseImagesBin(binPath);
const ours = JSON.parse(readFileSync(oursPath, 'utf8'));

// our center = -R^T t
const items = [];
for (const c of ours.cams) {
  const g = gt.get(c.name);
  if (!g) continue;
  const R = c.R, t = c.t;
  const C = [
    -(R[0]*t[0] + R[3]*t[1] + R[6]*t[2]),
    -(R[1]*t[0] + R[4]*t[1] + R[7]*t[2]),
    -(R[2]*t[0] + R[5]*t[1] + R[8]*t[2]),
  ];
  items.push({ name: c.name, ours: C, gt: g.C });
}
if (items.length < 3) { console.error(`only ${items.length} matched names`); process.exit(1); }
items.sort((a, b) => a.name.localeCompare(b.name));

// ---- Umeyama similarity alignment ours -> gt ----
const N = items.length;
const mean = (pts) => pts.reduce((m, p) => [m[0]+p[0]/N, m[1]+p[1]/N, m[2]+p[2]/N], [0,0,0]);
const muA = mean(items.map(i => i.ours)), muB = mean(items.map(i => i.gt));
let H = [[0,0,0],[0,0,0],[0,0,0]], varA = 0;
for (const it of items) {
  const a = it.ours.map((v, k) => v - muA[k]), b = it.gt.map((v, k) => v - muB[k]);
  varA += a[0]*a[0] + a[1]*a[1] + a[2]*a[2];
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) H[r][c] += b[r] * a[c];
}
// SVD of 3x3 via Jacobi on H^T H (good enough here)
function svd3(M) {
  // returns {U, S, V} with M = U diag(S) V^T ; use eigen of M^T M for V, M M^T for U
  const mt = (A) => A[0].map((_, c) => A.map(r => r[c]));
  const mm = (A, B) => A.map((r) => B[0].map((_, c) => r.reduce((s, v, k) => s + v * B[k][c], 0)));
  const jacobiEig = (S) => {
    let V = [[1,0,0],[0,1,0],[0,0,1]];
    let A = S.map(r => r.slice());
    for (let sweep = 0; sweep < 50; sweep++) {
      let offd = Math.abs(A[0][1]) + Math.abs(A[0][2]) + Math.abs(A[1][2]);
      if (offd < 1e-15) break;
      for (let p = 0; p < 2; p++) for (let q = p + 1; q < 3; q++) {
        if (Math.abs(A[p][q]) < 1e-18) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const tsig = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(tsig * tsig + 1), s = tsig * c;
        const G = [[1,0,0],[0,1,0],[0,0,1]];
        G[p][p] = c; G[q][q] = c; G[p][q] = s; G[q][p] = -s;
        A = mm(mm(mt(G), A), G);
        V = mm(V, G);
      }
    }
    return { vals: [A[0][0], A[1][1], A[2][2]], V };
  };
  const HtH = mm(mt(M), M);
  const { vals, V } = jacobiEig(HtH);
  // sort desc
  const idx = [0,1,2].sort((a,b) => vals[b] - vals[a]);
  const Vs = [0,1,2].map(r => idx.map(c => V[r][c]));
  const S = idx.map(i => Math.sqrt(Math.max(0, vals[i])));
  const U = [0,1,2].map(() => [0,0,0]);
  for (let c = 0; c < 3; c++) {
    const v = [Vs[0][c], Vs[1][c], Vs[2][c]];
    const mv = [0,1,2].map(r => M[r][0]*v[0] + M[r][1]*v[1] + M[r][2]*v[2]);
    const s = S[c] > 1e-12 ? S[c] : 1;
    for (let r = 0; r < 3; r++) U[r][c] = mv[r] / s;
  }
  return { U, S, V: Vs };
}
const { U, S, V } = svd3(H);
const det = (M) => M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1]) - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0]) + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]);
const D = [[1,0,0],[0,1,0],[0,0,det(U)*det(V) < 0 ? -1 : 1]];
const mm = (A, B) => A.map((r) => B[0].map((_, c) => r.reduce((s, v, k) => s + v * B[k][c], 0)));
const mt = (A) => A[0].map((_, c) => A.map(r => r[c]));
const Rot = mm(mm(U, D), mt(V));
const scale = (S[0] + S[1] + D[2][2] * S[2]) / varA;

const xf = (p) => {
  const a = p.map((v, k) => v - muA[k]);
  return [0,1,2].map(r => scale * (Rot[r][0]*a[0] + Rot[r][1]*a[1] + Rot[r][2]*a[2]) + muB[r]);
};

// ---- report ----
let se = 0, errs = [];
for (const it of items) {
  const p = xf(it.ours);
  const e = Math.hypot(p[0]-it.gt[0], p[1]-it.gt[1], p[2]-it.gt[2]);
  errs.push({ name: it.name, e });
  se += e * e;
}
// gt path length for context
let plen = 0;
for (let i = 1; i < items.length; i++)
  plen += Math.hypot(...[0,1,2].map(k => items[i].gt[k] - items[i-1].gt[k]));
const rms = Math.sqrt(se / N);
errs.sort((a, b) => a.e - b.e);
console.log(`matched ${N} cams, GT path length ${plen.toFixed(1)}`);
console.log(`ATE rms ${rms.toFixed(3)}  median ${errs[N>>1].e.toFixed(3)}  max ${errs[N-1].e.toFixed(3)}  (${(100*rms/plen).toFixed(2)}% of path)`);
console.log('worst 5:', errs.slice(-5).map(x => `${x.name}:${x.e.toFixed(2)}`).join(' '));
// per-camera error along the sequence
const seq = items.map(it => {
  const p = xf(it.ours);
  return Math.hypot(p[0]-it.gt[0], p[1]-it.gt[1], p[2]-it.gt[2]);
});
console.log('error along sequence (per cam):');
console.log(seq.map(e => e.toFixed(2)).join(' '));

// dump aligned centers for plotting / axis-wise analysis
const dumpPath = process.argv[4];
if (dumpPath) {
  const rows = items.map(it => ({ name: it.name, gt: it.gt.map(v => +v.toFixed(4)), ours: xf(it.ours).map(v => +v.toFixed(4)) }));
  const { writeFileSync } = await import('fs');
  writeFileSync(dumpPath, JSON.stringify(rows, null, 1));
  console.log(`aligned centers dumped to ${dumpPath}`);
}
