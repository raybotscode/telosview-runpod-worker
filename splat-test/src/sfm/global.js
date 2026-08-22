// global.js — GLOMAP-style global SfM initialization.
//
// Instead of growing the reconstruction incrementally (which locks into
// whatever basin early drift selects — the "dives into the ground" failure
// class), solve all camera rotations at once by ROTATION AVERAGING over
// pairwise relative rotations, then all camera positions at once from
// pairwise translation DIRECTIONS (cross-product least squares). The result
// only needs to land in the right basin — the existing bundle-adjustment
// stack polishes it.
//
// Conventions match the rest of the pipeline: R is world->cam (row-major
// 3x3), camera center C = -R^T t, i.e. t = -R C. A pair edge (i, j) carries
// the relative pose of cam j in cam i's frame: x_j = Rij x_i + tij,
// so globally Rij = Rj Ri^T and tij = Rj (Ci - Cj).

import { m3mul, m3t, m3mulv, so3Log, rodrigues, jacobiEigen } from './geometry.js';

/** Dense Cholesky factorization in place; returns false on failure. */
function cholFactor(A, dim) {
  for (let c = 0; c < dim; c++) {
    let d = A[c * dim + c];
    for (let k = 0; k < c; k++) d -= A[c * dim + k] * A[c * dim + k];
    if (d < 1e-14) return false;
    const s = Math.sqrt(d);
    A[c * dim + c] = s;
    for (let r = c + 1; r < dim; r++) {
      let v = A[r * dim + c];
      for (let k = 0; k < c; k++) v -= A[r * dim + k] * A[c * dim + k];
      A[r * dim + c] = v / s;
    }
  }
  return true;
}

/** Solve with a cholFactor'd matrix. */
function cholBack(L, b, dim) {
  const y = new Float64Array(dim);
  for (let r = 0; r < dim; r++) {
    let v = b[r];
    for (let k = 0; k < r; k++) v -= L[r * dim + k] * y[k];
    y[r] = v / L[r * dim + r];
  }
  const x = new Float64Array(dim);
  for (let r = dim - 1; r >= 0; r--) {
    let v = y[r];
    for (let k = r + 1; k < dim; k++) v -= L[k * dim + r] * x[k];
    x[r] = v / L[r * dim + r];
  }
  return x;
}

/** Dense Cholesky solve (one-shot convenience). Returns x or null. */
function cholSolveDense(Ain, b, dim) {
  const A = Float64Array.from(Ain);
  if (!cholFactor(A, dim)) return null;
  return cholBack(A, b, dim);
}

/** Nearest rotation to a 3x3 matrix (polar decomposition via eigen of M^T M). */
export function projectSO3(M) {
  const S = m3mul(m3t(M), M);
  const { vals, vecs } = jacobiEigen(S, 3);
  // R = M V diag(1/sqrt(l)) V^T  (flip last axis if det < 0)
  const inv = [1 / Math.sqrt(Math.max(1e-18, vals[0])),
               1 / Math.sqrt(Math.max(1e-18, vals[1])),
               1 / Math.sqrt(Math.max(1e-18, vals[2]))];
  const det = (A) =>
    A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
  if (det(M) < 0) inv[2] = -inv[2];
  // V diag(inv) V^T
  const W = new Float64Array(9);
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += vecs[r * 3 + k] * inv[k] * vecs[c * 3 + k];
      W[r * 3 + c] = s;
    }
  return m3mul(M, W);
}

/** Largest connected component of an edge list over n nodes. */
function largestComponent(n, edges) {
  const parent = new Int32Array(n);
  for (let i = 0; i < n; i++) parent[i] = i;
  const find = (x) => { let r = x; while (parent[r] !== r) r = parent[r]; while (parent[x] !== r) { const nx = parent[x]; parent[x] = r; x = nx; } return r; };
  for (const e of edges) { const a = find(e.i), b = find(e.j); if (a !== b) parent[a] = b; }
  const count = new Map();
  for (const e of edges) { const r = find(e.i); count.set(r, (count.get(r) || 0) + 1); }
  let bestRoot = -1, bestCount = -1;
  for (const [r, c] of count) if (c > bestCount) { bestCount = c; bestRoot = r; }
  const inComp = new Uint8Array(n);
  if (bestRoot >= 0)
    for (let i = 0; i < n; i++) if (find(i) === bestRoot) inComp[i] = 1;
  return inComp;
}

/** Robust rotation averaging (chordal IRLS, Jacobi sweeps).
 *  edges: [{ i, j, R (relative, x_j = R x_i), w }].
 *  Returns { R: Array(n) of world->cam or null, inComp: Uint8Array }. */
export function rotationAveraging(n, edges, log = () => {}) {
  const inComp = largestComponent(n, edges);
  let active = edges.filter((e) => inComp[e.i] && inComp[e.j]);
  if (!active.length) return null;
  const R = new Array(n).fill(null);
  let anchor = -1;

  // Outlier edges (wrong relative rotations that survived the E-gate) corrupt
  // both the spanning-tree init and the chordal L2 mean; solve -> drop
  // high-residual edges -> re-solve.
  for (let phase = 0; phase < 3; phase++) {
    const adj = Array.from({ length: n }, () => []);
    for (const e of active) {
      adj[e.i].push({ o: e.j, R: e.R, fwd: true, w: e.w ?? 1 });
      adj[e.j].push({ o: e.i, R: e.R, fwd: false, w: e.w ?? 1 });
    }
    if (phase === 0) {
      // anchor = highest-degree node; BFS spanning tree preferring strong edges
      let bestDeg = -1;
      for (let i = 0; i < n; i++) if (adj[i].length > bestDeg) { bestDeg = adj[i].length; anchor = i; }
      if (anchor < 0) return null;
      for (const a of adj) a.sort((x, y) => y.w - x.w);
      R[anchor] = Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]);
      const queue = [anchor];
      while (queue.length) {
        const i = queue.shift();
        for (const e of adj[i]) {
          if (R[e.o]) continue;
          R[e.o] = e.fwd ? m3mul(e.R, R[i]) : m3mul(m3t(e.R), R[i]);
          queue.push(e.o);
        }
      }
    }
    // Global L1-IRLS in the Lie algebra (Chatterjee-Govindaraju style):
    // per iteration linearize the edge residuals r_ij = log(R_j^T R_ij R_i)
    // as r + di - dj and solve one global sparse least-squares for all camera
    // updates at once. Converges from a spanning-tree init even with outlier
    // edges; local Jacobi relaxation does not.
    {
      const nodeIdx = new Int32Array(n).fill(-1);
      const nodes = [];
      for (let i = 0; i < n; i++) if (R[i]) { nodeIdx[i] = nodes.length; nodes.push(i); }
      const dim = 3 * nodes.length;
      for (let iter = 0; iter < 15; iter++) {
        const H = new Float64Array(dim * dim);
        const g = new Float64Array(dim);
        let maxR = 0;
        for (const e of active) {
          if (!R[e.i] || !R[e.j]) continue;
          const r = so3Log(m3mul(m3t(R[e.j]), m3mul(e.R, R[e.i])));
          const rn = Math.hypot(...r);
          maxR = Math.max(maxR, rn);
          // L2 first iteration, then L1-style reweighting
          const w = (e.w ?? 1) * (iter === 0 ? 1 : 1 / Math.max(0.02, rn));
          const bi = 3 * nodeIdx[e.i], bj = 3 * nodeIdx[e.j];
          for (let k = 0; k < 3; k++) {
            H[(bi + k) * dim + bi + k] += w;
            H[(bj + k) * dim + bj + k] += w;
            H[(bi + k) * dim + bj + k] -= w;
            H[(bj + k) * dim + bi + k] -= w;
            g[bi + k] -= w * r[k];
            g[bj + k] += w * r[k];
          }
        }
        const ba = 3 * nodeIdx[anchor];
        for (let k = 0; k < 3; k++) H[(ba + k) * dim + ba + k] += 1e6;
        for (let k = 0; k < dim; k++) H[k * dim + k] += 1e-9;
        const delta = cholSolveDense(H, g, dim);
        if (!delta) break;
        let moved = 0;
        for (let kk = 0; kk < nodes.length; kk++) {
          const i = nodes[kk];
          const d = [delta[3 * kk], delta[3 * kk + 1], delta[3 * kk + 2]];
          moved = Math.max(moved, Math.hypot(...d));
          R[i] = m3mul(R[i], rodrigues(d));
        }
        if (moved < 1e-6) break;
      }
    }
    // drop edges whose relative rotation disagrees with the solution
    const res = active.map((e) =>
      (R[e.i] && R[e.j]) ? Math.hypot(...so3Log(m3mul(m3mul(e.R, R[e.i]), m3t(R[e.j])))) : 1e9);
    const sorted = [...res].sort((a, b) => a - b);
    const med = sorted[sorted.length >> 1];
    const TH = Math.max(0.09, 4 * med); // >= ~5 deg
    const kept = active.filter((_, k) => res[k] <= TH);
    if (kept.length === active.length) break;
    log(`  rotation averaging: dropped ${active.length - kept.length}/${active.length} edges > ${(TH * 180 / Math.PI).toFixed(1)} deg (phase ${phase + 1})`);
    active = kept;
  }
  return { R, inComp, anchor, edgesUsed: active };
}

/** Global positioning from pairwise direction constraints.
 *  edges: [{ i, j, d (unit world direction of Ci - Cj), w }].
 *  Solves min sum w * ||cross(d, Ci - Cj)||^2 with anchor C=0 and the scale
 *  fixed by the strongest edge. IRLS for robustness. Returns Array(n) of C
 *  (null outside the component). */
export function globalPositions(n, edges, anchorIdx, log = () => {}) {
  const idx = new Int32Array(n).fill(-1);
  const nodes = [];
  for (const e of edges) {
    for (const v of [e.i, e.j]) if (idx[v] < 0) { idx[v] = nodes.length; nodes.push(v); }
  }
  const m = nodes.length;
  if (m < 3) return null;
  const N = 3 * m;
  const anchor = idx[anchorIdx] >= 0 ? idx[anchorIdx] : 0;

  // min sum w || Ci - Cj - s_ij d_ij ||^2  with per-edge scales s_ij >= 1.
  // The naive cross-product LS collapses (all-baselines-zero is a global
  // minimum); the s >= 1 floor fixes the gauge scale AND prevents collapse
  // (Govindaraju's linear translation registration). Alternate: closed-form
  // s given C, then one linear solve for C given s. Robust IRLS on top.
  const M2 = edges.length;
  const s = new Float64Array(M2).fill(1);
  let irlsW = edges.map((e) => e.w ?? 1);
  let C = null;
  for (let round = 0; round < 20; round++) {
    const H = new Float64Array(N * N);
    const g = new Float64Array(N);
    for (let ei = 0; ei < M2; ei++) {
      const e = edges[ei];
      const w = irlsW[ei];
      const bi = 3 * idx[e.i], bj = 3 * idx[e.j];
      for (let k = 0; k < 3; k++) {
        H[(bi + k) * N + bi + k] += w;
        H[(bj + k) * N + bj + k] += w;
        H[(bi + k) * N + bj + k] -= w;
        H[(bj + k) * N + bi + k] -= w;
        g[bi + k] += w * s[ei] * e.d[k];
        g[bj + k] -= w * s[ei] * e.d[k];
      }
    }
    for (let r = 0; r < 3; r++) H[(3 * anchor + r) * N + 3 * anchor + r] += 1e6;
    for (let r = 0; r < N; r++) H[r * N + r] += 1e-9;
    C = cholSolveDense(H, g, N);
    if (!C) return null;
    // update scales (floor at 1 pins the global scale to the min baseline)
    let sMoved = 0;
    for (let ei = 0; ei < M2; ei++) {
      const e = edges[ei];
      const bi = 3 * idx[e.i], bj = 3 * idx[e.j];
      const dotv = (C[bi] - C[bj]) * e.d[0] + (C[bi + 1] - C[bj + 1]) * e.d[1] + (C[bi + 2] - C[bj + 2]) * e.d[2];
      const ns = Math.max(1, dotv);
      sMoved = Math.max(sMoved, Math.abs(ns - s[ei]) / Math.max(1, s[ei]));
      s[ei] = ns;
    }
    // robust reweight: angular deviation between d and the current baseline
    if (round >= 1) {
      irlsW = edges.map((e, ei) => {
        const bi = 3 * idx[e.i], bj = 3 * idx[e.j];
        const v = [C[bi] - C[bj], C[bi + 1] - C[bj + 1], C[bi + 2] - C[bj + 2]];
        const len = Math.hypot(...v) + 1e-12;
        const cx = [
          e.d[1] * v[2] - e.d[2] * v[1],
          e.d[2] * v[0] - e.d[0] * v[2],
          e.d[0] * v[1] - e.d[1] * v[0],
        ];
        const rel = Math.hypot(...cx) / len; // sin(angle between d and baseline)
        return (e.w ?? 1) / (1 + (rel / 0.05) ** 2);
      });
    }
    if (round > 2 && sMoved < 1e-4) { log(`  positions converged after ${round + 1} rounds`); break; }
  }
  const out = new Array(n).fill(null);
  for (let k = 0; k < m; k++) out[nodes[k]] = [C[3 * k], C[3 * k + 1], C[3 * k + 2]];
  return out;
}

/** GLOMAP-style JOINT global positioning: camera centers AND 3D points from
 *  ray-direction constraints P_k = C_i + s_ik v_ik (v = world-frame unit ray
 *  of the observation, s >= floor), plus camera-camera baseline edges for the
 *  scale gauge. Camera-camera directions alone are hopeless on low-parallax
 *  video (~1 deg between neighbors); the tens of thousands of camera-POINT
 *  rays are what make the problem rigid — this is GLOMAP's actual core.
 *
 *  Structurally the point block of the normal equations is diagonal (scalar
 *  per point thanks to isotropic weights), so Schur elimination leaves a
 *  cameras-only system whose matrix is IDENTICAL for x/y/z — one Cholesky
 *  factorization, three back-substitutions per iteration.
 *
 *  camEdges: [{ i, j, d (unit world dir of Ci - Cj), w }]
 *  obs:      [{ i (cam), k (point), v ([3] unit world ray), w? }]
 *  Returns { C: Array(n) or null, P: Array(nPts) or null }. */
export function globalPositionsJoint(n, camEdges, obs, nPts, anchorIdx, log = () => {}) {
  const camIdx = new Int32Array(n).fill(-1);
  const cams = [];
  for (const e of camEdges) for (const v of [e.i, e.j]) if (camIdx[v] < 0) { camIdx[v] = cams.length; cams.push(v); }
  for (const o of obs) if (camIdx[o.i] < 0) { camIdx[o.i] = cams.length; cams.push(o.i); }
  const m = cams.length;
  if (m < 3) return null;
  const anchor = camIdx[anchorIdx] >= 0 ? camIdx[anchorIdx] : 0;

  // group obs by point
  const ptObs = Array.from({ length: nPts }, () => []);
  for (let oi = 0; oi < obs.length; oi++) ptObs[obs[oi].k].push(oi);

  const sEdge = new Float64Array(camEdges.length).fill(1);
  const sObs = new Float64Array(obs.length).fill(3); // start points ~3 baselines out
  let wEdge = camEdges.map((e) => (e.w ?? 1) * 4); // edges also carry the scale gauge
  let wObs = obs.map((o) => o.w ?? 1);
  const SMIN_OBS = 0.3;

  let C = null;
  const P = new Float64Array(3 * nPts);
  const t0 = performance.now();
  for (let round = 0; round < 25; round++) {
    const H = new Float64Array(m * m);
    const g0 = new Float64Array(m), g1 = new Float64Array(m), g2 = new Float64Array(m);
    for (let ei = 0; ei < camEdges.length; ei++) {
      const e = camEdges[ei], w = wEdge[ei];
      const a = camIdx[e.i], b = camIdx[e.j];
      H[a * m + a] += w; H[b * m + b] += w; H[a * m + b] -= w; H[b * m + a] -= w;
      const s = sEdge[ei];
      g0[a] += w * s * e.d[0]; g1[a] += w * s * e.d[1]; g2[a] += w * s * e.d[2];
      g0[b] -= w * s * e.d[0]; g1[b] -= w * s * e.d[1]; g2[b] -= w * s * e.d[2];
    }
    // camera-point rays, points Schur-eliminated
    const gP = new Float64Array(3 * nPts);
    const lam = new Float64Array(nPts);
    for (let oi = 0; oi < obs.length; oi++) {
      const o = obs[oi], w = wObs[oi];
      const a = camIdx[o.i];
      H[a * m + a] += w;
      lam[o.k] += w;
      const s = sObs[oi];
      // row: C_i - P_k = -s v   (b = -s v)
      g0[a] -= w * s * o.v[0]; g1[a] -= w * s * o.v[1]; g2[a] -= w * s * o.v[2];
      gP[3 * o.k] += w * s * o.v[0]; gP[3 * o.k + 1] += w * s * o.v[1]; gP[3 * o.k + 2] += w * s * o.v[2];
    }
    for (let k = 0; k < nPts; k++) {
      const list = ptObs[k];
      if (!list.length || lam[k] < 1e-12) continue;
      const il = 1 / lam[k];
      for (const oa of list) {
        const a = camIdx[obs[oa].i], wa = wObs[oa];
        // g fold: g'_a += w_a * gP_k / lam
        g0[a] += wa * il * gP[3 * k];
        g1[a] += wa * il * gP[3 * k + 1];
        g2[a] += wa * il * gP[3 * k + 2];
        for (const ob of list) {
          const b = camIdx[obs[ob].i];
          H[a * m + b] -= wa * wObs[ob] * il;
        }
      }
    }
    H[anchor * m + anchor] += 1e6;
    for (let r = 0; r < m; r++) H[r * m + r] += 1e-9;
    if (!cholFactor(H, m)) return null;
    const X = cholBack(H, g0, m), Y = cholBack(H, g1, m), Z = cholBack(H, g2, m);
    C = { X, Y, Z };
    // back-substitute points: P_k = (gP_k + sum_a w_a C_a) / lam_k
    for (let k = 0; k < nPts; k++) {
      const list = ptObs[k];
      if (!list.length || lam[k] < 1e-12) continue;
      let sx = gP[3 * k], sy = gP[3 * k + 1], sz = gP[3 * k + 2];
      for (const oa of list) {
        const a = camIdx[obs[oa].i], wa = wObs[oa];
        sx += wa * X[a]; sy += wa * Y[a]; sz += wa * Z[a];
      }
      P[3 * k] = sx / lam[k]; P[3 * k + 1] = sy / lam[k]; P[3 * k + 2] = sz / lam[k];
    }
    // scale updates + robust reweighting
    let sMoved = 0;
    for (let ei = 0; ei < camEdges.length; ei++) {
      const e = camEdges[ei];
      const a = camIdx[e.i], b = camIdx[e.j];
      const vx = X[a] - X[b], vy = Y[a] - Y[b], vz = Z[a] - Z[b];
      const dotv = vx * e.d[0] + vy * e.d[1] + vz * e.d[2];
      const ns = Math.max(1, dotv);
      sMoved = Math.max(sMoved, Math.abs(ns - sEdge[ei]) / Math.max(1, sEdge[ei]));
      sEdge[ei] = ns;
      if (round >= 1) {
        const len = Math.hypot(vx, vy, vz) + 1e-12;
        const cx = e.d[1] * vz - e.d[2] * vy, cy = e.d[2] * vx - e.d[0] * vz, cz = e.d[0] * vy - e.d[1] * vx;
        const rel = Math.hypot(cx, cy, cz) / len;
        wEdge[ei] = (camEdges[ei].w ?? 1) * 4 / (1 + (rel / 0.05) ** 2);
      }
    }
    for (let oi = 0; oi < obs.length; oi++) {
      const o = obs[oi];
      const a = camIdx[o.i];
      const vx = P[3 * o.k] - X[a], vy = P[3 * o.k + 1] - Y[a], vz = P[3 * o.k + 2] - Z[a];
      const dotv = vx * o.v[0] + vy * o.v[1] + vz * o.v[2];
      const ns = Math.max(SMIN_OBS, dotv);
      sMoved = Math.max(sMoved, Math.abs(ns - sObs[oi]) / Math.max(1, sObs[oi]));
      sObs[oi] = ns;
      if (round >= 1) {
        const len = Math.hypot(vx, vy, vz) + 1e-12;
        const cx = o.v[1] * vz - o.v[2] * vy, cy = o.v[2] * vx - o.v[0] * vz, cz = o.v[0] * vy - o.v[1] * vx;
        const rel = Math.hypot(cx, cy, cz) / len;
        wObs[oi] = (o.w ?? 1) / (1 + (rel / 0.03) ** 2);
      }
    }
    if (round > 3 && sMoved < 1e-3) { log(`  joint positioning converged after ${round + 1} rounds`); break; }
  }
  log(`  joint positioning: ${m} cams, ${nPts} pts, ${obs.length} rays in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  const outC = new Array(n).fill(null);
  for (let k = 0; k < m; k++) outC[cams[k]] = [C.X[k], C.Y[k], C.Z[k]];
  const outP = [];
  for (let k = 0; k < nPts; k++) outP.push([P[3 * k], P[3 * k + 1], P[3 * k + 2]]);
  return { C: outC, P: outP };
}
