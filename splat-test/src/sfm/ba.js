// ba.js — sparse bundle adjustment: Levenberg-Marquardt with Schur
// complement, robust (Huber) loss, shared focal + radial distortion (k1, k2).
//
// This is the piece that separates a drifting chain reconstruction from a
// COLMAP-grade one: the alternating pose-refine/retriangulate pass is
// coordinate descent and cannot remove smooth global trajectory bends; a
// joint solve over all cameras, points and intrinsics can.
//
// Projection model (pixels, principal point fixed at cx/cy):
//   p_c = R p + t;  x' = X/Z, y' = Y/Z;  r2 = x'^2 + y'^2
//   D = 1 + k1 r2 + k2 r2^2
//   u = f x' D + cx,  v = f y' D + cy
//
// State: per-camera se(3) increments (camera 0 fixed as gauge anchor),
// 3D points, and globals [log f, k1, k2].

import { rodrigues, m3mul } from './geometry.js';

/** Dense Cholesky solve (SPD, factors in place); returns x or null. */
function cholSolve(L, g, n) {
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = L[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      if (i === j) {
        if (s <= 0) return null;
        L[i * n + i] = Math.sqrt(s);
      } else {
        L[i * n + j] = s / L[j * n + j];
      }
    }
  }
  const x = Float64Array.from(g);
  for (let i = 0; i < n; i++) {
    let s = x[i];
    for (let k = 0; k < i; k++) s -= L[i * n + k] * x[k];
    x[i] = s / L[i * n + i];
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k];
    x[i] = s / L[i * n + i];
  }
  return x;
}

/**
 * problem: {
 *   cams:   [{R: 9-array, t: 3-array}],   // modified in place; cams[0] fixed
 *   points: [[x,y,z], ...],               // modified in place
 *   obs:    [{ci, pi, u, v}],             // pixel measurements
 *   f, cx, cy                             // shared intrinsics; f refined
 * }
 * opts: { maxIters, huberPx, refineDistortion, refineF, refineAspect, log }
 * returns { fScale, f, k1, k2, aspect, rmsBefore, rmsAfter, iters }
 */
export function bundleAdjust(problem, opts = {}) {
  const { cams, points, obs } = problem;
  const cx = problem.cx, cy = problem.cy;
  const maxIters = opts.maxIters ?? 25;
  const huber = opts.huberPx ?? 1.5;
  const doK = opts.refineDistortion ?? true;
  const doF = opts.refineF ?? true;
  const doA = opts.refineAspect ?? false; // fy = f * aspect (non-square pixels)
  const log = opts.log || (() => {});

  const nc = cams.length;
  const np = points.length;
  const K = obs.length;
  const nCamParams = 6 * (nc - 1); // cam 0 fixed (gauge anchor)
  const NG = 4;                     // logf, k1, k2, log-aspect
  const N = nCamParams + NG;
  const camBase = (ci) => (ci - 1) * 6;

  let f = problem.f;
  let k1 = 0, k2 = 0, aspect = 1;

  const ptObs = Array.from({ length: np }, () => []);
  for (let o = 0; o < K; o++) ptObs[obs[o].pi].push(o);

  const project = (R, t, p, fv, k1v, k2v, av) => {
    const x = R[0] * p[0] + R[1] * p[1] + R[2] * p[2] + t[0];
    const y = R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + t[1];
    const z = R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + t[2];
    if (z <= 1e-6) return null;
    const xp = x / z, yp = y / z;
    const r2 = xp * xp + yp * yp;
    const D = 1 + k1v * r2 + k2v * r2 * r2;
    return { x, y, z, xp, yp, r2, D, u: fv * xp * D + cx, v: fv * av * yp * D + cy };
  };

  const evalCostRms = (fv, k1v, k2v, av) => {
    let cost = 0, s = 0, c = 0;
    for (let o = 0; o < K; o++) {
      const ob = obs[o];
      const pr = project(cams[ob.ci].R, cams[ob.ci].t, points[ob.pi], fv, k1v, k2v, av);
      if (!pr) { cost += huber * 10; continue; }
      const ru = pr.u - ob.u, rv = pr.v - ob.v;
      const e = Math.sqrt(ru * ru + rv * rv);
      cost += e <= huber ? 0.5 * e * e : huber * (e - 0.5 * huber);
      s += ru * ru + rv * rv; c++;
    }
    return { cost, rms: Math.sqrt(s / Math.max(1, c)) };
  };

  const before = evalCostRms(f, k1, k2, aspect);
  let cost = before.cost;
  let lambda = 1e-3;
  let iters = 0;

  // per-observation scaled jacobian storage
  const Jp = new Float64Array(K * 6);   // 2x3
  const Jc = new Float64Array(K * 12);  // 2x6 (zero for cam 0)
  const Jg = new Float64Array(K * 8);   // 2x4
  const Rres = new Float64Array(K * 2);
  const valid = new Uint8Array(K);

  const H = new Float64Array(N * N);
  const g = new Float64Array(N);
  const Vinv = new Float64Array(np * 9);
  const gpAll = new Float64Array(np * 3);

  for (let iter = 0; iter < maxIters; iter++) {
    iters = iter + 1;
    H.fill(0); g.fill(0);

    // ---- build per-obs jacobians ----
    for (let o = 0; o < K; o++) {
      const ob = obs[o];
      const cam = cams[ob.ci];
      const pr = project(cam.R, cam.t, points[ob.pi], f, k1, k2, aspect);
      if (!pr) { valid[o] = 0; continue; }
      valid[o] = 1;
      const ru = pr.u - ob.u, rv = pr.v - ob.v;
      const e = Math.sqrt(ru * ru + rv * rv);
      const sw = Math.sqrt(e <= huber ? 1 : huber / e);

      const kp = k1 + 2 * k2 * pr.r2;
      const d00 = pr.D + 2 * pr.xp * pr.xp * kp;
      const d01 = 2 * pr.xp * pr.yp * kp;
      const d11 = pr.D + 2 * pr.yp * pr.yp * kp;
      const iz = 1 / pr.z;
      const fy = f * aspect;
      // Jproj = f * Ddist * d(x',y')/dp_c   (2x3); v-row uses fy
      const a00 = f * d00 * iz, a01 = f * d01 * iz, a02 = -f * (d00 * pr.xp + d01 * pr.yp) * iz;
      const a10 = fy * d01 * iz, a11 = fy * d11 * iz, a12 = -fy * (d01 * pr.xp + d11 * pr.yp) * iz;

      // Jp = Jproj * R
      const R = cam.R;
      for (let c = 0; c < 3; c++) {
        Jp[o * 6 + c] = sw * (a00 * R[c] + a01 * R[3 + c] + a02 * R[6 + c]);
        Jp[o * 6 + 3 + c] = sw * (a10 * R[c] + a11 * R[3 + c] + a12 * R[6 + c]);
      }
      // Jc = [Jproj * (-[p_c - t]x) | Jproj]
      if (ob.ci !== 0) {
        const px = pr.x - cam.t[0], py = pr.y - cam.t[1], pz = pr.z - cam.t[2];
        // dp_c/dw = -[p]x with p = R*point = p_c - t:
        // (-skew(p)) columns: col0 = (0, -pz, py); col1 = (pz, 0, -px); col2 = (-py, px, 0)
        Jc[o * 12 + 0] = sw * (-a01 * pz + a02 * py);
        Jc[o * 12 + 1] = sw * (a00 * pz - a02 * px);
        Jc[o * 12 + 2] = sw * (-a00 * py + a01 * px);
        Jc[o * 12 + 3] = sw * a00;
        Jc[o * 12 + 4] = sw * a01;
        Jc[o * 12 + 5] = sw * a02;
        Jc[o * 12 + 6] = sw * (-a11 * pz + a12 * py);
        Jc[o * 12 + 7] = sw * (a10 * pz - a12 * px);
        Jc[o * 12 + 8] = sw * (-a10 * py + a11 * px);
        Jc[o * 12 + 9] = sw * a10;
        Jc[o * 12 + 10] = sw * a11;
        Jc[o * 12 + 11] = sw * a12;
      } else {
        for (let k = 0; k < 12; k++) Jc[o * 12 + k] = 0;
      }
      // globals: [logf, k1, k2, log-aspect]
      const xd = pr.xp * pr.D, yd = pr.yp * pr.D;
      Jg[o * 8 + 0] = doF ? sw * f * xd : 0;
      Jg[o * 8 + 1] = doK ? sw * f * pr.xp * pr.r2 : 0;
      Jg[o * 8 + 2] = doK ? sw * f * pr.xp * pr.r2 * pr.r2 : 0;
      Jg[o * 8 + 3] = 0;                                     // du/dloga = 0
      Jg[o * 8 + 4] = doF ? sw * fy * yd : 0;
      Jg[o * 8 + 5] = doK ? sw * fy * pr.yp * pr.r2 : 0;
      Jg[o * 8 + 6] = doK ? sw * fy * pr.yp * pr.r2 * pr.r2 : 0;
      Jg[o * 8 + 7] = doA ? sw * fy * yd : 0;                // dv/dloga = v - cy
      Rres[o * 2] = sw * ru;
      Rres[o * 2 + 1] = sw * rv;
    }

    // ---- per-point V, gp; H,g direct terms + Schur folds ----
    for (let j = 0; j < np; j++) {
      const list = ptObs[j];
      if (!list.length) continue;
      const V = [0, 0, 0, 0, 0, 0, 0, 0, 0];
      const gp = [0, 0, 0];
      let any = false;
      for (const o of list) {
        if (!valid[o]) continue;
        any = true;
        const r0 = Rres[o * 2], r1 = Rres[o * 2 + 1];
        for (let a = 0; a < 3; a++) {
          gp[a] += Jp[o * 6 + a] * r0 + Jp[o * 6 + 3 + a] * r1;
          for (let b = 0; b < 3; b++)
            V[a * 3 + b] += Jp[o * 6 + a] * Jp[o * 6 + b] + Jp[o * 6 + 3 + a] * Jp[o * 6 + 3 + b];
        }
      }
      if (!any) { Vinv.fill(0, j * 9, j * 9 + 9); continue; }
      for (let a = 0; a < 3; a++) V[a * 3 + a] *= (1 + lambda);
      const det =
        V[0] * (V[4] * V[8] - V[5] * V[7]) -
        V[1] * (V[3] * V[8] - V[5] * V[6]) +
        V[2] * (V[3] * V[7] - V[4] * V[6]);
      if (Math.abs(det) < 1e-20) { Vinv.fill(0, j * 9, j * 9 + 9); continue; }
      const id = 1 / det;
      const Vi = Vinv.subarray(j * 9, j * 9 + 9);
      Vi[0] = (V[4] * V[8] - V[5] * V[7]) * id;
      Vi[1] = (V[2] * V[7] - V[1] * V[8]) * id;
      Vi[2] = (V[1] * V[5] - V[2] * V[4]) * id;
      Vi[3] = (V[5] * V[6] - V[3] * V[8]) * id;
      Vi[4] = (V[0] * V[8] - V[2] * V[6]) * id;
      Vi[5] = (V[2] * V[3] - V[0] * V[5]) * id;
      Vi[6] = (V[3] * V[7] - V[4] * V[6]) * id;
      Vi[7] = (V[1] * V[6] - V[0] * V[7]) * id;
      Vi[8] = (V[0] * V[4] - V[1] * V[3]) * id;
      gpAll[j * 3] = gp[0]; gpAll[j * 3 + 1] = gp[1]; gpAll[j * 3 + 2] = gp[2];

      // W blocks per valid obs: Wc = Jc^T Jp (6x3), Wg = Jg^T Jp (4x3)
      const vlist = list.filter((o) => valid[o]);
      const Wc = [], Wg = [];
      for (const o of vlist) {
        const wc = new Float64Array(18);
        for (let a = 0; a < 6; a++)
          for (let b = 0; b < 3; b++)
            wc[a * 3 + b] = Jc[o * 12 + a] * Jp[o * 6 + b] + Jc[o * 12 + 6 + a] * Jp[o * 6 + 3 + b];
        Wc.push(wc);
        const wg = new Float64Array(12);
        for (let a = 0; a < 4; a++)
          for (let b = 0; b < 3; b++)
            wg[a * 3 + b] = Jg[o * 8 + a] * Jp[o * 6 + b] + Jg[o * 8 + 4 + a] * Jp[o * 6 + 3 + b];
        Wg.push(wg);
      }

      // direct H/g contributions
      for (let ii = 0; ii < vlist.length; ii++) {
        const o = vlist[ii];
        const ob = obs[o];
        const r0 = Rres[o * 2], r1 = Rres[o * 2 + 1];
        if (ob.ci !== 0) {
          const base = camBase(ob.ci);
          for (let a = 0; a < 6; a++) {
            g[base + a] += Jc[o * 12 + a] * r0 + Jc[o * 12 + 6 + a] * r1;
            for (let b = 0; b < 6; b++)
              H[(base + a) * N + base + b] += Jc[o * 12 + a] * Jc[o * 12 + b] + Jc[o * 12 + 6 + a] * Jc[o * 12 + 6 + b];
            for (let b = 0; b < 4; b++) {
              const vAB = Jc[o * 12 + a] * Jg[o * 8 + b] + Jc[o * 12 + 6 + a] * Jg[o * 8 + 4 + b];
              H[(base + a) * N + nCamParams + b] += vAB;
              H[(nCamParams + b) * N + base + a] += vAB;
            }
          }
        }
        for (let a = 0; a < 4; a++) {
          g[nCamParams + a] += Jg[o * 8 + a] * r0 + Jg[o * 8 + 4 + a] * r1;
          for (let b = 0; b < 4; b++)
            H[(nCamParams + a) * N + nCamParams + b] += Jg[o * 8 + a] * Jg[o * 8 + b] + Jg[o * 8 + 4 + a] * Jg[o * 8 + 4 + b];
        }
      }

      // Schur folds
      const ViGp = [
        Vi[0] * gp[0] + Vi[1] * gp[1] + Vi[2] * gp[2],
        Vi[3] * gp[0] + Vi[4] * gp[1] + Vi[5] * gp[2],
        Vi[6] * gp[0] + Vi[7] * gp[1] + Vi[8] * gp[2],
      ];
      const foldG = (W, nA, base) => {
        for (let a = 0; a < nA; a++)
          g[base + a] -= W[a * 3] * ViGp[0] + W[a * 3 + 1] * ViGp[1] + W[a * 3 + 2] * ViGp[2];
      };
      const foldH = (Wa, nA, baseA, Wb, nB, baseB) => {
        for (let a = 0; a < nA; a++) {
          const w0 = Wa[a * 3], w1 = Wa[a * 3 + 1], w2 = Wa[a * 3 + 2];
          const x0 = w0 * Vi[0] + w1 * Vi[3] + w2 * Vi[6];
          const x1 = w0 * Vi[1] + w1 * Vi[4] + w2 * Vi[7];
          const x2 = w0 * Vi[2] + w1 * Vi[5] + w2 * Vi[8];
          for (let b = 0; b < nB; b++)
            H[(baseA + a) * N + baseB + b] -= x0 * Wb[b * 3] + x1 * Wb[b * 3 + 1] + x2 * Wb[b * 3 + 2];
        }
      };
      for (let ii = 0; ii < vlist.length; ii++) {
        const ci = obs[vlist[ii]].ci;
        if (ci !== 0) foldG(Wc[ii], 6, camBase(ci));
        foldG(Wg[ii], 4, nCamParams);
        for (let jj = 0; jj < vlist.length; jj++) {
          const cj = obs[vlist[jj]].ci;
          if (ci !== 0 && cj !== 0) foldH(Wc[ii], 6, camBase(ci), Wc[jj], 6, camBase(cj));
          if (ci !== 0) foldH(Wc[ii], 6, camBase(ci), Wg[jj], 4, nCamParams);
          if (cj !== 0) foldH(Wg[ii], 4, nCamParams, Wc[jj], 6, camBase(cj));
          foldH(Wg[ii], 4, nCamParams, Wg[jj], 4, nCamParams);
        }
      }
    }

    // damping + solve
    for (let a = 0; a < N; a++) H[a * N + a] = H[a * N + a] * (1 + lambda) + 1e-9;
    const negg = new Float64Array(N);
    for (let a = 0; a < N; a++) negg[a] = -g[a];
    const delta = cholSolve(Float64Array.from(H), negg, N);
    if (!delta) { lambda *= 10; if (lambda > 1e8) break; continue; }

    // ---- tentative update ----
    const oldCams = cams.map((c) => ({ R: Array.from(c.R), t: c.t.slice() }));
    const oldPts = points.map((p) => p.slice());
    const oldF = f, oldK1 = k1, oldK2 = k2, oldAspect = aspect;

    for (let ci = 1; ci < nc; ci++) {
      const base = camBase(ci);
      cams[ci].R = Array.from(m3mul(rodrigues([delta[base], delta[base + 1], delta[base + 2]]), cams[ci].R));
      cams[ci].t = [
        cams[ci].t[0] + delta[base + 3],
        cams[ci].t[1] + delta[base + 4],
        cams[ci].t[2] + delta[base + 5],
      ];
    }
    if (doF) f = f * Math.exp(delta[nCamParams]);
    if (doK) { k1 += delta[nCamParams + 1]; k2 += delta[nCamParams + 2]; }
    if (doA) aspect = aspect * Math.exp(delta[nCamParams + 3]);

    // back-substitute points: dp = -Vinv (gp + W^T dc)  where
    // (W^T dc)_b = sum_obs Jp_b . (Jc dc + Jg dg)   (residual-space form)
    for (let j = 0; j < np; j++) {
      const list = ptObs[j];
      if (!list.length) continue;
      const Vi = Vinv.subarray(j * 9, j * 9 + 9);
      let r0 = gpAll[j * 3], r1 = gpAll[j * 3 + 1], r2 = gpAll[j * 3 + 2];
      for (const o of list) {
        if (!valid[o]) continue;
        const ob = obs[o];
        let q0 = 0, q1 = 0;
        if (ob.ci !== 0) {
          const base = camBase(ob.ci);
          for (let a = 0; a < 6; a++) {
            q0 += Jc[o * 12 + a] * delta[base + a];
            q1 += Jc[o * 12 + 6 + a] * delta[base + a];
          }
        }
        for (let a = 0; a < 4; a++) {
          q0 += Jg[o * 8 + a] * delta[nCamParams + a];
          q1 += Jg[o * 8 + 4 + a] * delta[nCamParams + a];
        }
        r0 += Jp[o * 6] * q0 + Jp[o * 6 + 3] * q1;
        r1 += Jp[o * 6 + 1] * q0 + Jp[o * 6 + 4] * q1;
        r2 += Jp[o * 6 + 2] * q0 + Jp[o * 6 + 5] * q1;
      }
      points[j][0] -= Vi[0] * r0 + Vi[1] * r1 + Vi[2] * r2;
      points[j][1] -= Vi[3] * r0 + Vi[4] * r1 + Vi[5] * r2;
      points[j][2] -= Vi[6] * r0 + Vi[7] * r1 + Vi[8] * r2;
    }

    const now = evalCostRms(f, k1, k2, aspect);
    if (now.cost < cost) {
      const rel = 1 - now.cost / cost;
      cost = now.cost;
      lambda = Math.max(1e-9, lambda / 3);
      log(`  BA iter ${iters}: rms ${now.rms.toFixed(3)}px`);
      if (rel < 1e-5) break;
    } else {
      for (let ci = 0; ci < nc; ci++) { cams[ci].R = oldCams[ci].R; cams[ci].t = oldCams[ci].t; }
      for (let j = 0; j < np; j++) points[j] = oldPts[j];
      f = oldF; k1 = oldK1; k2 = oldK2; aspect = oldAspect;
      lambda *= 10;
      if (lambda > 1e8) break;
    }
  }

  const after = evalCostRms(f, k1, k2, aspect);
  return { fScale: f / problem.f, f, k1, k2, aspect, rmsBefore: before.rms, rmsAfter: after.rms, iters };
}
