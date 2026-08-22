// init.js — build initial Gaussian parameters from the sparse SfM point cloud.
// Layout per Gaussian (stride 8): [x, y, z, logScale, r, g, b, logitOpacity]

import { makeRng } from '../sfm/geometry.js';

const logit = (p) => Math.log(p / (1 - p));

/**
 * points: [{ X: [3], rgb: [3] }]
 * clones: extra jittered copies per point (increases capacity without densification)
 * Returns { data, n, center, radius }
 */
export function initGaussians(points, clones = 2, maxGaussians = 600000) {
  const rng = makeRng(777);
  const np = points.length;

  // scene bounds (robust-ish: median absolute extent)
  const center = [0, 0, 0];
  for (const p of points) {
    center[0] += p.X[0]; center[1] += p.X[1]; center[2] += p.X[2];
  }
  center[0] /= np; center[1] /= np; center[2] /= np;
  const dists = points.map((p) => Math.hypot(
    p.X[0] - center[0], p.X[1] - center[1], p.X[2] - center[2]));
  const sorted = Float64Array.from(dists).sort();
  const radius = Math.max(1e-3, sorted[Math.min(np - 1, Math.floor(np * 0.9))]);

  // nearest-neighbor distance estimate via a hash grid
  const cell = radius / 24;
  const grid = new Map();
  const key = (x, y, z) => `${Math.floor(x / cell)},${Math.floor(y / cell)},${Math.floor(z / cell)}`;
  points.forEach((p, i) => {
    const k = key(p.X[0], p.X[1], p.X[2]);
    if (!grid.has(k)) grid.set(k, []);
    grid.get(k).push(i);
  });
  const nnDist = (i) => {
    const [x, y, z] = points[i].X;
    let best = Infinity;
    const cx = Math.floor(x / cell), cy = Math.floor(y / cell), cz = Math.floor(z / cell);
    for (let dx = -1; dx <= 1; dx++)
      for (let dy = -1; dy <= 1; dy++)
        for (let dz = -1; dz <= 1; dz++) {
          const list = grid.get(`${cx + dx},${cy + dy},${cz + dz}`);
          if (!list) continue;
          for (const j of list) {
            if (j === i) continue;
            const q = points[j].X;
            const d2 = (q[0] - x) ** 2 + (q[1] - y) ** 2 + (q[2] - z) ** 2;
            if (d2 < best) best = d2;
          }
        }
    return best === Infinity ? cell * 2 : Math.sqrt(best);
  };

  // stride-16 layout: [pos3, logScale3, quat4 (w,x,y,z), colorLogit3,
  //                    logitOpacity, pad2] — isotropic identity-rotation init
  const STRIDE = 16;
  const perPoint = 1 + clones;
  const n = Math.min(maxGaussians, np * perPoint);
  const data = new Float32Array(n * STRIDE);
  const minS = radius * 3e-4, maxS = radius * 0.05;
  const lop = logit(0.35);

  let g = 0;
  for (let i = 0; i < np && g < n; i++) {
    const p = points[i];
    const s = Math.min(maxS, Math.max(minS, nnDist(i) * 0.7));
    for (let c = 0; c < perPoint && g < n; c++, g++) {
      const jit = c === 0 ? 0 : s * 0.6;
      const b = g * STRIDE;
      data[b]     = p.X[0] + (rng() - 0.5) * 2 * jit;
      data[b + 1] = p.X[1] + (rng() - 0.5) * 2 * jit;
      data[b + 2] = p.X[2] + (rng() - 0.5) * 2 * jit;
      const ls = Math.log(s * (c === 0 ? 1 : 0.8));
      data[b + 3] = ls;
      data[b + 4] = ls;
      data[b + 5] = ls;
      data[b + 6] = 1; // identity quaternion (w,x,y,z)
      // colors are stored as logits (sigmoid-activated in the shaders)
      data[b + 10] = logit(Math.min(0.98, Math.max(0.02, p.rgb[0])));
      data[b + 11] = logit(Math.min(0.98, Math.max(0.02, p.rgb[1])));
      data[b + 12] = logit(Math.min(0.98, Math.max(0.02, p.rgb[2])));
      data[b + 13] = lop;
    }
  }
  return { data, n: g, center, radius };
}
