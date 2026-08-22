// features.js — Shi-Tomasi corner detection + oriented BRIEF descriptors.
// Pure JS, operates on a grayscale Float32Array.

const DESC_WORDS = 8;          // 256-bit descriptor
const PATTERN_RADIUS = 13;
const BORDER = 20;

// Deterministic BRIEF sampling pattern (256 point pairs).
const PATTERN = (() => {
  let seed = 0x9E3779B9;
  const rnd = () => {
    seed ^= seed << 13; seed >>>= 0;
    seed ^= seed >>> 17;
    seed ^= seed << 5; seed >>>= 0;
    return seed / 4294967296;
  };
  const gauss = () => {
    const u = Math.max(1e-9, rnd()), v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const p = new Float32Array(256 * 4);
  for (let i = 0; i < 256 * 4; i++) {
    p[i] = Math.max(-PATTERN_RADIUS, Math.min(PATTERN_RADIUS, gauss() * PATTERN_RADIUS * 0.4));
  }
  return p;
})();

function boxBlur(src, w, h, r) {
  const tmp = new Float32Array(w * h);
  const dst = new Float32Array(w * h);
  const norm = 1 / (2 * r + 1);
  for (let y = 0; y < h; y++) {
    const off = y * w;
    let sum = 0;
    for (let x = -r; x <= r; x++) sum += src[off + Math.min(w - 1, Math.max(0, x))];
    for (let x = 0; x < w; x++) {
      tmp[off + x] = sum * norm;
      const xAdd = Math.min(w - 1, x + r + 1);
      const xSub = Math.max(0, x - r);
      sum += src[off + xAdd] - src[off + xSub];
    }
  }
  for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let y = -r; y <= r; y++) sum += tmp[Math.min(h - 1, Math.max(0, y)) * w + x];
    for (let y = 0; y < h; y++) {
      dst[y * w + x] = sum * norm;
      const yAdd = Math.min(h - 1, y + r + 1);
      const ySub = Math.max(0, y - r);
      sum += tmp[yAdd * w + x] - tmp[ySub * w + x];
    }
  }
  return dst;
}

function resample(src, sw, sh, dw, dh) {
  const dst = new Float32Array(dw * dh);
  const fx = sw / dw, fy = sh / dh;
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(sh - 1.001, (y + 0.5) * fy - 0.5);
    const y0 = Math.max(0, sy | 0), ty = sy - y0;
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(sw - 1.001, (x + 0.5) * fx - 0.5);
      const x0 = Math.max(0, sx | 0), tx = sx - x0;
      const o = y0 * sw + x0;
      dst[y * dw + x] =
        src[o] * (1 - tx) * (1 - ty) + src[o + 1] * tx * (1 - ty) +
        src[o + sw] * (1 - tx) * ty + src[o + sw + 1] * tx * ty;
    }
  }
  return dst;
}

/** Multi-scale detection (ORB-style): an image pyramid brings the two COLMAP
 *  front-end properties our single-scale BRIEF lacks — scale invariance
 *  (features detected + described at the octave where they live) and subpixel
 *  localization (quadratic fit on the corner response). Descriptors stay
 *  256-bit binary so matching stays Hamming-fast (full SIFT matching is
 *  ~100x slower in JS and is why COLMAP runs it on GPU).
 *  Returns { n, x, y, angle, scale, desc } with x/y at base resolution. */
export function detectAndDescribeMS(gray, w, h, maxFeats = 1500) {
  const FACTOR = 1.25, LEVELS = 6;
  const levels = [{ img: gray, w, h, s: 1 }];
  for (let l = 1; l < LEVELS; l++) {
    const p = levels[l - 1];
    const nw = Math.round(p.w / FACTOR), nh = Math.round(p.h / FACTOR);
    if (nw < 3 * BORDER || nh < 3 * BORDER) break;
    levels.push({ img: resample(p.img, p.w, p.h, nw, nh), w: nw, h: nh, s: w / nw });
  }
  // budget per level proportional to area
  const areas = levels.map((L) => L.w * L.h);
  const areaSum = areas.reduce((a, b) => a + b, 0);
  const all = [];
  for (let l = 0; l < levels.length; l++) {
    const L = levels[l];
    const budget = Math.max(40, Math.round(maxFeats * areas[l] / areaSum));
    const f = detectAndDescribe(L.img, L.w, L.h, budget, true);
    for (let k = 0; k < f.n; k++) {
      all.push({
        x: f.x[k] * L.s, y: f.y[k] * L.s, s: L.s,
        angle: f.angle[k], score: f.score[k],
        desc: f.desc.subarray(k * DESC_WORDS, k * DESC_WORDS + DESC_WORDS),
      });
    }
  }
  // cross-level non-max suppression at base resolution (strongest wins)
  all.sort((a, b) => b.score - a.score);
  const minDist = Math.max(4, Math.round(Math.min(w, h) / 90));
  const gw = Math.ceil(w / minDist), gh = Math.ceil(h / minDist);
  const grid = new Map();
  const keep = [];
  for (const c of all) {
    if (keep.length >= maxFeats) break;
    const cgx = (c.x / minDist) | 0, cgy = (c.y / minDist) | 0;
    let ok = true;
    for (let dy = -1; dy <= 1 && ok; dy++)
      for (let dx = -1; dx <= 1 && ok; dx++) {
        const list = grid.get((cgy + dy) * gw + (cgx + dx));
        if (!list) continue;
        for (const o of list)
          if ((o.x - c.x) ** 2 + (o.y - c.y) ** 2 < minDist * minDist) { ok = false; break; }
      }
    if (!ok) continue;
    const cell = cgy * gw + cgx;
    if (!grid.has(cell)) grid.set(cell, []);
    grid.get(cell).push(c);
    keep.push(c);
  }
  const n = keep.length;
  const out = {
    n,
    x: new Float32Array(n), y: new Float32Array(n),
    angle: new Float32Array(n), scale: new Float32Array(n),
    desc: new Uint32Array(n * DESC_WORDS),
  };
  keep.forEach((c, k) => {
    out.x[k] = c.x; out.y[k] = c.y;
    out.angle[k] = c.angle; out.scale[k] = c.s;
    out.desc.set(c.desc, k * DESC_WORDS);
  });
  return out;
}

/** Detect corners and compute descriptors.
 *  gray: Float32Array (0..1), w, h.
 *  Returns { n, x, y, angle, score, desc } (desc: Uint32Array n*8). */
export function detectAndDescribe(gray, w, h, maxFeats = 1500, subpixel = false) {
  const blur = boxBlur(gray, w, h, 2);

  // Gradients (central differences on blurred image)
  const gx = new Float32Array(w * h);
  const gy = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      gx[i] = (blur[i + 1] - blur[i - 1]) * 0.5;
      gy[i] = (blur[i + w] - blur[i - w]) * 0.5;
    }
  }
  // Structure tensor entries, box-smoothed
  const gxx = new Float32Array(w * h);
  const gyy = new Float32Array(w * h);
  const gxy = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    gxx[i] = gx[i] * gx[i];
    gyy[i] = gy[i] * gy[i];
    gxy[i] = gx[i] * gy[i];
  }
  const sxx = boxBlur(gxx, w, h, 2);
  const syy = boxBlur(gyy, w, h, 2);
  const sxy = boxBlur(gxy, w, h, 2);

  // Shi-Tomasi score = min eigenvalue of structure tensor
  const score = new Float32Array(w * h);
  let maxScore = 0;
  for (let y = BORDER; y < h - BORDER; y++) {
    for (let x = BORDER; x < w - BORDER; x++) {
      const i = y * w + x;
      const a = sxx[i], b = syy[i], c = sxy[i];
      const s = (a + b) / 2 - Math.sqrt(((a - b) / 2) ** 2 + c * c);
      score[i] = s;
      if (s > maxScore) maxScore = s;
    }
  }
  const thresh = maxScore * 0.002;

  // 3x3 local maxima above threshold
  const cands = [];
  for (let y = BORDER; y < h - BORDER; y++) {
    for (let x = BORDER; x < w - BORDER; x++) {
      const i = y * w + x;
      const s = score[i];
      if (s < thresh) continue;
      if (s >= score[i - 1] && s > score[i + 1] &&
          s >= score[i - w] && s > score[i + w] &&
          s >= score[i - w - 1] && s > score[i - w + 1] &&
          s >= score[i + w - 1] && s > score[i + w + 1]) {
        cands.push({ x, y, s });
      }
    }
  }
  cands.sort((a, b) => b.s - a.s);
  if (cands.length > 8000) cands.length = 8000;

  // Greedy min-distance suppression via occupancy grid
  const minDist = Math.max(4, Math.round(Math.min(w, h) / 90));
  const gw = Math.ceil(w / minDist), gh = Math.ceil(h / minDist);
  const grid = new Int32Array(gw * gh).fill(-1);
  const keep = [];
  for (const c of cands) {
    if (keep.length >= maxFeats) break;
    const cgx = (c.x / minDist) | 0, cgy = (c.y / minDist) | 0;
    let ok = true;
    for (let dy = -1; dy <= 1 && ok; dy++) {
      for (let dx = -1; dx <= 1 && ok; dx++) {
        const nx = cgx + dx, ny = cgy + dy;
        if (nx < 0 || ny < 0 || nx >= gw || ny >= gh) continue;
        const k = grid[ny * gw + nx];
        if (k >= 0) {
          const o = keep[k];
          if ((o.x - c.x) ** 2 + (o.y - c.y) ** 2 < minDist * minDist) ok = false;
        }
      }
    }
    if (!ok) continue;
    grid[cgy * gw + cgx] = keep.length;
    keep.push(c);
  }

  // Subpixel localization: 1D quadratic fits on the corner response.
  // OPT-IN ONLY: enabling it by default silently destabilized camping's
  // init-pair selection (E decompositions went rotation-dominant, the
  // reconstruction collapsed to path length 0.4 vs 44 and training lost 8dB).
  // The pipeline's subpixel accuracy comes from the LK obs refinement instead.
  for (const c of keep) {
    if (!subpixel) { c.sx = c.x; c.sy = c.y; continue; }
    const i = c.y * w + c.x;
    const dxDen = score[i - 1] - 2 * score[i] + score[i + 1];
    const dyDen = score[i - w] - 2 * score[i] + score[i + w];
    const ox = dxDen < -1e-12 ? 0.5 * (score[i - 1] - score[i + 1]) / dxDen : 0;
    const oy = dyDen < -1e-12 ? 0.5 * (score[i - w] - score[i + w]) / dyDen : 0;
    c.sx = c.x + Math.max(-0.6, Math.min(0.6, ox));
    c.sy = c.y + Math.max(-0.6, Math.min(0.6, oy));
  }

  // Orientation (intensity centroid) + BRIEF descriptors
  const n = keep.length;
  const xArr = new Float32Array(n);
  const yArr = new Float32Array(n);
  const angArr = new Float32Array(n);
  const scoreArr = new Float32Array(n);
  const desc = new Uint32Array(n * DESC_WORDS);

  const R = 7;
  for (let k = 0; k < n; k++) {
    const { x, y } = keep[k];
    xArr[k] = keep[k].sx; yArr[k] = keep[k].sy;
    scoreArr[k] = keep[k].s;
    let m10 = 0, m01 = 0;
    for (let dy = -R; dy <= R; dy++) {
      for (let dx = -R; dx <= R; dx++) {
        if (dx * dx + dy * dy > R * R) continue;
        const v = blur[(y + dy) * w + (x + dx)];
        m10 += dx * v; m01 += dy * v;
      }
    }
    const angle = Math.atan2(m01, m10);
    angArr[k] = angle;
    const ca = Math.cos(angle), sa = Math.sin(angle);
    for (let bit = 0; bit < 256; bit++) {
      const p0 = bit * 4;
      const x1 = PATTERN[p0], y1 = PATTERN[p0 + 1];
      const x2 = PATTERN[p0 + 2], y2 = PATTERN[p0 + 3];
      const rx1 = Math.round(x + ca * x1 - sa * y1);
      const ry1 = Math.round(y + sa * x1 + ca * y1);
      const rx2 = Math.round(x + ca * x2 - sa * y2);
      const ry2 = Math.round(y + sa * x2 + ca * y2);
      const v1 = blur[Math.min(h - 1, Math.max(0, ry1)) * w + Math.min(w - 1, Math.max(0, rx1))];
      const v2 = blur[Math.min(h - 1, Math.max(0, ry2)) * w + Math.min(w - 1, Math.max(0, rx2))];
      if (v1 < v2) desc[k * DESC_WORDS + (bit >> 5)] |= 1 << (bit & 31);
    }
  }
  return { n, x: xArr, y: yArr, angle: angArr, score: scoreArr, desc };
}
