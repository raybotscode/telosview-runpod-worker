// sift.js — SIFT features in plain JS (Lowe 2004, VLFeat/COLMAP conventions).
//
// This is the front end that separates COLMAP-quality geometry from ours:
// scale-space extrema (true scale invariance), 3D-interpolated subpixel
// localization, gradient-orientation invariance, and 128-D gradient-histogram
// descriptors with RootSIFT normalization (COLMAP's L1_ROOT default, matching
// its uint8 storage). Validated against COLMAP's own keypoints/descriptors
// extracted from the same images (scratch/colmap_train84/database.db).
//
// Cost notes: extraction is scalar JS (~1-2 s per 640px image). Descriptor
// matching is 128-D uint8 L2 — use matchSift() below (int32 accumulation,
// ratio + cross check); a WebGPU matcher can replace it transparently later.

const S = 3;              // scales per octave
const SIGMA0 = 1.6;       // base scale
const INIT_SIGMA = 0.5;   // assumed camera blur of the input
const PEAK_THRESH = 0.02 / S;  // DoG contrast threshold (COLMAP-ish)
const EDGE_R = 10;        // edge rejection ratio
const ORI_BINS = 36;
const MAX_ORI = 2;        // orientations per keypoint (COLMAP default 2)
const DW = 4;             // descriptor width (4x4 cells)
const DB = 8;             // orientation bins per cell

function gaussianBlur(src, w, h, sigma) {
  const r = Math.max(1, Math.ceil(sigma * 3));
  const k = new Float32Array(2 * r + 1);
  let sum = 0;
  for (let i = -r; i <= r; i++) { k[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma)); sum += k[i + r]; }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  const tmp = new Float32Array(w * h);
  const dst = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const off = y * w;
    for (let x = 0; x < w; x++) {
      let v = 0;
      for (let i = -r; i <= r; i++) {
        const xx = Math.min(w - 1, Math.max(0, x + i));
        v += src[off + xx] * k[i + r];
      }
      tmp[off + x] = v;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 0;
      for (let i = -r; i <= r; i++) {
        const yy = Math.min(h - 1, Math.max(0, y + i));
        v += tmp[yy * w + x] * k[i + r];
      }
      dst[y * w + x] = v;
    }
  }
  return dst;
}

function downsample2(src, w, h) {
  const nw = w >> 1, nh = h >> 1;
  const dst = new Float32Array(nw * nh);
  for (let y = 0; y < nh; y++)
    for (let x = 0; x < nw; x++)
      dst[y * nw + x] = src[(2 * y) * w + 2 * x];
  return { img: dst, w: nw, h: nh };
}

/** Extract SIFT keypoints + descriptors.
 *  gray: Float32Array 0..1. Returns { n, x, y, scale, angle, desc(Uint8Array n*128) }. */
export function detectSift(gray, w, h, maxFeats = 4000, firstOctave = 0, peakScale = 1) {
  const PT = PEAK_THRESH * peakScale;
  let baseImg = gray, bw = w, bh = h;
  if (firstOctave === -1) {
    // 2x upsample (bilinear) — more small-scale features, 4x the work
    const nw = w * 2, nh = h * 2;
    const up = new Float32Array(nw * nh);
    for (let y = 0; y < nh; y++) {
      const sy = Math.min(h - 1.001, y / 2);
      const y0 = sy | 0, ty = sy - y0;
      for (let x = 0; x < nw; x++) {
        const sx = Math.min(w - 1.001, x / 2);
        const x0 = sx | 0, tx = sx - x0;
        const o = y0 * w + x0;
        up[y * nw + x] = gray[o] * (1 - tx) * (1 - ty) + gray[o + 1] * tx * (1 - ty) +
                         gray[o + w] * (1 - tx) * ty + gray[o + w + 1] * tx * ty;
      }
    }
    baseImg = up; bw = nw; bh = nh;
  }
  const octScale0 = firstOctave === -1 ? 0.5 : 1;

  // blur base to SIGMA0 (input assumed at INIT_SIGMA / octScale0)
  const initS = INIT_SIGMA / octScale0;
  const d0 = Math.sqrt(Math.max(0.01, SIGMA0 * SIGMA0 - initS * initS));
  let oct = { img: gaussianBlur(baseImg, bw, bh, d0), w: bw, h: bh };

  const nOct = Math.max(1, Math.floor(Math.log2(Math.min(bw, bh))) - 3);
  const kStep = Math.pow(2, 1 / S);
  const feats = [];

  for (let o = 0; o < nOct; o++) {
    const { img, w: ow, h: oh } = oct;
    // gaussian stack
    const G = [img];
    let sigPrev = SIGMA0;
    for (let s = 1; s < S + 3; s++) {
      const sigCur = SIGMA0 * Math.pow(kStep, s);
      const dSig = Math.sqrt(sigCur * sigCur - sigPrev * sigPrev);
      G.push(gaussianBlur(G[s - 1], ow, oh, dSig));
      sigPrev = sigCur;
    }
    // DoG stack
    const D = [];
    for (let s = 0; s < S + 2; s++) {
      const d = new Float32Array(ow * oh);
      for (let i = 0; i < ow * oh; i++) d[i] = G[s + 1][i] - G[s][i];
      D.push(d);
    }
    // extrema in D[1..S]
    const B = 5;
    for (let s = 1; s <= S; s++) {
      const d0a = D[s - 1], d1 = D[s], d2 = D[s + 1];
      for (let y = B; y < oh - B; y++) {
        for (let x = B; x < ow - B; x++) {
          const i = y * ow + x;
          const v = d1[i];
          if (Math.abs(v) < 0.8 * PT) continue;
          let isMax = true, isMin = true;
          for (let dy = -1; dy <= 1 && (isMax || isMin); dy++)
            for (let dx = -1; dx <= 1 && (isMax || isMin); dx++) {
              const j = i + dy * ow + dx;
              for (const dd of [d0a, d1, d2]) {
                const u = dd[j];
                if (dd === d1 && j === i) continue;
                if (u >= v) isMax = false;
                if (u <= v) isMin = false;
              }
            }
          if (!isMax && !isMin) continue;
          // 3D quadratic refinement
          let xi = x, yi = y, si = s, ok = false;
          let ox = 0, oy = 0, os = 0, contrast = 0;
          for (let it = 0; it < 5; it++) {
            const ii = yi * ow + xi;
            const Da = D[si - 1], Dbm = D[si], Dc = D[si + 1];
            const dx1 = (Dbm[ii + 1] - Dbm[ii - 1]) / 2;
            const dy1 = (Dbm[ii + ow] - Dbm[ii - ow]) / 2;
            const ds1 = (Dc[ii] - Da[ii]) / 2;
            const dxx = Dbm[ii + 1] + Dbm[ii - 1] - 2 * Dbm[ii];
            const dyy = Dbm[ii + ow] + Dbm[ii - ow] - 2 * Dbm[ii];
            const dss = Dc[ii] + Da[ii] - 2 * Dbm[ii];
            const dxy = (Dbm[ii + ow + 1] - Dbm[ii + ow - 1] - Dbm[ii - ow + 1] + Dbm[ii - ow - 1]) / 4;
            const dxs = (Dc[ii + 1] - Dc[ii - 1] - Da[ii + 1] + Da[ii - 1]) / 4;
            const dys = (Dc[ii + ow] - Dc[ii - ow] - Da[ii + ow] + Da[ii - ow]) / 4;
            // solve H * off = -g (3x3)
            const H = [dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss];
            const det =
              H[0] * (H[4] * H[8] - H[5] * H[7]) - H[1] * (H[3] * H[8] - H[5] * H[6]) + H[2] * (H[3] * H[7] - H[4] * H[6]);
            if (Math.abs(det) < 1e-20) break;
            const inv = 1 / det;
            ox = -inv * ((H[4] * H[8] - H[5] * H[7]) * dx1 + (H[2] * H[7] - H[1] * H[8]) * dy1 + (H[1] * H[5] - H[2] * H[4]) * ds1);
            oy = -inv * ((H[5] * H[6] - H[3] * H[8]) * dx1 + (H[0] * H[8] - H[2] * H[6]) * dy1 + (H[2] * H[3] - H[0] * H[5]) * ds1);
            os = -inv * ((H[3] * H[7] - H[4] * H[6]) * dx1 + (H[1] * H[6] - H[0] * H[7]) * dy1 + (H[0] * H[4] - H[1] * H[3]) * ds1);
            if (Math.abs(ox) < 0.5 && Math.abs(oy) < 0.5 && Math.abs(os) < 0.5) {
              contrast = Dbm[ii] + 0.5 * (dx1 * ox + dy1 * oy + ds1 * os);
              // edge rejection on the 2x2 spatial Hessian
              const tr = dxx + dyy, dt = dxx * dyy - dxy * dxy;
              if (dt > 0 && tr * tr / dt < (EDGE_R + 1) * (EDGE_R + 1) / EDGE_R && Math.abs(contrast) >= PT) ok = true;
              break;
            }
            xi += Math.round(ox); yi += Math.round(oy); si += Math.round(os);
            if (si < 1 || si > S || xi < B || xi >= ow - B || yi < B || yi >= oh - B) break;
          }
          if (!ok) continue;
          const octMul = octScale0 * Math.pow(2, o);
          feats.push({
            xo: xi + ox, yo: yi + oy, so: si + os, oct: o, lvl: si,
            x: (xi + ox) * octMul, y: (yi + oy) * octMul,
            scale: SIGMA0 * Math.pow(kStep, si + os) * octMul,
            contrast: Math.abs(contrast),
            G: G[si], ow, oh, octMul,
          });
        }
      }
    }
    if (oct.w >= 48 && oct.h >= 48 && o + 1 < nOct) oct = downsample2(G[S], ow, oh);
    else break;
  }

  // keep strongest by contrast
  feats.sort((a, b) => b.contrast - a.contrast);
  if (feats.length > maxFeats) feats.length = maxFeats;

  // orientations + descriptors
  const out = [];
  const hist = new Float32Array(ORI_BINS);
  for (const f of feats) {
    const { G: L, ow, oh } = f;
    const sigW = 1.5 * SIGMA0 * Math.pow(kStep, f.so - f.lvl + f.lvl); // scale at octave coords
    const sig = 1.5 * (f.scale / f.octMul);
    const rad = Math.round(3 * sig);
    const cx = Math.round(f.xo), cy = Math.round(f.yo);
    if (cx < rad + 1 || cy < rad + 1 || cx >= ow - rad - 1 || cy >= oh - rad - 1) {
      if (cx < 2 || cy < 2 || cx >= ow - 2 || cy >= oh - 2) continue;
    }
    hist.fill(0);
    const r2max = rad * rad;
    for (let dy = -rad; dy <= rad; dy++) {
      const yy = cy + dy;
      if (yy < 1 || yy >= oh - 1) continue;
      for (let dx = -rad; dx <= rad; dx++) {
        const xx = cx + dx;
        if (xx < 1 || xx >= ow - 1) continue;
        if (dx * dx + dy * dy > r2max) continue;
        const i = yy * ow + xx;
        const gx = (L[i + 1] - L[i - 1]) / 2;
        const gy = (L[i + ow] - L[i - ow]) / 2;
        const mag = Math.hypot(gx, gy);
        const ang = Math.atan2(gy, gx);
        const wgt = Math.exp(-(dx * dx + dy * dy) / (2 * sig * sig));
        let bin = Math.round((ang / (2 * Math.PI)) * ORI_BINS);
        bin = ((bin % ORI_BINS) + ORI_BINS) % ORI_BINS;
        hist[bin] += mag * wgt;
      }
    }
    // smooth histogram
    for (let it = 0; it < 6; it++) {
      const first = hist[0], prevKeep = hist[ORI_BINS - 1];
      let prev = prevKeep;
      for (let b = 0; b < ORI_BINS; b++) {
        const cur = hist[b];
        hist[b] = (prev + cur + (b + 1 < ORI_BINS ? hist[b + 1] : first)) / 3;
        prev = cur;
      }
    }
    let hmax = 0;
    for (let b = 0; b < ORI_BINS; b++) hmax = Math.max(hmax, hist[b]);
    const oris = [];
    for (let b = 0; b < ORI_BINS && oris.length < MAX_ORI; b++) {
      const l = hist[(b + ORI_BINS - 1) % ORI_BINS], c = hist[b], r = hist[(b + 1) % ORI_BINS];
      if (c > l && c > r && c >= 0.8 * hmax) {
        const off = 0.5 * (l - r) / (l - 2 * c + r);
        oris.push(((b + off) / ORI_BINS) * 2 * Math.PI);
      }
    }
    for (const theta of oris) {
      const desc = describeSift(L, ow, oh, f.xo, f.yo, f.scale / f.octMul, theta);
      if (desc) out.push({ x: f.x, y: f.y, scale: f.scale, angle: theta, desc });
    }
  }

  const n = out.length;
  const res = {
    n,
    x: new Float32Array(n), y: new Float32Array(n),
    scale: new Float32Array(n), angle: new Float32Array(n),
    desc: new Uint8Array(n * 128),
  };
  out.forEach((f, i) => {
    res.x[i] = f.x; res.y[i] = f.y; res.scale[i] = f.scale; res.angle[i] = f.angle;
    res.desc.set(f.desc, i * 128);
  });
  return res;
}

function describeSift(L, ow, oh, xf, yf, sig, theta) {
  const binW = 3 * sig;                    // pixels per descriptor cell
  const radius = Math.round(binW * Math.SQRT1_2 * (DW + 1));
  const cx = Math.round(xf), cy = Math.round(yf);
  const ct = Math.cos(theta), st = Math.sin(theta);
  const acc = new Float32Array(DW * DW * DB);
  for (let dy = -radius; dy <= radius; dy++) {
    const yy = cy + dy;
    if (yy < 1 || yy >= oh - 1) continue;
    for (let dx = -radius; dx <= radius; dx++) {
      const xx = cx + dx;
      if (xx < 1 || xx >= ow - 1) continue;
      // rotate into the keypoint frame, in cell units, centered on the 4x4 grid
      const u = (ct * dx + st * dy) / binW + DW / 2 - 0.5;
      const v = (-st * dx + ct * dy) / binW + DW / 2 - 0.5;
      if (u <= -1 || u >= DW || v <= -1 || v >= DW) continue;
      const i = yy * ow + xx;
      const gx = (L[i + 1] - L[i - 1]) / 2;
      const gy = (L[i + ow] - L[i - ow]) / 2;
      const mag = Math.hypot(gx, gy);
      let ang = Math.atan2(gy, gx) - theta;
      ang = ((ang % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
      const ob = (ang / (2 * Math.PI)) * DB;
      const wgt = Math.exp(-(u - (DW / 2 - 0.5)) * (u - (DW / 2 - 0.5)) / (2 * (DW / 2) * (DW / 2))
                          - (v - (DW / 2 - 0.5)) * (v - (DW / 2 - 0.5)) / (2 * (DW / 2) * (DW / 2))) * mag;
      // trilinear scatter
      const u0 = Math.floor(u), v0 = Math.floor(v), o0 = Math.floor(ob);
      const fu = u - u0, fv = v - v0, fo = ob - o0;
      for (let du = 0; du <= 1; du++) {
        const ub = u0 + du;
        if (ub < 0 || ub >= DW) continue;
        const wu = du ? fu : 1 - fu;
        for (let dv = 0; dv <= 1; dv++) {
          const vb = v0 + dv;
          if (vb < 0 || vb >= DW) continue;
          const wv = dv ? fv : 1 - fv;
          for (let do_ = 0; do_ <= 1; do_++) {
            const obn = (o0 + do_) % DB;
            const wo = do_ ? fo : 1 - fo;
            acc[(vb * DW + ub) * DB + obn] += wgt * wu * wv * wo;
          }
        }
      }
    }
  }
  // normalize: L2 -> clamp 0.2 -> L2 -> RootSIFT (L1 + sqrt) -> uint8 x512
  let n2 = 0;
  for (let i = 0; i < acc.length; i++) n2 += acc[i] * acc[i];
  if (n2 < 1e-12) return null;
  let inv = 1 / Math.sqrt(n2);
  let n2b = 0;
  for (let i = 0; i < acc.length; i++) {
    acc[i] = Math.min(0.2, acc[i] * inv);
    n2b += acc[i] * acc[i];
  }
  inv = 1 / Math.sqrt(n2b);
  let l1 = 0;
  for (let i = 0; i < acc.length; i++) { acc[i] *= inv; l1 += acc[i]; }
  const out = new Uint8Array(128);
  for (let i = 0; i < acc.length; i++) {
    const v = Math.sqrt(acc[i] / l1) * 512;
    out[i] = Math.max(0, Math.min(255, Math.round(v)));
  }
  return out;
}

/** Ratio + cross-checked matching of uint8 SIFT descriptors (squared L2).
 *  Returns flat [ia0, ib0, ia1, ib1, ...]. */
export function matchSift(dA, nA, dB, nB, ratio = 0.8) {
  const best = (dX, nX, dY, nY) => {
    const out = new Int32Array(nX).fill(-1);
    for (let a = 0; a < nX; a++) {
      const oa = a * 128;
      let b1 = Infinity, b2 = Infinity, bi = -1;
      for (let b = 0; b < nY; b++) {
        const ob = b * 128;
        let s = 0;
        for (let k = 0; k < 128; k++) {
          const d = dX[oa + k] - dY[ob + k];
          s += d * d;
          if (s >= b2) break;
        }
        if (s < b1) { b2 = b1; b1 = s; bi = b; }
        else if (s < b2) b2 = s;
      }
      if (bi >= 0 && b1 < ratio * ratio * b2) out[a] = bi;
    }
    return out;
  };
  const ab = best(dA, nA, dB, nB);
  const ba = best(dB, nB, dA, nA);
  const res = [];
  for (let a = 0; a < nA; a++) {
    const b = ab[a];
    if (b >= 0 && ba[b] === a) res.push(a, b);
  }
  return res;
}
