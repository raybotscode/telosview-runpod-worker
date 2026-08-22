// sfm.js — minimal incremental Structure-from-Motion.
//
// features -> pairwise matching -> tracks (from raw mutual matches)
// -> focal search: for each candidate focal length, two-view initialization
//    (E-RANSAC over track correspondences) + incremental registration
// -> best reconstruction wins.
//
// Raw (ratio+cross-checked) matches build the track graph directly; geometric
// filtering happens later via reprojection checks. This keeps tracks connected
// on low-parallax video-style input where per-pair essential matrices are
// ill-conditioned. Tracks chain matches across many frames, connecting wide
// baselines that direct matching cannot.
//
// All geometry runs on normalized image coordinates. Since EXIF is not read,
// several focal-length guesses are tried (relative to f = 1.2 * max(w, h)).

import { detectAndDescribe, detectAndDescribeMS } from './features.js';
import { detectSift, matchSift } from './sift.js';
import { gpuMatchAll } from './gpumatch.js';
import { matchDescriptors } from './matching.js';
import {
  I3, ransacE, decomposeE, selectPose, triangulateN, parallaxAngle,
  pnpRansac, refinePose, reprojError, makeRng, m3t, m3mulv,
} from './geometry.js';
import { bundleAdjust } from './ba.js';
import { rotationAveraging, globalPositionsJoint } from './global.js';

const MAXF = 8192; // feature-id stride per image (must exceed per-image feature count; SIFT emits up to 2 orientations/keypoint)
const FOCAL_SCALES = [1.0, 0.8, 0.65, 1.3]; // relative to the 1.2*maxDim guess

const tick = () => new Promise((r) => setTimeout(r, 0));

class UnionFind {
  constructor(n) {
    this.p = new Int32Array(n);
    for (let i = 0; i < n; i++) this.p[i] = i;
  }
  find(x) {
    let r = x;
    while (this.p[r] !== r) r = this.p[r];
    while (this.p[x] !== r) { const nx = this.p[x]; this.p[x] = r; x = nx; }
    return r;
  }
  union(a, b) {
    const ra = this.find(a), rb = this.find(b);
    if (ra !== rb) this.p[ra] = rb;
  }
}

// Two graph profiles, declared per scene (index.json "graph"; default 'walk'):
//   walk  — neighbor window 10, sparse long-range. Safe for handheld paths:
//           on self-similar content (camping tents) wider windows admit
//           epipolar-consistent repeated-texture mismatches that poison the
//           union-find tracks and cost whole sections (113 -> 64 registered).
//   orbit — window 16 + dense long-range. Slow orbits (T&T train/truck)
//           genuinely overlap at those spans and the extra constraints stiffen
//           the chain: ATE 10% -> 4.5% on train-84.
function buildPairs(n, profile = 'walk') {
  const dense = profile === 'dense';
  const orbit = profile === 'orbit' || dense;
  const win = dense ? 20 : orbit ? 16 : 10;
  const pairs = [];
  if (n <= 30) {
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) pairs.push([i, j]);
  } else {
    const seen = new Set();
    const add = (i, j) => {
      if (i === j || i < 0 || j >= n) return;
      const k = i < j ? i * 10000 + j : j * 10000 + i;
      if (!seen.has(k)) { seen.add(k); pairs.push(i < j ? [i, j] : [j, i]); }
    };
    for (let i = 0; i < n; i++)
      for (let d = 1; d <= win; d++) add(i, i + d);
    // long-range pairs for loop closure ('dense': every i, every 2nd j —
    // the subsampled grid left long capture loops pinned too loosely)
    for (let i = 0; i < n; i += dense ? 1 : orbit ? 2 : 4)
      for (let j = i + win + 2; j < n; j += dense ? 2 : orbit ? 3 : 4) add(i, j);
  }
  return pairs;
}

// ---- subpixel observation refinement (Lucas-Kanade) ----
// Corner detections are pixel-quantized (~1.5px BA residual floor); a chain
// of short-baseline cameras with that much obs noise has smooth low-frequency
// bend modes as its NOISE FLOOR — no amount of bundle adjustment removes them.
// Align each observation's patch against a single reference patch of its track
// (zero-mean SSD, translation-only LK) so all obs of a track agree at subpixel
// on the SAME physical point.
function refineObsLK(images, feats, tracks, poses, vlog) {
  const W = 5;                 // half window -> 11x11 patch
  const MAXSHIFT = 2.5;        // px; larger moves are considered mismatches
  const P = 2 * W + 1;
  const refP = new Float32Array(P * P);
  const curP = new Float32Array(P * P);
  const gxP = new Float32Array(P * P);
  const gyP = new Float32Array(P * P);

  const bil = (g, w, h, x, y) => {
    const x0 = x | 0, y0 = y | 0;
    const fx = x - x0, fy = y - y0;
    const o = y0 * w + x0;
    return g[o] * (1 - fx) * (1 - fy) + g[o + 1] * fx * (1 - fy) +
           g[o + w] * (1 - fx) * fy + g[o + w + 1] * fx * fy;
  };

  let refined = 0, tried = 0, rejected = 0;
  const t0 = performance.now();
  for (const tr of tracks) {
    if (!tr.X) continue;
    const ok = tr.obs.filter((o) => o.ok && poses[o.img]);
    if (ok.length < 2) continue;
    const sorted = ok.slice().sort((a, b) => a.img - b.img);
    const ref = sorted[sorted.length >> 1];   // middle obs = reference
    const imR = images[ref.img];
    const rx = feats[ref.img].x[ref.feat], ry = feats[ref.img].y[ref.feat];
    if (rx < W + 2 || ry < W + 2 || rx > imR.fw - W - 3 || ry > imR.fh - W - 3) continue;
    let refMean = 0;
    for (let dy = -W, k = 0; dy <= W; dy++)
      for (let dx = -W; dx <= W; dx++, k++) refMean += refP[k] = bil(imR.gray, imR.fw, imR.fh, rx + dx, ry + dy);
    refMean /= P * P;

    for (const o of sorted) {
      if (o === ref) continue;
      tried++;
      const im = images[o.img];
      const x0 = feats[o.img].x[o.feat], y0 = feats[o.img].y[o.feat];
      let u = 0, v = 0, okConv = false;
      for (let it = 0; it < 10; it++) {
        const cx = x0 + u, cy = y0 + v;
        if (cx < W + 2 || cy < W + 2 || cx > im.fw - W - 3 || cy > im.fh - W - 3) break;
        let curMean = 0;
        for (let dy = -W, k = 0; dy <= W; dy++)
          for (let dx = -W; dx <= W; dx++, k++) {
            curP[k] = bil(im.gray, im.fw, im.fh, cx + dx, cy + dy);
            gxP[k] = 0.5 * (bil(im.gray, im.fw, im.fh, cx + dx + 1, cy + dy) - bil(im.gray, im.fw, im.fh, cx + dx - 1, cy + dy));
            gyP[k] = 0.5 * (bil(im.gray, im.fw, im.fh, cx + dx, cy + dy + 1) - bil(im.gray, im.fw, im.fh, cx + dx, cy + dy - 1));
            curMean += curP[k];
          }
        curMean /= P * P;
        let a00 = 0, a01 = 0, a11 = 0, b0 = 0, b1 = 0;
        for (let k = 0; k < P * P; k++) {
          const e = (curP[k] - curMean) - (refP[k] - refMean);
          a00 += gxP[k] * gxP[k]; a01 += gxP[k] * gyP[k]; a11 += gyP[k] * gyP[k];
          b0 -= gxP[k] * e; b1 -= gyP[k] * e;
        }
        const det = a00 * a11 - a01 * a01;
        if (det < 1e-12) break;
        const du = (a11 * b0 - a01 * b1) / det;
        const dv = (a00 * b1 - a01 * b0) / det;
        u += du; v += dv;
        if (u * u + v * v > MAXSHIFT * MAXSHIFT) { u = 1e9; break; }
        if (du * du + dv * dv < 1e-4) { okConv = true; break; }
      }
      if (okConv && u * u + v * v <= MAXSHIFT * MAXSHIFT) {
        feats[o.img].x[o.feat] = x0 + u;
        feats[o.img].y[o.feat] = y0 + v;
        refined++;
      } else rejected++;
    }
  }
  vlog(`subpixel LK refine: ${refined}/${tried} obs refined (${rejected} rejected) ` +
       `in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
}

// ---- guided track extension ----
// For every triangulated track, project its (bundle-adjusted) 3D point into
// each registered frame that does not observe it yet, and look for an
// unclaimed detected feature within a small pixel radius whose descriptor
// matches the track (min Hamming over the track's existing descriptors).
// Returns the list of added observations [{ tr, o }].
function extendTracks(feats, tracks, poses, K, regList, k1, k2, im0, vlog) {
  const DESC_WORDS = 8;
  const RADIUS = 4;        // px search radius around the projection
  const HAM_MAX = 64;      // stricter than the 90 used for blind matching
  const MARGIN = 6;        // stay away from the frame border
  const CELL = 8;          // spatial-grid cell size in px
  const w = im0.fw, h = im0.fh;

  const hamming = (dA, oA, dB, oB) => {
    let d = 0;
    for (let k = 0; k < DESC_WORDS; k++) {
      let v = (dA[oA + k] ^ dB[oB + k]) >>> 0;
      v = v - ((v >>> 1) & 0x55555555);
      v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
      d += (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
    }
    return d;
  };
  // SIFT descriptors are 128 x uint8 with L2 matching; binary BRIEF uses
  // Hamming. Pick per feature type.
  const isSift = !!feats[regList[0]].sift;
  const STRIDE = isSift ? 128 : DESC_WORDS;
  const DMAX = isSift ? 40000 : HAM_MAX; // L2^2 (rootSIFT x512) vs bits
  const descDist = isSift
    ? (dA, oA, dB, oB) => {
        let s = 0;
        for (let k = 0; k < 128; k++) {
          const d = dA[oA + k] - dB[oB + k];
          s += d * d;
          if (s > DMAX) return s;
        }
        return s;
      }
    : hamming;

  // spatial grid + claimed flags per registered image
  const gw = Math.ceil(w / CELL), gh = Math.ceil(h / CELL);
  const grid = new Map();     // img -> Map(cell -> [featIdx])
  const claimed = new Map();  // img -> Uint8Array(nFeats)
  for (const img of regList) {
    const f = feats[img];
    const g = new Map();
    for (let i = 0; i < f.n; i++) {
      const c = Math.min(gh - 1, Math.max(0, (f.y[i] / CELL) | 0)) * gw +
                Math.min(gw - 1, Math.max(0, (f.x[i] / CELL) | 0));
      let a = g.get(c);
      if (!a) g.set(c, a = []);
      a.push(i);
    }
    grid.set(img, g);
    claimed.set(img, new Uint8Array(f.n));
  }
  for (const tr of tracks)
    for (const o of tr.obs) {
      const cl = claimed.get(o.img);
      if (cl) cl[o.feat] = 1;
    }

  let lenBefore = 0, nTracks = 0;
  for (const tr of tracks) {
    if (!tr.X) continue;
    nTracks++;
    for (const o of tr.obs) if (o.ok && poses[o.img]) lenBefore++;
  }

  const extObs = [];
  for (const tr of tracks) {
    if (!tr.X) continue;
    const okObs = tr.obs.filter((o) => o.ok && poses[o.img]);
    if (okObs.length < 2) continue;
    const has = new Set(tr.obs.map((o) => o.img)); // incl. rejected obs
    // representative descriptors (cap for speed)
    const descs = okObs.slice(0, 6).map((o) => ({ d: feats[o.img].desc, off: o.feat * STRIDE }));
    for (const img of regList) {
      if (has.has(img)) continue;
      const R = poses[img].R, t = poses[img].t, X = tr.X;
      const pz = R[6]*X[0] + R[7]*X[1] + R[8]*X[2] + t[2];
      if (pz < 1e-9) continue;
      const x = (R[0]*X[0] + R[1]*X[1] + R[2]*X[2] + t[0]) / pz;
      const y = (R[3]*X[0] + R[4]*X[1] + R[5]*X[2] + t[1]) / pz;
      const r2 = x*x + y*y;
      const D = 1 + k1*r2 + k2*r2*r2;
      const u = K[img].f * x * D + K[img].cx;
      const v = K[img].f * y * D + K[img].cy;
      if (!(u > MARGIN && u < w - MARGIN && v > MARGIN && v < h - MARGIN)) continue;
      const f = feats[img], g = grid.get(img), cl = claimed.get(img);
      const cu = (u / CELL) | 0, cv = (v / CELL) | 0;
      let best = -1, bestHam = DMAX + 1, bestD2 = Infinity;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const cy2 = cv + dy, cx2 = cu + dx;
          if (cx2 < 0 || cy2 < 0 || cx2 >= gw || cy2 >= gh) continue;
          const cell = g.get(cy2 * gw + cx2);
          if (!cell) continue;
          for (const fi of cell) {
            if (cl[fi]) continue;
            const du = f.x[fi] - u, dv = f.y[fi] - v;
            const d2 = du*du + dv*dv;
            if (d2 > RADIUS * RADIUS) continue;
            let hm = Infinity;
            for (const dd of descs) hm = Math.min(hm, descDist(f.desc, fi * STRIDE, dd.d, dd.off));
            if (hm < bestHam || (hm === bestHam && d2 < bestD2)) { best = fi; bestHam = hm; bestD2 = d2; }
          }
        }
      if (best >= 0 && bestHam <= DMAX) {
        const o = { img, feat: best, ok: true };
        tr.obs.push(o);
        has.add(img);
        cl[best] = 1;
        extObs.push({ tr, o });
      }
    }
  }
  vlog(`track extension: +${extObs.length} obs ` +
       `(mean track length ${(lenBefore / Math.max(1, nTracks)).toFixed(2)} -> ` +
       `${((lenBefore + extObs.length) / Math.max(1, nTracks)).toFixed(2)})`);
  return extObs;
}

/**
 * Run incremental SfM.
 * images: [{ name, fw, fh, gray: Float32Array }]
 * log: (msg) => void
 * sampleColor: (imgIdx, x, y) => [r, g, b]
 * Returns { cams, points }.
 */
export async function runSfM(images, log, sampleColor, opts = {}) {
  const n = images.length;
  if (n < 2) throw new Error('need at least 2 images');
  const signal = opts.signal;
  const checkAbort = () => {
    if (signal && signal.aborted) throw new DOMException('SfM aborted', 'AbortError');
  };
  // structured progress alongside log(): { stage, done, total, detail }
  // stages: 'features' | 'matching' | 'focal' | 'register' | 'ba'
  const ev = opts.onEvent || (() => {});

  // ---- features ----
  // SIFT is the DEFAULT since 2026-08-19: it beat or tied oriented-BRIEF on
  // every validation scene (truck/train-84 ATE 0.04% = COLMAP parity, camping
  // 27.2, synthetic 43.0/40.8) with worker extraction + GPU matching keeping
  // it fast. features:'brief' restores the old binary pipeline.
  const useSift = opts.features !== 'brief';
  log(`detecting features in ${n} images ...`);
  const feats = [];
  if (useSift && typeof Worker !== 'undefined' && opts.workers !== false) {
    // SIFT extraction is ~1s/image of pure CPU — run it on a worker pool
    const t0f = performance.now();
    const nW = Math.min(8, Math.max(2, (navigator.hardwareConcurrency || 4) - 2));
    const workers = Array.from({ length: nW },
      () => new Worker(new URL('./featworker.js', import.meta.url), { type: 'module' }));
    const results = new Array(n);
    let doneF = 0;
    await new Promise((resolve, reject) => {
      let next = 0;
      const feed = (wk) => {
        if (next >= n) return;
        const id = next++;
        wk.onmessage = (e) => {
          results[e.data.id] = e.data;
          doneF++;
          // detail carries the REAL keypoints (references, feature scale) so a
          // UI can draw them as they arrive
          ev({ stage: 'features', done: doneF, total: n,
               detail: { image: e.data.id, n: e.data.n, x: e.data.x, y: e.data.y } });
          if (doneF % 16 === 0) log(`  features: ${doneF}/${n}`);
          if (doneF === n) resolve();
          else feed(wk);
        };
        wk.onerror = (err) => reject(new Error('feature worker: ' + (err.message || err)));
        // fine-scale defaults: firstOctave -1 (upsampled octave) flooded the
        // contrast-sorted cap with poorly-localized features and warped truck
        // to 8.5% ATE; firstOctave 0 + 1800 feats = 0.04% on truck AND train
        wk.postMessage({
          id, gray: images[id].gray, w: images[id].fw, h: images[id].fh,
          maxFeats: opts.siftFeats || 3900, firstOctave: opts.siftFirstOctave ?? 0,
          peakScale: opts.siftPeak ?? 1,
        });
      };
      workers.forEach(feed);
    });
    workers.forEach((w) => w.terminate());
    for (let i = 0; i < n; i++) {
      const f = results[i];
      f.sift = true;
      if (f.n > MAXF) throw new Error('too many features per image');
      f.xn = new Float32Array(f.n);
      f.yn = new Float32Array(f.n);
      feats.push(f);
    }
    log(`  SIFT extraction: ${n} images on ${nW} workers in ${((performance.now() - t0f) / 1000).toFixed(1)}s`);
  } else {
    for (let i = 0; i < n; i++) {
      // 1500 is deliberate: raising it (tried 2200) admits weaker corners whose
      // BRIEF descriptors mismatch, and pose accuracy measurably degrades.
      // features:'sift' = real scale-space SIFT (COLMAP-grade, validated at
      // 82.5% keypoint recall vs COLMAP's own extraction; slower).
      const f = useSift
        ? detectSift(images[i].gray, images[i].fw, images[i].fh, opts.siftFeats || 3900,
            opts.siftFirstOctave ?? 0, opts.siftPeak ?? 1)
        : opts.msFeatures
          ? detectAndDescribeMS(images[i].gray, images[i].fw, images[i].fh, 1500)
          : detectAndDescribe(images[i].gray, images[i].fw, images[i].fh, 1500);
      f.sift = useSift;
      if (f.n > MAXF) throw new Error('too many features per image');
      f.xn = new Float32Array(f.n);
      f.yn = new Float32Array(f.n);
      feats.push(f);
      ev({ stage: 'features', done: i + 1, total: n });
      if (i % 8 === 7) { log(`  features: ${i + 1}/${n}`); await tick(); checkAbort(); }
    }
  }
  checkAbort();
  log(`  avg features/image: ${Math.round(feats.reduce((s, f) => s + f.n, 0) / n)}`);
  await tick();

  // Adaptive rescue: images that came back feature-poor get a second pass
  // with the base octave upsampled 2x (COLMAP's own default). Soft indoor
  // texture — painted walls, ceilings — is invisible at octave 0 and those
  // frames starve registration (playroom: 150/225 -> 209/225). ONLY the
  // starved images: fine-scale features on rich images once flooded the
  // contrast-sorted cap and warped truck to 8.5% ATE (note above).
  if (useSift && (opts.siftFirstOctave ?? 0) >= 0) {
    // density, not an absolute count: fewer than one keypoint per ~700
    // feature-scale pixels marks a starved image. An absolute threshold
    // dragged camping's small video frames into the rescue and the
    // fine-scale flood cost it 2.15% ATE (0.38% at 1/500; 1/700 leaves its
    // marginal frames alone while playroom's blank walls still qualify).
    const starved = [];
    for (let i = 0; i < n; i++) {
      if (feats[i].n * 700 < images[i].fw * images[i].fh) starved.push(i);
    }
    if (starved.length) {
      log(`  ${starved.length} feature-poor images (density) — second pass at first octave -1`);
      const t0r = performance.now();
      const adopt = (data) => {
        const f = data;
        f.sift = true;
        if (f.n > MAXF) throw new Error('too many features per image');
        f.xn = new Float32Array(f.n);
        f.yn = new Float32Array(f.n);
        feats[f.id] = f;
      };
      if (typeof Worker !== 'undefined' && opts.workers !== false) {
        const nW = Math.min(8, Math.max(2, (navigator.hardwareConcurrency || 4) - 2));
        const workers = Array.from({ length: nW },
          () => new Worker(new URL('./featworker.js', import.meta.url), { type: 'module' }));
        let doneR = 0;
        await new Promise((resolve, reject) => {
          let next = 0;
          const feed = (wk) => {
            if (next >= starved.length) return;
            const id = starved[next++];
            wk.onmessage = (e) => {
              adopt(e.data);
              doneR++;
              ev({ stage: 'features', done: doneR, total: starved.length,
                   detail: { image: e.data.id, n: e.data.n, x: e.data.x, y: e.data.y } });
              if (doneR === starved.length) resolve();
              else feed(wk);
            };
            wk.onerror = (err) => reject(new Error('feature worker: ' + (err.message || err)));
            wk.postMessage({
              id, gray: images[id].gray, w: images[id].fw, h: images[id].fh,
              maxFeats: opts.siftFeats || 3900, firstOctave: -1,
              peakScale: opts.siftPeak ?? 1,
            });
          };
          workers.forEach(feed);
        });
        workers.forEach((w) => w.terminate());
      } else {
        for (const id of starved) {
          const f = detectSift(images[id].gray, images[id].fw, images[id].fh,
            opts.siftFeats || 3900, -1, opts.siftPeak ?? 1);
          f.id = id;
          adopt(f);
          await tick(); checkAbort();
        }
      }
      const avg2 = Math.round(starved.reduce((s, id) => s + feats[id].n, 0) / starved.length);
      log(`  rescue: avg ${avg2} features on those images ` +
        `(${((performance.now() - t0r) / 1000).toFixed(1)}s)`);
      checkAbort();
      await tick();
    }
  }

  // ---- pairwise matching ----
  // Mutual matches build the track graph. When an essential matrix fits a
  // pair well, its inliers replace the raw matches (removes contamination on
  // wide-baseline pairs); when E is unreliable (low-parallax video pairs) the
  // raw matches are kept so tracks stay connected.
  const matchRng = makeRng(24680);
  const K0 = images.map((im) => ({
    f: 1.2 * Math.max(im.fw, im.fh), cx: im.fw / 2, cy: im.fh / 2,
  }));
  // With SIFT the wide 'orbit' graph is safe everywhere (the walk-window
  // limit was a BRIEF pathology: repeated-texture mismatches at wide
  // baselines poisoned tracks — camping went 113 -> 64 registered). SIFT
  // camping with the wide graph: 113/113, best-ever PSNR, tail drift down.
  // 'dense' since 2026-08-21: the subsampled long-range grid left a 250-image
  // loop pinned so loosely that two 8-16 camera clusters registered ~5 units
  // off (a ghost truck in training). Dense long-range: truck-250 ATE
  // 2.18% -> 0.00% vs COLMAP. GPU matching absorbs the extra pairs.
  const pairs = buildPairs(n, opts.graph || (useSift ? 'dense' : 'walk'));
  log(`matching ${pairs.length} image pairs ...`);
  const pairInfo = []; // { i, j, matches: [[fa, fb], ...] }
  const failedRich = []; // many matches but failed the E-gate (rescue candidates)
  let done = 0, filtered = 0;
  // SIFT matching runs on GPU when available (CPU 128-D L2 is the dominant
  // cost of a SIFT run: ~350s of 484s on train-84; the GPU does the whole
  // graph in ~1s of compute)
  let gpuAll = null;
  if (useSift && typeof navigator !== 'undefined' && navigator.gpu) {
    try {
      gpuAll = await gpuMatchAll(feats, pairs, 0.8, log, opts.gpu && opts.gpu.device);
    } catch (e) {
      log(`  GPU matcher unavailable (${e.message}) — CPU fallback`);
    }
  }
  for (let pi = 0; pi < pairs.length; pi++) {
    const [i, j] = pairs[pi];
    const m = gpuAll
      ? gpuAll[pi]
      : feats[i].sift
        ? matchSift(feats[i].desc, feats[i].n, feats[j].desc, feats[j].n)
        : matchDescriptors(feats[i].desc, feats[i].n, feats[j].desc, feats[j].n);
    done++;
    if (m.length / 2 >= 25) {
      let matches = [];
      for (let k = 0; k < m.length; k += 2) matches.push([m[k], m[k + 1]]);
      const x1s = matches.map(([fa]) => [(feats[i].x[fa] - K0[i].cx) / K0[i].f, (feats[i].y[fa] - K0[i].cy) / K0[i].f]);
      const x2s = matches.map(([, fb]) => [(feats[j].x[fb] - K0[j].cx) / K0[j].f, (feats[j].y[fb] - K0[j].cy) / K0[j].f]);
      const favg = (K0[i].f + K0[j].f) / 2;
      const res = ransacE(x1s, x2s, 2.5 / favg, matchRng, 600);
      // pairs whose matches can't support an essential matrix are junk —
      // letting their raw matches into the track graph merges unrelated
      // tracks, which then get dropped as conflicted
      if (res && res.inliers.length >= 25 && res.inliers.length >= 0.4 * matches.length) {
        matches = res.inliers.map((idx) => matches[idx]);
        filtered++;
        pairInfo.push({ i, j, matches });
      } else if (matches.length >= 60) {
        failedRich.push({ i, j, matches });
      }
    }
    const lastPair = pairInfo[pairInfo.length - 1];
    ev({ stage: 'matching', done, total: pairs.length, detail: {
      usable: pairInfo.length,
      // a drawable sample of the latest surviving pair (feature indices)
      pair: lastPair ? { i: lastPair.i, j: lastPair.j, sample: lastPair.matches.slice(0, 70) } : null,
    } });
    if (done % 40 === 0) { log(`  pairs: ${done}/${pairs.length} (${pairInfo.length} usable)`); await tick(); checkAbort(); }
  }
  checkAbort();
  log(`  usable pairs: ${pairInfo.length} (${filtered} E-filtered, rest raw)`);
  if (pairInfo.length === 0) throw new Error('no image pair with enough matches');
  await tick();

  // ---- component rescue ----
  // The strict 40%-ratio E-gate keeps the dense track graph clean, but a
  // capture-session seam (truck frames 51|52) can leave the IMAGE graph split
  // with every bridging pair sitting just under the gate — losing one pair
  // then costs 32 of 84 frames. Retry only the match-rich failed pairs that
  // would connect two components, with a bigger RANSAC budget and an
  // absolute-consensus acceptance. In-component pairs never get the looser
  // rule (a global relaxation measurably poisons the graph: 49/84 registered,
  // spurious k1 0.22).
  if (failedRich.length) {
    const comp = new Int32Array(n);
    for (let i = 0; i < n; i++) comp[i] = i;
    const findC = (x) => {
      let r = x;
      while (comp[r] !== r) r = comp[r];
      while (comp[x] !== r) { const nx = comp[x]; comp[x] = r; x = nx; }
      return r;
    };
    for (const p of pairInfo) { const a = findC(p.i), b = findC(p.j); if (a !== b) comp[a] = b; }
    const pre = Array.from({ length: n }, (_, i) => findC(i)); // snapshot BEFORE rescue
    // only bridge SUBSTANTIAL components (two solid capture sessions, like
    // truck 52|32). Wide-baseline photo sets fragment into confetti — gluing
    // those with marginal geometry produces flipped cameras (playroom went
    // 7 -> 15 "registered" with visibly wrong poses before this guard).
    const compSize = new Map();
    for (const r of pre) compSize.set(r, (compSize.get(r) || 0) + 1);
    const cand = failedRich
      .filter((c) => pre[c.i] !== pre[c.j] &&
                     compSize.get(pre[c.i]) >= 6 && compSize.get(pre[c.j]) >= 6)
      .sort((a, b) => b.matches.length - a.matches.length)
      .slice(0, 20);
    let rescued = 0;
    for (const c of cand) {
      const x1s = c.matches.map(([fa]) => [(feats[c.i].x[fa] - K0[c.i].cx) / K0[c.i].f, (feats[c.i].y[fa] - K0[c.i].cy) / K0[c.i].f]);
      const x2s = c.matches.map(([, fb]) => [(feats[c.j].x[fb] - K0[c.j].cx) / K0[c.j].f, (feats[c.j].y[fb] - K0[c.j].cy) / K0[c.j].f]);
      const favg = (K0[c.i].f + K0[c.j].f) / 2;
      const res = ransacE(x1s, x2s, 2.5 / favg, matchRng, 3000);
      if (res && res.inliers.length >= 30 && res.inliers.length >= 0.15 * c.matches.length) {
        pairInfo.push({ i: c.i, j: c.j, matches: res.inliers.map((idx) => c.matches[idx]) });
        rescued++;
        log(`  component rescue: pair ${c.i}-${c.j} accepted (${res.inliers.length}/${c.matches.length} inliers)`);
      }
    }
    if (rescued) log(`  component rescue: +${rescued} bridging pairs`);
    await tick();
  }

  // ---- feature tracks (union-find over raw matches) ----
  const uf = new UnionFind(n * MAXF);
  for (const p of pairInfo)
    for (const [fa, fb] of p.matches)
      uf.union(p.i * MAXF + fa, p.j * MAXF + fb);

  const groups = new Map();
  for (const p of pairInfo) {
    for (const [fa, fb] of p.matches) {
      for (const id of [p.i * MAXF + fa, p.j * MAXF + fb]) {
        const root = uf.find(id);
        if (!groups.has(root)) groups.set(root, new Set());
        groups.get(root).add(id);
      }
    }
  }
  // tracks: keep only unambiguous observations (one feature per image)
  const tracks = [];
  const featTrack = images.map((im, i) => new Int32Array(feats[i].n).fill(-1));
  for (const g of groups.values()) {
    if (g.size < 2) continue;
    const perImg = new Map();
    for (const id of g) {
      const img = (id / MAXF) | 0;
      if (!perImg.has(img)) perImg.set(img, []);
      perImg.get(img).push(id % MAXF);
    }
    // a track with two features in one image is a bad union-find merge —
    // drop it entirely rather than salvaging (mixed tracks poison PnP)
    let conflicted = false;
    const obs = [];
    for (const [img, fs] of perImg) {
      if (fs.length > 1) { conflicted = true; break; }
      obs.push({ img, feat: fs[0], ok: true });
    }
    if (conflicted || obs.length < 2) continue;
    const tid = tracks.length;
    tracks.push({ obs, X: null });
    for (const o of obs) featTrack[o.img][o.feat] = tid;
  }
  log(`  tracks: ${tracks.length}`);
  await tick();

  // shared-track counts per image pair (focal-independent)
  const sharedCount = new Map();
  for (const tr of tracks) {
    const imgs = tr.obs.map((o) => o.img).sort((a, b) => a - b);
    for (let a = 0; a < imgs.length; a++)
      for (let b = a + 1; b < imgs.length; b++) {
        const k = imgs[a] * 10000 + imgs[b];
        sharedCount.set(k, (sharedCount.get(k) || 0) + 1);
      }
  }

  // =========================================================================
  // Geometry stage, parameterized by focal scale. Fully self-contained so the
  // focal search can run it repeatedly on reset track state.
  // =========================================================================
  async function runGeometry(fScale, verbose, withBA = false) {
    const vlog = verbose ? log : () => {};
    const rng = makeRng(1234567);
    const K = images.map((im) => ({
      f: fScale * 1.2 * Math.max(im.fw, im.fh),
      cx: im.fw / 2,
      cy: im.fh / 2,
    }));
    for (let i = 0; i < n; i++) {
      const f = feats[i];
      for (let k = 0; k < f.n; k++) {
        f.xn[k] = (f.x[k] - K[i].cx) / K[i].f;
        f.yn[k] = (f.y[k] - K[i].cy) / K[i].f;
      }
    }
    // reset track state
    for (const tr of tracks) { tr.X = null; for (const o of tr.obs) o.ok = true; }

    const obsNorm = (o) => [feats[o.img].xn[o.feat], feats[o.img].yn[o.feat]];
    const poses = new Array(n).fill(null);
    const registered = new Set();
    const failed = new Set();
    // Generous reprojection budget (feature-scale px): triangulation from
    // low-parallax pairs is noisy; global refinement tightens it afterwards.
    const thN = (img) => 6.0 / K[img].f;

    function triangulateTrack(tr) {
      for (let attempt = 0; attempt < 2; attempt++) {
        const cams = [], obsPts = [], used = [];
        for (const o of tr.obs) {
          if (!o.ok || !poses[o.img]) continue;
          cams.push(poses[o.img]);
          obsPts.push(obsNorm(o));
          used.push(o);
        }
        if (cams.length < 2) { tr.X = null; return false; }
        const X = triangulateN(cams, obsPts);
        if (!X) { tr.X = null; return false; }
        let anyBad = false;
        for (let k = 0; k < cams.length; k++) {
          const err = reprojError(cams[k].R, cams[k].t, X, obsPts[k]);
          if (!(err < thN(used[k].img))) { used[k].ok = false; anyBad = true; }
        }
        if (anyBad) { tr.X = null; continue; }
        let maxAng = 0;
        for (let a = 0; a < cams.length - 1 && maxAng < 0.005; a++)
          for (let b = a + 1; b < cams.length && maxAng < 0.005; b++)
            maxAng = Math.max(maxAng, parallaxAngle(cams[a], cams[b], X));
        if (maxAng < 0.005) { tr.X = null; return false; }
        tr.X = X;
        return true;
      }
      return false;
    }

    const sfmOpts = opts;
    const useGlobal = !!opts.globalInit;

    // ---- initialization pair: ranked by shared tracks ----
    const candPairs = [...sharedCount.entries()]
      .map(([k, c]) => ({ i: (k / 10000) | 0, j: k % 10000, c }))
      .filter((p) => p.c >= 30)
      .sort((a, b) => b.c * (1 + Math.abs(b.j - b.i)) - a.c * (1 + Math.abs(a.j - a.i)))
      .slice(0, 30);

    const trackCorrs = (i, j) => {
      const x1s = [], x2s = [];
      for (const tr of tracks) {
        let oi = null, oj = null;
        for (const o of tr.obs) {
          if (!o.ok) continue;
          if (o.img === i) oi = o;
          if (o.img === j) oj = o;
        }
        if (oi && oj) { x1s.push(obsNorm(oi)); x2s.push(obsNorm(oj)); }
      }
      return { x1s, x2s };
    };

    // ---- GLOMAP-style global initialization ----
    // Solve all rotations at once (robust rotation averaging over pairwise
    // relative rotations), then all positions at once (translation-direction
    // registration). No incremental growth = no basin lock-in; the BA stack
    // downstream polishes. Opt-in: sfmOpts/opts.globalInit.
    if (useGlobal) {
      const t0g = performance.now();
      const gPairs = [...sharedCount.entries()]
        .map(([k, c]) => ({ i: (k / 10000) | 0, j: k % 10000, c }))
        .filter((p) => p.c >= 20);
      const rotEdges = [];
      const posRaw = [];
      for (const p of gPairs) {
        const { x1s, x2s } = trackCorrs(p.i, p.j);
        if (x1s.length < 20) continue;
        const favg = (K[p.i].f + K[p.j].f) / 2;
        const res = ransacE(x1s, x2s, 2.5 / favg, rng, 600);
        if (!res || res.inliers.length < 20 || res.inliers.length < 0.4 * x1s.length) continue;
        const pose = selectPose(decomposeE(res.U, res.V), x1s, x2s, res.inliers);
        if (!pose || pose.goodFrac < 0.6) continue;
        const w = Math.min(4, res.inliers.length / 50);
        rotEdges.push({ i: p.i, j: p.j, R: pose.R, w });
        // translation direction is meaningless without parallax
        if (pose.medianAngle > 0.01)
          posRaw.push({ i: p.i, j: p.j, t: pose.t, w: w * Math.min(1, pose.medianAngle / 0.03) });
      }
      vlog(`global init: ${rotEdges.length} rotation edges, ${posRaw.length} direction edges`);
      const ra = rotEdges.length >= n - 1 ? rotationAveraging(n, rotEdges, vlog) : null;
      if (!ra) { vlog('global init failed: rotation averaging'); return null; }
      // world direction of (C_i - C_j): the pair's t lives in cam-j coords,
      // so d_w = R_j^T t (using the AVERAGED R_j, not the noisy pairwise one)
      const posEdges = [];
      for (const e of posRaw) {
        if (!ra.R[e.i] || !ra.R[e.j]) continue;
        const d = m3mulv(m3t(ra.R[e.j]), e.t);
        const len = Math.hypot(d[0], d[1], d[2]);
        if (len < 1e-9) continue;
        posEdges.push({ i: e.i, j: e.j, d: [d[0] / len, d[1] / len, d[2] / len], w: e.w });
      }
      // camera-point rays: the tens of thousands of track observations are
      // what make positioning rigid on low-parallax video (camera-camera
      // directions alone gave 8-16% ATE; GLOMAP's core idea is exactly this
      // joint solve)
      const raysObs = [];
      let nPts = 0;
      for (const tr of tracks) {
        const ok = tr.obs.filter((o) => o.ok && ra.R[o.img]);
        if (ok.length < 2) continue;
        const k = nPts++;
        for (const o of ok) {
          const xn = feats[o.img].xn[o.feat], yn = feats[o.img].yn[o.feat];
          const nr = Math.hypot(xn, yn, 1);
          const vc = [xn / nr, yn / nr, 1 / nr];
          const R = ra.R[o.img]; // world ray = R^T vc
          raysObs.push({
            i: o.img, k,
            v: [
              R[0] * vc[0] + R[3] * vc[1] + R[6] * vc[2],
              R[1] * vc[0] + R[4] * vc[1] + R[7] * vc[2],
              R[2] * vc[0] + R[5] * vc[1] + R[8] * vc[2],
            ],
          });
        }
      }
      const jp = (posEdges.length >= 3 && raysObs.length >= 100)
        ? globalPositionsJoint(n, posEdges, raysObs, nPts, ra.anchor, vlog)
        : null;
      if (!jp || !jp.C) { vlog('global init failed: positioning'); return null; }
      const Cs = jp.C;
      let reg = 0;
      for (let i = 0; i < n; i++) {
        if (!ra.R[i] || !Cs[i]) continue;
        const R = ra.R[i], C = Cs[i];
        poses[i] = {
          R: Array.from(R),
          t: [
            -(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]),
            -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]),
            -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2]),
          ],
        };
        registered.add(i);
        reg++;
      }
      vlog(`global init: ${reg}/${n} cameras in ${((performance.now() - t0g) / 1000).toFixed(1)}s`);
      if (reg < 3) return null;
      for (const tr of tracks) triangulateTrack(tr);
      globalRefine(2);
      await tick();
    }

    const scored = [];
    for (const p of useGlobal ? [] : candPairs) {
      const { x1s, x2s } = trackCorrs(p.i, p.j);
      if (x1s.length < 30) continue;
      const favg = (K[p.i].f + K[p.j].f) / 2;
      const res = ransacE(x1s, x2s, 2.5 / favg, rng);
      if (!res || res.inliers.length < 30) continue;
      const pose = selectPose(decomposeE(res.U, res.V), x1s, x2s, res.inliers);
      if (!pose || pose.goodFrac < 0.7 || pose.medianAngle < 0.015) continue;
      // cap the parallax bonus so a sparse wide-baseline pair can't beat a
      // dense moderate one
      scored.push({
        p, pose, nInl: res.inliers.length,
        score: res.inliers.length * Math.min(pose.medianAngle, 0.06),
      });
    }
    // A near-degenerate init pair poisons EVERYTHING downstream: camping once
    // initialized from a 1.1-deg-parallax pair and the whole reconstruction
    // collapsed (path length 0.36 vs 44) while every internal check still
    // passed (rms 0.67px!). Demand >= 1.4 deg median parallax when any such
    // candidate exists; the low-parallax tier is only a last resort.
    {
      const strong = scored.filter((s) => s.pose.medianAngle >= 0.025);
      if (strong.length) { scored.length = 0; scored.push(...strong); }
    }
    scored.sort((a, b) => b.score - a.score);
    if (!useGlobal && !scored.length) return null;

    let initDone = false;
    for (const { p, pose, nInl } of scored) {
      poses[p.i] = { R: I3(), t: [0, 0, 0] };
      poses[p.j] = { R: pose.R, t: pose.t.slice() };
      registered.add(p.i); registered.add(p.j);
      let tri = 0;
      for (const tr of tracks) {
        const hasI = tr.obs.some((o) => o.ok && o.img === p.i);
        const hasJ = tr.obs.some((o) => o.ok && o.img === p.j);
        if (hasI && hasJ && triangulateTrack(tr)) tri++;
      }
      if (tri < 30) {
        poses[p.i] = poses[p.j] = null;
        registered.clear();
        for (const tr of tracks) { tr.X = null; for (const o of tr.obs) o.ok = true; }
        continue;
      }
      vlog(`init pair: images ${p.i} + ${p.j} (${nInl} E-inliers, ${tri} points, ` +
           `median parallax ${(pose.medianAngle * 180 / Math.PI).toFixed(1)} deg)`);
      initDone = true;
      break;
    }
    if (!useGlobal && !initDone) return null;

    // ---- incremental registration ----
    function corrsFor(img) {
      const obj = [], im = [];
      const ft = featTrack[img];
      for (let f = 0; f < feats[img].n; f++) {
        const tid = ft[f];
        if (tid < 0) continue;
        const tr = tracks[tid];
        if (!tr.X) continue;
        const o = tr.obs.find((oo) => oo.img === img);
        if (!o || !o.ok) continue;
        obj.push(tr.X);
        im.push([feats[img].xn[f], feats[img].yn[f]]);
      }
      return { obj, im };
    }

    function refineCamera(img) {
      const { obj, im } = corrsFor(img);
      if (obj.length < 6) return;
      poses[img] = refinePose(poses[img].R, poses[img].t, obj, im, 8);
    }

    function globalRefine(rounds) {
      for (let r = 0; r < rounds; r++) {
        for (const img of registered) refineCamera(img);
        for (const tr of tracks) {
          if (tr.X) {
            for (const o of tr.obs) {
              if (!o.ok || !poses[o.img]) continue;
              const err = reprojError(poses[o.img].R, poses[o.img].t, tr.X, obsNorm(o));
              if (!(err < thN(o.img) * 1.5)) o.ok = false;
            }
          }
          triangulateTrack(tr);
        }
      }
    }

    /** K changed (BA rescales f) or feats.x/y moved (LK refine): the cached
     *  normalized coords everything else uses must follow. */
    function refreshNorm() {
      for (let i = 0; i < n; i++) {
        const f = feats[i];
        for (let k = 0; k < f.n; k++) {
          f.xn[k] = (f.x[k] - K[i].cx) / K[i].f;
          f.yn[k] = (f.y[k] - K[i].cy) / K[i].f;
        }
      }
    }

    /** Joint sparse BA over the whole current reconstruction. Applies the
     *  result (poses, shared f, point positions) and returns the ba result,
     *  or null when the problem is too small / dimensions are mixed. */
    function runGlobalBA(tag, o = {}) {
      if (registered.size < 8) return null;
      const regList = [...registered].sort((a, b) => a - b);
      const sameDims = regList.every((img) =>
        images[img].fw === images[regList[0]].fw && images[img].fh === images[regList[0]].fh);
      if (!sameDims) { vlog(`BA ${tag} skipped: mixed image dimensions`); return null; }
      const camMap = new Map(regList.map((img, i) => [img, i]));
      const baCams = regList.map((img) => ({ R: Array.from(poses[img].R), t: poses[img].t.slice() }));
      const baPoints = [], baTracks = [], baObs = [];
      for (const tr of tracks) {
        if (!tr.X) continue;
        const pi = baPoints.length;
        let nOk = 0;
        for (const ob of tr.obs) {
          if (!ob.ok || !poses[ob.img]) continue;
          baObs.push({ ci: camMap.get(ob.img), pi, u: feats[ob.img].x[ob.feat], v: feats[ob.img].y[ob.feat] });
          nOk++;
        }
        if (nOk < 2) { baObs.length -= nOk; continue; }
        baPoints.push(tr.X.slice());
        baTracks.push(tr);
      }
      if (baPoints.length < 50) return null;
      const t0 = performance.now();
      const res = bundleAdjust(
        { cams: baCams, points: baPoints, obs: baObs, f: K[regList[0]].f, cx: K[regList[0]].cx, cy: K[regList[0]].cy },
        { maxIters: o.maxIters ?? 30, huberPx: 1.5,
          refineDistortion: o.refineDistortion ?? (sfmOpts.refineDistortion ?? true),
          refineAspect: sfmOpts.refineAspect ?? false,
          log: o.verbose ? vlog : () => {} });
      vlog(`BA ${tag} (${baCams.length} cams, ${baPoints.length} pts, ${baObs.length} obs) ` +
           `in ${((performance.now() - t0) / 1000).toFixed(1)}s: ` +
           `rms ${res.rmsBefore.toFixed(2)} -> ${res.rmsAfter.toFixed(2)}px, ` +
           `f x${res.fScale.toFixed(4)}, k1 ${res.k1.toFixed(4)}, k2 ${res.k2.toFixed(4)}` +
           (res.aspect !== 1 ? `, aspect ${res.aspect.toFixed(4)}` : ''));
      regList.forEach((img, i) => { poses[img] = { R: baCams[i].R, t: baCams[i].t }; });
      if (Math.abs(res.fScale - 1) > 1e-9) {
        for (let i = 0; i < n; i++) K[i].f *= res.fScale;
        refreshNorm();
      }
      baTracks.forEach((tr, j) => { tr.X = baPoints[j]; });
      return res;
    }

    /** Iterative outlier rejection (what COLMAP does between BA rounds).
     *  Measured on truck vs COLMAP ground truth: ~25% of our observations are
     *  outliers (p90 17px under the true geometry). Huber only softens them —
     *  left in the problem they bend the whole trajectory (7% ATE); the same
     *  BA holds the true solution at 0.3% ATE once they are filtered. */
    function baFilterObs(k1v, k2v, tag) {
      const items = [];
      const errsAll = [];
      for (const tr of tracks) {
        if (!tr.X) continue;
        for (const o of tr.obs) {
          if (!o.ok || !poses[o.img]) continue;
          const R = poses[o.img].R, t = poses[o.img].t, X = tr.X;
          const pz = R[6]*X[0] + R[7]*X[1] + R[8]*X[2] + t[2];
          let e = 1e9;
          if (pz > 1e-9) {
            const x = (R[0]*X[0] + R[1]*X[1] + R[2]*X[2] + t[0]) / pz;
            const y = (R[3]*X[0] + R[4]*X[1] + R[5]*X[2] + t[1]) / pz;
            const r2 = x*x + y*y;
            const D = 1 + k1v*r2 + k2v*r2*r2;
            const du = feats[o.img].x[o.feat] - (K[o.img].f*x*D + K[o.img].cx);
            const dv = feats[o.img].y[o.feat] - (K[o.img].f*y*D + K[o.img].cy);
            e = Math.hypot(du, dv);
          }
          items.push({ o, e });
          errsAll.push(e);
        }
      }
      if (!items.length) return 0;
      errsAll.sort((a, b) => a - b);
      const med = errsAll[errsAll.length >> 1];
      const TH = Math.max(2.0, 3 * med);
      let dropped = 0;
      for (const it of items) if (it.e > TH) { it.o.ok = false; dropped++; }
      vlog(`  obs filter (${tag}): dropped ${dropped}/${items.length} obs > ${TH.toFixed(1)}px (med ${med.toFixed(2)})`);
      return dropped / items.length;
    }

    /** Periodic global BA during incremental registration. Coordinate-descent
     *  refinement alone lets drift accumulate until the reconstruction settles
     *  into a bent basin that no final BA can escape (two self-consistent
     *  minima at ~0.6px rms exist on the truck set; only one matches COLMAP).
     *  Frequent joint solves + filtering keep growth inside the right basin. */
    function interimBA() {
      const res = runGlobalBA(`interim @${registered.size}`, { maxIters: 12, refineDistortion: false });
      if (!res) return;
      for (const tr of tracks) triangulateTrack(tr);
      baFilterObs(res.k1, res.k2, `interim @${registered.size}`);
      for (const tr of tracks) triangulateTrack(tr);
    }

    /** Register from the pose of the registered image sharing the most
     *  tracks, then alternate LM refinement and inlier re-selection.
     *  Falls back to DLT-PnP RANSAC. */
    function registerImage(img) {
      const { obj, im } = corrsFor(img);
      if (obj.length < 8) return null;
      const th = thN(img) * 1.5;

      const shared = new Map();
      const ft = featTrack[img];
      for (let f = 0; f < feats[img].n; f++) {
        const tid = ft[f];
        if (tid < 0 || !tracks[tid].X) continue;
        for (const o of tracks[tid].obs) {
          if (o.ok && o.img !== img && poses[o.img])
            shared.set(o.img, (shared.get(o.img) || 0) + 1);
        }
      }
      let ref = -1, refCount = 0;
      for (const [r, c] of shared) if (c > refCount) { refCount = c; ref = r; }

      const tryFrom = (R0, t0) => {
        let R = R0, t = t0;
        let curObj = obj, curIm = im;
        let inl = 0;
        for (let round = 0; round < 4; round++) {
          const p = refinePose(R, t, curObj, curIm, 10);
          R = p.R; t = p.t;
          const io = [], ii = [];
          for (let k = 0; k < obj.length; k++)
            if (reprojError(R, t, obj[k], im[k]) < th) { io.push(obj[k]); ii.push(im[k]); }
          inl = io.length;
          if (inl < 8) return null;
          curObj = io; curIm = ii;
        }
        return { R, t, inliers: inl };
      };

      if (ref >= 0 && refCount >= 8) {
        const res = tryFrom(poses[ref].R, poses[ref].t.slice());
        if (res && res.inliers >= Math.max(8, obj.length * 0.25)) return res;
      }
      const pnp = pnpRansac(obj, im, th, rng);
      if (pnp) return { R: pnp.R, t: pnp.t, inliers: pnp.inliers.length };
      return null;
    }

    // Registration in passes: failed images get another chance as the
    // reconstruction grows.
    for (let pass = 0; pass < 5; pass++) {
      failed.clear();
      let addedThisPass = 0;
      let sinceRefine = 0;
      let sinceBA = 0;
      while (registered.size + failed.size < n) {
        let bestImg = -1, bestCount = 0;
        for (let img = 0; img < n; img++) {
          if (registered.has(img) || failed.has(img)) continue;
          let c = 0;
          const ft = featTrack[img];
          for (let f = 0; f < feats[img].n; f++) {
            const tid = ft[f];
            if (tid >= 0 && tracks[tid].X) c++;
          }
          if (c > bestCount) { bestCount = c; bestImg = img; }
        }
        if (bestImg < 0 || bestCount < 8) break;

        const reg = registerImage(bestImg);
        if (!reg) { failed.add(bestImg); continue; }
        poses[bestImg] = { R: reg.R, t: reg.t };
        registered.add(bestImg);
        addedThisPass++;

        let newPts = 0;
        const ft = featTrack[bestImg];
        for (let f = 0; f < feats[bestImg].n; f++) {
          const tid = ft[f];
          if (tid < 0 || tracks[tid].X) continue;
          if (triangulateTrack(tracks[tid])) newPts++;
        }
        vlog(`registered image ${bestImg} (${reg.inliers}/${bestCount} inliers, +${newPts} points)`);
        if (withBA) {
          // a downsampled snapshot of the triangulated cloud rides along so a
          // UI can show the reconstruction growing, not just the frustums —
          // with real colours (sampleColor is a plain array read, ~free)
          const cloud = [];
          const cloudRgb = [];
          let nPts = 0;
          const stride = Math.max(1, (tracks.length / 4000) | 0);
          for (let ti = 0; ti < tracks.length; ti++) {
            const X = tracks[ti].X;
            if (!X) continue;
            if (nPts % stride === 0) {
              cloud.push(X[0], X[1], X[2]);
              const o = tracks[ti].obs.find((ob) => ob.ok) || tracks[ti].obs[0];
              const c = sampleColor(o.img, feats[o.img].x[o.feat], feats[o.img].y[o.feat]);
              cloudRgb.push(c[0], c[1], c[2]);
            }
            nPts++;
          }
          ev({ stage: 'register', done: registered.size, total: n, detail: {
            image: bestImg, R: Array.from(reg.R), t: Array.from(reg.t), f: K[bestImg].f,
            cloud, cloudRgb, points: nPts,
          } });
        }

        if (++sinceRefine >= 3) { globalRefine(1); sinceRefine = 0; }
        // periodic joint BA so the growing chain never drifts into a bent
        // basin (only on the final run — the focal search stays cheap)
        if (withBA && sfmOpts.interimBA !== false && ++sinceBA >= 6 && registered.size >= 12) {
          interimBA();
          sinceRefine = 0;
          sinceBA = 0;
        }
        await tick();
        checkAbort();
      }
      if (registered.size === n) break;
      globalRefine(1);
      if (addedThisPass === 0) break;
      await tick();
    }
    globalRefine(2);
    await tick();

    // ---- true sparse bundle adjustment (final pass only) ----
    // The alternating refine above is coordinate descent and cannot remove
    // smooth global trajectory bends; the joint LM solve (with shared focal
    // and radial distortion) can — this is what makes COLMAP paths clean.
    let baResult = null;
    if (withBA && sfmOpts.ba !== false && registered.size >= 3) {
      ev({ stage: 'ba', done: 0, total: 1 });
      if (sfmOpts.lkRefine !== false) { refineObsLK(images, feats, tracks, poses, vlog); refreshNorm(); }
      baResult = runGlobalBA('pass 1', { verbose: true });
      if (baResult && sfmOpts.obsFilter !== false) {
        if (baFilterObs(baResult.k1, baResult.k2, 'after pass 1') > 0.002)
          baResult = runGlobalBA('pass 1b', { verbose: true }) || baResult;
      }
      // ---- guided track extension ----
      // Mean track length ~3 obs makes the camera chain bendable; project
      // each adjusted point into every registered frame it is not yet
      // observed in and claim descriptor-verified features nearby. Requires
      // pass-1 distortion so edge-of-frame projections land where the raw
      // (distorted) features actually are.
      if (baResult && sfmOpts.trackExtend !== false) {
        const regList = [...registered].sort((a, b) => a - b);
        // two rounds: the second round extends from the improved pass-2 poses
        // and reaches frames the first projection missed
        for (let round = 1; round <= 2; round++) {
          const extObs = extendTracks(feats, tracks, poses, K, regList,
                                      baResult.k1, baResult.k2, images[regList[0]], vlog);
          if (!extObs.length) break;
          // refine the freshly added obs to subpixel too (idempotent for
          // already-refined ones — they are at their converged position)
          if (sfmOpts.lkRefine !== false) { refineObsLK(images, feats, tracks, poses, vlog); refreshNorm(); }
          baResult = runGlobalBA(`pass ${1 + round}`, { verbose: true }) || baResult;
        }
      }
      if (baResult && sfmOpts.obsFilter !== false) {
        // iterate filter -> BA until the drop rate is negligible (each round
        // reveals outliers the previous bent model was hiding)
        for (let round = 1; round <= 3; round++) {
          if (baFilterObs(baResult.k1, baResult.k2, `final round ${round}`) <= 0.002) break;
          baResult = runGlobalBA(`final ${round}`, { verbose: true }) || baResult;
        }
      }
      await tick();
    }

    // ---- collect output + median reprojection error ----
    const points = [];
    const errs = [];
    for (const tr of tracks) {
      if (!tr.X) continue;
      const okObs = tr.obs.filter((o) => o.ok && poses[o.img]);
      if (okObs.length < 2) continue;
      let r = 0, g = 0, b = 0, c = 0;
      for (const o of okObs.slice(0, 3)) {
        const rgb = sampleColor(o.img, feats[o.img].x[o.feat], feats[o.img].y[o.feat]);
        r += rgb[0]; g += rgb[1]; b += rgb[2]; c++;
        errs.push(reprojError(poses[o.img].R, poses[o.img].t, tr.X, obsNorm(o)) * K[o.img].f);
      }
      points.push({ X: tr.X, rgb: [r / c, g / c, b / c], nObs: okObs.length });
    }
    errs.sort((a, b) => a - b);
    const medErr = errs.length ? errs[errs.length >> 1] : Infinity;

    const cams = [];
    for (const img of registered) {
      cams.push({
        imgIdx: img,
        R: poses[img].R, t: poses[img].t,
        f: K[img].f, cx: K[img].cx, cy: K[img].cy,
      });
    }
    cams.sort((a, b) => a.imgIdx - b.imgIdx);

    // internals hook for dev tooling / UI beats (feature marks, match lines)
    if (opts.debug) opts.debug({ feats, tracks, poses, K, registered, corrsFor, thN });

    return {
      cams, points, medErr, fScale,
      k1: baResult ? baResult.k1 : 0,
      k2: baResult ? baResult.k2 : 0,
      fFeat: K[0].f,
      rmsBA: baResult ? baResult.rmsAfter : null,
    };
  }

  // ---- focal search ----
  // Evaluate ALL candidates (an early-exit at the first full registration
  // once picked a focal 24% off on the synthetic GT while a near-perfect
  // candidate sat later in the grid). Rank by registered cameras, then by
  // reprojection error. Incremental runs rank the tiebreak by ANGULAR error
  // (raw pixel error interacts with the 1/f registration thresholds). Global
  // init registers everything at every focal, so the count never
  // discriminates and angular error trivially favors the largest f (it
  // divides by f) — measured: it picked 1.56x over the true 0.78x and BA
  // invented k1=0.70 to cope. There the PIXEL median is the right metric
  // (detector noise is in pixels; model misfit adds to it).
  const useGlobalSearch = !!opts.globalInit;
  const candidates = [];
  let fi = 0;
  for (const s of FOCAL_SCALES) {
    ev({ stage: 'focal', done: fi++, total: FOCAL_SCALES.length, detail: { fScale: s * 1.2 } });
    checkAbort();
    const res = await runGeometry(s, false);
    const fEff = (s * 1.2).toFixed(2);
    if (!res) {
      log(`focal ${fEff}x maxDim: no valid initialization`);
      continue;
    }
    res.angErr = res.medErr / (s * 1.2 * 640); // relative units; constant factor irrelevant
    res.rankErr = useGlobalSearch ? res.medErr : res.angErr;
    log(`focal ${fEff}x maxDim: ${res.cams.length}/${n} cams, ${res.points.length} pts, ` +
        `median reproj ${res.medErr.toFixed(2)}px (angular ${(res.angErr * 1e4).toFixed(2)}e-4)`);
    candidates.push(res);
    await tick();
  }
  if (!candidates.length) throw new Error('no focal candidate produced a valid initialization — need more parallax/overlap');
  // A candidate must register (nearly) the most cameras to compete, but among
  // those the PIXEL median error picks the winner: a too-long focal can bend
  // the reconstruction to register EVERYTHING (measured on truck at 960px
  // features: 1.56x registered 42/42 and won on count while the true 0.78x
  // sat at 37/42 with clearly better pixel error -> ATE 0.04% vs 2.28%). The
  // angular metric is no tiebreak either — it divides by f and favored the
  // same wrong candidate. The near-count window (88%) keeps the completeness
  // prior for genuinely fragile candidates; the BA rerun of the winner
  // recovers the few cameras the cheap search pass missed.
  const maxCams = Math.max(...candidates.map((r) => r.cams.length));
  const eligible = candidates.filter((r) => r.cams.length >= Math.ceil(0.88 * maxCams));
  let best = null;
  for (const res of eligible) {
    if (!best || res.medErr < best.medErr) best = res;
  }

  // re-run the winner verbosely (with bundle adjustment) to produce the
  // final reconstruction
  log(`focal search winner: ${(best.fScale * 1.2).toFixed(2)}x maxDim — rerunning with BA ...`);
  const final = await runGeometry(best.fScale, true, true);
  if (!final) throw new Error('focal winner failed on rerun (unexpected)');

  log(`SfM done: ${final.cams.length}/${n} cameras registered, ${final.points.length} points, ` +
      (final.rmsBA != null ? `BA rms ${final.rmsBA.toFixed(2)}px` : `median reproj ${final.medErr.toFixed(2)}px`));
  if (final.cams.length < 2 || final.points.length < 50)
    throw new Error('reconstruction too sparse to continue');
  return final;
}
