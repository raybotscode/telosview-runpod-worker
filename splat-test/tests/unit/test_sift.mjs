// Validate js SIFT against COLMAP's reference extraction on the same image.
import { createRequire } from 'module';
import fs from 'fs';
import { detectSift, matchSift } from '../../src/sfm/sift.js';

const require = createRequire(import.meta.url);
const jpeg = require('jpeg-js');

function loadGray(path) {
  const { data, width, height } = jpeg.decode(fs.readFileSync(path), { useTArray: true });
  const g = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++)
    g[i] = (0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2]) / 255;
  return { g, width, height };
}

function loadRef(img) {
  const kpBuf = fs.readFileSync(`C:/Dev/arrival.space/Browser_3DGS/scratch/sift_ref_img${img}_kp.bin`);
  const kp = new Float32Array(kpBuf.buffer, kpBuf.byteOffset, kpBuf.length / 4);
  const n = kp.length / 6;
  const descBuf = fs.readFileSync(`C:/Dev/arrival.space/Browser_3DGS/scratch/sift_ref_img${img}_desc.bin`);
  return { n, kp, desc: new Uint8Array(descBuf.buffer, descBuf.byteOffset, descBuf.length) };
}

const t0 = Date.now();
const im1 = loadGray('C:/Dev/arrival.space/Browser_3DGS/data/train/00001.jpg');
const f1 = detectSift(im1.g, im1.width, im1.height, 4000, -1);
console.log(`mine img1: ${f1.n} features in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

const ref = loadRef(1);
console.log(`colmap img1: ${ref.n} features`);

// keypoint recall: colmap kp with one of mine within 2px and scale ratio < 1.6
let hits = 0, scaleHits = 0;
const cell = 8;
const grid = new Map();
for (let i = 0; i < f1.n; i++) {
  const key = ((f1.y[i] / cell) | 0) * 10000 + ((f1.x[i] / cell) | 0);
  if (!grid.has(key)) grid.set(key, []);
  grid.get(key).push(i);
}
for (let r = 0; r < ref.n; r++) {
  const rx = ref.kp[r * 6], ry = ref.kp[r * 6 + 1];
  const a11 = ref.kp[r * 6 + 2], a12 = ref.kp[r * 6 + 3], a21 = ref.kp[r * 6 + 4], a22 = ref.kp[r * 6 + 5];
  const rscale = Math.sqrt(Math.abs(a11 * a22 - a12 * a21));
  let found = false, sFound = false;
  const cgx = (rx / cell) | 0, cgy = (ry / cell) | 0;
  for (let dy = -1; dy <= 1; dy++)
    for (let dx = -1; dx <= 1; dx++) {
      const list = grid.get((cgy + dy) * 10000 + (cgx + dx));
      if (!list) continue;
      for (const i of list) {
        const d = Math.hypot(f1.x[i] - rx, f1.y[i] - ry);
        if (d < 2) {
          found = true;
          const ratio = f1.scale[i] / rscale;
          if (ratio > 0.6 && ratio < 1.7) sFound = true;
        }
      }
    }
  if (found) hits++;
  if (sFound) scaleHits++;
}
console.log(`keypoint recall (2px): ${(100 * hits / ref.n).toFixed(1)}%, with scale agreement: ${(100 * scaleHits / ref.n).toFixed(1)}%`);

// matching: mine img1 vs img2 -> compare count vs colmap's verified 2520
const im2 = loadGray('C:/Dev/arrival.space/Browser_3DGS/data/train/00002.jpg');
const f2 = detectSift(im2.g, im2.width, im2.height, 4000, -1);
console.log(`mine img2: ${f2.n} features`);
const tM = Date.now();
const m = matchSift(f1.desc, f1.n, f2.desc, f2.n);
console.log(`my matches 1-2: ${m.length / 2} in ${((Date.now() - tM) / 1000).toFixed(1)}s (colmap verified: 2520)`);

// geometric consistency of my matches (median displacement coherence)
const dxs = [], dys = [];
for (let k = 0; k < m.length; k += 2) {
  dxs.push(f2.x[m[k + 1]] - f1.x[m[k]]);
  dys.push(f2.y[m[k + 1]] - f1.y[m[k]]);
}
dxs.sort((a, b) => a - b); dys.sort((a, b) => a - b);
const mdx = dxs[dxs.length >> 1], mdy = dys[dys.length >> 1];
let coherent = 0;
for (let k = 0; k < m.length; k += 2) {
  const ddx = f2.x[m[k + 1]] - f1.x[m[k]] - mdx;
  const ddy = f2.y[m[k + 1]] - f1.y[m[k]] - mdy;
  if (Math.hypot(ddx, ddy) < 30) coherent++;
}
console.log(`coherent matches (within 30px of median flow): ${(100 * coherent / (m.length / 2)).toFixed(1)}%`);
