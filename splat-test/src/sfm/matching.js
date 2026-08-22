// matching.js — brute-force Hamming matching of 256-bit BRIEF descriptors
// with ratio test and mutual (cross) check.

const W = 8; // words per descriptor

function popcnt(x) {
  x = x - ((x >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  x = (x + (x >>> 4)) & 0x0f0f0f0f;
  return Math.imul(x, 0x01010101) >>> 24;
}

function bestMatches(dA, nA, dB, nB, maxDist, ratio) {
  // For each descriptor in A: index of best match in B (or -1).
  const best = new Int32Array(nA).fill(-1);
  for (let a = 0; a < nA; a++) {
    const oa = a * W;
    let b1 = 256, b2 = 256, bi = -1;
    for (let b = 0; b < nB; b++) {
      const ob = b * W;
      let d = 0;
      for (let k = 0; k < W; k++) d += popcnt(dA[oa + k] ^ dB[ob + k]);
      if (d < b1) { b2 = b1; b1 = d; bi = b; }
      else if (d < b2) { b2 = d; }
    }
    if (b1 <= maxDist && b1 < ratio * b2) best[a] = bi;
  }
  return best;
}

/** Match descriptors A -> B. Returns flat Int32Array [ia0, ib0, ia1, ib1, ...]. */
export function matchDescriptors(dA, nA, dB, nB, maxDist = 90, ratio = 0.85) {
  const ab = bestMatches(dA, nA, dB, nB, maxDist, ratio);
  const ba = bestMatches(dB, nB, dA, nA, maxDist, ratio);
  const out = [];
  for (let a = 0; a < nA; a++) {
    const b = ab[a];
    if (b >= 0 && ba[b] === a) { out.push(a, b); }
  }
  return Int32Array.from(out);
}
