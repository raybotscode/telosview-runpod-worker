// synthetic.js — renders a synthetic multi-view dataset (a textured room corner)
// with a 2D canvas, so the full SfM -> 3DGS pipeline can be tested end-to-end
// without real photos.
//
// Rendering is deliberately sort-free and view-INDEPENDENT (a global painter's
// sort z-fights on coplanar decals and mis-sorts large quads, which made
// rectangles pop between views — an impossible training target):
//   1. the three corner planes are concave from every camera in the arc, so
//      they never occlude each other — draw in any fixed order, decals after
//      their own base plane
//   2. the floating box is convex and always in front of the corner — draw
//      last, front-facing faces only (visible faces of a convex solid never
//      overlap in projection)

import { processSource, adaptiveTrainCap } from './io/frames.js';
import { makeRng } from './sfm/geometry.js';

const W = 640, H = 480, F = 620;

const lerp3 = (o, u, v, a, b) => [
  o[0] + u[0] * a + v[0] * b,
  o[1] + u[1] * a + v[1] * b,
  o[2] + u[2] * a + v[2] * b,
];

// world: y points DOWN (matches the camera convention), scene centered at origin
function buildScene(rng) {
  const planeDefs = [
    // floor: y = +1.0
    { o: [-1.6, 1.0, -1.6], u: [3.2, 0, 0], v: [0, 0, 3.2], base: '#8a7f6d' },
    // back wall: z = +1.6
    { o: [-1.6, 1.0, 1.6], u: [3.2, 0, 0], v: [0, -2.4, 0], base: '#7d8894' },
    // side wall: x = +1.6
    { o: [1.6, 1.0, 1.6], u: [0, 0, -3.2], v: [0, -2.4, 0], base: '#94847d' },
  ];
  const planes = [];
  for (const pl of planeDefs) {
    const base = {
      pts: [lerp3(pl.o, pl.u, pl.v, 0, 0), lerp3(pl.o, pl.u, pl.v, 1, 0),
            lerp3(pl.o, pl.u, pl.v, 1, 1), lerp3(pl.o, pl.u, pl.v, 0, 1)],
      color: pl.base,
    };
    const decals = [];
    for (let k = 0; k < 180; k++) {
      const a = rng() * 0.92, b = rng() * 0.92;
      const sa = 0.015 + rng() * 0.09, sb = 0.015 + rng() * 0.09;
      const hue = (rng() * 360) | 0;
      const light = 25 + ((rng() * 55) | 0);
      decals.push({
        pts: [lerp3(pl.o, pl.u, pl.v, a, b), lerp3(pl.o, pl.u, pl.v, a + sa, b),
              lerp3(pl.o, pl.u, pl.v, a + sa, b + sb), lerp3(pl.o, pl.u, pl.v, a, b + sb)],
        color: `hsl(${hue} ${40 + ((rng() * 50) | 0)}% ${light}%)`,
      });
    }
    planes.push({ base, decals });
  }

  // a floating box for extra parallax
  const bx = [-0.3, 0.45, -0.2], bs = 0.55;
  const boxCenter = [bx[0] + 0.5 * bs, bx[1] - 0.5 * bs, bx[2] + 0.5 * bs];
  const c = (dx, dy, dz) => [bx[0] + dx * bs, bx[1] + dy * bs, bx[2] + dz * bs];
  const faceQuads = [
    [c(0, 0, 0), c(1, 0, 0), c(1, -1, 0), c(0, -1, 0)],   // z- side
    [c(0, 0, 1), c(1, 0, 1), c(1, -1, 1), c(0, -1, 1)],   // z+ side
    [c(0, 0, 0), c(0, 0, 1), c(0, -1, 1), c(0, -1, 0)],   // x- side
    [c(1, 0, 0), c(1, 0, 1), c(1, -1, 1), c(1, -1, 0)],   // x+ side
    [c(0, -1, 0), c(1, -1, 0), c(1, -1, 1), c(0, -1, 1)], // top (y-)
  ];
  const boxColors = ['#c25b4e', '#4e8ac2', '#c2a44e', '#5bc24e', '#8a4ec2'];
  const boxFaces = faceQuads.map((pts, fi) => {
    // outward normal via cross of edges, oriented away from the box center
    const e1 = [pts[1][0] - pts[0][0], pts[1][1] - pts[0][1], pts[1][2] - pts[0][2]];
    const e2 = [pts[3][0] - pts[0][0], pts[3][1] - pts[0][1], pts[3][2] - pts[0][2]];
    let nrm = [
      e1[1] * e2[2] - e1[2] * e2[1],
      e1[2] * e2[0] - e1[0] * e2[2],
      e1[0] * e2[1] - e1[1] * e2[0],
    ];
    const mid = [
      (pts[0][0] + pts[2][0]) / 2, (pts[0][1] + pts[2][1]) / 2, (pts[0][2] + pts[2][2]) / 2,
    ];
    const out = [mid[0] - boxCenter[0], mid[1] - boxCenter[1], mid[2] - boxCenter[2]];
    if (nrm[0] * out[0] + nrm[1] * out[1] + nrm[2] * out[2] < 0) nrm = nrm.map((v) => -v);
    // dots on the face (view-independent, drawn right after their face)
    const dots = [];
    for (let k = 0; k < 25; k++) {
      const a = 0.05 + Math.min(0.85, rng() * 0.8), b = 0.05 + Math.min(0.85, rng() * 0.8);
      const s = 0.04 + rng() * 0.10;
      const p = (aa, bb) => [
        pts[0][0] + (pts[1][0] - pts[0][0]) * aa + (pts[3][0] - pts[0][0]) * bb,
        pts[0][1] + (pts[1][1] - pts[0][1]) * aa + (pts[3][1] - pts[0][1]) * bb,
        pts[0][2] + (pts[1][2] - pts[0][2]) * aa + (pts[3][2] - pts[0][2]) * bb,
      ];
      dots.push({
        pts: [p(a, b), p(a + s, b), p(a + s, b + s), p(a, b + s)],
        color: `hsl(${(fi * 67 + ((rng() * 120) | 0)) % 360} 70% ${20 + ((rng() * 60) | 0)}%)`,
      });
    }
    return { pts, color: boxColors[fi], nrm, mid, dots };
  });

  return { planes, boxFaces };
}

function renderView(scene, pose, jitterRng) {
  const cv = document.createElement('canvas');
  cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#1c1c20';
  ctx.fillRect(0, 0, W, H);

  const { R, t } = pose;
  const proj = (p) => {
    const x = R[0] * p[0] + R[1] * p[1] + R[2] * p[2] + t[0];
    const y = R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + t[1];
    const z = R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + t[2];
    return [F * x / z + W / 2, F * y / z + H / 2, z];
  };
  const camPos = [
    -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
    -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
    -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
  ];

  const drawQuad = (q) => {
    const pp = [];
    for (const p of q.pts) {
      const s = proj(p);
      if (s[2] < 0.2) return;
      pp.push(s);
    }
    ctx.fillStyle = q.color;
    ctx.beginPath();
    ctx.moveTo(pp[0][0], pp[0][1]);
    for (let i = 1; i < 4; i++) ctx.lineTo(pp[i][0], pp[i][1]);
    ctx.closePath();
    ctx.fill();
  };

  // 1. corner planes (mutually non-occluding), each base then its decals
  for (const pl of scene.planes) {
    drawQuad(pl.base);
    for (const d of pl.decals) drawQuad(d);
  }
  // 2. box: front-facing faces only, each followed by its dots
  for (const f of scene.boxFaces) {
    const toCam = [camPos[0] - f.mid[0], camPos[1] - f.mid[1], camPos[2] - f.mid[2]];
    if (f.nrm[0] * toCam[0] + f.nrm[1] * toCam[1] + f.nrm[2] * toCam[2] <= 0) continue;
    drawQuad(f);
    for (const d of f.dots) drawQuad(d);
  }

  // mild sensor-like noise so BRIEF has gradients everywhere
  const id = ctx.getImageData(0, 0, W, H);
  for (let i = 0; i < id.data.length; i += 4) {
    const nz = (jitterRng() - 0.5) * 6;
    id.data[i] += nz; id.data[i + 1] += nz; id.data[i + 2] += nz;
  }
  ctx.putImageData(id, 0, 0);
  return cv;
}

/** Generate `count` raw synthetic views (deterministic, seed 42): the exact
 *  640x480 canvases + ground-truth world-to-camera poses (OpenCV convention:
 *  x right, y down, z forward; pc = R*p + t). */
export function generateSyntheticRaw(count = 12) {
  const rng = makeRng(42);
  const scene = buildScene(rng);
  const out = [];
  const center = [0, 0.2, 0];
  for (let i = 0; i < count; i++) {
    const a = -0.55 + 1.1 * (i / (count - 1));       // yaw arc
    const elev = -0.55 - 0.15 * Math.sin(i * 1.7);   // slight height variation
    const dist = 3.6 + 0.3 * Math.cos(i * 2.3);
    const eye = [
      center[0] + Math.sin(a) * dist * 0.9 - 0.5,
      center[1] + elev,
      center[2] - Math.cos(a) * dist,
    ];
    const pose = lookAtPose(eye, center);
    const cv = renderView(scene, pose, rng);
    out.push({
      name: `synthetic_${String(i).padStart(2, '0')}`,
      canvas: cv, pose, eye, f: F, cx: W / 2, cy: H / 2, w: W, h: H,
    });
  }
  return out;
}

/** Generate `count` synthetic views. Returns the same format as loadImageFiles. */
export function generateSyntheticDataset(count = 12, trainCap, opts = {}) {
  const cap = trainCap || adaptiveTrainCap(count, W, H, opts);
  return generateSyntheticRaw(count).map((v) => processSource(v.canvas, W, H, v.name, cap, opts));
}

function lookAtPose(eye, center) {
  // world up (y is down) => up direction is -y
  const worldUp = [0, -1, 0];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => { const l = Math.hypot(...a); return [a[0] / l, a[1] / l, a[2] / l]; };
  const zc = norm(sub(center, eye));
  const xc = norm(cross(zc, worldUp));
  const yc = cross(zc, xc);
  const R = [xc[0], xc[1], xc[2], yc[0], yc[1], yc[2], zc[0], zc[1], zc[2]];
  const t = [
    -(R[0] * eye[0] + R[1] * eye[1] + R[2] * eye[2]),
    -(R[3] * eye[0] + R[4] * eye[1] + R[5] * eye[2]),
    -(R[6] * eye[0] + R[7] * eye[1] + R[8] * eye[2]),
  ];
  return { R, t };
}
