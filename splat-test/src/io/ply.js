// ply.js — export trained Gaussians as a standard 3DGS PLY
// (INRIA layout: positions, normals, f_dc, opacity (pre-sigmoid),
//  log-scales, rotation quaternion).
//
// Trainer parameter layout (stride 16):
//   [0-2 pos, 3-5 logScale xyz, 6-9 quat (w,x,y,z, raw), 10-12 color logits,
//    13 logitOpacity, 14-15 pad]

const SH_C0 = 0.28209479177387814;
const STRIDE = 16;

/** The trainer's splat weights include a Mip-Splatting opacity compensation
 *  factor that standard sorted renderers don't apply. Bake an approximation
 *  into the opacities (screen size estimated per splat at its NEAREST
 *  training camera, isotropic approximation via the mean scale).
 *  camPositions: flat [x,y,z,...]. Returns a transformed copy. */
export function bakeOpacityCompensation(data, n, f, camPositions) {
  const out = Float32Array.from(data);
  const nc = camPositions.length / 3;
  for (let i = 0; i < n; i++) {
    const b = i * STRIDE;
    let z2min = Infinity;
    for (let c = 0; c < nc; c++) {
      const dx = data[b] - camPositions[c * 3];
      const dy = data[b + 1] - camPositions[c * 3 + 1];
      const dz = data[b + 2] - camPositions[c * 3 + 2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < z2min) z2min = d2;
    }
    const z = Math.max(1e-3, Math.sqrt(z2min));
    const sMean = Math.exp((data[b + 3] + data[b + 4] + data[b + 5]) / 3);
    const s2d = f * sMean / z;
    const comp = (s2d * s2d) / (s2d * s2d + 0.3);
    const opa = comp / (1 + Math.exp(-data[b + 13]));
    const clamped = Math.min(1 - 1e-6, Math.max(1e-6, opa));
    out[b + 13] = Math.log(clamped / (1 - clamped));
  }
  return out;
}

/** sh: optional channel-major SH rest buffer (n * 3K floats: K red coeffs,
 *  K green, K blue per splat) — the trainer's color model is EXACTLY the
 *  standard one (f_dc = (sigmoid(logit)-0.5)/C0, f_rest = raw coeffs), so
 *  standard viewers reproduce the trained view dependence bit-for-bit. */
export function gaussiansToPly(data, n, sh = null, shK = 0) {
  const K = sh ? shK : 0;
  const props = [
    'x', 'y', 'z', 'nx', 'ny', 'nz',
    'f_dc_0', 'f_dc_1', 'f_dc_2',
    // INRIA layout: f_rest is channel-major (all red band coeffs, then green,
    // then blue) and sits between f_dc and opacity
    ...Array.from({ length: 3 * K }, (_, k) => `f_rest_${k}`),
    'opacity',
    'scale_0', 'scale_1', 'scale_2',
    'rot_0', 'rot_1', 'rot_2', 'rot_3',
  ];
  const header =
    'ply\n' +
    'format binary_little_endian 1.0\n' +
    `element vertex ${n}\n` +
    props.map((p) => `property float ${p}`).join('\n') + '\n' +
    'end_header\n';

  const headerBytes = new TextEncoder().encode(header);
  const stride = props.length * 4;
  const body = new ArrayBuffer(n * stride);
  const dv = new DataView(body);

  const sig = (v) => 1 / (1 + Math.exp(-v));
  for (let i = 0; i < n; i++) {
    const b = i * STRIDE;
    const o = i * stride;
    dv.setFloat32(o, data[b], true);
    dv.setFloat32(o + 4, data[b + 1], true);
    dv.setFloat32(o + 8, data[b + 2], true);
    // normals = 0; colors stored as logits -> activate exactly like the trainer
    dv.setFloat32(o + 24, (sig(data[b + 10]) - 0.5) / SH_C0, true);
    dv.setFloat32(o + 28, (sig(data[b + 11]) - 0.5) / SH_C0, true);
    dv.setFloat32(o + 32, (sig(data[b + 12]) - 0.5) / SH_C0, true);
    for (let k = 0; k < 3 * K; k++) {
      dv.setFloat32(o + 36 + k * 4, sh[i * 3 * K + k], true);
    }
    const q = o + 36 + 3 * K * 4;
    dv.setFloat32(q, data[b + 13], true);               // logit opacity
    dv.setFloat32(q + 4, data[b + 3], true);            // log scales
    dv.setFloat32(q + 8, data[b + 4], true);
    dv.setFloat32(q + 12, data[b + 5], true);
    // normalized quaternion (w, x, y, z)
    let qw = data[b + 6], qx = data[b + 7], qy = data[b + 8], qz = data[b + 9];
    const ql = Math.hypot(qw, qx, qy, qz);
    if (ql < 1e-6) { qw = 1; qx = qy = qz = 0; }
    else { qw /= ql; qx /= ql; qy /= ql; qz /= ql; }
    dv.setFloat32(q + 16, qw, true);
    dv.setFloat32(q + 20, qx, true);
    dv.setFloat32(q + 24, qy, true);
    dv.setFloat32(q + 28, qz, true);
  }
  return new Blob([headerBytes, body], { type: 'application/octet-stream' });
}

