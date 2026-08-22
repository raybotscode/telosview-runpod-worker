// shaders.js — WGSL for the WebGPU 3DGS trainer.
//
// Rasterization is standard 3DGS: anisotropic Gaussians (per-axis scale +
// rotation quaternion), EWA projection to a 2D covariance, GLOBAL sorted
// binning (count -> prefix scan -> scatter into exact-size per-tile segments
// -> per-segment depth sort; no per-tile capacity cap), sorted front-to-back
// alpha compositing, and the matching back-to-front transmittance-recursion
// backward pass with the full covariance chain rule
// (conic -> 2D cov -> 3D cov -> scales/quaternion, incl. the J position term).
//
// Remaining prototype simplifications vs. reference 3DGS:
//  * Charbonnier loss (no SSIM)
//
// Gaussian parameter layout (stride 16 f32):
//   [0-2 pos, 3-5 logScale xyz, 6-9 quat (w,x,y,z, raw), 10-12 color logits,
//    13 logitOpacity, 14-15 pad]
// View-dependent color (shDeg > 0) lives in a SEPARATE per-splat SH buffer
// (channel-major: K red rest coeffs, K green, K blue). The color model is
//   c(dir) = max(0, sigmoid(logit) + sum_k sh_k Y_k(dir))
// — the DC stays the bounded sigmoid (identical to shDeg 0 when sh == 0),
// the rest bands are the STANDARD basis in color space, so the PLY export is
// exact: f_dc = (sigmoid(logit) - 0.5)/C0, f_rest_k = sh_k.
// Per-frame projected layout (stride 16 f32):
//   [0 meanX, 1 meanY, 2 depth, 3 conicA, 4 conicB, 5 conicC, 6 comp,
//    7 opacity, 8-10 rgb, 11 visible flag, 12-14 cov2D (a,b,c), 15 radius]
// Screen-space gradient accumulator (stride 16 atomic<i32>, fixed point):
//   [0 dMeanX, 1 dMeanY, 2 dConicA, 3 dConicB, 4 dConicC, 5 dComp,
//    6 dLogitOpacity, 7-9 dRGB]
// Entry buffer (interleaved pairs): [2k] = depth key (f32 bits), [2k+1] = id.

export const STRIDE = 16;
export const TILE = 16;
export const SHARED_SORT = 2048;    // shared-memory sort fast path (16KB workgroup mem)
export const ENTRIES_CAP = 12000000; // global (key,id) pair budget across all tiles (96MB)
export const FIXED = 16384.0;
// same scale as FIXED since the render pass normalizes conic grads by
// (1 + rad^2) per splat (they are px^2-scaled otherwise; the old coarser
// 4096 scale still hit the i32 ceiling on big splats at native res)
export const FIXED_CONIC = 16384.0;

// Cutoffs (parameterized so gradcheck can use a "strict" variant whose
// boundary discontinuities are negligible): E_CUT is the Gaussian exponent
// cutoff, A_MIN the minimum alpha, RADM the matching binning radius in sigmas.
export const DEFAULT_E_CUT = 4.5;
export const DEFAULT_A_MIN = 0.0039;
// RC (binning radius clamp, fraction of image width) defaults to full frame:
// clamping smaller shows up as square-clipped splats when the camera gets
// close (the Gaussian renders only inside its binned tiles).
const cutConsts = (E, A, RC = 1.0) => /* wgsl */ `
const E_CUT = ${E.toExponential()};
const A_MIN = ${A.toExponential()};
const RADM = ${Math.sqrt(2 * E).toExponential()};
const RADCL = ${RC.toExponential()};
`;

// ---- spherical harmonics (view-dependent color) ----
// Real SH basis, INRIA constant convention (deg 1-3). Generated per compiled
// degree; the RUNTIME active degree ramps via cam.misc3.x (INRIA-style, one
// band per 1000 iters) — inactive bands contribute nothing and get 0 grads.
const SH_C1 = 0.4886025119029199;
const SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
  -1.0925484305920792, 0.5462742152960396];
const SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
  0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435];
export const shRestCoefs = (deg) => (deg + 1) * (deg + 1) - 1;

// WGSL: basis values Y_k(v) and (for the chain pass) their gradients dY_k/dv
// at a unit direction v. Emitted only for shDeg >= 1 compiles.
const shFns = (deg) => {
  const K = shRestCoefs(deg);
  const e = (v) => v.toExponential();
  let y = `
  Y[0] = ${e(-SH_C1)} * y;
  Y[1] = ${e(SH_C1)} * z;
  Y[2] = ${e(-SH_C1)} * x;`;
  let dy = `
  D[0] = vec3f(0.0, ${e(-SH_C1)}, 0.0);
  D[1] = vec3f(0.0, 0.0, ${e(SH_C1)});
  D[2] = vec3f(${e(-SH_C1)}, 0.0, 0.0);`;
  if (deg >= 2) {
    y += `
  Y[3] = ${e(SH_C2[0])} * x * y;
  Y[4] = ${e(SH_C2[1])} * y * z;
  Y[5] = ${e(SH_C2[2])} * (2.0 * zz - xx - yy);
  Y[6] = ${e(SH_C2[3])} * x * z;
  Y[7] = ${e(SH_C2[4])} * (xx - yy);`;
    dy += `
  D[3] = ${e(SH_C2[0])} * vec3f(y, x, 0.0);
  D[4] = ${e(SH_C2[1])} * vec3f(0.0, z, y);
  D[5] = ${e(SH_C2[2])} * vec3f(-2.0 * x, -2.0 * y, 4.0 * z);
  D[6] = ${e(SH_C2[3])} * vec3f(z, 0.0, x);
  D[7] = ${e(SH_C2[4])} * vec3f(2.0 * x, -2.0 * y, 0.0);`;
  }
  if (deg >= 3) {
    y += `
  Y[8]  = ${e(SH_C3[0])} * y * (3.0 * xx - yy);
  Y[9]  = ${e(SH_C3[1])} * x * y * z;
  Y[10] = ${e(SH_C3[2])} * y * (4.0 * zz - xx - yy);
  Y[11] = ${e(SH_C3[3])} * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
  Y[12] = ${e(SH_C3[4])} * x * (4.0 * zz - xx - yy);
  Y[13] = ${e(SH_C3[5])} * z * (xx - yy);
  Y[14] = ${e(SH_C3[6])} * x * (xx - yy);`;
    dy += `
  D[8]  = ${e(SH_C3[0])} * vec3f(6.0 * x * y, 3.0 * xx - 3.0 * yy, 0.0);
  D[9]  = ${e(SH_C3[1])} * vec3f(y * z, x * z, x * y);
  D[10] = ${e(SH_C3[2])} * vec3f(-2.0 * x * y, 4.0 * zz - xx - 3.0 * yy, 8.0 * y * z);
  D[11] = ${e(SH_C3[3])} * vec3f(-6.0 * x * z, -6.0 * y * z, 6.0 * zz - 3.0 * xx - 3.0 * yy);
  D[12] = ${e(SH_C3[4])} * vec3f(4.0 * zz - 3.0 * xx - yy, -2.0 * x * y, 8.0 * x * z);
  D[13] = ${e(SH_C3[5])} * vec3f(2.0 * x * z, -2.0 * y * z, xx - yy);
  D[14] = ${e(SH_C3[6])} * vec3f(3.0 * xx - yy, -2.0 * x * y, 0.0);`;
  }
  const pre = deg >= 2
    ? 'let x = v.x; let y = v.y; let z = v.z;\n  let xx = x * x; let yy = y * y; let zz = z * z;'
    : 'let x = v.x; let y = v.y; let z = v.z;';
  return /* wgsl */ `
const SHK = ${K}u;
fn shActiveK() -> u32 {
  let ad = u32(cam.misc3.x + 0.5);
  return min((ad + 1u) * (ad + 1u) - 1u, ${K}u);
}
fn camPosWorld() -> vec3f {
  return -vec3f(
    cam.R0.x * cam.t.x + cam.R1.x * cam.t.y + cam.R2.x * cam.t.z,
    cam.R0.y * cam.t.x + cam.R1.y * cam.t.y + cam.R2.y * cam.t.z,
    cam.R0.z * cam.t.x + cam.R1.z * cam.t.y + cam.R2.z * cam.t.z);
}
fn shBasis(v: vec3f) -> array<f32, ${K}> {
  var Y: array<f32, ${K}>;
  ${pre}${y}
  return Y;
}
fn shBasisGrad(v: vec3f) -> array<vec3f, ${K}> {
  var D: array<vec3f, ${K}>;
  ${pre}${dy}
  return D;
}
`;
};

const CAM_STRUCT = /* wgsl */ `
struct Cam {
  R0: vec4f,      // row 0 of world-to-cam rotation
  R1: vec4f,
  R2: vec4f,
  t: vec4f,       // xyz = translation, w = near plane
  proj: vec4f,    // x = focal(px), y = cx, z = cy, w = exposure gain
  size: vec4f,    // x = width, y = height, z = tilesX, w = numGaussians
  misc: vec4f,    // xyz = background color, w = target offset (u32 bits)
  misc2: vec4f,   // x = trainMode (1/0), y = camera index, z = numCams, w = exposure bias
  misc3: vec4f,   // x = active SH degree
};
@group(0) @binding(0) var<uniform> cam: Cam;
const TILEF = ${TILE}.0;
const SHSORT = ${SHARED_SORT}u;
override ENTCAP: u32 = ${ENTRIES_CAP}u;
const FIXED = ${FIXED.toExponential()};
const FIXEDC = ${FIXED_CONIC.toExponential()};
const FIXCAM = 64.0; // camera grads sum over all splats: coarse fixed point
`;

// Shared per-splat geometry: params -> normalized quat, rotation, cam-space
// point, projection T = J*W, 2D covariance (va, vb, vc). Used identically by
// project and chain so both see the same forward quantities.
const GEOM_FNS = /* wgsl */ `
struct Geom {
  ok: f32,
  pc: vec3f,          // cam-space point
  q: vec4f,           // normalized quat (w,x,y,z)
  r0: vec3f, r1: vec3f, r2: vec3f,   // rows of R(q)
  s: vec3f,           // scales
  t0: vec3f, t1: vec3f,              // rows of T = J*W
  va: f32, vb: f32, vc: f32,         // 2D covariance
  s00: f32, s01: f32, s02: f32, s11: f32, s12: f32, s22: f32, // 3D cov
};

fn computeGeom(pbase: u32) -> Geom {
  var g: Geom;
  g.ok = 0.0;
  let p = vec3f(params[pbase], params[pbase + 1u], params[pbase + 2u]);
  g.pc = vec3f(dot(cam.R0.xyz, p), dot(cam.R1.xyz, p), dot(cam.R2.xyz, p)) + cam.t.xyz;
  if (g.pc.z < cam.t.w) { return g; }

  var q = vec4f(params[pbase + 6u], params[pbase + 7u], params[pbase + 8u], params[pbase + 9u]);
  let ql = length(q);
  if (ql < 1e-6) { q = vec4f(1.0, 0.0, 0.0, 0.0); } else { q = q / ql; }
  g.q = q;
  let qw = q.x; let qx = q.y; let qy = q.z; let qz = q.w;
  g.r0 = vec3f(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy));
  g.r1 = vec3f(2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx));
  g.r2 = vec3f(2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy));

  g.s = vec3f(
    exp(clamp(params[pbase + 3u], -12.0, 6.0)),
    exp(clamp(params[pbase + 4u], -12.0, 6.0)),
    exp(clamp(params[pbase + 5u], -12.0, 6.0)));

  // M = R * diag(s); Sigma3D = M M^T
  let m0 = g.r0 * g.s;
  let m1 = g.r1 * g.s;
  let m2 = g.r2 * g.s;
  g.s00 = dot(m0, m0); g.s01 = dot(m0, m1); g.s02 = dot(m0, m2);
  g.s11 = dot(m1, m1); g.s12 = dot(m1, m2); g.s22 = dot(m2, m2);

  // T = J * W (2x3): J = [[f/z, 0, -f x/z^2], [0, f/z, -f y/z^2]]
  let f = cam.proj.x;
  let iz = 1.0 / g.pc.z;
  g.t0 = (f * iz) * cam.R0.xyz + (-f * g.pc.x * iz * iz) * cam.R2.xyz;
  g.t1 = (f * iz) * cam.R1.xyz + (-f * g.pc.y * iz * iz) * cam.R2.xyz;

  // V = T Sigma T^T
  let st0 = vec3f(
    g.s00 * g.t0.x + g.s01 * g.t0.y + g.s02 * g.t0.z,
    g.s01 * g.t0.x + g.s11 * g.t0.y + g.s12 * g.t0.z,
    g.s02 * g.t0.x + g.s12 * g.t0.y + g.s22 * g.t0.z);
  let st1 = vec3f(
    g.s00 * g.t1.x + g.s01 * g.t1.y + g.s02 * g.t1.z,
    g.s01 * g.t1.x + g.s11 * g.t1.y + g.s12 * g.t1.z,
    g.s02 * g.t1.x + g.s12 * g.t1.y + g.s22 * g.t1.z);
  g.va = max(dot(g.t0, st0), 0.0);
  g.vb = dot(g.t1, st0);
  g.vc = max(dot(g.t1, st1), 0.0);
  g.ok = 1.0;
  return g;
}
`;

// Pass 1: project each splat and COUNT the tiles it touches.
export const makeProjectSrc = (E = DEFAULT_E_CUT, A = DEFAULT_A_MIN, RC = 1.0, shDeg = 0) =>
  CAM_STRUCT + cutConsts(E, A, RC) + /* wgsl */ `
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read_write> proj: array<f32>;
@group(0) @binding(3) var<storage, read_write> tileCnt: array<atomic<u32>>;
` + (shDeg > 0 ? `@group(0) @binding(4) var<storage, read> sh: array<f32>;\n` : '')
  + GEOM_FNS + (shDeg > 0 ? shFns(shDeg) : '') + /* wgsl */ `
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u32(cam.size.w)) { return; }
  let b = i * 16u;
  proj[b + 11u] = 0.0; // culled until proven visible

  let g = computeGeom(b);
  if (g.ok < 0.5) { return; }

  let detV = max(g.va * g.vc - g.vb * g.vb, 0.0);
  let ad = g.va + 0.3;
  let cd = g.vc + 0.3;
  let detVd = ad * cd - g.vb * g.vb;
  if (detVd < 1e-8) { return; }
  // Mip-Splatting opacity compensation: dilation must not add energy
  let comp = sqrt(max(detV / detVd, 0.0));
  let opa = 1.0 / (1.0 + exp(-clamp(params[b + 13u], -9.0, 9.0)));
  if (opa * comp < A_MIN) { return; }

  let f = cam.proj.x;
  let mx = f * g.pc.x / g.pc.z + cam.proj.y;
  let my = f * g.pc.y / g.pc.z + cam.proj.z;
  // bounding radius from the largest eigenvalue of the dilated covariance,
  // shrunk opacity-aware: bin only where alpha can still exceed A_MIN
  let mid = 0.5 * (ad + cd);
  let disc = sqrt(max(mid * mid - detVd, 0.0));
  let eMax = min(E_CUT, log(max(opa * comp / A_MIN, 1.0001)));
  let rad = min(sqrt(2.0 * eMax) * sqrt(mid + disc), RADCL * cam.size.x);
  let W = cam.size.x;
  let H = cam.size.y;
  if (mx + rad < 0.0 || my + rad < 0.0 || mx - rad > W || my - rad > H) { return; }

  let inv = 1.0 / detVd;
  proj[b]       = mx;
  proj[b + 1u]  = my;
  proj[b + 2u]  = g.pc.z;
  proj[b + 3u]  = cd * inv;        // conic A
  proj[b + 4u]  = -g.vb * inv;     // conic B
  proj[b + 5u]  = ad * inv;        // conic C
  proj[b + 6u]  = comp;
  proj[b + 7u]  = opa;
  var col = vec3f(
    1.0 / (1.0 + exp(-clamp(params[b + 10u], -9.0, 9.0))),
    1.0 / (1.0 + exp(-clamp(params[b + 11u], -9.0, 9.0))),
    1.0 / (1.0 + exp(-clamp(params[b + 12u], -9.0, 9.0))));
${shDeg > 0 ? /* wgsl */ `
  // view-dependent color: SH rest bands added to the sigmoid DC, clamp at 0
  {
    let un = vec3f(params[b], params[b + 1u], params[b + 2u]) - camPosWorld();
    let v = un / max(length(un), 1e-9);
    var Y = shBasis(v);
    let aK = shActiveK();
    let sb = i * ${3 * shRestCoefs(shDeg)}u;
    for (var k = 0u; k < aK; k++) {
      col += vec3f(sh[sb + k], sh[sb + SHK + k], sh[sb + 2u * SHK + k]) * Y[k];
    }
    col = max(col, vec3f(0.0));
  }` : ''}
  proj[b + 8u]  = col.x;
  proj[b + 9u]  = col.y;
  proj[b + 10u] = col.z;
  proj[b + 11u] = 1.0;
  proj[b + 12u] = g.va;
  proj[b + 13u] = g.vb;
  proj[b + 14u] = g.vc;
  proj[b + 15u] = rad;

  let tilesX = u32(cam.size.z);
  let tilesY = u32(ceil(H / TILEF));
  let tx0 = u32(clamp(floor((mx - rad) / TILEF), 0.0, f32(tilesX - 1u)));
  let tx1 = u32(clamp(floor((mx + rad) / TILEF), 0.0, f32(tilesX - 1u)));
  let ty0 = u32(clamp(floor((my - rad) / TILEF), 0.0, f32(tilesY - 1u)));
  let ty1 = u32(clamp(floor((my + rad) / TILEF), 0.0, f32(tilesY - 1u)));
  for (var ty = ty0; ty <= ty1; ty++) {
    for (var tx = tx0; tx <= tx1; tx++) {
      atomicAdd(&tileCnt[ty * tilesX + tx], 1u);
    }
  }
}
`;

// Pass 2: single-workgroup two-level exclusive scan over per-tile counts.
// Segments over SHSORT get padded to the next power of two so the global
// bitonic path has real padding slots. Writes tileStart (segment starts,
// +1 sentinel), initializes tileCursor, flags overflow in stats[3].
export const SCAN_SRC = CAM_STRUCT + /* wgsl */ `
@group(0) @binding(1) var<storage, read> tileCnt: array<u32>;
@group(0) @binding(2) var<storage, read_write> tileStart: array<u32>;
@group(0) @binding(3) var<storage, read_write> tileCursor: array<u32>;
@group(0) @binding(4) var<storage, read_write> stats: array<atomic<u32>>;

const CHUNK = 64u; // 256 threads x 64 = up to 16384 tiles
var<workgroup> sums: array<u32, 256>;

fn paddedSize(c: u32) -> u32 {
  if (c <= SHSORT) { return c; }
  return 1u << (32u - countLeadingZeros(c - 1u));
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  let tilesX = u32(cam.size.z);
  let tilesY = u32(ceil(cam.size.y / TILEF));
  let numTiles = tilesX * tilesY;

  var local = 0u;
  for (var k = 0u; k < CHUNK; k++) {
    let t = li * CHUNK + k;
    if (t < numTiles) { local += paddedSize(tileCnt[t]); }
  }
  sums[li] = local;
  workgroupBarrier();
  if (li == 0u) {
    var acc = 0u;
    for (var k = 0u; k < 256u; k++) {
      let v = sums[k];
      sums[k] = acc;
      acc += v;
    }
  }
  workgroupBarrier();
  var acc = sums[li];
  for (var k = 0u; k < CHUNK; k++) {
    let t = li * CHUNK + k;
    if (t >= numTiles) { break; }
    var p = paddedSize(tileCnt[t]);
    if (acc + p > ENTCAP) { // out of entry budget: drop this tile, flag it
      p = 0u;
      atomicAdd(&stats[3], 1u);
    }
    tileStart[t] = acc;
    tileCursor[t] = acc;
    // dropped tiles get zero-length segments (start == next start)
    if (p == 0u && tileCnt[t] > 0u) { tileStart[t] = acc; }
    acc += p;
    if (t == numTiles - 1u) { tileStart[numTiles] = acc; }
  }
}
`;

// Pass 3: scatter (depthKey, id) pairs into each splat's tiles (bounds from
// the proj buffer, identical to the count pass).
export const SCATTER_SRC = CAM_STRUCT + /* wgsl */ `
@group(0) @binding(1) var<storage, read> proj: array<f32>;
@group(0) @binding(2) var<storage, read_write> tileCursor: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> entries: array<u32>;
@group(0) @binding(4) var<storage, read> tileStart: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u32(cam.size.w)) { return; }
  let b = i * 16u;
  if (proj[b + 11u] <= 0.0) { return; }
  let mx = proj[b];
  let my = proj[b + 1u];
  let rad = proj[b + 15u];
  let key = bitcast<u32>(proj[b + 2u]); // positive depth: bits are monotonic

  let tilesX = u32(cam.size.z);
  let tilesY = u32(ceil(cam.size.y / TILEF));
  let tx0 = u32(clamp(floor((mx - rad) / TILEF), 0.0, f32(tilesX - 1u)));
  let tx1 = u32(clamp(floor((mx + rad) / TILEF), 0.0, f32(tilesX - 1u)));
  let ty0 = u32(clamp(floor((my - rad) / TILEF), 0.0, f32(tilesY - 1u)));
  let ty1 = u32(clamp(floor((my + rad) / TILEF), 0.0, f32(tilesY - 1u)));
  let numTiles = tilesX * tilesY;
  for (var ty = ty0; ty <= ty1; ty++) {
    for (var tx = tx0; tx <= tx1; tx++) {
      let t = ty * tilesX + tx;
      // zero-length segments (budget-dropped tiles) take no entries
      if (tileStart[t + 1u] == tileStart[t]) { continue; }
      let pos = atomicAdd(&tileCursor[t], 1u);
      entries[2u * pos] = key;
      entries[2u * pos + 1u] = i;
    }
  }
}
`;

// Pass 4: sort each tile segment front-to-back (key asc, id tie-break).
// Fast path: bitonic in shared memory (<= SHSORT entries). Large segments:
// bitonic in global memory over the pow2-padded segment.
export const SORT_SRC = /* wgsl */ `
@group(0) @binding(0) var<storage, read> tileStart: array<u32>;
@group(0) @binding(1) var<storage, read> tileCursor: array<u32>;
@group(0) @binding(2) var<storage, read_write> entries: array<u32>;

const SHSORT = ${SHARED_SORT}u;
var<workgroup> sk: array<u32, ${SHARED_SORT}>;
var<workgroup> sv: array<u32, ${SHARED_SORT}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u,
        @builtin(local_invocation_index) li: u32) {
  let tile = wg.x;
  let s = tileStart[tile];
  let segCap = tileStart[tile + 1u] - s; // pow2 for large segments
  if (segCap == 0u) { return; }
  let cnt = tileCursor[tile] - s;

  if (cnt <= SHSORT) {
    // ---- shared-memory bitonic over SHSORT slots ----
    for (var i = li; i < SHSORT; i += 256u) {
      if (i < cnt) {
        sk[i] = entries[2u * (s + i)];
        sv[i] = entries[2u * (s + i) + 1u];
      } else {
        sk[i] = 0xFFFFFFFFu;
        sv[i] = 0xFFFFFFFFu;
      }
    }
    workgroupBarrier();
    for (var k = 2u; k <= SHSORT; k = k << 1u) {
      for (var j = k >> 1u; j > 0u; j = j >> 1u) {
        for (var i = li; i < SHSORT; i += 256u) {
          let l = i ^ j;
          if (l > i) {
            let asc = (i & k) == 0u;
            let gt = sk[i] > sk[l] || (sk[i] == sk[l] && sv[i] > sv[l]);
            if (gt == asc) {
              let tk = sk[i]; sk[i] = sk[l]; sk[l] = tk;
              let tv = sv[i]; sv[i] = sv[l]; sv[l] = tv;
            }
          }
        }
        workgroupBarrier();
      }
    }
    for (var i = li; i < cnt; i += 256u) {
      entries[2u * (s + i)] = sk[i];
      entries[2u * (s + i) + 1u] = sv[i];
    }
    return;
  }

  // ---- global-memory bitonic over the pow2-padded segment ----
  for (var i = li; i < segCap; i += 256u) {
    if (i >= cnt) {
      entries[2u * (s + i)] = 0xFFFFFFFFu;
      entries[2u * (s + i) + 1u] = 0xFFFFFFFFu;
    }
  }
  storageBarrier();
  for (var k = 2u; k <= segCap; k = k << 1u) {
    for (var j = k >> 1u; j > 0u; j = j >> 1u) {
      for (var i = li; i < segCap; i += 256u) {
        let l = i ^ j;
        if (l > i) {
          let asc = (i & k) == 0u;
          let ki = entries[2u * (s + i)];
          let kl = entries[2u * (s + l)];
          let vi = entries[2u * (s + i) + 1u];
          let vl = entries[2u * (s + l) + 1u];
          let gt = ki > kl || (ki == kl && vi > vl);
          if (gt == asc) {
            entries[2u * (s + i)] = kl; entries[2u * (s + l)] = ki;
            entries[2u * (s + i) + 1u] = vl; entries[2u * (s + l) + 1u] = vi;
          }
        }
      }
      storageBarrier();
    }
  }
}
`;

// The render pass comes in two gradient-accumulation flavours:
//   tileGrad=false  every pixel atomicAdds its 10 gradient slots straight to
//                   global memory — fastest on desktop GPUs (L2-side atomics)
//   tileGrad=true   the whole tile walks entries in lockstep, sums each
//                   entry's gradients in on-chip workgroup memory and flushes
//                   ONE global atomicAdd per slot per splat per tile. Apple
//                   (TBDR) GPUs pay dearly for contended global atomics —
//                   this is the difference between ~12 it/s and usable on an
//                   M1. Integer sums commute, so results are bit-identical.
export const makeRenderSrc = (E = DEFAULT_E_CUT, A = DEFAULT_A_MIN, tileGrad = false) =>
  CAM_STRUCT + cutConsts(E, A) + /* wgsl */ `
@group(0) @binding(1) var<storage, read> proj: array<f32>;
@group(0) @binding(2) var<storage, read> tileStart: array<u32>;
@group(0) @binding(3) var<storage, read> entries: array<u32>;
@group(0) @binding(4) var<storage, read> tgtImg: array<u32>; // packed RGBA8, alpha 0 = invalid
@group(0) @binding(5) var<storage, read_write> outImg: array<f32>;
@group(0) @binding(6) var<storage, read_write> gradP: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> stats: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> gradCam: array<atomic<i32>>;

fn camAdd(idx: u32, v: f32) {
  atomicAdd(&gradCam[idx], i32(clamp(v * FIXCAM, -1.0e9, 1.0e9)));
}
` + (tileGrad ? /* wgsl */ `
var<workgroup> wgEnd: atomic<u32>;
var<workgroup> wgEndU: u32;
var<workgroup> sg: array<atomic<i32>, 10>;
fn atomAdd(slot: u32, v: f32) {
  atomicAdd(&sg[slot], i32(clamp(v * FIXED, -1.0e9, 1.0e9)));
}
fn atomAddC(slot: u32, v: f32) {
  atomicAdd(&sg[slot], i32(clamp(v * FIXEDC, -1.0e9, 1.0e9)));
}
` : /* wgsl */ `
fn atomAdd(idx: u32, v: f32) {
  atomicAdd(&gradP[idx], i32(clamp(v * FIXED, -1.0e9, 1.0e9)));
}
fn atomAddC(idx: u32, v: f32) {
  atomicAdd(&gradP[idx], i32(clamp(v * FIXEDC, -1.0e9, 1.0e9)));
}
`) + /* wgsl */ `

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3u,
        @builtin(workgroup_id) wid: vec3u,
        @builtin(local_invocation_index) li: u32) {
  let W = u32(cam.size.x);
  let H = u32(cam.size.y);
${tileGrad ? '  let pxOk = g.x < W && g.y < H;' : '  if (g.x >= W || g.y >= H) { return; }'}
  let px = vec2f(f32(g.x) + 0.5, f32(g.y) + 0.5);
  let tile = wid.y * u32(cam.size.z) + wid.x;
  let segS = tileStart[tile];
  let segE = tileStart[tile + 1u];

  // ---- forward: sorted front-to-back alpha compositing ----
  var T = 1.0;
  var Crgb = vec3f(0.0);
  var end = segS; // one past the last processed entry
${tileGrad ? '  if (pxOk) {' : '  {'}
  for (var k = segS; k < segE; k++) {
    if (entries[2u * k] == 0xFFFFFFFFu) { break; } // segment padding
    end = k + 1u;
    let i = entries[2u * k + 1u];
    let b = i * 16u;
    let d = px - vec2f(proj[b], proj[b + 1u]);
    let e = max(0.5 * (proj[b + 3u] * d.x * d.x + proj[b + 5u] * d.y * d.y) + proj[b + 4u] * d.x * d.y, 0.0);
    if (e > E_CUT) { continue; }
    let araw = proj[b + 7u] * proj[b + 6u] * exp(-e);
    let alpha = min(0.99, araw);
    if (alpha < A_MIN) { continue; }
    Crgb += T * alpha * vec3f(proj[b + 8u], proj[b + 9u], proj[b + 10u]);
    T *= 1.0 - alpha;
    if (T < 1e-4) { break; }
  }
  }
  let bg = cam.misc.xyz;
  let C = Crgb + T * bg;
  let pi = g.y * W + g.x;
${tileGrad ? '  if (pxOk) {' : '  {'}
  outImg[pi * 4u]      = C.r;
  outImg[pi * 4u + 1u] = C.g;
  outImg[pi * 4u + 2u] = C.b;
  outImg[pi * 4u + 3u] = 1.0 - T;
  }

  if (cam.misc2.x < 0.5) { return; } // uniform: view render, no gradients

  // ---- loss ----
  let off = bitcast<u32>(cam.misc.w); // raw u32 PIXEL offset (f32 exact only to 2^24)
${tileGrad ? '  var lossOk = pxOk;' : '  var lossOk = true;'}
  var gC = vec3f(0.0);
  if (lossOk) {
    let packed = tgtImg[off + pi];
    if ((packed >> 24u) == 0u) {
      lossOk = false; // invalid pixel (undistortion out-of-frame sentinel)
    } else {
      let tcol = unpack4x8unorm(packed).rgb;
      atomicAdd(&stats[2], 1u); // valid-pixel count (PSNR denominator)
      // per-image exposure compensation (gain = cam.proj.w, bias = cam.misc2.w)
      let gain = cam.proj.w;
      let err = (gain * C + vec3f(cam.misc2.w)) - tcol;
      // squared error for the PSNR metric, DITHERED before quantization: plain
      // truncation zeroes sub-quantum pixels and inflates PSNR above ~40dB
      let dith = fract(sin(f32(pi) * 12.9898) * 43758.5453);
      atomicAdd(&stats[0], u32(dot(err, err) * 16.0 + dith));
      // Charbonnier (smooth L1); gC = dL/dC up to a constant
      const DELTA = 0.03;
      let root = sqrt(err * err + vec3f(DELTA * DELTA));
      let eg = err / root;         // dL / d(exposure-adjusted color)
      gC = gain * eg;              // dL / d(rendered color)
      let lossv = (root.x + root.y + root.z) - 3.0 * DELTA;
      atomicAdd(&stats[1], u32(lossv * 32768.0)); // training loss (grad-check)
      let ci8 = u32(cam.misc2.y) * 8u;
      camAdd(ci8 + 6u, dot(eg, C) * gain); // d/d(log gain)
      camAdd(ci8 + 7u, eg.x + eg.y + eg.z); // d/d(bias)
    }
  }
${tileGrad ? '' : '  if (!lossOk) { return; }'}

  // ---- backward: back-to-front transmittance recursion ----
  // dC/da_i = c_i T_i - S_i / (1 - a_i),
  // S_i = sum_{k>i} c_k a_k T_k + bg T_N   (everything behind splat i)
${tileGrad ? /* wgsl */ `
  // the whole tile walks the same entries so per-entry sums can live in
  // workgroup memory; pixels beyond their own 'end' simply contribute zero
  atomicMax(&wgEnd, end);
  workgroupBarrier();
  if (li == 0u) { wgEndU = atomicLoad(&wgEnd); }
  let endMax = workgroupUniformLoad(&wgEndU);
` : '  let endMax = end;'}
  var S = bg * T;
  var Ta = T;
  for (var kk = endMax; kk > segS; kk--) {
${tileGrad ? /* wgsl */ `
    if (li < 10u) { atomicStore(&sg[li], 0); }
    workgroupBarrier();
` : ''}
    let i = entries[2u * (kk - 1u) + 1u];
    let b = i * 16u;
${tileGrad ? '    if (lossOk && kk <= end) {' : '    {'}
    let d = px - vec2f(proj[b], proj[b + 1u]);
    let cA = proj[b + 3u];
    let cB = proj[b + 4u];
    let cC = proj[b + 5u];
    let e = max(0.5 * (cA * d.x * d.x + cC * d.y * d.y) + cB * d.x * d.y, 0.0);
    if (e <= E_CUT) {
    let comp = proj[b + 6u];
    let opa = proj[b + 7u];
    let G = exp(-e);
    let araw = opa * comp * G;
    let alpha = min(0.99, araw);
    if (alpha >= A_MIN) {
    let c = vec3f(proj[b + 8u], proj[b + 9u], proj[b + 10u]);

    let Tb = Ta / (1.0 - alpha); // transmittance in front of this splat
    // raw dL/dcolor — the activation chain (sigmoid DC + SH bands) is
    // per-splat constant, so the chain pass applies it once after summation
    let gcv = gC * (alpha * Tb);
    var galpha = dot(gC, c * Tb - S / (1.0 - alpha));
    if (araw > 0.99) { galpha = 0.0; } // alpha clamped: no gradient through it

    let ga = galpha * araw;
    let gmean = ga * vec2f(cA * d.x + cB * d.y, cB * d.x + cC * d.y);
    atomAdd(${tileGrad ? '0u' : 'b'},      gmean.x);
    atomAdd(${tileGrad ? '1u' : 'b + 1u'}, gmean.y);
    // conic grads scale with d^2 (px^2): measured at native res, big-splat
    // accumulators hit the i32 ceiling and silently wrapped (max 2.14e9 with
    // FIXEDC 4096). Normalize per splat by (1 + lambda_max) of the dilated 2D
    // covariance — d^2/lambda_max is bounded by 2*E_CUT, so per-add values
    // stay O(1) without amplifying quantization noise (a radius^2 normalizer
    // overshoots by (rad/sigma)^2 and BREAKS the FD gradcheck). The chain
    // pass recomputes the identical factor from proj[12..14] and undoes it.
    let cva = proj[b + 12u] + 0.3;
    let cvc = proj[b + 14u] + 0.3;
    let cmid = 0.5 * (cva + cvc);
    let lmax = cmid + sqrt(max(cmid * cmid - (cva * cvc - proj[b + 13u] * proj[b + 13u]), 0.0));
    let cnorm = 1.0 / (1.0 + lmax);
    atomAddC(${tileGrad ? '2u' : 'b + 2u'}, -ga * 0.5 * d.x * d.x * cnorm);
    atomAddC(${tileGrad ? '3u' : 'b + 3u'}, -ga * d.x * d.y * cnorm);
    atomAddC(${tileGrad ? '4u' : 'b + 4u'}, -ga * 0.5 * d.y * d.y * cnorm);
    atomAdd(${tileGrad ? '5u' : 'b + 5u'}, galpha * opa * G);          // d/dcomp
    atomAdd(${tileGrad ? '6u' : 'b + 6u'}, ga * (1.0 - opa));          // d/dlogitOpacity
    atomAdd(${tileGrad ? '7u' : 'b + 7u'}, gcv.r);
    atomAdd(${tileGrad ? '8u' : 'b + 8u'}, gcv.g);
    atomAdd(${tileGrad ? '9u' : 'b + 9u'}, gcv.b);

    S += c * alpha * Tb;
    Ta = Tb;
    }
    }
    }
${tileGrad ? /* wgsl */ `
    workgroupBarrier();
    if (li < 10u) {
      let v = atomicLoad(&sg[li]);
      if (v != 0) { atomicAdd(&gradP[b + li], v); }
    }
` : ''}
  }
}
`;

export const makeChainSrc = (AREG = 0.02, shDeg = 0) => CAM_STRUCT + /* wgsl */ `
const AREG = ${AREG.toExponential()};
` + /* wgsl */ `
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read> proj: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradP: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> gradF: array<f32>;
// per-camera pose gradients: row per camera x 8 slots
// [dwx, dwy, dwz, dtx, dty, dtz, dlogGain, dBias]; row numCams slot 0 = dlogf
@group(0) @binding(5) var<storage, read_write> gradCam: array<atomic<i32>>;
${shDeg > 0 ? `@group(0) @binding(6) var<storage, read> sh: array<f32>;
@group(0) @binding(7) var<storage, read_write> shGrad: array<f32>;` : ''}

fn camAdd(idx: u32, v: f32) {
  atomicAdd(&gradCam[idx], i32(clamp(v * FIXCAM, -1.0e9, 1.0e9)));
}
` + GEOM_FNS + (shDeg > 0 ? shFns(shDeg) : '') + /* wgsl */ `
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u32(cam.size.w)) { return; }
  let b = i * 16u;

  var gp: array<f32, 10>;
  for (var k = 0u; k < 10u; k++) {
    let scale = select(FIXED, FIXEDC, k >= 2u && k <= 4u);
    gp[k] = f32(atomicLoad(&gradP[b + k])) / scale;
    atomicStore(&gradP[b + k], 0);
  }
  // undo the render pass's per-splat conic range normalization (identical
  // lambda_max formula from the same proj values)
  {
    let cva = proj[b + 12u] + 0.3;
    let cvc = proj[b + 14u] + 0.3;
    let cmid = 0.5 * (cva + cvc);
    let lmax = cmid + sqrt(max(cmid * cmid - (cva * cvc - proj[b + 13u] * proj[b + 13u]), 0.0));
    let cdenorm = 1.0 + lmax;
    gp[2] *= cdenorm;
    gp[3] *= cdenorm;
    gp[4] *= cdenorm;
  }
  for (var k = 0u; k < 16u; k++) { gradF[b + k] = 0.0; }
${shDeg > 0 ? /* wgsl */ `
  let sbz = i * ${3 * shRestCoefs(shDeg)}u;
  for (var k = 0u; k < ${3 * shRestCoefs(shDeg)}u; k++) { shGrad[sbz + k] = 0.0; }` : ''}
  if (proj[b + 11u] <= 0.0) { return; }

  let g = computeGeom(b);
  if (g.ok < 0.5) { return; }

  let f = cam.proj.x;
  let z = g.pc.z;
  let iz = 1.0 / z;

  // ---- conic -> dilated 2D covariance ----
  let ad = g.va + 0.3;
  let cd = g.vc + 0.3;
  let detVd = ad * cd - g.vb * g.vb;
  let inv = 1.0 / detVd;
  let kA = cd * inv;
  let kB = -g.vb * inv;
  let kC = ad * inv;
  let g11 = gp[2]; let g12 = 0.5 * gp[3]; let g22 = gp[4];
  let m100 = g11 * kA + g12 * kB;
  let m101 = g11 * kB + g12 * kC;
  let m110 = g12 * kA + g22 * kB;
  let m111 = g12 * kB + g22 * kC;
  let p00 = kA * m100 + kB * m110;
  let p01 = kA * m101 + kB * m111;
  let p10 = kB * m100 + kC * m110;
  let p11 = kB * m101 + kC * m111;
  var gva = -p00;
  var gvb = -(p01 + p10);
  var gvc = -p11;

  // ---- comp -> raw 2D covariance ----
  let detV = max(g.va * g.vc - g.vb * g.vb, 0.0);
  let comp = sqrt(max(detV / detVd, 0.0));
  if (comp > 1e-4) {
    let gcomp = gp[5];
    let denom = 2.0 * comp * detVd;
    gva += gcomp * (g.vc - comp * comp * cd) / denom;
    gvc += gcomp * (g.va - comp * comp * ad) / denom;
    gvb += gcomp * (2.0 * g.vb * (comp * comp - 1.0)) / denom;
  }

  // ---- V = T Sigma T^T backward ----
  let h11 = gva; let h12 = 0.5 * gvb; let h22 = gvc;
  let u0 = h11 * g.t0 + h12 * g.t1;
  let u1 = h12 * g.t0 + h22 * g.t1;
  let dS00 = g.t0.x * u0.x + g.t1.x * u1.x;
  let dS01 = 0.5 * ((g.t0.x * u0.y + g.t1.x * u1.y) + (g.t0.y * u0.x + g.t1.y * u1.x));
  let dS02 = 0.5 * ((g.t0.x * u0.z + g.t1.x * u1.z) + (g.t0.z * u0.x + g.t1.z * u1.x));
  let dS11 = g.t0.y * u0.y + g.t1.y * u1.y;
  let dS12 = 0.5 * ((g.t0.y * u0.z + g.t1.y * u1.z) + (g.t0.z * u0.y + g.t1.z * u1.y));
  let dS22 = g.t0.z * u0.z + g.t1.z * u1.z;

  let sT0 = vec3f(
    g.s00 * g.t0.x + g.s01 * g.t0.y + g.s02 * g.t0.z,
    g.s01 * g.t0.x + g.s11 * g.t0.y + g.s12 * g.t0.z,
    g.s02 * g.t0.x + g.s12 * g.t0.y + g.s22 * g.t0.z);
  let sT1 = vec3f(
    g.s00 * g.t1.x + g.s01 * g.t1.y + g.s02 * g.t1.z,
    g.s01 * g.t1.x + g.s11 * g.t1.y + g.s12 * g.t1.z,
    g.s02 * g.t1.x + g.s12 * g.t1.y + g.s22 * g.t1.z);
  let dT0 = 2.0 * (h11 * sT0 + h12 * sT1);
  let dT1 = 2.0 * (h12 * sT0 + h22 * sT1);

  // ---- T = J W: gradient to cam-space position through J ----
  let dJ00 = dot(dT0, cam.R0.xyz);
  let dJ02 = dot(dT0, cam.R2.xyz);
  let dJ11 = dot(dT1, cam.R1.xyz);
  let dJ12 = dot(dT1, cam.R2.xyz);
  var dpc = vec3f(0.0);
  dpc.x += dJ02 * (-f * iz * iz);
  dpc.y += dJ12 * (-f * iz * iz);
  dpc.z += (dJ00 + dJ11) * (-f * iz * iz)
         + dJ02 * (2.0 * f * g.pc.x * iz * iz * iz)
         + dJ12 * (2.0 * f * g.pc.y * iz * iz * iz);

  // ---- mean path ----
  dpc.x += gp[0] * f * iz;
  dpc.y += gp[1] * f * iz;
  dpc.z += -f * (gp[0] * g.pc.x + gp[1] * g.pc.y) * iz * iz;

  // dL/dp_world = W^T dpc
  gradF[b]      = cam.R0.x * dpc.x + cam.R1.x * dpc.y + cam.R2.x * dpc.z;
  gradF[b + 1u] = cam.R0.y * dpc.x + cam.R1.y * dpc.y + cam.R2.y * dpc.z;
  gradF[b + 2u] = cam.R0.z * dpc.x + cam.R1.z * dpc.y + cam.R2.z * dpc.z;

  // ---- camera pose gradients (train mode): p_c = exp(w^) R p + t ----
  if (cam.misc2.x > 0.5) {
    let ci = u32(cam.misc2.y) * 8u;
    camAdd(ci + 3u, dpc.x); // dL/dt
    camAdd(ci + 4u, dpc.y);
    camAdd(ci + 5u, dpc.z);
    var dw = cross(g.pc - cam.t.xyz, dpc);
    let dW0 = (f * iz) * dT0;
    let dW1 = (f * iz) * dT1;
    let dW2 = (-f * g.pc.x * iz * iz) * dT0 + (-f * g.pc.y * iz * iz) * dT1;
    let wc0 = vec3f(cam.R0.x, cam.R1.x, cam.R2.x);
    let wc1 = vec3f(cam.R0.y, cam.R1.y, cam.R2.y);
    let wc2 = vec3f(cam.R0.z, cam.R1.z, cam.R2.z);
    dw += cross(wc0, vec3f(dW0.x, dW1.x, dW2.x))
        + cross(wc1, vec3f(dW0.y, dW1.y, dW2.y))
        + cross(wc2, vec3f(dW0.z, dW1.z, dW2.z));
    camAdd(ci,      dw.x);
    camAdd(ci + 1u, dw.y);
    camAdd(ci + 2u, dw.z);
    // shared focal: dL/dlogf = f dL/df (mean path + J path, J entries all ~f)
    let dlogf = gp[0] * f * g.pc.x * iz + gp[1] * f * g.pc.y * iz
      + (dJ00 + dJ11) * (f * iz)
      + dJ02 * (-f * g.pc.x * iz * iz) + dJ12 * (-f * g.pc.y * iz * iz);
    camAdd(u32(cam.misc2.z) * 8u, dlogf);
  }

  // ---- Sigma = M M^T backward: dL/dM = 2 dLdSigma M, M = R diag(s) ----
  let m0 = g.r0 * g.s;
  let m1 = g.r1 * g.s;
  let m2 = g.r2 * g.s;
  let dM0 = 2.0 * (dS00 * m0 + dS01 * m1 + dS02 * m2);
  let dM1 = 2.0 * (dS01 * m0 + dS11 * m1 + dS12 * m2);
  let dM2 = 2.0 * (dS02 * m0 + dS12 * m1 + dS22 * m2);

  // dL/ds_k = sum_i dM[i][k] R[i][k];  dlogs = ds * s
  let dsv = vec3f(
    dM0.x * g.r0.x + dM1.x * g.r1.x + dM2.x * g.r2.x,
    dM0.y * g.r0.y + dM1.y * g.r1.y + dM2.y * g.r2.y,
    dM0.z * g.r0.z + dM1.z * g.r1.z + dM2.z * g.r2.z);
  // Anisotropy regularizer: pull each log-scale toward the splat's mean.
  // On low-parallax data the thin dimensions are unconstrained and random-walk
  // to the clamps (needle artifact); this restoring force wins exactly where
  // data gradients are absent, while data-supported plates override it.
  let ls = vec3f(
    clamp(params[b + 3u], -12.0, 6.0),
    clamp(params[b + 4u], -12.0, 6.0),
    clamp(params[b + 5u], -12.0, 6.0));
  let mls = (ls.x + ls.y + ls.z) / 3.0;
  gradF[b + 3u] = dsv.x * g.s.x + AREG * (ls.x - mls);
  gradF[b + 4u] = dsv.y * g.s.y + AREG * (ls.y - mls);
  gradF[b + 5u] = dsv.z * g.s.z + AREG * (ls.z - mls);

  // dL/dR = dM * diag(s)
  let dR0 = dM0 * g.s;
  let dR1 = dM1 * g.s;
  let dR2 = dM2 * g.s;

  // quaternion backward (normalized q = (w,x,y,z))
  let qw = g.q.x; let qx = g.q.y; let qy = g.q.z; let qz = g.q.w;
  let gw = 2.0 * (dR0.y * (-qz) + dR0.z * qy + dR1.x * qz + dR1.z * (-qx) + dR2.x * (-qy) + dR2.y * qx);
  let gx = 2.0 * (dR0.y * qy + dR0.z * qz + dR1.x * qy + dR1.y * (-2.0 * qx) + dR1.z * (-qw) + dR2.x * qz + dR2.y * qw + dR2.z * (-2.0 * qx));
  let gy = 2.0 * (dR0.x * (-2.0 * qy) + dR0.y * qx + dR0.z * qw + dR1.x * qx + dR1.z * qz + dR2.x * (-qw) + dR2.y * qz + dR2.z * (-2.0 * qy));
  let gz = 2.0 * (dR0.x * (-2.0 * qz) + dR0.y * (-qw) + dR0.z * qx + dR1.x * qw + dR1.y * (-2.0 * qz) + dR1.z * qy + dR2.x * qx + dR2.y * qy);
  let gq = vec4f(gw, gx, gy, gz);
  var qraw = vec4f(params[b + 6u], params[b + 7u], params[b + 8u], params[b + 9u]);
  let ql = max(length(qraw), 1e-6);
  let gqr = (gq - g.q * dot(g.q, gq)) / ql;
  gradF[b + 6u] = gqr.x;
  gradF[b + 7u] = gqr.y;
  gradF[b + 8u] = gqr.z;
  gradF[b + 9u] = gqr.w;

  // ---- color: DC sigmoid chain + SH band grads (+ position via view dir) ----
  let sCol = vec3f(
    1.0 / (1.0 + exp(-clamp(params[b + 10u], -9.0, 9.0))),
    1.0 / (1.0 + exp(-clamp(params[b + 11u], -9.0, 9.0))),
    1.0 / (1.0 + exp(-clamp(params[b + 12u], -9.0, 9.0))));
  var dRGB = vec3f(gp[7], gp[8], gp[9]);
${shDeg > 0 ? /* wgsl */ `
  {
    let un = vec3f(params[b], params[b + 1u], params[b + 2u]) - camPosWorld();
    let ulen = max(length(un), 1e-9);
    let v = un / ulen;
    var Y = shBasis(v);
    var D = shBasisGrad(v);
    let aK = shActiveK();
    let sb = i * ${3 * shRestCoefs(shDeg)}u;
    var col = sCol;
    for (var k = 0u; k < aK; k++) {
      col += vec3f(sh[sb + k], sh[sb + SHK + k], sh[sb + 2u * SHK + k]) * Y[k];
    }
    // clamp-at-zero gate: a channel clamped in the forward has no gradient
    dRGB *= vec3f(
      select(0.0, 1.0, col.x > 0.0),
      select(0.0, 1.0, col.y > 0.0),
      select(0.0, 1.0, col.z > 0.0));
    var gv = vec3f(0.0);
    for (var k = 0u; k < aK; k++) {
      shGrad[sb + k] = dRGB.x * Y[k];
      shGrad[sb + SHK + k] = dRGB.y * Y[k];
      shGrad[sb + 2u * SHK + k] = dRGB.z * Y[k];
      let w = dRGB.x * sh[sb + k] + dRGB.y * sh[sb + SHK + k] + dRGB.z * sh[sb + 2u * SHK + k];
      gv += w * D[k];
    }
    // dL/dp through the view direction v = (p - cam)/|p - cam|
    let shPos = (gv - v * dot(v, gv)) / ulen;
    gradF[b]      += shPos.x;
    gradF[b + 1u] += shPos.y;
    gradF[b + 2u] += shPos.z;
  }` : ''}
  gradF[b + 10u] = dRGB.x * sCol.x * (1.0 - sCol.x);
  gradF[b + 11u] = dRGB.y * sCol.y * (1.0 - sCol.y);
  gradF[b + 12u] = dRGB.z * sCol.z * (1.0 - sCol.z);
  gradF[b + 13u] = gp[6];
}
`;

// Adam for the SH coefficient buffer (single lr, no clamps, NaN-guarded).
export const SH_ADAM_SRC = /* wgsl */ `
struct SHA {
  hp: vec4f,   // beta1, beta2, eps, step t
  cfg: vec4f,  // x = lr, y = total coeffs (n * 3K)
};
@group(0) @binding(0) var<uniform> au: SHA;
@group(0) @binding(1) var<storage, read_write> sh: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read_write> mBuf: array<f32>;
@group(0) @binding(4) var<storage, read_write> vBuf: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(num_workgroups) nw: vec3u) {
  // 2D dispatch: big splat counts exceed 65535 workgroups in one dimension
  let j = gid.x + gid.y * nw.x * 256u;
  if (j >= u32(au.cfg.y)) { return; }
  var gr = g[j];
  if (!(abs(gr) < 1e18)) { gr = 0.0; }
  let b1 = au.hp.x;
  let b2 = au.hp.y;
  let m = b1 * mBuf[j] + (1.0 - b1) * gr;
  let v = b2 * vBuf[j] + (1.0 - b2) * gr * gr;
  mBuf[j] = m;
  vBuf[j] = v;
  let t = au.hp.w;
  let mh = m / (1.0 - pow(b1, t));
  let vh = v / (1.0 - pow(b2, t));
  sh[j] = sh[j] - au.cfg.x * mh / (sqrt(vh) + au.hp.z);
}
`;

export const ADAM_SRC = /* wgsl */ `
struct AdamU {
  lr0: vec4f,   // lr slots 0..3   (pos xyz, logScale x)
  lr1: vec4f,   // lr slots 4..7   (logScale yz, quat wx)
  lr2: vec4f,   // lr slots 8..11  (quat yz, color rg)
  lr3: vec4f,   // lr slots 12..15 (color b, logitOpacity, pads)
  hp: vec4f,    // beta1, beta2, eps, step t
  cl: vec4f,    // minLogScale, maxLogScale, maxAbsLogit, totalParams (n*16)
  reg: vec4f,   // x = opacity regularization weight
};
@group(0) @binding(0) var<uniform> au: AdamU;
@group(0) @binding(1) var<storage, read_write> params: array<f32>;
@group(0) @binding(2) var<storage, read> gradF: array<f32>;
@group(0) @binding(3) var<storage, read_write> mBuf: array<f32>;
@group(0) @binding(4) var<storage, read_write> vBuf: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(num_workgroups) nw: vec3u) {
  // 2D dispatch: big splat counts exceed 65535 workgroups in one dimension
  let j = gid.x + gid.y * nw.x * 256u;
  if (j >= u32(au.cl.w)) { return; }
  let slot = j % 16u;
  var lr: f32;
  if (slot < 4u) { lr = au.lr0[slot]; }
  else if (slot < 8u) { lr = au.lr1[slot - 4u]; }
  else if (slot < 12u) { lr = au.lr2[slot - 8u]; }
  else { lr = au.lr3[slot - 12u]; }
  if (lr == 0.0) { return; }

  var g = gradF[j];
  if (!(abs(g) < 1e18)) { g = 0.0; } // NaN/Inf guard
  if (slot == 13u) {
    let sg = 1.0 / (1.0 + exp(-clamp(params[j], -9.0, 9.0)));
    g += au.reg.x * sg * (1.0 - sg); // opacity regularizer
  }

  let b1 = au.hp.x;
  let b2 = au.hp.y;
  var m = b1 * mBuf[j] + (1.0 - b1) * g;
  var v = b2 * vBuf[j] + (1.0 - b2) * g * g;
  mBuf[j] = m;
  vBuf[j] = v;
  let t = au.hp.w;
  let mh = m / (1.0 - pow(b1, t));
  let vh = v / (1.0 - pow(b2, t));
  var p = params[j] - lr * mh / (sqrt(vh) + au.hp.z);

  if (slot >= 3u && slot <= 5u) { p = clamp(p, au.cl.x, au.cl.y); }
  if (slot >= 10u && slot <= 13u) { p = clamp(p, -au.cl.z, au.cl.z); } // logits
  params[j] = p;
}
`;

export const BLIT_SRC = CAM_STRUCT + /* wgsl */ `
@group(0) @binding(1) var<storage, read> outImg: array<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[vi], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  if (fc.x >= cam.size.x || fc.y >= cam.size.y) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let x = u32(fc.x);
  let y = u32(fc.y);
  let i = (y * u32(cam.size.x) + x) * 4u;
  return vec4f(
    clamp(outImg[i], 0.0, 1.0),
    clamp(outImg[i + 1u], 0.0, 1.0),
    clamp(outImg[i + 2u], 0.0, 1.0),
    1.0);
}
`;
