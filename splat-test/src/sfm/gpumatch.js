// gpumatch.js — brute-force SIFT descriptor matching on WebGPU.
//
// CPU matching dominates SIFT SfM time (~350s of a 484s train-84 run): each
// pair is nA*nB 128-D uint8 L2 distances. On GPU that is a tiled reduction —
// one workgroup handles 64 query descriptors, streaming the other image's
// descriptors through workgroup memory. The whole 1500-pair match graph is
// ~1 TFLOP of int math: ~1s of RTX-class compute.
//
// Semantics identical to matchSift(): squared-L2 nearest + second-nearest,
// Lowe ratio 0.8, mutual cross-check (done on CPU from the GPU's best/second
// tables).

const WGSL = /* wgsl */`
struct Job { offA: u32, nA: u32, offB: u32, nB: u32, outOff: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> descs: array<u32>;
@group(0) @binding(1) var<storage, read> jobs: array<Job>;
@group(0) @binding(2) var<storage, read_write> outBuf: array<u32>;

var<workgroup> tileB: array<u32, 2048>; // 64 descriptors x 32 u32

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let job = jobs[wg.y];
  let a = wg.x * 64u + li.x;
  var best = 0xFFFFFFFFu;
  var second = 0xFFFFFFFFu;
  var bestIdx = 0xFFFFFFFFu;
  var da: array<u32, 32>;
  if (a < job.nA) {
    for (var k = 0u; k < 32u; k = k + 1u) { da[k] = descs[job.offA + a * 32u + k]; }
  }
  let nTiles = (job.nB + 63u) / 64u;
  for (var t = 0u; t < nTiles; t = t + 1u) {
    let bBase = t * 64u;
    let bLoad = bBase + li.x;
    for (var k = 0u; k < 32u; k = k + 1u) {
      tileB[li.x * 32u + k] = select(0u, descs[job.offB + bLoad * 32u + k], bLoad < job.nB);
    }
    workgroupBarrier();
    if (a < job.nA) {
      let tEnd = min(64u, job.nB - bBase);
      for (var bi = 0u; bi < tEnd; bi = bi + 1u) {
        var s = 0u;
        for (var k = 0u; k < 32u; k = k + 1u) {
          let x = da[k];
          let y = tileB[bi * 32u + k];
          let d0 = i32(x & 0xFFu) - i32(y & 0xFFu);
          let d1 = i32((x >> 8u) & 0xFFu) - i32((y >> 8u) & 0xFFu);
          let d2 = i32((x >> 16u) & 0xFFu) - i32((y >> 16u) & 0xFFu);
          let d3 = i32(x >> 24u) - i32(y >> 24u);
          s = s + u32(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3);
        }
        if (s < best) { second = best; best = s; bestIdx = bBase + bi; }
        else if (s < second) { second = s; }
      }
    }
    workgroupBarrier();
  }
  if (a < job.nA) {
    outBuf[job.outOff + a * 3u] = bestIdx;
    outBuf[job.outOff + a * 3u + 1u] = best;
    outBuf[job.outOff + a * 3u + 2u] = second;
  }
}
`;

// pipeline cache per injected device (a host page owns ONE device shared by
// the matcher and the trainer; a second device would double descriptor VRAM)
const pipelines = new WeakMap();

function getPipeline(device) {
  let p = pipelines.get(device);
  if (!p) {
    const module = device.createShaderModule({ code: WGSL });
    p = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    pipelines.set(device, p);
  }
  return p;
}

/** Match all pairs on GPU. feats: array of { n, desc: Uint8Array(n*128) }.
 *  pairs: [[i, j], ...]. Returns Array (parallel to pairs) of flat match
 *  arrays [ia0, ib0, ia1, ib1, ...].
 *  extDevice: a caller-owned GPUDevice (recommended — share it with the
 *  trainer); when omitted a temporary device is created and destroyed. */
export async function gpuMatchAll(feats, pairs, ratio = 0.8, log = () => {}, extDevice = null) {
  const t0 = performance.now();
  let device = extDevice, ownDevice = false;
  if (!device) {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('no WebGPU adapter');
    device = await adapter.requestDevice();
    ownDevice = true;
  }
  const pipeline = getPipeline(device);

  // upload all descriptors once (concatenated, u32 view of the uint8 data)
  const offsets = new Array(feats.length);
  let totalWords = 0;
  for (let i = 0; i < feats.length; i++) {
    offsets[i] = totalWords;
    totalWords += feats[i].n * 32;
  }
  const descData = new Uint32Array(totalWords);
  for (let i = 0; i < feats.length; i++) {
    const bytes = feats[i].desc;
    descData.set(new Uint32Array(bytes.buffer, bytes.byteOffset, feats[i].n * 32), offsets[i]);
  }
  const descBuf = device.createBuffer({ size: Math.max(16, descData.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(descBuf, 0, descData);

  // jobs: two per pair (A->B and B->A)
  const jobList = [];
  let outWords = 0;
  let maxNA = 0;
  for (const [i, j] of pairs) {
    jobList.push({ offA: offsets[i], nA: feats[i].n, offB: offsets[j], nB: feats[j].n, outOff: outWords });
    outWords += feats[i].n * 3;
    jobList.push({ offA: offsets[j], nA: feats[j].n, offB: offsets[i], nB: feats[i].n, outOff: outWords });
    outWords += feats[j].n * 3;
    maxNA = Math.max(maxNA, feats[i].n, feats[j].n);
  }

  // chunk jobs so the output buffer stays modest
  const results = new Array(pairs.length);
  const CHUNK_WORDS = 24_000_000; // 96MB
  let jStart = 0;
  while (jStart < jobList.length) {
    let jEnd = jStart, words = 0;
    while (jEnd + 1 < jobList.length + 1 && jEnd < jobList.length) {
      const w = jobList[jEnd].nA * 3 + jobList[jEnd + 1].nA * 3;
      if (words + w > CHUNK_WORDS && jEnd > jStart) break;
      words += w;
      jEnd += 2; // keep the two jobs of a pair together
    }
    const chunk = jobList.slice(jStart, jEnd);
    const base = chunk[0].outOff;
    const chunkWords = chunk.reduce((s, jb) => s + jb.nA * 3, 0);
    const jobData = new Uint32Array(chunk.length * 8);
    chunk.forEach((jb, k) => {
      jobData[k * 8] = jb.offA; jobData[k * 8 + 1] = jb.nA;
      jobData[k * 8 + 2] = jb.offB; jobData[k * 8 + 3] = jb.nB;
      jobData[k * 8 + 4] = jb.outOff - base;
    });
    const jobBuf = device.createBuffer({ size: jobData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(jobBuf, 0, jobData);
    const outBuf = device.createBuffer({ size: chunkWords * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuf = device.createBuffer({ size: chunkWords * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: descBuf } },
        { binding: 1, resource: { buffer: jobBuf } },
        { binding: 2, resource: { buffer: outBuf } },
      ],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(maxNA / 64), chunk.length);
    pass.end();
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, chunkWords * 4);
    device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const data = new Uint32Array(readBuf.getMappedRange()).slice();
    readBuf.unmap();
    outBuf.destroy(); jobBuf.destroy(); readBuf.destroy();

    // CPU: ratio + cross-check per pair in this chunk
    const r2 = ratio * ratio;
    for (let k = 0; k + 1 < chunk.length; k += 2) {
      const jA = chunk[k], jB = chunk[k + 1];
      const offA = jA.outOff - base, offB = jB.outOff - base;
      const pairIdx = (jStart + k) / 2;
      const m = [];
      for (let a = 0; a < jA.nA; a++) {
        const bi = data[offA + a * 3];
        const d1 = data[offA + a * 3 + 1];
        const d2 = data[offA + a * 3 + 2];
        if (bi === 0xFFFFFFFF || d1 >= r2 * d2) continue;
        // cross check + B-side ratio
        const ba = data[offB + bi * 3];
        const bd1 = data[offB + bi * 3 + 1];
        const bd2 = data[offB + bi * 3 + 2];
        if (ba === a && bd1 < r2 * bd2) m.push(a, bi);
      }
      results[pairIdx] = m;
    }
    jStart = jEnd;
  }
  descBuf.destroy();
  if (ownDevice) device.destroy();
  log(`  GPU matching: ${pairs.length} pairs in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  return results;
}
