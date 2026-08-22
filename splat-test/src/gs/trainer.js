// trainer.js — WebGPU 3DGS optimizer (anisotropic, sorted; see shaders.js).

import {
  STRIDE, TILE, ENTRIES_CAP, makeProjectSrc, makeRenderSrc, makeChainSrc,
  SCAN_SRC, SCATTER_SRC, SORT_SRC, ADAM_SRC, SH_ADAM_SRC, BLIT_SRC, shRestCoefs,
} from './shaders.js';
import { rodrigues, m3mul } from '../sfm/geometry.js';
import { createGpu } from '../gpu/context.js';

export class GSTrainer {
  /** opts.gpu: a GpuContext from createGpu() — share ONE device between the
   *  trainer and the SIFT matcher. When omitted, a private one is created. */
  static async create(opts = {}) {
    const gpu = opts.gpu || await createGpu(opts);
    return new GSTrainer(gpu.device, opts, gpu.info || {});
  }

  constructor(device, opts = {}, gpuInfo = {}) {
    this.device = device;
    this.opts = opts; // { eCut, aMin, opacityReg } — gradcheck uses strict cutoffs
    this.gpuInfo = gpuInfo;
    // Tile-shared gradient accumulation (see makeRenderSrc): built for Apple
    // (TBDR) GPUs where contended global atomics are ruinous, but measured
    // FASTER on desktop NVIDIA too (+11% synthetic it/s) — default ON
    // everywhere; opts.tileGrad = false restores the direct path.
    this.tileGrad = opts.tileGrad ?? true;
    this.iter = 0;
    this.pixelsSeen = 0;
    this.stride = STRIDE;
    // view-dependent color: SH degree (0 disables; coeffs live in a separate
    // channel-major buffer of 3*shK floats per splat)
    this.shDeg = opts.shDeg ?? 2;
    this.shK = this.shDeg > 0 ? shRestCoefs(this.shDeg) : 0;
    this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    this._buildPipelines();
  }

  _buildPipelines() {
    const d = this.device;
    const mk = (code, label) => d.createShaderModule({ code, label });
    this.pipeProject = d.createComputePipeline({
      label: 'project', layout: 'auto',
      compute: { module: mk(makeProjectSrc(this.opts.eCut, this.opts.aMin, this.opts.radClamp, this.shDeg), 'project'), entryPoint: 'main' },
    });
    // the (key,id) entry budget scales with an explicit splat ceiling — the
    // fixed 12M cap silently dropped tiles at 800k splats (fast iterations,
    // collapsing PSNR); default unchanged unless maxSplats is raised
    this.entriesCap = this.opts.entriesCap ??
      (this.opts.maxSplats ? Math.max(ENTRIES_CAP, this.opts.maxSplats * 24) : ENTRIES_CAP);
    this.pipeScan = d.createComputePipeline({
      label: 'tile-scan', layout: 'auto',
      compute: {
        module: mk(SCAN_SRC, 'tile-scan'), entryPoint: 'main',
        constants: { ENTCAP: this.entriesCap },
      },
    });
    this.pipeScatter = d.createComputePipeline({
      label: 'tile-scatter', layout: 'auto',
      compute: { module: mk(SCATTER_SRC, 'tile-scatter'), entryPoint: 'main' },
    });
    this.pipeSort = d.createComputePipeline({
      label: 'tile-sort', layout: 'auto',
      compute: { module: mk(SORT_SRC, 'tile-sort'), entryPoint: 'main' },
    });
    this.pipeRender = d.createComputePipeline({
      label: 'render', layout: 'auto',
      compute: { module: mk(makeRenderSrc(this.opts.eCut, this.opts.aMin, this.tileGrad), 'render'), entryPoint: 'main' },
    });
    this.pipeChain = d.createComputePipeline({
      label: 'chain', layout: 'auto',
      // anisoReg default 0.005 (was 0.02): with SIFT-grade poses the needle
      // pathology is gone (camping p99 ratio 42:1) and the stronger pull
      // toward isotropy measurably blurs edges (-0.8dB holdout on train-84)
      compute: { module: mk(makeChainSrc(this.opts.anisoReg ?? 0.005, this.shDeg), 'chain'), entryPoint: 'main' },
    });
    this.pipeAdam = d.createComputePipeline({
      label: 'adam', layout: 'auto',
      compute: { module: mk(ADAM_SRC, 'adam'), entryPoint: 'main' },
    });
    if (this.shK) {
      this.pipeSHAdam = d.createComputePipeline({
        label: 'sh-adam', layout: 'auto',
        compute: { module: mk(SH_ADAM_SRC, 'sh-adam'), entryPoint: 'main' },
      });
    }
    const blitModule = mk(BLIT_SRC, 'blit');
    this.pipeBlit = d.createRenderPipeline({
      label: 'blit', layout: 'auto',
      vertex: { module: blitModule, entryPoint: 'vs' },
      fragment: { module: blitModule, entryPoint: 'fs', targets: [{ format: this.canvasFormat }] },
      primitive: { topology: 'triangle-list' },
    });
  }

  /**
   * gaussians: { data: Float32Array (n*STRIDE), n }
   * cams: [{ R, t, f, cx, cy, w, h, imgIdx }]  (training-resolution intrinsics)
   * images: [{ tw, th, rgb }] indexed by cam.imgIdx
   */
  setup(gaussians, cams, images, maxViewW, maxViewH, sceneRadius) {
    const d = this.device;
    this.n = gaussians.n;
    // buffers are allocated for `cap` so refine() can also GROW the splat
    // count (MCMC-style) without reallocating
    this.cap = Math.min(
      Math.max(Math.floor(gaussians.n * (this.opts.capMult ?? 4)), gaussians.n),
      this.opts.maxSplats ?? 600000);
    this.cams = cams;
    this.sceneRadius = sceneRadius;

    // Targets are packed RGBA8 (one u32 per pixel; alpha 0 marks pixels the
    // undistortion resampled out of frame). The source photographs are 8-bit,
    // so this is lossless vs f32 — and 4x less GPU memory and loss-read
    // bandwidth, which is what lets full sets fit on phones.
    let total = 0; // in PIXELS
    this.camMeta = cams.map((c) => {
      const im = images[c.imgIdx];
      const meta = { ...c, w: im.tw, h: im.th, offset: total };
      total += im.tw * im.th;
      return meta;
    });
    const limit = this.device.limits.maxStorageBufferBindingSize;
    if (total * 4 > limit) {
      throw new Error(`training targets (${(total * 4 / 1e6).toFixed(0)}MB) exceed the device ` +
        `binding limit (${(limit / 1e6).toFixed(0)}MB) — reduce image count or resolution`);
    }
    const targetData = new Uint32Array(total);
    for (const meta of this.camMeta) {
      const rgb = images[meta.imgIdx].rgb;
      const np = meta.w * meta.h;
      for (let p = 0; p < np; p++) {
        const r = rgb[p * 3];
        if (r < 0) { targetData[meta.offset + p] = 0; continue; } // invalid sentinel
        targetData[meta.offset + p] = (255 << 24)
          | (Math.min(255, Math.round(rgb[p * 3 + 2] * 255)) << 16)
          | (Math.min(255, Math.round(rgb[p * 3 + 1] * 255)) << 8)
          | Math.min(255, Math.round(r * 255));
      }
    }

    const maxPix = Math.max(maxViewW * maxViewH,
      ...this.camMeta.map((m) => m.w * m.h));
    const maxTiles = Math.max(
      Math.ceil(maxViewW / TILE) * Math.ceil(maxViewH / TILE),
      ...this.camMeta.map((m) => Math.ceil(m.w / TILE) * Math.ceil(m.h / TILE)));
    this.maxTiles = maxTiles;
    this.tileZero = new Uint32Array(maxTiles);

    const B = GPUBufferUsage;
    const buf = (size, usage, label) => d.createBuffer({ size, usage, label });
    const nb = this.cap * STRIDE * 4;
    this.bufParams = buf(nb, B.STORAGE | B.COPY_DST | B.COPY_SRC, 'params');
    d.queue.writeBuffer(this.bufParams, 0, gaussians.data.buffer, gaussians.data.byteOffset,
      this.n * STRIDE * 4);
    this.bufProj = buf(nb, B.STORAGE | B.COPY_SRC, 'proj');
    this.bufGradP = buf(nb, B.STORAGE | B.COPY_SRC, 'gradP'); // COPY_SRC: precision diagnostics
    this.bufGradF = buf(nb, B.STORAGE | B.COPY_SRC, 'gradF');
    this.bufM = buf(nb, B.STORAGE | B.COPY_DST | B.COPY_SRC, 'adam-m');
    this.bufV = buf(nb, B.STORAGE | B.COPY_DST | B.COPY_SRC, 'adam-v');
    d.queue.writeBuffer(this.bufM, 0, new Float32Array(this.cap * STRIDE));
    d.queue.writeBuffer(this.bufV, 0, new Float32Array(this.cap * STRIDE));
    this.bufTileCnt = buf(maxTiles * 4, B.STORAGE | B.COPY_DST | B.COPY_SRC, 'tileCnt');
    this.bufTileStart = buf((maxTiles + 1) * 4, B.STORAGE | B.COPY_SRC, 'tileStart');
    this.bufTileCursor = buf(maxTiles * 4, B.STORAGE | B.COPY_SRC, 'tileCursor');
    this.bufEntries = buf(this.entriesCap * 2 * 4, B.STORAGE | B.COPY_SRC, 'entries');
    this.bufOut = buf(maxPix * 4 * 4, B.STORAGE | B.COPY_SRC, 'outImg');
    this.bufTarget = buf(Math.max(16, total * 4), B.STORAGE | B.COPY_DST, 'targets');
    d.queue.writeBuffer(this.bufTarget, 0, targetData);
    this.bufStats = buf(16, B.STORAGE | B.COPY_DST | B.COPY_SRC, 'stats');
    d.queue.writeBuffer(this.bufStats, 0, new Uint32Array(4));
    this.bufStatsRead = buf(16, B.COPY_DST | B.MAP_READ, 'statsRead');
    this.bufParamsRead = buf(nb, B.COPY_DST | B.MAP_READ, 'paramsRead'); // cap-sized
    this.bufCamGrad = buf((cams.length + 1) * 8 * 4, B.STORAGE | B.COPY_DST | B.COPY_SRC, 'camGrad');
    d.queue.writeBuffer(this.bufCamGrad, 0, new Int32Array((cams.length + 1) * 8));
    if (this.shK) {
      // SH coeffs + their (non-atomic: written once per splat by the chain
      // pass) gradients + Adam moments; zero-initialized per the WebGPU spec
      const shb = this.cap * this.shK * 3 * 4;
      this.bufSH = buf(shb, B.STORAGE | B.COPY_DST | B.COPY_SRC, 'sh');
      this.bufSHGrad = buf(shb, B.STORAGE | B.COPY_SRC, 'shGrad');
      // COPY_SRC is load-bearing: refine() reads these back to relocate donor
      // moments — without it the readback encoder errors and refine silently
      // wrote back ZEROED SH Adam moments every 2500 iters
      this.bufSHM = buf(shb, B.STORAGE | B.COPY_DST | B.COPY_SRC, 'sh-adam-m');
      this.bufSHV = buf(shb, B.STORAGE | B.COPY_DST | B.COPY_SRC, 'sh-adam-v');
      this.uniSHAdam = buf(32, B.UNIFORM | B.COPY_DST, 'uniSHAdam');
    }

    this.uniTrain = buf(144, B.UNIFORM | B.COPY_DST, 'uniTrain');
    this.uniView = buf(144, B.UNIFORM | B.COPY_DST, 'uniView');
    this.uniAdam = buf(112, B.UNIFORM | B.COPY_DST, 'uniAdam');

    const bgProject = (uni) => d.createBindGroup({
      layout: this.pipeProject.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uni } },
        { binding: 1, resource: { buffer: this.bufParams } },
        { binding: 2, resource: { buffer: this.bufProj } },
        { binding: 3, resource: { buffer: this.bufTileCnt } },
        ...(this.shK ? [{ binding: 4, resource: { buffer: this.bufSH } }] : []),
      ],
    });
    const bgScan = (uni) => d.createBindGroup({
      layout: this.pipeScan.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uni } },
        { binding: 1, resource: { buffer: this.bufTileCnt } },
        { binding: 2, resource: { buffer: this.bufTileStart } },
        { binding: 3, resource: { buffer: this.bufTileCursor } },
        { binding: 4, resource: { buffer: this.bufStats } },
      ],
    });
    const bgScatter = (uni) => d.createBindGroup({
      layout: this.pipeScatter.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uni } },
        { binding: 1, resource: { buffer: this.bufProj } },
        { binding: 2, resource: { buffer: this.bufTileCursor } },
        { binding: 3, resource: { buffer: this.bufEntries } },
        { binding: 4, resource: { buffer: this.bufTileStart } },
      ],
    });
    const bgRender = (uni) => d.createBindGroup({
      layout: this.pipeRender.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uni } },
        { binding: 1, resource: { buffer: this.bufProj } },
        { binding: 2, resource: { buffer: this.bufTileStart } },
        { binding: 3, resource: { buffer: this.bufEntries } },
        { binding: 4, resource: { buffer: this.bufTarget } },
        { binding: 5, resource: { buffer: this.bufOut } },
        { binding: 6, resource: { buffer: this.bufGradP } },
        { binding: 7, resource: { buffer: this.bufStats } },
        { binding: 8, resource: { buffer: this.bufCamGrad } },
      ],
    });
    const bgBlit = (uni) => d.createBindGroup({
      layout: this.pipeBlit.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uni } },
        { binding: 1, resource: { buffer: this.bufOut } },
      ],
    });
    this.bgProjectTrain = bgProject(this.uniTrain);
    this.bgProjectView = bgProject(this.uniView);
    this.bgScanTrain = bgScan(this.uniTrain);
    this.bgScanView = bgScan(this.uniView);
    this.bgScatterTrain = bgScatter(this.uniTrain);
    this.bgScatterView = bgScatter(this.uniView);
    this.bgRenderTrain = bgRender(this.uniTrain);
    this.bgRenderView = bgRender(this.uniView);
    this.bgBlitTrain = bgBlit(this.uniTrain);
    this.bgBlitView = bgBlit(this.uniView);
    this.bgSort = d.createBindGroup({
      layout: this.pipeSort.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.bufTileStart } },
        { binding: 1, resource: { buffer: this.bufTileCursor } },
        { binding: 2, resource: { buffer: this.bufEntries } },
      ],
    });
    this.bgChain = d.createBindGroup({
      layout: this.pipeChain.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniTrain } },
        { binding: 1, resource: { buffer: this.bufParams } },
        { binding: 2, resource: { buffer: this.bufProj } },
        { binding: 3, resource: { buffer: this.bufGradP } },
        { binding: 4, resource: { buffer: this.bufGradF } },
        { binding: 5, resource: { buffer: this.bufCamGrad } },
        ...(this.shK ? [
          { binding: 6, resource: { buffer: this.bufSH } },
          { binding: 7, resource: { buffer: this.bufSHGrad } },
        ] : []),
      ],
    });
    this.bgAdam = d.createBindGroup({
      layout: this.pipeAdam.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniAdam } },
        { binding: 1, resource: { buffer: this.bufParams } },
        { binding: 2, resource: { buffer: this.bufGradF } },
        { binding: 3, resource: { buffer: this.bufM } },
        { binding: 4, resource: { buffer: this.bufV } },
      ],
    });
    if (this.shK) {
      this.bgSHAdam = d.createBindGroup({
        layout: this.pipeSHAdam.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniSHAdam } },
          { binding: 1, resource: { buffer: this.bufSH } },
          { binding: 2, resource: { buffer: this.bufSHGrad } },
          { binding: 3, resource: { buffer: this.bufSHM } },
          { binding: 4, resource: { buffer: this.bufSHV } },
        ],
      });
      // Adam hp + lr for the SH coeffs; INRIA convention: rest lr = DC/20,
      // directly applicable since our bands also add in color space
      this.shAdamData = new Float32Array([
        0.9, 0.999, 1e-15, 1,
        this.opts.shLr ?? 1.25e-4, this.n * this.shK * 3, 0, 0,
      ]);
    }

    for (const m of this.camMeta) m.f0 = m.f; // original focal (shared scale is optimized)
    this.camUniforms = this.camMeta.map((m, i) => this._camUniform(m, 1, m.offset, i));
    // camera-pose optimizer state (opts.camOpt enables it)
    this.holdout = -1;
    this.logfScale = 0;
    this.camStep = 0;
    this.camM = new Float64Array((cams.length + 1) * 8);
    this.camV = new Float64Array((cams.length + 1) * 8);

    // hyperparameters loosely following LichtFeld/3DGS conventions
    // schedule horizon: pos-lr decay and growth scale with the intended
    // training length (default matches main.js auto-stop) instead of a
    // hardcoded 30k that predates the longer default runs
    this.horizon = this.opts.maxIters ?? 60000;
    this.adamData = new Float32Array(28);
    const r = sceneRadius;
    this.basePosLr = 3e-4 * r;
    const lrs = new Float32Array(16);
    lrs[0] = lrs[1] = lrs[2] = this.basePosLr;      // position
    lrs[3] = lrs[4] = lrs[5] = 5e-3;                // log scales
    lrs[6] = lrs[7] = lrs[8] = lrs[9] = 1e-3;       // quaternion
    lrs[10] = lrs[11] = lrs[12] = 1.5e-2;           // color logits
    lrs[13] = 2.5e-2;                               // opacity logit
    this.adamData.set(lrs, 0);
    this.adamData.set([
      0.9, 0.999, 1e-15, 1,                          // beta1, beta2, eps, t
      // min scale default 1e-4*r: the old 1e-3*r floor is a WORLD-space size
      // and projects inversely with depth — close surfaces hit a 2-5px floor
      // ("inverted depth of field", user-diagnosed) while far ones stay
      // sub-pixel. Dropping it: truck +2.9dB train / +1.6dB holdout. Needle
      // risk is handled by anisoReg (ratio bound), not this absolute floor.
      // zero-width needle axes the optimizer happily saturated)
      Math.log(r * (this.opts.minScale ?? 1e-4)), Math.log(r * 0.05), 8.0, this.n * STRIDE,
      // 0.01 (was 0.05): matches standard 3DGS-MCMC; the strong early-era pull
      // kept splats semi-transparent and layered ("milky")
      this.opts.opacityReg ?? 0.01, 0, 0, 0,
    ], 16);
    this.lastRefine = 0;
    this.rand = () => Math.random();

    // gradcheck metadata
    this.hBySlot = [
      r * 2e-3, r * 2e-3, r * 2e-3, 0.02, 0.02, 0.02,
      0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 0, 0,
    ];
    this.slotNames = [
      'pos.x', 'pos.y', 'pos.z', 'logSx', 'logSy', 'logSz',
      'q.w', 'q.x', 'q.y', 'q.z', 'c.r', 'c.g', 'c.b', 'opa', 'pad', 'pad',
    ];
  }

  _camUniform({ R, t, f, cx, cy, w, h, g = 0, b = 0 }, trainMode, offset, camIdx = 0) {
    const u = new Float32Array(36);
    u.set([R[0], R[1], R[2], 0], 0);
    u.set([R[3], R[4], R[5], 0], 4);
    u.set([R[6], R[7], R[8], 0], 8);
    u.set([t[0], t[1], t[2], 0.05], 12);          // near plane
    u.set([f, cx, cy, Math.exp(g)], 16);          // .w = exposure gain
    u.set([w, h, Math.ceil(w / TILE), this.n], 20);
    u.set([0, 0, 0, 0], 24);                      // black background
    u.set([trainMode, camIdx, this.camMeta ? this.camMeta.length : 0, b], 28); // .w = exposure bias
    u[32] = this.shDeg; // active SH degree (stepOnce ramps this during training)
    // target offset as raw u32 bits (f32 is exact only to 2^24; full-res
    // target buffers exceed that) — shader reads it via bitcast
    new Uint32Array(u.buffer)[27] = offset >>> 0;
    return u;
  }

  /** Encode the full raster sequence (project -> scan -> scatter -> sort ->
   *  render) into an open compute pass for the given camera dims. */
  encodeRaster(p, meta, train) {
    const numTiles = Math.ceil(meta.w / TILE) * Math.ceil(meta.h / TILE);
    p.setPipeline(this.pipeProject);
    p.setBindGroup(0, train ? this.bgProjectTrain : this.bgProjectView);
    p.dispatchWorkgroups(Math.ceil(this.n / 256));
    p.setPipeline(this.pipeScan);
    p.setBindGroup(0, train ? this.bgScanTrain : this.bgScanView);
    p.dispatchWorkgroups(1);
    p.setPipeline(this.pipeScatter);
    p.setBindGroup(0, train ? this.bgScatterTrain : this.bgScatterView);
    p.dispatchWorkgroups(Math.ceil(this.n / 256));
    p.setPipeline(this.pipeSort);
    p.setBindGroup(0, this.bgSort);
    p.dispatchWorkgroups(numTiles);
    p.setPipeline(this.pipeRender);
    p.setBindGroup(0, train ? this.bgRenderTrain : this.bgRenderView);
    p.dispatchWorkgroups(Math.ceil(meta.w / TILE), Math.ceil(meta.h / TILE));
  }

  /** Run one training iteration (own submit; queue-ordered).
   *  Set trainer.holdout = <camIdx> to exclude a camera from training
   *  (evaluate it with evalCamPsnr for an honest novel-view metric). */
  stepOnce() {
    const d = this.device;
    let ci = (this.rand() * this.camMeta.length) | 0;
    // skip the holdout and any excluded (e.g. motion-blurred) cameras
    for (let tries = 0; tries < 16; tries++) {
      const bad = ci === this.holdout || (this.excluded && this.excluded.has(ci));
      if (!bad) break;
      ci = (ci + 1) % this.camMeta.length;
    }
    const meta = this.camMeta[ci];
    this.lastCam = ci; // which camera this step trains on (UI pulse)
    if (this.shK) {
      // INRIA-style band ramp: one SH degree per 1000 iters
      this.camUniforms[ci][32] = Math.min(this.shDeg, Math.floor(this.iter / 1000));
    }
    d.queue.writeBuffer(this.uniTrain, 0, this.camUniforms[ci]);
    d.queue.writeBuffer(this.bufTileCnt, 0, this.tileZero);
    this.iter++;
    this.pixelsSeen += meta.w * meta.h;
    this.adamData[19] = this.iter;
    // exponential position-lr decay to 1% at 75% of the horizon, then a
    // floor-lr polish phase. A/B'd vs INRIA-style full-length decay on
    // camping @40k: full-length gains +0.18 train but LOSES 0.15dB holdout
    // (positions moving late = overfit); the polish phase wins.
    const posLr = this.basePosLr * Math.pow(0.01, Math.min(1, this.iter / (0.75 * this.horizon)));
    this.adamData[0] = this.adamData[1] = this.adamData[2] = posLr;
    d.queue.writeBuffer(this.uniAdam, 0, this.adamData);
    if (this.shK) {
      this.shAdamData[3] = this.iter;
      this.shAdamData[5] = this.n * this.shK * 3;
      d.queue.writeBuffer(this.uniSHAdam, 0, this.shAdamData);
    }

    const enc = d.createCommandEncoder();
    const p = enc.beginComputePass();
    this.encodeRaster(p, meta, true);
    p.setPipeline(this.pipeChain);
    p.setBindGroup(0, this.bgChain);
    p.dispatchWorkgroups(Math.ceil(this.n / 256));
    // 2D-safe dispatch: at high splat counts these linear passes exceed the
    // 65535 workgroups-per-dimension limit (SH-Adam broke first: n*24/256 >
    // 65535 above ~620k splats — invalid command buffers, whole frames
    // silently no-oping)
    const dispatch1D = (pass, total) => {
      const groups = Math.ceil(total / 256);
      if (groups <= 65535) pass.dispatchWorkgroups(groups);
      else {
        const x = 65535;
        pass.dispatchWorkgroups(x, Math.ceil(groups / x));
      }
    };
    p.setPipeline(this.pipeAdam);
    p.setBindGroup(0, this.bgAdam);
    dispatch1D(p, this.n * STRIDE);
    if (this.shK) {
      p.setPipeline(this.pipeSHAdam);
      p.setBindGroup(0, this.bgSHAdam);
      dispatch1D(p, this.n * this.shK * 3);
    }
    p.end();
    d.queue.submit([enc.finish()]);

    // camera pose/focal optimization (default ON — validated +0.4..0.7dB
    // holdout on both test scenes): apply accumulated gradients every 25
    // iterations after a splat warmup
    // default OFF since SIFT-grade SfM (2026-08-19): poses arrive at the
    // information limit (train-84 ATE 0.04% before AND after training), so
    // photometric pose gradients only drift the global frame and cost the
    // holdout ~1dB raw. Re-enable via trainerOpts { camOpt: true } for
    // captures where SfM quality is in doubt.
    if ((this.opts.camOpt ?? false) && this.iter > 1500 && this.iter % 25 === 0) {
      this._applyCamGrads();
    }
  }

  /** Read the accumulated per-camera gradients and take one Adam step on
   *  every camera pose (R <- exp(dw)R, t += dt) plus the shared log-focal.
   *  Camera 0 is pinned (gauge anchor); the holdout camera is skipped. */
  async _applyCamGrads() {
    if (this._camApplying) return;
    this._camApplying = true;
    try {
      const d = this.device;
      const rows = this.camMeta.length + 1;
      const rb = d.createBuffer({ size: rows * 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = d.createCommandEncoder();
      enc.copyBufferToBuffer(this.bufCamGrad, 0, rb, 0, rows * 32);
      d.queue.submit([enc.finish()]);
      d.queue.writeBuffer(this.bufCamGrad, 0, new Int32Array(rows * 8));
      await rb.mapAsync(GPUMapMode.READ);
      const g = new Int32Array(rb.getMappedRange()).slice();
      rb.unmap(); rb.destroy();

      this.camStep++;
      const t = this.camStep;
      const decay = Math.pow(0.02, Math.min(1, this.iter / (0.75 * this.horizon)));
      const rotLr = 2e-4 * decay;
      const trnLr = 2e-4 * this.sceneRadius * decay;
      const focLr = 1e-4 * decay;
      const b1 = 0.9, b2 = 0.99, eps = 1e-15;
      const step = (row, slot, lr) => {
        const j = row * 8 + slot;
        const grad = g[j] / 64;
        this.camM[j] = b1 * this.camM[j] + (1 - b1) * grad;
        this.camV[j] = b2 * this.camV[j] + (1 - b2) * grad * grad;
        const mh = this.camM[j] / (1 - Math.pow(b1, t));
        const vh = this.camV[j] / (1 - Math.pow(b2, t));
        return lr * mh / (Math.sqrt(vh) + eps);
      };
      // default OFF: A/B on the camping video showed no holdout gain (its
      // exposure is stable; the freedom only absorbs error). Enable for
      // captures with real auto-exposure drift via opts.expComp.
      const expLr = 5e-3;
      const doExp = this.opts.expComp ?? false;
      for (let r = 1; r < this.camMeta.length; r++) { // cam 0 pinned (gauge + exposure anchor)
        if (r === this.holdout) continue;
        const meta = this.camMeta[r];
        const dw = [-step(r, 0, rotLr), -step(r, 1, rotLr), -step(r, 2, rotLr)];
        meta.R = Array.from(m3mul(rodrigues(dw), meta.R));
        meta.t = [
          meta.t[0] - step(r, 3, trnLr),
          meta.t[1] - step(r, 4, trnLr),
          meta.t[2] - step(r, 5, trnLr),
        ];
        if (doExp) {
          // tight clamps: real auto-exposure drift is a few percent; wider
          // freedom gets abused as per-image error absorption (measured:
          // -0.55dB holdout with +-60% range)
          meta.g = Math.max(-0.1, Math.min(0.1, (meta.g || 0) - step(r, 6, expLr)));
          meta.b = Math.max(-0.05, Math.min(0.05, (meta.b || 0) - step(r, 7, expLr)));
        }
      }
      const nr = this.camMeta.length;
      this.logfScale = Math.max(-0.3, Math.min(0.3, this.logfScale - step(nr, 0, focLr)));
      for (const m of this.camMeta) m.f = m.f0 * Math.exp(this.logfScale);
      this.camUniforms = this.camMeta.map((m, i) => this._camUniform(m, 1, m.offset, i));
    } finally {
      this._camApplying = false;
    }
  }

  /** One training-path pass for camera ci (optionally with a pose override);
   *  returns { psnr, camGrad (Int32Array row ci + focal row) }. Drains all
   *  gradient pollution so training state stays clean. */
  async _evalPass(ci, override) {
    const d = this.device;
    const meta = this.camMeta[ci];
    const uni = override
      ? this._camUniform({ ...meta, ...override }, 1, meta.offset, ci)
      : this.camUniforms[ci];
    const rows = this.camMeta.length + 1;
    d.queue.writeBuffer(this.bufStats, 0, new Uint32Array(4));
    d.queue.writeBuffer(this.uniTrain, 0, uni);
    d.queue.writeBuffer(this.bufTileCnt, 0, this.tileZero);
    const enc = d.createCommandEncoder();
    const p = enc.beginComputePass();
    this.encodeRaster(p, meta, true);
    p.setPipeline(this.pipeChain); p.setBindGroup(0, this.bgChain); // zeroes gradP
    p.dispatchWorkgroups(Math.ceil(this.n / 256));
    p.end();
    enc.copyBufferToBuffer(this.bufStats, 0, this.bufStatsRead, 0, 16);
    const rbC = d.createBuffer({ size: rows * 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(this.bufCamGrad, 0, rbC, 0, rows * 32);
    d.queue.submit([enc.finish()]);
    d.queue.writeBuffer(this.bufCamGrad, 0, new Int32Array(rows * 8)); // drain
    await Promise.all([this.bufStatsRead.mapAsync(GPUMapMode.READ), rbC.mapAsync(GPUMapMode.READ)]);
    const sarr = new Uint32Array(this.bufStatsRead.getMappedRange());
    const v = sarr[0];
    const validPx = Math.max(1, sarr[2]);
    this.bufStatsRead.unmap();
    const camGrad = new Int32Array(rbC.getMappedRange()).slice();
    rbC.unmap(); rbC.destroy();
    d.queue.writeBuffer(this.bufStats, 0, new Uint32Array(4));
    this.pixelsSeen = 0; // running train-psnr window was clobbered
    const mse = v / 16 / (validPx * 3);
    return { psnr: mse > 0 ? -10 * Math.log10(mse) : Infinity, camGrad };
  }

  /** PSNR of one camera (typically the holdout) at its current pose. */
  async evalCamPsnr(ci) {
    return (await this._evalPass(ci)).psnr;
  }

  /** PSNR of the holdout after test-time pose refinement: optimizes ONLY this
   *  camera's 6-DOF pose against its photo (splats frozen), the standard
   *  protocol when training refines camera poses. */
  async evalCamPsnrRefined(ci, steps = 200) {
    const meta = this.camMeta[ci];
    let R = Array.from(meta.R);
    let t = meta.t.slice();
    let g = 0, b = 0; // holdout exposure refined at test time too
    const doExp = this.opts.expComp ?? false;
    const m = new Float64Array(8), v = new Float64Array(8);
    const b1 = 0.9, b2 = 0.99, eps = 1e-15;
    const rotLr = 3e-4, trnLr = 3e-4 * this.sceneRadius, expLr = 5e-3;
    for (let s = 1; s <= steps; s++) {
      const { camGrad } = await this._evalPass(ci, { R, t, g, b });
      const upd = new Float64Array(8);
      for (let k = 0; k < 8; k++) {
        const grad = camGrad[ci * 8 + k] / 64;
        m[k] = b1 * m[k] + (1 - b1) * grad;
        v[k] = b2 * v[k] + (1 - b2) * grad * grad;
        const mh = m[k] / (1 - Math.pow(b1, s));
        const vh = v[k] / (1 - Math.pow(b2, s));
        upd[k] = (k < 3 ? rotLr : (k < 6 ? trnLr : expLr)) * mh / (Math.sqrt(vh) + eps);
      }
      R = Array.from(m3mul(rodrigues([-upd[0], -upd[1], -upd[2]]), R));
      t = [t[0] - upd[3], t[1] - upd[4], t[2] - upd[5]];
      if (doExp) {
        g = Math.max(-0.1, Math.min(0.1, g - upd[6]));
        b = Math.max(-0.05, Math.min(0.05, b - upd[7]));
      }
    }
    return (await this._evalPass(ci, { R, t, g, b })).psnr;
  }

  /** Read and reset accumulated squared-error stats -> mean L2 per pixel-channel. */
  async readLoss() {
    const d = this.device;
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(this.bufStats, 0, this.bufStatsRead, 0, 16);
    d.queue.submit([enc.finish()]);
    d.queue.writeBuffer(this.bufStats, 0, new Uint32Array(4));
    const px = this.pixelsSeen;
    this.pixelsSeen = 0;
    await this.bufStatsRead.mapAsync(GPUMapMode.READ);
    const s = new Uint32Array(this.bufStatsRead.getMappedRange());
    const v = s[0];
    const validPx = s[2]; // valid-pixel count (undistortion borders excluded)
    this.entryOverflowTiles = (this.entryOverflowTiles || 0) + s[3];
    this.bufStatsRead.unmap();
    if (px === 0 || validPx === 0) return null;
    return v / 16 / (validPx * 3);
  }

  /** Render an arbitrary camera into a WebGPU canvas context. */
  renderView(camParams, ctx, trainMode = 0, offset = 0) {
    const d = this.device;
    const u = this._camUniform(camParams, trainMode, offset);
    d.queue.writeBuffer(this.uniView, 0, u);
    d.queue.writeBuffer(this.bufTileCnt, 0, this.tileZero);
    const enc = d.createCommandEncoder();
    const p = enc.beginComputePass();
    this.encodeRaster(p, camParams, false);
    p.end();
    const rp = enc.beginRenderPass({
      colorAttachments: [{
        view: ctx.getCurrentTexture().createView(),
        loadOp: 'clear', storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    rp.setPipeline(this.pipeBlit);
    rp.setBindGroup(0, this.bgBlitView);
    rp.draw(3);
    rp.end();
    d.queue.submit([enc.finish()]);
  }

  renderTrainCam(ci, ctx) {
    const meta = this.camMeta[ci];
    this.renderView(meta, ctx, 0, meta.offset);
  }

  /** MCMC-lite refinement: relocate dead splats onto jittered copies of
   *  well-supported donors (resetting their Adam state) and grow the splat
   *  count by up to 5% per call until the buffer cap is reached.
   *  Returns { moved, grown, n }. */
  async refine(rng = Math.random) {
    const d = this.device;
    const nb = this.cap * STRIDE * 4;
    const readBuf = async (buf, bytes = nb) => {
      const rb = d.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = d.createCommandEncoder();
      enc.copyBufferToBuffer(buf, 0, rb, 0, bytes);
      d.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(rb.getMappedRange()).slice();
      rb.unmap(); rb.destroy();
      return out;
    };
    const params = await readBuf(this.bufParams);
    const m = await readBuf(this.bufM);
    const v = await readBuf(this.bufV);
    const shr = this.shK * 3;
    const sh = this.shK ? await readBuf(this.bufSH, this.cap * shr * 4) : null;
    const shM = this.shK ? await readBuf(this.bufSHM, this.cap * shr * 4) : null;
    const shV = this.shK ? await readBuf(this.bufSHV, this.cap * shr * 4) : null;
    const sig = (x) => 1 / (1 + Math.exp(-x));

    let dead = [];
    const donors = [];
    for (let i = 0; i < this.n; i++) {
      const o = sig(params[i * STRIDE + 13]);
      if (o < 0.02) dead.push(i);
      else if (o > 0.4) donors.push(i);
    }
    if (donors.length < 16) return { moved: 0, grown: 0, n: this.n };
    const moveCap = Math.ceil(this.n * 0.05);
    if (dead.length > moveCap) dead = dead.slice(0, moveCap);

    const gauss = () => {
      const u = Math.max(1e-9, rng()), w = rng();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * w);
    };
    // Blur is carried by LARGE well-supported splats (typically background —
    // SfM points cluster on foreground, and uniform cloning compounds that
    // imbalance: measured 5x capacity = +0.4dB only). Growth therefore
    // prefers SPLITTING the biggest donors, classic-3DGS style: both halves
    // shrink /1.6 so the footprint is conserved and each half can sharpen.
    const meanLogScale = (i) =>
      (params[i * STRIDE + 3] + params[i * STRIDE + 4] + params[i * STRIDE + 5]) / 3;
    const bigDonors = [...donors]
      .sort((a, b) => meanLogScale(b) - meanLogScale(a))
      .slice(0, Math.max(16, donors.length >> 2));
    const spawnAt = (bi, allowSplit) => {
      const doSplit = allowSplit && bigDonors.length > 16 && rng() < 0.7;
      let don;
      if (doSplit) {
        const di = (rng() * bigDonors.length) | 0;
        don = bigDonors[di];
        bigDonors[di] = bigDonors[bigDonors.length - 1];
        bigDonors.pop(); // a splat splits at most once per refine call
      } else {
        don = donors[(rng() * donors.length) | 0];
      }
      const bd = don * STRIDE;
      const s = Math.exp(meanLogScale(don));
      params[bi] = params[bd] + gauss() * s * 0.7;
      params[bi + 1] = params[bd + 1] + gauss() * s * 0.7;
      params[bi + 2] = params[bd + 2] + gauss() * s * 0.7;
      for (let k = 3; k <= 12; k++) params[bi + k] = params[bd + k];
      if (doSplit) {
        const shrink = Math.log(1 / 1.6);
        for (let k = 3; k <= 5; k++) { params[bi + k] += shrink; params[bd + k] += shrink; }
        params[bi + 13] = params[bd + 13]; // split keeps the donor's opacity
        for (let k = 0; k < STRIDE; k++) { m[bd + k] = 0; v[bd + k] = 0; } // donor changed shape: reset its moments
      } else {
        params[bi + 3] += Math.log(0.85);
        params[bi + 4] += Math.log(0.85);
        params[bi + 5] += Math.log(0.85);
        params[bi + 13] = Math.log(0.25 / 0.75);
      }
      params[bi + 14] = 0; params[bi + 15] = 0;
      for (let k = 0; k < STRIDE; k++) { m[bi + k] = 0; v[bi + k] = 0; }
      if (sh) { // clones/splits inherit the donor's view-dependence, fresh moments
        const so = (bi / STRIDE) * shr;
        const sd = don * shr;
        for (let k = 0; k < shr; k++) {
          sh[so + k] = sh[sd + k];
          shM[so + k] = 0; shV[so + k] = 0;
          if (doSplit) { shM[sd + k] = 0; shV[sd + k] = 0; }
        }
      }
    };
    for (const i of dead) spawnAt(i * STRIDE, false); // relocation: to mass, as before
    // growth: new capacity where mass already is (stop late in training so
    // the last iterations refine a stable population)
    // 0.15/step with the 1e-4 minScale floor: capacity converts to real
    // sharpness now (truck 52k -> 235k splats = +0.84dB holdout); the old
    // timid 0.05 predates the floor fix, when extra splats bought nothing
    const grown = this.iter < (this.opts.growUntil ?? 0.75 * this.horizon)
      ? Math.min(Math.ceil(this.n * (this.opts.growRate ?? 0.15)), this.cap - this.n) : 0;
    for (let k = 0; k < grown; k++) spawnAt((this.n + k) * STRIDE, true);
    if (grown > 0) {
      this.n += grown;
      this.adamData[23] = this.n * STRIDE;
      this.camUniforms = this.camMeta.map((mm) => this._camUniform(mm, 1, mm.offset));
    }
    d.queue.writeBuffer(this.bufParams, 0, params);
    d.queue.writeBuffer(this.bufM, 0, m);
    d.queue.writeBuffer(this.bufV, 0, v);
    if (sh) {
      d.queue.writeBuffer(this.bufSH, 0, sh);
      d.queue.writeBuffer(this.bufSHM, 0, shM);
      d.queue.writeBuffer(this.bufSHV, 0, shV);
    }

    // tile-pressure telemetry (counts from the most recent render)
    const rbT = d.createBuffer({ size: this.maxTiles * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encT = d.createCommandEncoder();
    encT.copyBufferToBuffer(this.bufTileCnt, 0, rbT, 0, this.maxTiles * 4);
    d.queue.submit([encT.finish()]);
    await rbT.mapAsync(GPUMapMode.READ);
    const cnts = new Uint32Array(rbT.getMappedRange());
    let maxTile = 0;
    for (let i = 0; i < this.maxTiles; i++) {
      if (cnts[i] > maxTile) maxTile = cnts[i];
    }
    rbT.unmap(); rbT.destroy();
    // entry-budget drops (whole tiles skipped) are counted by the scan pass
    // into stats[3] and surfaced via readLoss -> this.entryOverflowTiles
    return { moved: dead.length, grown, n: this.n, maxTile, overflow: this.entryOverflowTiles || 0 };
  }

  /** Read back current Gaussian parameters (+ SH coeffs when enabled). */
  async readGaussians() {
    const d = this.device;
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(this.bufParams, 0, this.bufParamsRead, 0, this.n * STRIDE * 4);
    d.queue.submit([enc.finish()]);
    await this.bufParamsRead.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(this.bufParamsRead.getMappedRange()).slice();
    this.bufParamsRead.unmap();
    let sh = null;
    if (this.shK) {
      const bytes = this.n * this.shK * 3 * 4;
      const rb = d.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const e2 = d.createCommandEncoder();
      e2.copyBufferToBuffer(this.bufSH, 0, rb, 0, bytes);
      d.queue.submit([e2.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      sh = new Float32Array(rb.getMappedRange()).slice();
      rb.unmap(); rb.destroy();
    }
    return { data, n: this.n, sh, shK: this.shK };
  }
}
