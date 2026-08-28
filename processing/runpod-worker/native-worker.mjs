/**
 * TelosView Native WebGPU Worker (the pivot from Chromium).
 *
 * Runs splat.js's Gaussian splatting pipeline directly in Node using native
 * Dawn (the `webgpu` package) instead of launching headless Chrome. Native Dawn
 * reaches the NVIDIA GPU through Vulkan with no Chromium browser checks.
 *
 * Endpoints (same shape as the old Chromium worker, so the orchestrator is unchanged):
 *   GET  /health
 *   GET  /gpustat
 *   POST /upload-frames/:projectId        - multipart 'frames' JPEG upload
 *   POST /process                          - { projectId, maxIters }
 *   GET  /process/:projectId/progress      - SSE progress stream
 *   GET  /process/:projectId/result        - PLY download
 */
import { create, globals } from 'webgpu';
import { createCanvas, loadImage } from '@napi-rs/canvas';
import { createSession } from './src/index.js';
import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = 8080;

// ── Polyfills: make splat.js (browser code) run in Node ─────────────────────
// WebGPU types (GPUBufferUsage, GPUMapMode, etc.) onto globalThis
Object.assign(globalThis, globals);

// navigator.gpu via native Dawn (Vulkan backend). No Worker / no rAF on
// globalThis — splat.js falls back to synchronous SIFT and a setTimeout loop.
// Node 22 exposes navigator as a read-only getter, so override via defineProperty.
Object.defineProperty(globalThis, 'navigator', {
  value: { gpu: create(['backend=vulkan']), hardwareConcurrency: 4 },
  configurable: true,
  writable: true,
});

// Minimal document for canvas creation (frames.js mkCanvas uses
// document.createElement('canvas') when OffscreenCanvas is absent).
globalThis.document = {
  createElement: (tag) => {
    if (tag === 'canvas') return createCanvas(1, 1);
    throw new Error(`unsupported element: ${tag}`);
  },
};
globalThis.OffscreenCanvas = undefined;

// createImageBitmap -> @napi-rs/canvas loadImage (returns a drawable Image)
globalThis.createImageBitmap = async (source) => {
  let buf;
  if (Buffer.isBuffer(source)) buf = source;
  else if (source instanceof Uint8Array) buf = Buffer.from(source);
  else if (source instanceof Blob) buf = Buffer.from(await source.arrayBuffer());
  else if (ArrayBuffer.isView(source)) buf = Buffer.from(source.buffer, source.byteOffset, source.byteLength);
  else buf = Buffer.from(await source.arrayBuffer());
  const img = await loadImage(buf);
  if (!img.close) img.close = () => {}; // frames.js calls bmp.close()
  return img;
};

// ── Directories ─────────────────────────────────────────────────────────────
const FRAMES_DIR = path.join(__dirname, 'frames');
const OUTPUT_DIR = path.join(__dirname, 'output');
fs.mkdirSync(FRAMES_DIR, { recursive: true });
fs.mkdirSync(OUTPUT_DIR, { recursive: true });

const upload = multer({
  storage: multer.diskStorage({
    destination: (req, _f, cb) => {
      const dir = path.join(FRAMES_DIR, req.params.projectId);
      fs.mkdirSync(dir, { recursive: true });
      cb(null, dir);
    },
    filename: (_req, file, cb) => cb(null, file.originalname),
  }),
  limits: { fileSize: 20 * 1024 * 1024 },
});

const activeJobs = new Map();

// ── Express app ─────────────────────────────────────────────────────────────
const app = express();
app.use(express.json());

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', gpu: process.env.NVIDIA_VISIBLE_DEVICES || 'unknown', native: true });
});

app.get('/gpustat', (_req, res) => {
  import('child_process').then(({ execSync }) => {
    try {
      const out = execSync('nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader', { timeout: 3000 }).toString().trim();
      res.json({ gpu: out });
    } catch (e) { res.json({ error: e.message }); }
  });
});

app.post('/upload-frames/:projectId', upload.array('frames', 500), (req, res) => {
  const { projectId } = req.params;
  const files = req.files;
  console.log(`[native] Received ${files?.length || 0} frames for ${projectId}`);
  res.json({ projectId, frameCount: files?.length || 0 });
});

app.post('/process', async (req, res) => {
  const { projectId, maxIters = 60000 } = req.body;
  if (!projectId) return res.status(400).json({ error: 'projectId required' });
  if (activeJobs.has(projectId)) return res.status(409).json({ error: 'already processing' });

  const framesDir = path.join(FRAMES_DIR, projectId);
  if (!fs.existsSync(framesDir)) return res.status(404).json({ error: 'no frames' });

  const job = {
    projectId,
    status: 'processing',
    stage: 'initializing',
    progress: 0,
    message: 'Starting...',
    error: null,
    plyPath: null,
    sseClients: new Set(),
  };
  activeJobs.set(projectId, job);
  res.json({ projectId, status: 'processing', message: 'Processing started' });

  // Defer the sync file reads so the /process response flushes first (the
  // synchronous fs.readFileSync for 60+ frames blocks the event loop, which
  // delays the response and can make the RunPod proxy time out).
  setImmediate(() => runProcessing(job, framesDir, maxIters).catch((err) => {
    console.error(`[native] processing failed:`, err);
    job.status = 'error';
    job.error = err.message;
    job.message = `Error: ${err.message}`;
    broadcast(job);
  }));
});

app.get('/process/:projectId/progress', (req, res) => {
  const job = activeJobs.get(req.params.projectId);
  if (!job) return res.status(404).json({ error: 'no active job' });
  res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
  res.write(`data: ${JSON.stringify(state(job))}\n\n`);
  job.sseClients.add(res);
  req.on('close', () => job.sseClients.delete(res));
});

app.get('/process/:projectId/result', (req, res) => {
  const { projectId } = req.params;
  const job = activeJobs.get(projectId);
  const plyPath = (job && job.plyPath) || path.join(OUTPUT_DIR, `${projectId}.ply`);
  if (fs.existsSync(plyPath)) return res.download(plyPath, `${projectId}.ply`);
  if (job && job.status === 'error') return res.status(500).json({ error: job.error });
  return res.status(409).json({ error: 'not complete' });
});

// ── PLY Coordinate Transform ────────────────────────────────────────────────
// splat.js trains in OpenCV convention (x right, y down, z forward).
// Spark.js/Three.js renders in y-up, z-back convention.
// Transform: negate Y, Z positions and qy, qz quaternion components.
function transformPlyToThreeJS(buf) {
  const headerEnd = buf.indexOf(Buffer.from('end_header\n')) + Buffer.from('end_header\n').length;
  const header = buf.slice(0, headerEnd).toString();

  const propLines = header.split('\n').filter(l => l.startsWith('property'));
  const propNames = propLines.map(l => l.trim().split(/\s+/).pop());
  const stride = propLines.length * 4;

  // Find indices of properties to negate
  const yIdx = propNames.indexOf('y');
  const zIdx = propNames.indexOf('z');
  const qyIdx = propNames.indexOf('rot_1'); // quaternion y component
  const qzIdx = propNames.indexOf('rot_2'); // quaternion z component

  if (yIdx < 0 || zIdx < 0 || qyIdx < 0 || qzIdx < 0) {
    console.warn('[transform] Could not find y/z/rot_1/rot_2 properties, skipping transform');
    return buf;
  }

  // Copy buffer and transform in-place
  const out = Buffer.from(buf);
  const vertexMatch = header.match(/element vertex (\d+)/);
  const vertexCount = vertexMatch ? parseInt(vertexMatch[1]) : 0;

  for (let i = 0; i < vertexCount; i++) {
    const o = headerEnd + i * stride;
    // Negate Y and Z positions
    out.writeFloatLE(-buf.readFloatLE(o + yIdx * 4), o + yIdx * 4);
    out.writeFloatLE(-buf.readFloatLE(o + zIdx * 4), o + zIdx * 4);
    // Negate qy and qz quaternion components
    out.writeFloatLE(-buf.readFloatLE(o + qyIdx * 4), o + qyIdx * 4);
    out.writeFloatLE(-buf.readFloatLE(o + qzIdx * 4), o + qzIdx * 4);
  }

  console.log(`[transform] Converted ${vertexCount} vertices from OpenCV to Three.js convention`);
  return out;
}

// ── Processing ──────────────────────────────────────────────────────────────
async function runProcessing(job, framesDir, maxIters) {
  const { projectId } = job;

  // Heartbeat: keep the SSE stream alive through long SfM phases (bundle
  // adjustment emits no stage/metrics events, so without this the connection
  // can idle-timeout and the orchestrator sees the pod as dead).
  const heartbeat = setInterval(() => broadcast(job), 15000);

  try {
  // Load JPEG frames as Buffers
  update(job, 'decode', 5, 'Reading frames...');
  const names = fs.readdirSync(framesDir).filter((n) => /^frame_\d{5}\.jpg$/i.test(n)).sort();
  if (names.length < 2) throw new Error('need at least 2 frames');
  const fileBlobs = names.map((name) => ({ source: fs.readFileSync(path.join(framesDir, name)), name }));
  update(job, 'decode', 10, `Loaded ${fileBlobs.length} frames`);

  const session = createSession({
    maxIters,
    initTarget: 40000,       // reduced from 60000 to fit GPU binding limit
    maxViewW: 1920,          // cap training resolution
    maxViewH: 1080,          // cap training resolution
    trainer: {
      anisoReg: 0.01,        // stronger anti-needle regularization (default 0.005)
      opacityReg: 0.015,     // slightly stronger opacity regularization (default 0.01)
      minScale: 5e-4,        // higher floor to prevent sub-pixel needles (default 1e-4)
      camOpt: true,          // camera pose optimization for phone video
    }
  });
  console.log(`[native] Training config: maxIters=${maxIters}, initTarget=40000, anisoReg=0.01, opacityReg=0.015, minScale=5e-4, camOpt=true`);
  session.on('stage', (e) => {
    job.stage = e.stage;
    job.message = `${e.stage}: ${e.detail || ''}`;
    const pct = 15 + (e.done / Math.max(1, e.total)) * 20;
    job.progress = Math.min(35, pct);
    broadcast(job);
  });
  session.on('log', (m) => console.log(`[splat] ${m}`));
  session.on('metrics', (e) => {
    job.stage = 'train';
    job.message = `Iter ${e.iter} · ${e.splats} splats · ${Math.round(e.itersPerSec)} it/s`;
    job.progress = 35 + Math.min(60, (e.iter / maxIters) * 60);
    broadcast(job);
  });

  // Decode + SfM + seed
  update(job, 'decode', 12, 'Decoding frames...');
  await session.load(fileBlobs);

  update(job, 'sfm', 20, 'Structure from Motion...');
  await session.solve();

  update(job, 'seed', 30, 'Seeding Gaussians...');
  await session.seed();

  // Train
  update(job, 'train', 35, 'Training...');
  session.start();
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('Training timed out (30 min)')), 30 * 60 * 1000);
    session.on('event', (e) => {
      if (e.kind === 'train-complete') { clearTimeout(timeout); resolve(); }
    });
  });

  // Export PLY
  update(job, 'export', 97, 'Exporting .ply...');
  const plyBlob = await session.exportPlyBlob();
  const plyBuf = Buffer.from(await plyBlob.arrayBuffer());

  // Transform PLY from OpenCV convention (splat.js training) to Three.js convention
  // (Spark.js rendering). Negate Y, Z positions and qy, qz quaternion components.
  // This eliminates the need for rotation.x = Math.PI hack in the viewer.
  const transformedPly = transformPlyToThreeJS(plyBuf);
  const plyPath = path.join(OUTPUT_DIR, `${projectId}.ply`);
  fs.writeFileSync(plyPath, transformedPly);

  job.plyPath = plyPath;
  job.status = 'done';
  job.progress = 100;
  job.stage = 'done';
  job.message = `Complete! PLY: ${(plyBuf.length / 1024 / 1024).toFixed(1)} MB`;
  broadcast(job);
  console.log(`[native] Complete: ${plyPath} (${(plyBuf.length / 1024 / 1024).toFixed(1)} MB)`);
  activeJobs.delete(projectId);
  } finally {
    clearInterval(heartbeat);
  }
}

function update(job, stage, progress, message) { job.stage = stage; job.progress = progress; job.message = message; }
function state(job) { return { status: job.status, stage: job.stage, progress: job.progress, message: job.message, error: job.error }; }
function broadcast(job) {
  const data = JSON.stringify(state(job));
  for (const c of job.sseClients) { try { c.write(`data: ${data}\n\n`); } catch { job.sseClients.delete(c); } }
}

app.listen(PORT, '0.0.0.0', () => console.log(`[native] TelosView native worker on ${PORT}`));
