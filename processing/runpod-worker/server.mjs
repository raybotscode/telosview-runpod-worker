/**
 * RunPod GPU Worker Server
 *
 * Express server that runs inside the RunPod GPU pod.
 * Accepts frame uploads, launches Chromium with real GPU (WebGPU/Vulkan),
 * runs the splat.js Gaussian splatting pipeline, and streams progress.
 *
 * Endpoints:
 *   GET  /health                          - Health check
 *   POST /upload-frames/:projectId        - Upload JPEG frames (multipart)
 *   GET  /api/projects/:projectId/frames  - List frames (for worker.html)
 *   GET  /frames/:projectId/:filename     - Serve individual frame images
 *   GET  /src/*                           - Serve splat.js source files
 *   GET  /worker.html                     - Serve the worker page
 *   POST /process                         - Start processing { projectId }
 *   GET  /process/:projectId/progress     - SSE progress stream
 *   GET  /process/:projectId/result       - Download PLY result
 */

import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { chromium } from 'playwright';

// CRITICAL: RunPod forces NVIDIA_VISIBLE_DEVICES=void which hides the GPU from
// CUDA/Vulkan. nvidia-smi still works (direct driver access) so the GPU IS there.
// Override before any GPU code runs.
if (process.env.NVIDIA_VISIBLE_DEVICES === 'void' || !process.env.NVIDIA_VISIBLE_DEVICES) {
  process.env.NVIDIA_VISIBLE_DEVICES = 'all';
  console.log('[worker] Overrode NVIDIA_VISIBLE_DEVICES=void → all');
}
if (!process.env.NVIDIA_DRIVER_CAPABILITIES) {
  process.env.NVIDIA_DRIVER_CAPABILITIES = 'compute,utility,graphics';
}
// Ensure a DISPLAY is set (Xvfb is started in the Dockerfile CMD). Chromium's
// WebGPU/Vulkan backend needs it to init surface/swapchain even for compute.
if (!process.env.DISPLAY) {
  process.env.DISPLAY = ':99';
  console.log('[worker] Set DISPLAY=:99 (was unset)');
}

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = 8080;

// Directories
const FRAMES_DIR = path.join(__dirname, 'frames');
const OUTPUT_DIR = path.join(__dirname, 'output');
const SRC_DIR = path.join(__dirname, 'src');

fs.mkdirSync(FRAMES_DIR, { recursive: true });
fs.mkdirSync(OUTPUT_DIR, { recursive: true });

// Multer for frame uploads
const upload = multer({
  storage: multer.diskStorage({
    destination: (req, _file, cb) => {
      const dir = path.join(FRAMES_DIR, req.params.projectId);
      fs.mkdirSync(dir, { recursive: true });
      cb(null, dir);
    },
    filename: (_req, file, cb) => cb(null, file.originalname),
  }),
  limits: { fileSize: 20 * 1024 * 1024 }, // 20MB per frame
});

// Track active processing jobs
const activeJobs = new Map();

// ── Static file serving ──

app.get('/worker.html', (_req, res) => {
  res.sendFile(path.join(__dirname, 'worker.html'));
});

app.use('/src', express.static(SRC_DIR, {
  setHeaders: (res, filePath) => {
    if (filePath.endsWith('.js')) {
      res.setHeader('Content-Type', 'application/javascript');
    }
  }
}));

app.use('/frames', express.static(FRAMES_DIR));

// ── Health check ──

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', gpu: process.env.NVIDIA_VISIBLE_DEVICES || 'unknown' });
});

// ── GPU/Vulkan diagnostics ──
app.get('/diag', async (_req, res) => {
  const { execSync } = await import('child_process');
  const diag = {};

  // Environment vars
  diag.env = {
    NVIDIA_VISIBLE_DEVICES: process.env.NVIDIA_VISIBLE_DEVICES,
    NVIDIA_DRIVER_CAPABILITIES: process.env.NVIDIA_DRIVER_CAPABILITIES,
    DISPLAY: process.env.DISPLAY,
  };

  // Check Vulkan ICD files
  const icdPaths = [
    '/usr/share/vulkan/icd.d/nvidia_icd.json',
    '/etc/vulkan/icd.d/nvidia_icd.json',
    '/usr/share/vulkan/icd.d/nvidia_layers.json',
    '/etc/vulkan/icd.d/nvidia_layers.json',
  ];
  diag.vulkanIcdFiles = {};
  for (const p of icdPaths) {
    try {
      diag.vulkanIcdFiles[p] = fs.existsSync(p);
      if (diag.vulkanIcdFiles[p]) {
        diag.vulkanIcdFiles[p + '_content'] = fs.readFileSync(p, 'utf8').substring(0, 500);
      }
    } catch { diag.vulkanIcdFiles[p] = false; }
  }

  // nvidia-smi
  try {
    diag.nvidiaSmi = execSync('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1', { timeout: 5000 }).toString().trim();
  } catch (e) { diag.nvidiaSmi = `error: ${e.message}`; }

  // vulkaninfo (no VK_ICD_FILENAMES override — let default discovery find RunPod's ICD)
  try {
    const vkOut = execSync('vulkaninfo --summary 2>&1', { timeout: 15000 }).toString();
    diag.vulkaninfo = vkOut.substring(0, 3000);
  } catch (e) {
    // execSync with 2>&1 puts the merged output in e.stdout, not e.stderr
    diag.vulkaninfo = `error: ${e.message}\n${e.stdout || ''}\n${e.stderr || ''}`.substring(0, 2000);
  }

  // Check if libGLX_nvidia exists
  try {
    diag.nvidiaLib = execSync('ls -la /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so* 2>&1', { timeout: 3000 }).toString().trim();
  } catch (e) { diag.nvidiaLib = `error: ${e.message}`; }

  // Check Vulkan loader
  try {
    diag.vulkanLoader = execSync('ls -la /usr/lib/x86_64-linux-gnu/libvulkan* 2>&1', { timeout: 3000 }).toString().trim();
  } catch (e) { diag.vulkanLoader = `error: ${e.message}`; }

  // Comprehensive WebGPU test via full Chromium (channel:'chromium' = new headless w/ GPU)
  try {
    const browser = await chromium.launch({
      headless: true,
      channel: 'chromium',
      args: [
        '--no-sandbox',
        '--headless=new',
        '--enable-unsafe-webgpu',
        '--enable-features=Vulkan,UseSkiaRenderer',
        '--enable-webgpu-developer-features',
        '--use-angle=vulkan',
        '--ignore-gpu-blocklist',
        '--disable-gpu-sandbox',
        '--disable-dawn-features=adapter_blocklist',
      ],
      ignoreDefaultArgs: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader-webgl'],
    });
    diag.browserVersion = browser.version();
    diag.browserExecutable = chromium.executablePath();

    const ctx = await browser.newContext();
    const page = await ctx.newPage();

    // 1) Check navigator.gpu — MUST be a secure context (http://localhost, NOT data:)
    await page.goto(`http://localhost:${PORT}/health`);
    const gpuInfo = await page.evaluate(async () => {
      try {
        if (!navigator.gpu) return { error: 'navigator.gpu is undefined' };
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) return { error: 'requestAdapter() returned null (no adapter)' };
        const info = adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {};
        return {
          ok: true,
          isFallbackAdapter: adapter.isFallbackAdapter,
          description: info.description || 'unknown',
          vendor: info.vendor || 'unknown',
          architecture: info.architecture || 'unknown',
          features: [...adapter.features].slice(0, 20),
        };
      } catch (e) { return { error: e.message }; }
    });
    diag.webgpu = gpuInfo;

    // 2) Dump chrome://gpu — extract Problems Detected + Driver Info + command line
    try {
      await page.goto('chrome://gpu');
      await page.waitForTimeout(1500);
      const gpuText = await page.evaluate(() => {
        // innerText excludes <style> CSS and includes shadow-DOM rendered text
        const full = document.body.innerText || '';
        return full;
      });
      diag.chromeGpu = (gpuText || '(empty)').substring(0, 6000);
      // Also extract the key sections separately for easier parsing
      const pick = (label) => {
        const i = gpuText.indexOf(label);
        return i >= 0 ? gpuText.substring(i, i + 1200) : null;
      };
      diag.problems = pick('Problems Detected');
      diag.driverInfo = pick('Driver Information');
      diag.vulkanInfo = pick('Vulkan information') || pick('Vulkan Information');
      diag.logMessages = pick('Log Messages');
    } catch (e) {
      diag.chromeGpu = `error reading chrome://gpu: ${e.message}`;
    }

    await ctx.close();
    await browser.close();
  } catch (e) { diag.webgpu = { error: e.message }; }

  res.json(diag);
});

// ── Frame upload ──

app.post('/upload-frames/:projectId', upload.array('frames', 500), (req, res) => {
  const { projectId } = req.params;
  const files = req.files;
  console.log(`[worker] Received ${files?.length || 0} frames for project ${projectId}`);
  res.json({
    projectId,
    frameCount: files?.length || 0,
    message: `Uploaded ${files?.length || 0} frames`,
  });
});

// ── Frame list API (mimics the main API for worker.html compatibility) ──

app.get('/api/projects/:projectId/frames', (req, res) => {
  const { projectId } = req.params;
  const framesDir = path.join(FRAMES_DIR, projectId);

  if (!fs.existsSync(framesDir)) {
    res.status(404).json({ error: 'No frames found' });
    return;
  }

  const names = fs.readdirSync(framesDir)
    .filter(n => /^frame_\d{5}\.jpg$/i.test(n))
    .sort();

  res.json({
    count: names.length,
    frames: names.map(name => ({
      name,
      url: `/frames/${encodeURIComponent(projectId)}/${encodeURIComponent(name)}`,
    })),
  });
});

// ── Processing ──

app.use(express.json());

app.post('/process', async (req, res) => {
  const { projectId, maxIters = 20000 } = req.body;

  if (!projectId) {
    res.status(400).json({ error: 'projectId is required' });
    return;
  }

  if (activeJobs.has(projectId)) {
    res.status(409).json({ error: 'Project is already being processed' });
    return;
  }

  // Check frames exist
  const framesDir = path.join(FRAMES_DIR, projectId);
  if (!fs.existsSync(framesDir)) {
    res.status(404).json({ error: 'No frames found for this project' });
    return;
  }

  // Initialize job state
  const job = {
    projectId,
    status: 'processing',
    stage: 'initializing',
    progress: 0,
    message: 'Starting...',
    metrics: null,
    error: null,
    plyPath: null,
    log: [],
    sseClients: new Set(),
  };
  activeJobs.set(projectId, job);

  // Respond immediately — processing runs async
  res.json({ projectId, status: 'processing', message: 'Processing started' });

  // Run processing in background
  runProcessing(job, maxIters).catch(err => {
    console.error(`[worker] Processing failed for ${projectId}:`, err);
    job.status = 'error';
    job.error = err.message;
    job.message = `Error: ${err.message}`;
    broadcastSSE(job);
  });
});

// SSE progress endpoint
app.get('/process/:projectId/progress', (req, res) => {
  const { projectId } = req.params;
  const job = activeJobs.get(projectId);

  if (!job) {
    res.status(404).json({ error: 'No active job for this project' });
    return;
  }

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
  });

  // Send current state immediately
  res.write(`data: ${JSON.stringify(getJobState(job))}\n\n`);

  // Register SSE client
  job.sseClients.add(res);
  req.on('close', () => job.sseClients.delete(res));
});

// Result download endpoint
app.get('/process/:projectId/result', (req, res) => {
  const { projectId } = req.params;
  const job = activeJobs.get(projectId);

  if (!job) {
    // Check if result file exists on disk
    const plyPath = path.join(OUTPUT_DIR, `${projectId}.ply`);
    if (fs.existsSync(plyPath)) {
      res.download(plyPath, `${projectId}.ply`);
      return;
    }
    res.status(404).json({ error: 'No job found for this project' });
    return;
  }

  if (job.status === 'error') {
    res.status(500).json({ error: job.error });
    return;
  }

  if (job.status !== 'done') {
    res.status(409).json({ error: 'Processing not complete', status: job.status });
    return;
  }

  const plyPath = job.plyPath || path.join(OUTPUT_DIR, `${projectId}.ply`);
  if (!fs.existsSync(plyPath)) {
    res.status(500).json({ error: 'PLY file not found' });
    return;
  }

  res.download(plyPath, `${projectId}.ply`);
});

// ── Processing logic ──

async function runProcessing(job, maxIters) {
  const { projectId } = job;
  const baseUrl = `http://localhost:${PORT}`;

  console.log(`[worker] Starting processing for project ${projectId} (maxIters: ${maxIters})`);

  // Launch Chromium with real GPU (no SwiftShader)
  updateJob(job, 'launching', 2, 'Launching browser with GPU...');
  broadcastSSE(job);

  const args = [
    '--no-sandbox',
    '--headless=new',
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan,UseSkiaRenderer',
    '--enable-webgpu-developer-features',
    '--use-angle=vulkan',
    '--ignore-gpu-blocklist',
    '--disable-gpu-sandbox',
    '--disable-dawn-features=adapter_blocklist',
    '--disable-dev-shm-usage',
  ];

  // On a real GPU server, we do NOT add --use-vulkan=swiftshader
  // The browser will use the real Vulkan/GPU backend
  console.log(`[worker] GPU env: NVIDIA_VISIBLE_DEVICES=${process.env.NVIDIA_VISIBLE_DEVICES || 'not set'}`);
  console.log(`[worker] Chromium args: ${args.join(' ')}`);

  let browser;
  try {
    // channel:'chromium' forces the full Chromium build (new headless mode).
    // The default headless shell (chrome-headless-shell) does NOT support WebGPU.
    // ignoreDefaultArgs strips Playwright's injected SwiftShader/software fallbacks
    // that would otherwise override our Vulkan flags.
    browser = await chromium.launch({
      headless: true,
      channel: 'chromium',
      args,
      ignoreDefaultArgs: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader-webgl'],
    });
  } catch (err) {
    console.log(`[worker] Full Chromium failed, trying with SwiftShader only: ${err.message}`);
    browser = await chromium.launch({
      headless: true,
      channel: 'chromium',
      args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--disable-dev-shm-usage'],
    });
  }

  let context;
  try {
    context = await browser.newContext();
    const page = await context.newPage();

    // Navigate to worker page (served by this same Express server)
    const workerUrl = `${baseUrl}/worker.html`;
    console.log(`[worker] Navigating to ${workerUrl}`);
    await page.goto(workerUrl, { waitUntil: 'domcontentloaded' });

    // Check what WebGPU adapter we got
    const gpuCheck = await page.evaluate(async () => {
      try {
        if (!navigator.gpu) return { error: 'navigator.gpu not available' };
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) return { error: 'No WebGPU adapter' };
        const info = adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {};
        return {
          adapter: info.description || info.device || 'unknown',
          vendor: info.vendor || 'unknown',
          features: [...adapter.features],
          limits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize },
        };
      } catch (e) { return { error: e.message }; }
    });
    console.log(`[worker] WebGPU adapter: ${JSON.stringify(gpuCheck)}`);

    // Wait for worker to be ready
    updateJob(job, 'loading', 5, 'Worker page loaded, waiting for init...');
    broadcastSSE(job);

    await page.waitForFunction(
      () => window.__worker?.status === 'ready',
      null,
      { timeout: 2 * 60 * 1000 }
    );

    // Inject maxIters config
    await page.evaluate((iters) => {
      window.__maxIters = iters;
    }, maxIters);

    // Start processing
    updateJob(job, 'processing', 8, 'Starting Gaussian splatting pipeline...');
    broadcastSSE(job);

    await page.evaluate((pid) => {
      window.__process(pid);
    }, projectId);

    // Poll for progress
    const progressInterval = setInterval(async () => {
      try {
        const state = await page.evaluate(() => window.__worker);
        if (state) {
          job.stage = state.stage;
          job.progress = Math.round(state.progress);
          job.message = state.message;
          job.metrics = state.metrics;
          job.status = state.status === 'done' ? 'processing' : state.status;
          broadcastSSE(job);
        }
      } catch {
        // Page may have navigated or closed
      }
    }, 2000);

    // Wait for completion (30 min timeout)
    console.log(`[worker] Waiting for processing to complete...`);
    await page.waitForFunction(
      () => {
        const w = window.__worker;
        return w?.status === 'done' || w?.status === 'error';
      },
      null,
      { timeout: 30 * 60 * 1000 }
    );

    clearInterval(progressInterval);

    // Get final state
    const finalState = await page.evaluate(() => window.__worker);

    if (finalState.status === 'error') {
      job.status = 'error';
      job.error = finalState.error || 'Unknown processing error';
      job.message = `Error: ${job.error}`;
      broadcastSSE(job);
      return;
    }

    // Save PLY to disk
    if (!finalState.plyBase64) {
      job.status = 'error';
      job.error = 'No PLY data produced';
      job.message = 'Error: No PLY data produced';
      broadcastSSE(job);
      return;
    }

    const plyPath = path.join(OUTPUT_DIR, `${projectId}.ply`);
    const plyBuf = Buffer.from(finalState.plyBase64, 'base64');
    fs.writeFileSync(plyPath, plyBuf);

    job.plyPath = plyPath;
    job.status = 'done';
    job.progress = 100;
    job.stage = 'done';
    job.message = `Complete! PLY: ${(plyBuf.length / 1024 / 1024).toFixed(1)} MB`;
    job.metrics = finalState.metrics;
    broadcastSSE(job);

    console.log(`[worker] Processing complete: ${plyPath} (${(plyBuf.length / 1024 / 1024).toFixed(1)} MB)`);

  } finally {
    await context?.close().catch(() => {});
    await browser?.close().catch(() => {});
  }
}

function updateJob(job, stage, progress, message) {
  job.stage = stage;
  job.progress = progress;
  job.message = message;
}

function getJobState(job) {
  return {
    status: job.status,
    stage: job.stage,
    progress: job.progress,
    message: job.message,
    metrics: job.metrics,
    error: job.error,
  };
}

function broadcastSSE(job) {
  const data = JSON.stringify(getJobState(job));
  for (const client of job.sseClients) {
    try {
      client.write(`data: ${data}\n\n`);
    } catch {
      job.sseClients.delete(client);
    }
  }
}

// ── Start server ──

app.listen(PORT, '0.0.0.0', () => {
  console.log(`[worker] TelosView RunPod GPU Worker listening on port ${PORT}`);
  console.log(`[worker] GPU: ${process.env.NVIDIA_VISIBLE_DEVICES || 'detecting...'}`);
  console.log(`[worker] Ready to process projects`);
});
