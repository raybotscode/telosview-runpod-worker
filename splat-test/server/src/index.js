import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { chromium } from 'playwright';
import { randomUUID } from 'crypto';
import { execFile } from 'child_process';
import { readFile, writeFile, mkdir, readdir, unlink } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..', '..');
const JOBS_DIR = join(ROOT, 'jobs');
const UPLOADS_DIR = join(ROOT, 'uploads');
const PLY_DIR = join(ROOT, 'output');
const FRAMES_DIR = join(ROOT, 'server', 'frames');

// Ensure directories exist
for (const d of [JOBS_DIR, UPLOADS_DIR, PLY_DIR, FRAMES_DIR]) {
  await mkdir(d, { recursive: true });
}

const upload = multer({
  dest: UPLOADS_DIR,
  limits: { fileSize: 500 * 1024 * 1024 }, // 500MB
  fileFilter: (_req, file, cb) => {
    if (/^video\//.test(file.mimetype) || /\.(mp4|mov|webm|mkv|m4v)$/i.test(file.originalname)) {
      cb(null, true);
    } else {
      cb(new Error('Only video files are accepted'));
    }
  }
});

// ── Job State ──
const jobs = new Map();

// Load persisted jobs on startup
try {
  const files = await readdir(JOBS_DIR);
  for (const f of files) {
    if (!f.endsWith('.json')) continue;
    const data = JSON.parse(await readFile(join(JOBS_DIR, f), 'utf-8'));
    jobs.set(data.id, data);
  }
  console.log(`Loaded ${jobs.size} persisted jobs`);
} catch {}

function readJob(id) {
  return jobs.get(id) || null;
}

async function persistJob(job) {
  jobs.set(job.id, job);
  await writeFile(join(JOBS_DIR, `${job.id}.json`), JSON.stringify(job, null, 2));
}

// ── Playwright Worker Pool ──
let browser = null;
let busy = false;
const queue = [];

async function getBrowser() {
  if (browser && browser.isConnected()) return browser;
  console.log('Launching Chromium with WebGPU support...');
  browser = await chromium.launch({
    headless: true,
    args: [
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan',
      '--use-vulkan=swiftshader',
      '--enable-webgpu-developer-features',
      '--disable-gpu-sandbox',
      '--no-sandbox',
    ],
  });
  browser.on('disconnected', () => { browser = null; });
  return browser;
}

async function extractFrames(job) {
  const frameDir = join(FRAMES_DIR, job.id);
  await mkdir(frameDir, { recursive: true });

  job.status = 'processing';
  job.stage = 'extracting';
  job.progress = 1;
  job.message = 'Extracting JPEG frames with ffmpeg...';
  job.frameDir = frameDir;
  job.frameCount = 0;
  await persistJob(job);

  await new Promise((resolve, reject) => {
    const child = execFile('ffmpeg', [
      '-nostdin',
      '-hide_banner',
      '-loglevel', 'error',
      '-progress', 'pipe:2',
      '-i', job.videoPath,
      '-vf', 'fps=2',
      '-q:v', '2',
      '-start_number', '1',
      join(frameDir, 'frame_%05d.jpg'),
    ], { maxBuffer: 10 * 1024 * 1024 }, (error) => {
      if (error) reject(new Error(`ffmpeg frame extraction failed: ${error.message}`));
      else resolve();
    });

    let stderr = '';
    child.stderr?.on('data', async (chunk) => {
      stderr += chunk.toString();
      const lines = stderr.split(/\r?\n/);
      stderr = lines.pop() || '';
      for (const line of lines) {
        const match = line.match(/^frame=(\d+)$/);
        if (!match) continue;
        const count = Number(match[1]);
        if (count <= job.frameCount) continue;
        job.frameCount = count;
        // Sixty frames is the normal 30-second target. Cap extraction at 20%
        // so longer videos cannot consume progress reserved for reconstruction.
        job.progress = Math.min(19, 1 + (count / 60) * 18);
        job.message = `Extracted ${count} JPEG frame${count === 1 ? '' : 's'}...`;
        await persistJob(job).catch(() => {});
      }
    });
  });

  const frames = (await readdir(frameDir))
    .filter(name => /^frame_\d{5}\.jpg$/i.test(name))
    .sort();
  if (frames.length === 0) throw new Error('ffmpeg did not extract any frames');

  job.frameCount = frames.length;
  job.progress = 20;
  job.message = `Extracted ${frames.length} JPEG frames`;
  await persistJob(job);
}

async function processJob(job) {
  let context;
  let progressInterval;

  try {
    await extractFrames(job);

    const b = await getBrowser();
    context = await b.newContext();
    const page = await context.newPage();

    // Navigate to the worker page served by our Express server
    await page.goto(`http://localhost:${PORT}/worker.html`, { waitUntil: 'domcontentloaded' });

    // Wait for the worker to be ready
    await page.waitForFunction(
      () => window.__worker?.status === 'ready',
      null,
      { timeout: 2 * 60 * 1000 }
    );

    // The worker receives only the job ID and fetches its JPEG frames over HTTP.
    await page.evaluate((jobId) => { window.__process(jobId); }, job.id);

    // Poll for progress
    progressInterval = setInterval(async () => {
      try {
        const state = await page.evaluate(() => window.__worker);
        if (state) {
          job.progress = state.progress;
          job.stage = state.stage;
          job.message = state.message;
          job.metrics = state.metrics;
          job.log = state.log;
          await persistJob(job);
        }
      } catch {}
    }, 2000);

    // Wait for completion (up to 30 min)
    await page.waitForFunction(
      () => window.__worker?.status === 'done' || window.__worker?.status === 'error',
      null,
      { timeout: 30 * 60 * 1000 }
    );

    clearInterval(progressInterval);
    progressInterval = null;

    const finalState = await page.evaluate(() => window.__worker);

    if (finalState.status === 'error') {
      job.status = 'failed';
      job.error = finalState.error;
      job.message = `Failed: ${finalState.error}`;
      job.log = finalState.log;
      await persistJob(job);
      return;
    }

    // Extract PLY
    if (finalState.plyBase64) {
      const plyPath = join(PLY_DIR, `${job.id}.ply`);
      const plyBuf = Buffer.from(finalState.plyBase64, 'base64');
      await writeFile(plyPath, plyBuf);
      job.plyPath = plyPath;
      job.plySize = plyBuf.length;
    }

    job.status = 'completed';
    job.progress = 100;
    job.message = 'Complete';
    job.metrics = finalState.metrics;
    job.log = finalState.log;
    job.completedAt = Date.now();
    await persistJob(job);

    console.log(`Job ${job.id} completed. PLY: ${(job.plySize / 1024 / 1024).toFixed(1)} MB`);

  } catch (err) {
    console.error(`Job ${job.id} failed:`, err.message);
    job.status = 'failed';
    job.error = err.message;
    job.message = `Failed: ${err.message}`;
    await persistJob(job);
  } finally {
    if (progressInterval) clearInterval(progressInterval);
    await context?.close().catch(() => {});
    busy = false;
    processQueue();
  }
}

function processQueue() {
  if (busy || queue.length === 0) return;
  busy = true;
  const job = queue.shift();
  processJob(job);
}

function enqueueJob(job) {
  job.status = 'queued';
  job.queuePosition = queue.length;
  persistJob(job);
  queue.push(job);
  processQueue();
  return job;
}

// ── Express API ──
const app = express();
const PORT = Number(process.env.PORT ?? 3457);

app.use(cors());
app.use(express.json());

// Serve static files (worker page + splat.js source)
app.use('/src', express.static(join(ROOT, 'src')));
app.use('/telosview', express.static(join(ROOT, 'telosview')));
app.use('/frames', express.static(FRAMES_DIR));
app.get('/worker.html', (_req, res) => res.sendFile(join(ROOT, 'server', 'worker.html')));

// Health check
app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', browser: browser?.isConnected() || false, queue: queue.length, busy });
});

// Upload video and start processing
app.post('/api/projects', upload.single('video'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No video file provided' });

    const name = req.body.name || 'Untitled';
    const id = randomUUID();

    // Move uploaded file to a stable path
    const ext = req.file.originalname.match(/\.[^.]+$/)?.[0] || '.mp4';
    const videoPath = join(UPLOADS_DIR, `${id}${ext}`);
    await readFile(req.file.path).then(buf => writeFile(videoPath, buf));
    await unlink(req.file.path).catch(() => {});

    const job = {
      id,
      name,
      fileName: req.file.originalname,
      videoPath,
      status: 'pending',
      stage: '',
      progress: 0,
      message: 'Queued for processing...',
      metrics: null,
      error: null,
      plyPath: null,
      plySize: null,
      log: [],
      createdAt: Date.now(),
      completedAt: null,
    };

    await persistJob(job);
    enqueueJob(job);

    res.json({ id: job.id, name: job.name, status: job.status, queuePosition: job.queuePosition });
  } catch (err) {
    console.error('Upload error:', err);
    res.status(500).json({ error: err.message });
  }
});

// Get job status
app.get('/api/projects/:id', async (req, res) => {
  const job = readJob(req.params.id);
  if (!job) {
    // Try reading from disk
    try {
      const data = await readFile(join(JOBS_DIR, `${req.params.id}.json`), 'utf-8');
      const parsed = JSON.parse(data);
      jobs.set(parsed.id, parsed);
      return res.json(formatJob(parsed));
    } catch {
      return res.status(404).json({ error: 'Project not found' });
    }
  }
  res.json(formatJob(job));
});

// List the extracted JPEGs for a browser worker. Static URLs keep the binary
// image data out of job JSON and Playwright's page.evaluate boundary.
app.get('/api/projects/:id/frames', async (req, res) => {
  const job = readJob(req.params.id);
  if (!job) return res.status(404).json({ error: 'Project not found' });
  if (!job.frameDir) return res.status(409).json({ error: 'Frames are not ready' });

  try {
    const names = (await readdir(job.frameDir))
      .filter(name => /^frame_\d{5}\.jpg$/i.test(name))
      .sort();
    res.json({
      count: names.length,
      frames: names.map(name => ({
        name,
        url: `/frames/${encodeURIComponent(job.id)}/${encodeURIComponent(name)}`,
      })),
    });
  } catch {
    res.status(409).json({ error: 'Frames are not ready' });
  }
});

// SSE stream for real-time updates
app.get('/api/projects/:id/events', async (req, res) => {
  const id = req.params.id;
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
  });

  const interval = setInterval(() => {
    const job = readJob(id);
    if (job) {
      res.write(`data: ${JSON.stringify(formatJob(job))}\n\n`);
      if (job.status === 'completed' || job.status === 'failed') {
        clearInterval(interval);
        res.end();
      }
    }
  }, 500);

  req.on('close', () => clearInterval(interval));
});

// Download PLY
app.get('/api/projects/:id/ply', async (req, res) => {
  const job = readJob(req.params.id);
  if (!job || !job.plyPath) return res.status(404).json({ error: 'PLY not available' });
  res.download(job.plyPath, `${job.name.replace(/\s+/g, '_')}.ply`);
});

// List all projects
app.get('/api/projects', async (_req, res) => {
  try {
    const files = await readdir(JOBS_DIR);
    const allJobs = [];
    for (const f of files) {
      if (!f.endsWith('.json')) continue;
      const data = JSON.parse(await readFile(join(JOBS_DIR, f), 'utf-8'));
      allJobs.push(formatJob(data));
    }
    allJobs.sort((a, b) => b.createdAt - a.createdAt);
    res.json(allJobs);
  } catch {
    res.json([]);
  }
});

function formatJob(job) {
  return {
    id: job.id,
    name: job.name,
    fileName: job.fileName,
    status: job.status,
    stage: job.stage,
    progress: Math.round(job.progress || 0),
    message: job.message,
    metrics: job.metrics,
    error: job.error,
    hasPly: !!job.plyPath,
    plySize: job.plySize,
    createdAt: job.createdAt,
    completedAt: job.completedAt,
    log: job.log?.slice(-20) || [],
  };
}

// Error handler
app.use((err, _req, res, _next) => {
  console.error(err);
  if (err instanceof multer.MulterError) {
    return res.status(400).json({ error: `Upload error: ${err.message}` });
  }
  res.status(500).json({ error: err.message });
});

app.listen(PORT, () => {
  console.log(`TelosView server running on http://localhost:${PORT}`);
  console.log(`Serving splat.js source from ${ROOT}/src`);
  console.log(`Jobs directory: ${JOBS_DIR}`);
  // Pre-warm browser
  getBrowser().then(() => console.log('Browser ready')).catch(e => console.warn('Browser init deferred:', e.message));
});
